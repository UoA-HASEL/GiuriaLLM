"""Evaluation views reuse calls and distinguish order votes from panel votes."""

import pytest

from giuriallm.aggregate.evaluation import distribution_view, evaluation_views
from giuriallm.engine import Engine
from giuriallm.metaeval import build_alignment
from giuriallm.providers import LLMResponse, Provider, RawUsage
from giuriallm.spec import parse_config
from giuriallm.types import (
    CriterionScore,
    EvaluationItem,
    ItemResult,
    Judgement,
    Status,
    Usage,
    Variant,
    Verdict,
)


def config_for(
    tmp_path, kind="best_choice", aggregation="majority_vote", panel=True, weights=None, samples=1
):
    scale = {"kind": "binary"} if kind == "binary" else {"kind": "likert", "min": 0, "max": 4}
    graded = kind in {"binary", "graded"}
    model = {"provider": "mock", "id": "a", "label": "a", "samples": samples}
    data = {
        "judge": {
            "type": "pointwise" if graded else "listwise",
            "criteria": [{"name": "quality", "definition": "quality", "scale": scale}],
            "outputs": {"result": "graded" if graded else kind, "explanation": False},
        },
        "model": model,
        "dataset": {
            "path": str(tmp_path / "unused.jsonl"),
            "fields": {"candidates": ["candidate"] if graded else ["a", "b", "c"]},
        },
        "engine": {"cache": False, "cache_dir": str(tmp_path / "cache")},
        "debias": {"permutations": 1 if graded else 3},
    }
    if panel:
        data["panel"] = {
            "members": [dict(model, id=m, label=m) for m in "abc"],
            "aggregation": aggregation,
            "weights": weights or {},
        }
    return parse_config(data)


def judgement(model, order, value, kind="best_choice", sample=0, ok=True):
    if kind == "ranking":
        verdict = Verdict(ranking=value)
    elif kind in {"binary", "graded"}:
        verdict = Verdict(scores=[CriterionScore(criterion="quality", score=value)])
    else:
        verdict = Verdict(best_choice=value)
    return Judgement(
        item_id="i",
        model=model,
        sample=sample,
        variant=Variant(index=order, order=[[0, 1, 2], [2, 1, 0], [1, 2, 0]][order]),
        verdict=verdict if ok else None,
        status=Status.OK if ok else Status.PARSE_ERROR,
        usage=Usage(input_tokens=10, output_tokens=2, latency_s=3, cost_usd=0.1),
    )


def views(config, js):
    return evaluation_views(config, ItemResult(item_id="i", judgements=js))


def test_panel_votes_per_order_before_combining_orders(tmp_path):
    config = config_for(tmp_path)
    # Pooled votes favour 1 by 5:4; order winners are 0, 0, 1.
    js = [
        judgement(m, order, value)
        for order, values in enumerate([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
        for m, value in zip("abc", values, strict=True)
    ]
    result = views(config, js)["panel"]
    assert result.order_choices == {0: 0, 1: 0, 2: 1}
    assert result.selected_candidate == 0


def test_binary_majority_is_label_not_mean(tmp_path):
    config = config_for(tmp_path, "binary")
    result = views(
        config,
        [judgement(m, 0, value, "binary") for m, value in zip("abc", [0, 1, 1], strict=True)],
    )
    assert result["panel"].aggregate.score_for("quality") == 1
    assert result["a"].aggregate.score_for("quality") == 0


def test_ranking_top1_flip_ignores_other_rank_changes_and_preserves_raw(tmp_path):
    config = config_for(tmp_path, "ranking")
    js = [
        judgement(m, order, rank, "ranking")
        for m in "abc"
        for order, rank in enumerate([[0, 1, 2], [0, 2, 1]])
    ]
    before = [j.model_dump() for j in js]
    result = views(config, js)
    for view in result.values():
        assert view.selected_candidate == 0
        assert view.flipped is False
        assert view.aggregate is None  # No fabricated complete panel ranking.
    assert [j.model_dump() for j in js] == before


def test_panel_tie_prefers_first_configured_member_not_input_order(tmp_path):
    config = config_for(tmp_path)
    result = views(config, [judgement("c", 0, 2), judgement("b", 0, 1), judgement("a", 0, 0)])
    assert result["panel"].selected_candidate == 0


def test_cross_order_tie_prefers_original_choice(tmp_path):
    config = config_for(tmp_path)
    result = views(
        config, [judgement(m, order, value) for m in "abc" for order, value in [(1, 0), (0, 2)]]
    )
    assert all(view.selected_candidate == 2 for view in result.values())


def test_member_can_flip_while_panel_is_stable(tmp_path):
    config = config_for(tmp_path)
    js = [
        judgement(m, order, 1 if m == "a" and order == 1 else 0) for m in "abc" for order in [0, 1]
    ]
    result = views(config, js)
    assert result["a"].flipped is True
    assert result["panel"].flipped is False


def test_incomplete_panel_order_not_counted_as_vote_or_flip(tmp_path):
    config = config_for(tmp_path)
    js = [
        judgement(m, order, order, ok=not (m == "c" and order == 1))
        for m in "abc"
        for order in [0, 1]
    ]
    result = views(config, js)
    assert result["panel"].order_choices == {0: 0}
    assert result["panel"].flipped is None
    assert result["a"].flipped is True
    assert result["c"].flipped is None
    assert result["panel"].usage.total_tokens == 72
    assert result["panel"].usage.latency_s == 18


def test_no_complete_panel_order_has_no_final_candidate(tmp_path):
    result = views(config_for(tmp_path), [judgement("a", 0, 0), judgement("b", 0, 0)])
    assert result["panel"].selected_candidate is None
    assert result["panel"].aggregate is None
    assert result["panel"].flipped is None


def test_invalid_panel_cannot_enter_metrics_via_partial_member_distribution(tmp_path):
    config = config_for(tmp_path)
    result = views(config, [judgement("a", 0, 0), judgement("b", 0, 0)])
    metric_view = distribution_view(result["panel"], [result[m] for m in "abc"])
    alignment = build_alignment(config, [metric_view], {"i": [0]})[0]
    assert alignment.items() == []


def test_self_consistency_variation_is_not_position_flip(tmp_path):
    config = config_for(tmp_path, panel=False, samples=3)
    result = views(config, [judgement("a", 0, v, sample=s) for s, v in enumerate([0, 1, 1])])["a"]
    assert result.selected_candidate == 1
    assert result.flipped is None


def test_distribution_uses_one_final_score_per_member_not_all_calls(tmp_path):
    config = config_for(tmp_path, "graded", aggregation="mean", samples=2)
    js = [
        judgement(m, 0, score, "graded", sample=s)
        for m, scores in zip("abc", [[0, 2], [2, 4], [4, 4]], strict=True)
        for s, score in enumerate(scores)
    ]
    result = views(config, js)
    distribution = distribution_view(result["panel"], [result[m] for m in "abc"])
    assert [j.verdict.score_for("quality") for j in distribution.judgements] == [1, 3, 4]
    assert result["panel"].aggregate.score_for("quality") == pytest.approx(8 / 3)
    assert len(result["panel"].judgements) == 6


def test_existing_weighted_choice_is_not_silently_unweighted(tmp_path):
    config = config_for(tmp_path, aggregation="weighted", weights={"a": 4, "b": 1, "c": 1})
    result = views(
        config, [judgement(m, 0, value) for m, value in zip("abc", [0, 1, 1], strict=True)]
    )
    assert result["panel"].aggregate.best_choice == 0


def test_nonmajority_ranking_preserves_borda_and_uses_its_winner(tmp_path):
    config = config_for(tmp_path, "ranking", aggregation="mean")
    js = [
        judgement(m, 0, ranking, "ranking")
        for m, ranking in zip("abc", [[0, 1, 2], [1, 2, 0], [2, 1, 0]], strict=True)
    ]
    result = views(config, js)["panel"]
    # Top-choice plurality ties and would select member a's 0; Borda selects 1.
    assert result.aggregate.ranking == [1, 2, 0]
    assert result.selected_candidate == 1
    assert result.order_choices == {0: 1}


class CountingProvider(Provider):
    name = "mock"

    def __init__(self):
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return LLMResponse(
            payload={"best_choice": "A"},
            model=request.model.id,
            usage=RawUsage(input_tokens=10, output_tokens=2),
        )


async def test_engine_reports_members_and_panel_without_extra_calls(tmp_path):
    config = config_for(tmp_path)
    provider = CountingProvider()
    engine = Engine(config, provider=provider)
    try:
        result = await engine.run([EvaluationItem(id="i", candidates=["a", "b", "c"])])
    finally:
        await engine.aclose()
    assert provider.calls == 9  # Three members x three orders, no panel API call.
    assert set(result.evaluations) == {"a", "b", "c", "panel"}
    assert result.evaluations["panel"].total_tokens == 108
    assert all(result.evaluations[m].total_tokens == 36 for m in "abc")
    panel = result.evaluations["panel"]
    assert panel.latency_per_item_mean_s == pytest.approx(
        sum(result.evaluations[m].latency_per_item_mean_s for m in "abc")
    )
    assert panel.usage.latency_s == pytest.approx(result.usage.latency_s)
