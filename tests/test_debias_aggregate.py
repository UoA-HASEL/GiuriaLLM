"""Debiasing variants, flip-rate detection, and aggregation."""

from __future__ import annotations

import pytest

from giuriallm.aggregate import aggregate_item, bradley_terry, dawid_skene
from giuriallm.debias import flip_rate, flipped, position_bias, variants_for
from giuriallm.spec import parse_config
from giuriallm.types import CriterionScore, ItemResult, Judgement, Status, Variant, Verdict

PAIRWISE = {
    "judge": {
        "type": "pairwise",
        "criteria": [{"name": "correctness", "definition": "works?"}],
        "outputs": {"result": "best_choice"},
    },
    "dataset": {"path": "d.jsonl", "fields": {"candidates": ["gen_a", "gen_b"]}},
}

POINTWISE = {
    "judge": {
        "type": "pointwise",
        "criteria": [{"name": "correctness", "definition": "works?"}],
        "outputs": {"result": "graded"},
    },
    "dataset": {"path": "d.jsonl", "fields": {"candidates": ["candidate"]}},
    "debias": {"position_swap": False},
}


def choice_judgement(order, choice, model="m", sample=0) -> Judgement:
    return Judgement(
        item_id="i",
        model=model,
        sample=sample,
        variant=Variant(order=order, index=0 if order == sorted(order) else 1),
        verdict=Verdict(best_choice=choice),
    )


def test_position_swap_produces_both_orders():
    config = parse_config(PAIRWISE)
    variants = variants_for(config, 2)
    assert [v.order for v in variants] == [[0, 1], [1, 0]]


def test_position_swap_can_be_turned_off():
    config = parse_config({**PAIRWISE, "debias": {"position_swap": False}})
    assert [v.order for v in variants_for(config, 2)] == [[0, 1]]


def test_pointwise_has_a_single_variant():
    config = parse_config(POINTWISE)
    assert [v.order for v in variants_for(config, 1)] == [[0]]


def test_listwise_permutations_are_distinct_and_deterministic():
    config = parse_config(
        {
            "judge": {
                "type": "listwise",
                "criteria": [{"name": "c", "definition": "d"}],
                "outputs": {"result": "ranking"},
            },
            "dataset": {"path": "d.jsonl", "fields": {"candidates": ["a", "b", "c", "d"]}},
            "debias": {"position_swap": False, "permutations": 3},
        }
    )
    orders = [tuple(v.order) for v in variants_for(config, 4)]
    assert len(set(orders)) == 3
    assert orders == [tuple(v.order) for v in variants_for(config, 4)]


def test_flip_detected_when_the_verdict_follows_the_position():
    """The judge picks whatever is shown first: candidate 0, then candidate 1."""
    judgements = [choice_judgement([0, 1], 0), choice_judgement([1, 0], 1)]
    assert flipped(judgements) is True


def test_no_flip_when_the_verdict_survives_the_swap():
    judgements = [choice_judgement([0, 1], 1), choice_judgement([1, 0], 1)]
    assert flipped(judgements) is False


def test_flip_is_unknown_without_a_second_order():
    assert flipped([choice_judgement([0, 1], 0)]) is None


def test_failed_judgements_do_not_count_as_agreement():
    bad = choice_judgement([1, 0], 0)
    bad.status = Status.PROVIDER_ERROR
    bad.verdict = None
    assert flipped([choice_judgement([0, 1], 0), bad]) is None


def test_flip_rate_over_items():
    stable = ItemResult(item_id="a", flipped=False)
    unstable = ItemResult(item_id="b", flipped=True)
    unknown = ItemResult(item_id="c", flipped=None)
    assert flip_rate([stable, unstable, unknown]) == pytest.approx(0.5)


def test_position_bias_counts_shown_positions():
    result = ItemResult(
        item_id="i",
        judgements=[choice_judgement([0, 1], 0), choice_judgement([1, 0], 1)],
    )
    bias = position_bias([result])
    assert bias == {"position_1": 1.0}


def test_majority_vote_across_samples():
    config = parse_config(PAIRWISE)
    judgements = [
        choice_judgement([0, 1], 1, sample=0),
        choice_judgement([1, 0], 1, sample=1),
        choice_judgement([0, 1], 0, sample=2),
    ]
    verdict, disagreement = aggregate_item(config, judgements)
    assert verdict.best_choice == 1
    assert disagreement == pytest.approx(1 / 3)


def test_graded_aggregation_averages_and_reports_spread():
    config = parse_config(POINTWISE)
    judgements = [
        Judgement(
            item_id="i",
            model="m",
            sample=i,
            variant=Variant(order=[0]),
            verdict=Verdict(scores=[CriterionScore(criterion="correctness", score=score)]),
        )
        for i, score in enumerate([3.0, 4.0, 5.0])
    ]
    verdict, disagreement = aggregate_item(config, judgements)
    assert verdict.score_for("correctness") == pytest.approx(4.0)
    # pstdev([3,4,5]) = 0.8165, normalised by the 1-5 scale range of 4.
    assert disagreement == pytest.approx(0.8165 / 4, rel=1e-3)


def test_aggregation_ignores_failed_judgements():
    config = parse_config(PAIRWISE)
    good = choice_judgement([0, 1], 0)
    bad = choice_judgement([0, 1], 1, sample=1)
    bad.status = Status.PARSE_ERROR
    bad.verdict = None
    verdict, _ = aggregate_item(config, [good, bad])
    assert verdict.best_choice == 0


def test_aggregation_returns_none_when_everything_failed():
    config = parse_config(PAIRWISE)
    bad = choice_judgement([0, 1], 1)
    bad.status = Status.PROVIDER_ERROR
    bad.verdict = None
    verdict, disagreement = aggregate_item(config, [bad])
    assert verdict is None and disagreement is None


def test_bradley_terry_orders_by_win_rate():
    comparisons = [("a", "b")] * 8 + [("b", "a")] * 2 + [("b", "c")] * 7 + [("c", "b")] * 3
    strengths = bradley_terry(comparisons)
    assert strengths["a"] > strengths["b"] > strengths["c"]
    assert sum(strengths.values()) == pytest.approx(len(strengths))


def test_dawid_skene_recovers_labels_and_ranks_annotators():
    truth = {f"i{i}": "0" if i % 2 == 0 else "1" for i in range(12)}
    labels = {}
    for i, (item, true_label) in enumerate(truth.items()):
        other = "1" if true_label == "0" else "0"
        labels[item] = {
            "reliable": true_label,
            "also_reliable": true_label,
            "noisy": true_label if i % 3 else other,
        }
    inferred, accuracy = dawid_skene(labels)
    assert inferred == truth
    assert accuracy["reliable"] > accuracy["noisy"]
