"""Final verdict metrics, explicit human identities, and leave-one-out baselines."""

from __future__ import annotations

import json

import pytest

from giuriallm.dataset import (
    DatasetError,
    HumanAnnotations,
    load_human_annotations,
    load_human_labels,
)
from giuriallm.metaeval import (
    HIGH,
    LOW,
    Alignment,
    build_alignment,
    build_human_baseline,
    evaluate_alignment,
)
from giuriallm.spec import Metric, parse_config
from giuriallm.types import CriterionScore, ItemResult, Judgement, Variant, Verdict


def config_for(scale=None, metrics=None, threshold=0.8):
    return parse_config(
        {
            "judge": {
                "type": "pointwise",
                "criteria": [
                    {
                        "name": "quality",
                        "definition": "quality",
                        "scale": scale or {"kind": "likert", "min": 0, "max": 4},
                    }
                ],
                "outputs": {"result": "graded"},
            },
            "dataset": {"path": "items.jsonl"},
            "meta_evaluation": {
                "human_labels": "human.jsonl",
                "metrics": metrics or ["error", "spearman", "earth_movers_distance"],
                "agreement_threshold": threshold,
            },
        }
    )


def score_verdict(score):
    return Verdict(scores=[CriterionScore(criterion="quality", score=score)])


def judgement(score, model="a"):
    return Judgement(
        item_id="i", model=model, sample=0, variant=Variant(order=[0]), verdict=score_verdict(score)
    )


def test_point_metrics_use_final_aggregate_but_emd_uses_member_distribution():
    config = config_for()
    result = ItemResult(
        item_id="i",
        aggregate=score_verdict(2),
        judgements=[judgement(0), judgement(4, "b")],
    )
    alignment = build_alignment(config, [result], {"i": [2]})
    report = evaluate_alignment(alignment, [Metric.ERROR, Metric.EARTH_MOVERS_DISTANCE])
    assert report.targets[0].metrics == pytest.approx({"error": 0, "earth_movers_distance": 0.5})
    # The final point may differ from the raw-distribution mean.
    result.aggregate = score_verdict(4)
    report = evaluate_alignment(build_alignment(config, [result], {"i": [2]}), [Metric.ERROR])
    assert report.targets[0].metrics["error"] == pytest.approx(0.5)


def test_choice_metrics_use_derived_selection_not_raw_mode():
    class SelectedResult(ItemResult):
        selected_candidate: int | None = None

    config = parse_config(
        {
            "judge": {
                "type": "listwise",
                "criteria": [{"name": "quality", "definition": "quality"}],
                "outputs": {"result": "ranking"},
            },
            "dataset": {"path": "items.jsonl", "fields": {"candidates": ["a", "b", "c"]}},
        }
    )
    result = SelectedResult(
        item_id="i",
        selected_candidate=1,
        judgements=[
            Judgement(
                item_id="i",
                model="a",
                sample=0,
                variant=Variant(order=[0, 1, 2]),
                verdict=Verdict(ranking=[0, 1, 2]),
            )
        ],
    )
    report = evaluate_alignment(build_alignment(config, [result], {"i": [1]}), [Metric.ERROR])
    assert report.targets[0].metrics["error"] == 0


def test_error_uses_configured_continuous_scale_range():
    config = config_for({"kind": "continuous", "min": 10, "max": 30})
    result = ItemResult(item_id="i", aggregate=score_verdict(20), judgements=[judgement(20)])
    report = evaluate_alignment(build_alignment(config, [result], {"i": [10]}), [Metric.ERROR])
    assert report.targets[0].metrics["error"] == 0.5


def test_existing_alignment_without_point_map_still_works():
    alignment = Alignment(
        target="quality", numeric=True, support=[0, 4], judge={"i": [0, 4]}, human={"i": [0]}
    )
    report = evaluate_alignment([alignment], [Metric.ERROR])
    assert report.targets[0].metrics["error"] == 0.5


def test_fractional_member_means_are_not_dropped_from_distributions():
    alignment = Alignment(
        target="quality",
        numeric=True,
        support=[0, 1, 2, 3, 4],
        judge={"i": [0.5, 2]},
        human={"i": [2]},
    )
    report = evaluate_alignment([alignment], [Metric.EARTH_MOVERS_DISTANCE, Metric.JENSEN_SHANNON])
    # Half the mass moves 1.5 units on the configured four-unit scale.
    assert report.targets[0].metrics["earth_movers_distance"] == pytest.approx(0.5 * 1.5 / 4)
    assert report.targets[0].metrics["jensen_shannon"] == pytest.approx(0.31127812445913283)


def test_fractional_human_means_retain_configured_emd_normalization():
    alignment = Alignment(
        target="quality",
        numeric=True,
        support=[1, 2, 3, 4, 5],
        judge={"i": [2.5]},
        human={"i": [3.5]},
    )
    report = evaluate_alignment([alignment], [Metric.EARTH_MOVERS_DISTANCE])
    assert report.targets[0].metrics["earth_movers_distance"] == pytest.approx(0.25)


def test_binary_is_mismatch_not_absolute_error_to_mean():
    config = config_for({"kind": "binary"}, ["error", "jensen_shannon"], 1)
    result = ItemResult(item_id="i", aggregate=score_verdict(1), judgements=[judgement(1)])
    report = evaluate_alignment(build_alignment(config, [result], {"i": [0, 1]}), [Metric.ERROR])
    assert report.targets[0].metrics["error"] == 0


def test_human_labels_backwards_compatible_and_optional_ids(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text('{"id":"i","labels":[0,1]}\n{"id":"i","label":2}\n')
    assert load_human_labels(path) == {"i": [0, 1, 2]}
    assert load_human_annotations(path)["i"].annotators is None
    path.write_text('{"id":"i","labels":[0,1],"annotators":["a","b"]}\n')
    assert load_human_annotations(path)["i"] == HumanAnnotations([0, 1], ["a", "b"])


@pytest.mark.parametrize("annotators", [["a"], ["a", "a"], ["a", ""], "a b", [None, "b"]])
def test_invalid_annotator_metadata_is_rejected(tmp_path, annotators):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps({"id": "i", "labels": [0, 1], "annotators": annotators}))
    with pytest.raises(DatasetError):
        load_human_annotations(path)


def test_duplicate_annotator_across_rows_rejected(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"id":"i","labels":[0],"annotators":["a"]}\n{"id":"i","labels":[1],"annotators":["a"]}\n'
    )
    with pytest.raises(DatasetError, match="duplicate annotator"):
        load_human_annotations(path)


def test_loo_has_no_minimum_count_and_only_uses_run_item_ids():
    annotations = {
        "i": HumanAnnotations([0, 4], ["a", "b"]),
        "outside": HumanAnnotations([0, 0], ["outsider", "a"]),
    }
    report = build_human_baseline(config_for(), ["i"], annotations)
    target = report.targets[0]
    assert report.available
    assert report.n_items == 1
    assert target.n_annotators == 2
    assert set(target.per_annotator) == {"a", "b"}
    assert target.metrics["error"] == 1
    assert target.metric_annotators["error"] == 2
    assert target.metrics["spearman"] is None
    assert target.metric_annotators["spearman"] == 0


def test_loo_two_item_spearman_is_defined():
    annotations = {
        "i": HumanAnnotations([0, 1], ["a", "b"]),
        "j": HumanAnnotations([4, 3], ["a", "b"]),
    }
    target = build_human_baseline(config_for(), ["i", "j"], annotations).targets[0]
    assert target.metrics["spearman"] == 1
    assert target.metric_annotators["spearman"] == 2


def test_loo_strata_are_fixed_and_use_total_annotator_weights():
    annotations = {
        "i": HumanAnnotations([0, 0, 0], ["a", "b", "c"]),
        "j": HumanAnnotations([0, 0, 4], ["a", "b", "c"]),
        "k": HumanAnnotations([4, 4], ["a", "d"]),
    }
    target = build_human_baseline(config_for(), list(annotations), annotations).targets[0]
    assert target.strata[HIGH]["error"] == 0
    # a has 3 items, b and c each have 2; the low stratum has one item each.
    assert target.strata[LOW]["error"] == pytest.approx((3 * 0.5 + 2 * 0.5 + 2 * 1) / 7)
    assert target.per_annotator["c"].stratum_sizes[LOW] == 1
    assert target.per_annotator["c"].strata[LOW]["error"] == 1
    # Leaving c out makes j unanimous, but it must remain in the fixed low stratum.
    assert target.stratum_annotators[LOW]["error"] == 3


def test_loo_binary_ties_are_positive_and_degenerate_kappa_is_excluded():
    config = config_for({"kind": "binary"}, ["error", "cohen_kappa", "jensen_shannon"], 1)
    annotations = {"i": HumanAnnotations([0, 0, 1], ["a", "b", "c"])}
    target = build_human_baseline(config, ["i"], annotations).targets[0]
    assert target.metrics["error"] == 1
    annotations = {"i": HumanAnnotations([0, 0, 0], ["a", "b", "c"])}
    target = build_human_baseline(config, ["i"], annotations).targets[0]
    assert target.metrics["cohen_kappa"] is None
    assert target.metric_annotators["cohen_kappa"] == 0
    assert target.metrics["error"] == 0
    assert target.metric_annotators["error"] == 3
    assert target.metrics["jensen_shannon"] == 0


def test_loo_unavailable_without_explicit_ids():
    report = build_human_baseline(config_for(), ["i"], {"i": HumanAnnotations([0, 4])})
    assert not report.available
    assert report.notes


def test_error_metric_config_round_trip():
    config = config_for()
    assert (
        Metric.ERROR in parse_config(json.loads(config.model_dump_json())).meta_evaluation.metrics
    )
