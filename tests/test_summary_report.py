"""Default reports stay compact, task-independent and faithful to saved results."""

import json
from copy import deepcopy

import pytest

from giuriallm.engine import Engine, write_run
from giuriallm.metaeval import Alignment, evaluate_alignment
from giuriallm.providers.mock import MockProvider
from giuriallm.report import render_json, render_report, write_reports
from giuriallm.report.saved import regenerate_reports, saved_manifest
from giuriallm.report.summary import per_item_usage, render_summary, report_config
from giuriallm.spec import Metric, parse_config
from giuriallm.types import EvaluationItem


def configuration(tmp_path, kind="pointwise", panel=False):
    candidates = {"pointwise": ["candidate"], "pairwise": ["a", "b"], "listwise": ["a", "b", "c"]}[
        kind
    ]
    raw = {
        "judge": {
            "type": kind,
            "criteria": [{"name": "score", "definition": "quality"}],
            "outputs": {
                "result": {"pointwise": "graded", "pairwise": "best_choice", "listwise": "ranking"}[
                    kind
                ]
            },
        },
        "dataset": {"path": str(tmp_path / "data.jsonl"), "fields": {"candidates": candidates}},
        "model": {"provider": "mock", "id": "model", "label": "One"},
        "engine": {"cache": False, "cache_dir": str(tmp_path / "cache")},
        "report": {"formats": ["json", "markdown"]},
    }
    if panel:
        raw["panel"] = {
            "aggregation": "majority_vote",
            "members": [
                {"provider": "mock", "id": f"model-{i}", "label": f"Model {i}"} for i in range(5)
            ],
        }
    return parse_config(raw)


@pytest.mark.parametrize("kind", ["pointwise", "pairwise", "listwise"])
@pytest.mark.parametrize("panel", [False, True])
async def test_one_template_all_task_types_and_complete_json(tmp_path, kind, panel):
    config = configuration(tmp_path, kind, panel)
    count = len(config.dataset.fields.candidates)
    result = await Engine(config, MockProvider()).run(
        [EvaluationItem(id="do-not-print-this-id", candidates=["code"] * count)]
    )
    before = result.model_dump()
    markdown = render_report(config, result, markdown=True)
    console = render_report(config, result)
    assert [line for line in markdown.splitlines() if line.startswith("## ")] == [
        "## Run summary",
        "## Judges",
        "## Evaluation",
        "## Resource use",
        "## Notes",
    ]
    assert "do-not-print-this-id" not in markdown + console
    assert "Input tokens (tokens/item)" in markdown + console
    payload = json.loads(render_json(config, result))
    item = payload["items"][0]
    assert set(item) == {
        "item_id", "aggregate", "selected_candidate", "flipped", "n_judgements", "n_failed"
    }
    primary = result.results[0]
    assert item["aggregate"] == (primary.aggregate.model_dump() if primary.aggregate else None)
    assert item["selected_candidate"] == primary.selected_candidate
    assert item["flipped"] == primary.flipped
    assert item["n_judgements"] == len(primary.judgements)
    assert item["n_failed"] == 0
    assert len(payload["evaluations"]) == (6 if panel else 1)
    assert "report_config" not in payload
    assert all("usage_per_item" not in value for value in payload["evaluations"].values())
    assert result.model_dump() == before


def test_per_item_denominator_includes_failed_items_and_zero_is_undefined():
    summary = {
        "n_items": 100,
        "n_valid": 90,
        "total_tokens": 350,
        "usage": {
            "input_tokens": 250,
            "output_tokens": 100,
            "cost_usd": 0.1,
            "energy_wh": 0.14,
            "latency_s": 200,
        },
    }
    before = deepcopy(summary)
    usage = per_item_usage(summary)
    assert usage == pytest.approx(
        {
            "input_tokens": 2.5,
            "output_tokens": 1.0,
            "total_tokens": 3.5,
            "cost_usd": 0.001,
            "energy_wh": 0.0014,
            "latency_s": 2.0,
        }
    )
    assert summary == before
    assert all(v is None for v in per_item_usage({**summary, "n_items": 0}).values())


def test_annotation_groups_require_multiple_labels_not_annotator_ids():
    alignment = Alignment(
        target="choice",
        numeric=False,
        support=["0", "1"],
        judge={"single": ["0"], "multi": ["1"]},
        human={"single": ["0"], "multi": ["1", "1"]},
    )
    target = evaluate_alignment([alignment], [Metric.ERROR]).targets[0]
    assert target.metrics == {"error": 0.0}
    assert target.stratum_sizes == {"high_agreement": 1, "low_agreement": 0}
    assert target.n_ungrouped == 1
    assert target.stratum_unavailable["low_agreement"]["error"] == "empty_group"


def test_null_metrics_have_specific_reason_codes():
    alignment = Alignment(
        target="score",
        numeric=True,
        support=[0.0, 1.0],
        judge={"one": [0.0]},
        human={"one": [1.0, 1.0]},
    )
    target = evaluate_alignment([alignment], [Metric.SPEARMAN]).targets[0]
    assert target.unavailable["spearman"] == "insufficient_pairs"
    alignment.judge["two"], alignment.human["two"] = [0.0], [1.0, 1.0]
    target = evaluate_alignment([alignment], [Metric.SPEARMAN]).targets[0]
    assert target.unavailable["spearman"] == "constant_values"


def test_endpoint_credentials_are_not_reported(tmp_path):
    config = configuration(tmp_path)
    config.model.options["base_url"] = "https://name:secret@example.org/v1?key=hidden#token"
    metadata = report_config(config)
    text = json.dumps(metadata)
    assert metadata["judges"][0]["endpoint"] == "https://example.org/v1"
    assert all(secret not in text for secret in ("secret", "hidden", "token"))


def test_wide_tables_split_without_losing_judges_or_values():
    evaluations = {
        f"provider:long-model-name-{i}": {
            "n_items": 100,
            "n_valid": 100,
            "n_failed": 0,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 100,
                "cost_usd": 1e-10,
                "energy_wh": 0.08,
                "latency_s": 1,
            },
            "total_tokens": 200,
            "agreement": {
                "targets": [{"target": "score", "n_items": 100, "metrics": {"error": i / 10}}]
            },
        }
        for i in range(8)
    }
    manifest = {"evaluations": evaluations}
    before = deepcopy(manifest)
    text = render_summary(manifest, width=40)
    assert max(map(len, text.splitlines())) <= 40
    assert "provider:long-model-name-7" in text and "0.7000" in text and "1e-12" in text
    assert "Judge 8" not in text
    assert manifest == before


async def test_100_items_do_not_expand_default_report_and_saved_refresh_is_offline(tmp_path):
    config = configuration(tmp_path)
    result = await Engine(config, MockProvider()).run(
        [EvaluationItem(id=f"hidden-item-{i}", candidates=["code"]) for i in range(100)]
    )
    run_dir = write_run(config, result, tmp_path / "runs")
    write_reports(config, result, run_dir)
    original = json.loads((run_dir / "results.json").read_text())
    protected = {
        name: (run_dir / name).read_bytes()
        for name in ("manifest.json", "config.yaml", "judgements.jsonl")
    }
    regenerate_reports(run_dir)
    updated = json.loads((run_dir / "results.json").read_text())
    assert updated == original
    assert all((run_dir / name).read_bytes() == contents for name, contents in protected.items())
    assert "hidden-item" not in (run_dir / "console.txt").read_text()
    assert len(updated["items"]) == 100
    assert render_summary(saved_manifest(run_dir)) == (run_dir / "console.txt").read_text()

    # Previously expanded files compact back to the same results and readable reports.
    reports = {name: (run_dir / name).read_bytes() for name in ("report.md", "console.txt")}
    expanded = deepcopy(original)
    expanded["report_config"] = report_config(config)
    for evaluation in expanded["evaluations"].values():
        evaluation["usage_per_item"] = per_item_usage(evaluation)
    for item in expanded["items"]:
        item.update(judges={"One": {"orders": {"0": {}}}}, orders={}, disagreement=0.0)
    (run_dir / "results.json").write_text(json.dumps(expanded))
    regenerate_reports(run_dir)
    assert json.loads((run_dir / "results.json").read_text()) == original
    for name, contents in {**protected, **reports}.items():
        assert (run_dir / name).read_bytes() == contents


def test_missing_and_empty_data_still_have_default_sections():
    text = render_summary({"counts": {"items": 0}, "evaluations": {}}, markdown=True)
    assert text.count("\n## ") == 5
    assert "No evaluation metrics configured." in text


def test_disabled_flip_reporting_is_not_reenabled_by_available_orders():
    manifest = {
        "report_config": {"report_flip_rate": False},
        "evaluations": {"Judge": {"n_items": 1, "n_flip_items": 1, "flip_rate": None}},
    }
    text = render_summary(manifest)
    assert "Flip rate" not in text
    assert "Items checked for flips" not in text


def test_model_names_separate_groups_empty_group_and_counts():
    target = {
        "target": "correctness",
        "n_items": 1,
        "metrics": {"error": 0.0},
        "strata": {"high_agreement": {"error": 0.0}, "low_agreement": {"error": None}},
        "stratum_sizes": {"high_agreement": 1, "low_agreement": 0},
    }
    manifest = {
        "run_id": "example",
        "counts": {"items": 1, "judgements": 1, "failed_judgements": 0},
        "usage": {"total_tokens": 200, "cost_usd": 0.001},
        "report_config": {
            "judges": [
                {"label": "GPT", "id": "full-model-name", "endpoint": "https://example.org"}
            ],
            "error_definitions": {"correctness": "proportion of incorrect predictions."},
        },
        "evaluations": {
            "GPT": {
                "n_items": 1,
                "n_valid": 1,
                "n_failed": 0,
                "agreement": {"targets": [target]},
            }
        },
    }
    original = deepcopy(manifest)
    md = render_summary(manifest, markdown=True)
    assert "Total tokens: 200 \\| Estimated cost: $0.001" in md
    assert "| Metric | full-model-name |" in md
    assert "| Items | 1 |" in md
    assert "### Overall — correctness" in md
    assert "### High human agreement" in md
    assert "### Low human agreement\n\nNo items in this group." in md
    assert md.count("| Number of items |") == 2
    assert "Matched items" not in md and "| Group |" not in md and "Judge 1" not in md
    assert "Error: proportion of incorrect predictions." in md
    assert manifest == original


def test_order_count_has_same_label_and_does_not_require_references():
    manifest = {
        "report_config": {"evaluation_type": "pairwise", "report_flip_rate": True},
        "evaluations": {"a": {"n_items": 10, "n_valid": 9, "n_flip_items": 8, "flip_rate": 0.25}},
    }
    text = render_summary(manifest, markdown=True)
    assert "### Order sensitivity" in text
    assert "| Number of items | 8 |" in text
    assert "| Flip rate | 0.2500 |" in text
    assert "Items checked" not in text
    manifest["evaluations"]["a"].update(n_flip_items=0, flip_rate=None)
    text = render_summary(manifest, markdown=True)
    assert "Flip rate is unavailable: no items have comparable presentation orders." in text


def test_error_description_comes_from_scale_not_benchmark(tmp_path):
    config = configuration(tmp_path)
    assert report_config(config)["error_definitions"] == {
        "score": "mean absolute error normalized by the score range."
    }
    raw = config.model_dump(mode="json")
    raw["judge"]["criteria"][0]["scale"] = {"kind": "binary"}
    assert report_config(parse_config(raw))["error_definitions"] == {
        "score": "proportion of incorrect predictions."
    }
    assert report_config(configuration(tmp_path, kind="pairwise"))["error_definitions"] == {
        "choice": "proportion of incorrect predictions."
    }


def test_same_model_configurations_have_distinct_headers():
    manifest = {
        "report_config": {
            "judges": [
                {"label": "cold", "id": "same-model"},
                {"label": "warm", "id": "same-model"},
            ]
        },
        "evaluations": {"cold": {"n_items": 1}, "warm": {"n_items": 1}},
    }
    md = render_summary(manifest, markdown=True)
    assert "same-model (cold)" in md and "same-model (warm)" in md
    assert "Judge 1" not in md


def test_full_model_headers_wrap_before_splitting_off_human():
    from giuriallm.report.presentation import Table
    from giuriallm.report.summary import _table

    table = Table(
        "",
        [
            "Metric",
            "gpt-6.1-sol",
            "gemini-4-argon",
            "claude-opus-5-5",
            "qwen3.5-35b-a3b",
            "mistral-large-3",
            "Panel",
            "Human",
        ],
        [["Jensen-Shannon divergence", *["0.0000"] * 6, "0.0000 (n=5)"]],
    )
    lines = _table(table, False, 120)
    assert max(map(len, lines)) <= 120
    assert sum("Metric" in line for line in lines) == 1
    assert sum(set(line.replace(" ", "")) == {"-"} for line in lines) == 1


async def test_json_is_saved_even_when_only_markdown_is_requested(tmp_path):
    config = configuration(tmp_path)
    config = config.model_copy(
        update={"report": config.report.model_copy(update={"formats": ["markdown"]})}
    )
    result = await Engine(config, MockProvider()).run(
        [EvaluationItem(id="item", candidates=["code"])]
    )
    write_reports(config, result, tmp_path / "output")
    data = json.loads((tmp_path / "output/results.json").read_text())
    assert data["items"][0]["aggregate"] is not None
    assert data["items"][0]["n_failed"] == 0
    assert "judges" not in data["items"][0]


@pytest.mark.parametrize("names", [["same", "same"], ["Panel", "other"], ["Human", "other"]])
def test_duplicate_or_reserved_judge_names_are_rejected(tmp_path, names):
    config = configuration(tmp_path, panel=True).model_dump(mode="python")
    config["panel"]["members"] = config["panel"]["members"][:2]
    for model, name in zip(config["panel"]["members"], names, strict=True):
        model["label"] = name
    with pytest.raises(ValueError, match="Judge labels must be unique"):
        parse_config(config)
