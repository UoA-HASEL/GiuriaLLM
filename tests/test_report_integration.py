"""Reporting and repeated-run statistics use persisted measurements, not API calls."""

import json
from copy import deepcopy
from math import sqrt

import pytest

from giuriallm.cli import main
from giuriallm.engine.runner import RunResult
from giuriallm.report import evaluation_values, render_console, write_reports
from giuriallm.report.repetitions import RepetitionError, render_repetitions, summarize_runs
from giuriallm.spec import parse_config
from giuriallm.types import CriterionScore, ItemResult, Verdict


def evaluation(score=0.5):
    return {
        "n_items": 100,
        "n_valid": 99,
        "n_failed": 1,
        "total_tokens": 1200,
        "latency_per_item_mean_s": 2.0,
        "flip_rate": 0.25,
        "n_flip_items": 100,
        "n_flipped": 25,
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 200,
            "cost_usd": 0.2,
            "energy_wh": 0.48,
            "latency_s": 200.0,
        },
        "agreement": {
            "targets": [
                {
                    "target": "correctness",
                    "n_items": 99,
                    "metrics": {"cohen_kappa": score},
                    "strata": {"high_agreement": {"error": 0.1}},
                    "stratum_sizes": {"high_agreement": 61},
                }
            ]
        },
    }


def write_manifest(tmp_path, run_id, score=0.5, **overrides):
    path = tmp_path / run_id
    path.mkdir()
    manifest = {
        "run_id": run_id,
        "config_hash": "configuration",
        "dataset_hash": "data",
        "human_labels_hash": "labels",
        "evaluations": {"mock/member": evaluation(score), "panel": evaluation(score)},
        "human_baseline": None,
        "counts": {"items": 100, "judgements": 100, "failed_judgements": 1},
        "usage": {"total_tokens": 1200, "cost_usd": 0.2, "energy_wh_estimate": 0.48},
        "diagnostics": {},
    }
    manifest.update(overrides)
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def test_flat_fields_and_console_preserve_members_and_strata():
    values = evaluation_values(evaluation())
    assert values["correctness/high_agreement/error"] == 0.1
    assert values["correctness/high_agreement/n_items"] == 61
    assert values["total_tokens"] == 1200
    text = render_console({"mock/member": evaluation(), "panel": evaluation()})
    assert "mock/member" in text and "Panel" in text
    assert "API time (s/item)" in text
    assert "Cohen's kappa" in text
    assert "latency_per_item_mean_s" not in text
    assert "Coverage" in text and "Resource use" in text
    assert "Failed judgements" in text
    assert "Overall — correctness" in text and "High human agreement" in text


def test_console_does_not_round_small_costs_to_zero():
    summary = evaluation()
    summary["usage"]["cost_usd"] = 0.00011076
    text = render_console({"judge": summary})
    assert "1.1076e-06" in text
    assert "1200.000" not in text


def test_human_console_reports_valid_counts_without_dumping_individuals():
    human = {
        "targets": [
            {
                "target": "score",
                "metrics": {"spearman": None},
                "metric_annotators": {"spearman": 0},
                "strata": {"high_agreement": {"error": 0.05}},
                "stratum_annotators": {"high_agreement": {"error": 16}},
                "per_annotator": {"hidden-person": {}},
            }
        ]
    }
    text = render_console({}, human)
    assert "Spearman's rho" in text and "n/a (n=0)" in text
    assert "Overall — score" in text and "0.0500 (n=16)" in text
    assert "n/a" in text
    assert "hidden-person" not in text


def test_repetitions_use_sample_sd_and_valid_counts(tmp_path):
    paths = [
        write_manifest(tmp_path, "r1", 0.2),
        write_manifest(tmp_path, "r2", 0.8),
        write_manifest(tmp_path, "r3", None),
    ]
    summary = summarize_runs(paths)
    stats = summary["evaluations"]["panel"]["correctness/cohen_kappa"]
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["sd"] == pytest.approx(sqrt(0.18))
    assert stats["n"] == 2
    single = summarize_runs(paths[:1])["evaluations"]["panel"]["correctness/cohen_kappa"]
    assert single == {"mean": 0.2, "sd": None, "n": 1}


@pytest.mark.parametrize("field", ["config_hash", "dataset_hash", "human_labels_hash"])
def test_repetitions_reject_incompatible_data_or_settings(tmp_path, field):
    a = write_manifest(tmp_path, "a")
    b = write_manifest(tmp_path, "b", **{field: "different"})
    with pytest.raises(RepetitionError, match=field):
        summarize_runs([a, b])


def test_repetitions_reject_duplicates_and_missing_provenance(tmp_path):
    a = write_manifest(tmp_path, "a")
    with pytest.raises(RepetitionError, match="same run"):
        summarize_runs([a, a])
    b = write_manifest(tmp_path, "b", dataset_hash=None)
    with pytest.raises(RepetitionError, match="dataset_hash"):
        summarize_runs([b])


def test_cli_show_and_summarize_read_saved_measurements(tmp_path, capsys):
    a = write_manifest(tmp_path, "a")
    b = write_manifest(tmp_path, "b", 0.7)
    assert main(["show", str(a)]) == 0
    assert "High human agreement" in capsys.readouterr().out
    assert main(["show", str(a), "--verbose"]) == 0
    assert "High human agreement" in capsys.readouterr().out
    output = tmp_path / "summary.json"
    assert main(["summarize", str(a), str(b), "--json", "-o", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == payload
    assert payload["n_runs"] == 2
    assert main(["summarize", str(a), str(b)]) == 0
    assert "sample standard deviation" in capsys.readouterr().out


def test_cli_incompatible_runs_is_actionable_error(tmp_path, capsys):
    a = write_manifest(tmp_path, "a")
    b = write_manifest(tmp_path, "b", config_hash="other")
    assert main(["summarize", str(a), str(b)]) == 1
    assert "config_hash" in capsys.readouterr().err


def test_narrow_console_preserves_all_judges_without_overflow():
    judges = {f"provider:very-long-model-identifier-{i}": evaluation() for i in range(7)}
    judges["panel"] = evaluation()
    text = render_console(judges, width=40)
    assert max(map(len, text.splitlines())) <= 40
    assert "provider:very-long-model-identifier-6" in text and "Panel" in text
    assert text.count("0.5000") == 8


def test_pointwise_omits_inapplicable_flip_but_preserves_requested_nulls():
    summary = evaluation()
    summary.update(flip_rate=None, n_flip_items=0, n_flipped=0)
    summary["agreement"]["targets"][0]["metrics"] = {"spearman": None}
    text = render_console({"judge": summary})
    assert "Order consistency" not in text
    assert "Flip rate" not in text
    assert "Spearman's rho" in text and "n/a" in text
    # Historical nulls have no reason code: do not invent a cause.
    assert "constant values" not in text


def test_distinguish_unreported_from_undefined():
    first, second = evaluation(), evaluation()
    first["agreement"]["targets"][0]["metrics"] = {"spearman": None}
    text = render_console({"a": first, "b": second}, width=120)
    line = next(line for line in text.splitlines() if "Spearman's rho" in line)
    assert "n/a" in line and "--" in line


def test_markdown_escapes_identifiers_and_does_not_mutate_data():
    summary = evaluation()
    summary["agreement"]["targets"][0]["target"] = "custom|<script>"
    values = {"provider:model|<script>": summary}
    before = deepcopy(values)
    text = render_console(values, markdown=True)
    assert "custom\\|&lt;script&gt;" in text
    assert "provider:model\\|&lt;script&gt;" in text
    assert "## Resource use" in text
    assert "| Metric | provider:model\\|&lt;script&gt; |" in text
    assert values == before


def test_console_without_human_labels_keeps_usage_and_coverage():
    summary = evaluation()
    summary["agreement"] = None
    text = render_console({"model": summary})
    assert "Flip rate" in text
    assert "Human" not in text
    assert "Coverage" in text and "Resource use" in text


def test_repeated_run_display_marks_missing_values_and_keeps_statistics(tmp_path):
    paths = [write_manifest(tmp_path, "one", 0.2), write_manifest(tmp_path, "two", None)]
    summary = summarize_runs(paths)
    before = deepcopy(summary)
    text = render_repetitions(summary, width=120)
    assert "0.2000 ± n/a [runs=1]" in text
    assert "Cohen's kappa" in text
    assert summary == before


def test_show_json_is_not_formatted_as_console(tmp_path, capsys):
    path = write_manifest(tmp_path, "saved")
    assert main(["show", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads((path / "manifest.json").read_text())


def test_repetitions_normalize_each_run_before_averaging(tmp_path):
    first, second = evaluation(), evaluation()
    first["n_items"], first["n_valid"] = 100, 90
    second["n_items"], second["n_valid"] = 50, 45
    first["usage"]["cost_usd"], second["usage"]["cost_usd"] = 0.2, 0.4
    a = write_manifest(tmp_path, "a", evaluations={"member": first})
    b = write_manifest(tmp_path, "b", evaluations={"member": second})
    original = [(p / "manifest.json").read_bytes() for p in (a, b)]
    summary = summarize_runs([a, b])
    usage = summary["evaluations"]["member"]
    assert usage["usage_per_item/cost_usd"]["mean"] == pytest.approx(0.005)
    assert usage["usage_per_item/cost_usd"]["sd"] == pytest.approx(sqrt(0.000018))
    # Preserve the old JSON keys as totals; never silently change their units.
    assert usage["usage/cost_usd"]["mean"] == pytest.approx(0.3)
    assert usage["usage_per_item/total_tokens"]["mean"] == 18
    assert original == [(p / "manifest.json").read_bytes() for p in (a, b)]
    md = render_repetitions(summary, markdown=True)
    assert "Estimated cost (USD/item)" in md
    assert "0.005 ± 0.00424264" in md


def test_repetitions_reuse_default_layout_and_keep_human_fixed(tmp_path):
    human = {
        "available": True,
        "targets": [
            {
                "target": "correctness",
                "metrics": {"cohen_kappa": 0.79},
                "metric_annotators": {"cohen_kappa": 5},
            }
        ],
    }
    cfg = {
        "judges": [
            {"label": "mock/member", "id": "full-model-id", "endpoint": "https://example.org/v1"}
        ]
    }
    a = write_manifest(tmp_path, "a", 0.2, human_baseline=human, report_config=cfg)
    b = write_manifest(tmp_path, "b", 0.8, human_baseline=human, report_config=cfg)
    summary = summarize_runs([a, b])
    before = deepcopy(summary)
    md = render_repetitions(summary, markdown=True)
    assert [line for line in md.splitlines() if line.startswith("## ")] == [
        "## Run summary",
        "## Judges",
        "## Evaluation",
        "## Resource use",
        "## Notes",
    ]
    assert "| Metric | full-model-id | Panel | Human |" in md
    assert "| Group |" not in md and "Judge 1" not in md
    assert "0.5000 ± 0.4243" in md
    assert "0.7900 (n=5)" in md and "0.7900 ±" not in md
    assert "Human is a fixed baseline" in md
    assert "### Coverage" in md and "### High human agreement" in md
    assert "Number of items" in md and "Items checked" not in md
    assert summary == before


def test_repetitions_use_refreshed_results_not_obsolete_manifest_groups(tmp_path):
    a = write_manifest(tmp_path, "a")
    b = write_manifest(tmp_path, "b")
    for path in (a, b):
        payload = json.loads((path / "manifest.json").read_text())
        for result in payload["evaluations"].values():
            target = result["agreement"]["targets"][0]
            target["strata"], target["stratum_sizes"] = {}, {}
        (path / "results.json").write_text(json.dumps(payload))
    summary = summarize_runs([a, b])
    assert not any("high_agreement" in key for key in summary["evaluations"]["panel"])
    assert "High human agreement" not in render_repetitions(summary)


def test_repetitions_zero_items_do_not_produce_zero_resource_estimates(tmp_path):
    record = evaluation()
    record["n_items"] = 0
    path = write_manifest(tmp_path, "empty", evaluations={"member": record})
    summary = summarize_runs([path])
    assert summary["evaluations"]["member"]["usage_per_item/cost_usd"] == {
        "mean": None,
        "sd": None,
        "n": 0,
    }
    assert "n/a ± n/a [runs=0]" in render_repetitions(summary, markdown=True)


def test_summary_cli_saves_all_formats_without_changing_json_stdout(tmp_path, capsys):
    a, b = write_manifest(tmp_path, "a"), write_manifest(tmp_path, "b")
    output = tmp_path / "summary" / "comparison.json"
    assert main(["summarize", str(a), str(b), "--json", "-o", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text())
    markdown = output.with_name("comparison.report.md").read_text()
    text = output.with_name("comparison.console.txt").read_text()
    assert markdown.startswith("# GiuriaLLM Report")
    assert text.startswith("GiuriaLLM Report")
    assert "±" in markdown and "±" in text
    assert main(["summarize", str(a), str(b), "--markdown"]) == 0
    assert capsys.readouterr().out.strip() == markdown.strip()


def test_repetitions_do_not_average_different_human_baselines(tmp_path):
    a = write_manifest(tmp_path, "a", human_baseline={"available": False})
    b = write_manifest(tmp_path, "b", human_baseline={"available": True})
    with pytest.raises(RepetitionError, match="different Human references"):
        summarize_runs([a, b])


def test_repetitions_preserve_run_and_evaluation_notes(tmp_path):
    record = evaluation()
    record["agreement"]["notes"] = ["Reference coverage is incomplete."]
    a = write_manifest(
        tmp_path,
        "a",
        evaluations={"member": record},
        notes=["Mock results are not evaluation evidence."],
    )
    b = write_manifest(tmp_path, "b", evaluations={"member": record}, notes=["Second run note."])
    text = render_repetitions(summarize_runs([a, b]), markdown=True)
    assert "Mock results are not evaluation evidence." in text
    assert "Second run note." in text
    assert text.count("Reference coverage is incomplete.") == 1


@pytest.mark.parametrize("disagreement", [None, 0.0, 0.25])
@pytest.mark.parametrize(
    "evaluation_type,result_kind,verdict",
    [
        ("pointwise", "graded", Verdict(scores=[CriterionScore(criterion="score", score=2)])),
        ("pairwise", "best_choice", Verdict(best_choice=0)),
        ("listwise", "ranking", Verdict(ranking=[0, 1, 2])),
    ],
)
def test_reports_omit_disagreement_without_changing_results(
    tmp_path, disagreement, evaluation_type, result_kind, verdict
):
    config = parse_config(
        {
            "judge": {
                "type": evaluation_type,
                "criteria": [{"name": "score", "definition": "quality"}],
                "outputs": {"result": result_kind},
            },
            "model": {"provider": "mock", "id": "test"},
            "dataset": {
                "path": str(tmp_path / "unused.jsonl"),
                "fields": {
                    "candidates": ["a", "b", "c"][
                        : {"pointwise": 1, "pairwise": 2, "listwise": 3}[evaluation_type]
                    ]
                },
            },
            "report": {"formats": ["json", "markdown", "html", "latex"]},
        }
    )
    item = ItemResult(item_id="example", aggregate=verdict, disagreement=disagreement, flipped=True)
    result = RunResult(
        run_id="saved",
        config_hash="configuration",
        started_at="2026-10-15T00:00:00Z",
        results=[item],
        flip_rate=1.0,
    )
    before = result.model_dump()
    paths = write_reports(config, result, tmp_path / "reports")
    assert len(paths) == 5
    for path in paths:
        assert "disagreement" not in path.read_text().lower()
    payload = json.loads((tmp_path / "reports/results.json").read_text())
    assert payload["items"][0]["aggregate"] == verdict.model_dump()
    assert payload["items"][0]["flipped"] is True
    assert payload["flip_rate"] == 1.0
    markdown = (tmp_path / "reports/report.md").read_text()
    assert "Per-item results" not in markdown
    assert [line for line in markdown.splitlines() if line.startswith("## ")] == [
        "## Run summary",
        "## Judges",
        "## Evaluation",
        "## Resource use",
        "## Notes",
    ]
    assert result.model_dump() == before
