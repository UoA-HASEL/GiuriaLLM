"""Offline summaries of repeated runs of the same evaluation."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from . import evaluation_values
from .saved import saved_manifest
from .summary import per_item_usage, render_summary


class RepetitionError(ValueError):
    """Runs cannot safely be treated as repetitions of one experiment."""


def summarize_runs(run_dirs: list[Path]) -> dict[str, Any]:
    if not run_dirs:
        raise RepetitionError("provide at least one run directory")
    manifests = []
    for path in run_dirs:
        try:
            manifests.append(saved_manifest(path))
        except (OSError, ValueError) as exc:
            raise RepetitionError(f"cannot read run {path}: {exc}") from exc
    ids = [m["run_id"] for m in manifests]
    if len(set(ids)) != len(ids):
        raise RepetitionError("the same run cannot be counted more than once")
    first = manifests[0]
    for key in ("config_hash", "dataset_hash", "human_labels_hash"):
        if key not in first or (key != "human_labels_hash" and not first[key]):
            raise RepetitionError(f"missing {key}; regenerate the run report before summarizing")
        if any(key not in m or m[key] != first[key] for m in manifests):
            raise RepetitionError(f"runs have different or missing {key}")
    judges = set(first.get("evaluations", {}))
    if not judges:
        raise RepetitionError("run has no per-judge evaluations; regenerate its report")
    if any(set(m.get("evaluations", {})) != judges for m in manifests):
        raise RepetitionError("runs contain different sets of judges")
    summaries = {}
    for judge in first["evaluations"]:
        repetitions = [evaluation_values(m["evaluations"][judge]) for m in manifests]
        for values, manifest in zip(repetitions, manifests, strict=True):
            values.update(
                {
                    f"usage_per_item/{key}": value
                    for key, value in per_item_usage(manifest["evaluations"][judge]).items()
                }
            )
        keys = sorted(set().union(*(r.keys() for r in repetitions)))
        summaries[judge] = {key: _statistics([r.get(key) for r in repetitions]) for key in keys}
    human = first.get("human_baseline")
    if any(m.get("human_baseline") != human for m in manifests):
        raise RepetitionError("runs have different Human references; do not average them silently")
    counts = {
        key: _statistics([(m.get("counts") or {}).get(key) for m in manifests])
        for key in ("items", "judgements", "failed_judgements")
    }
    usage = {
        key: _statistics([(m.get("usage") or {}).get(key) for m in manifests])
        for key in ("total_tokens", "cost_usd")
    }
    return {
        "run_ids": ids,
        "n_runs": len(ids),
        "config_hash": first["config_hash"],
        "dataset_hash": first["dataset_hash"],
        "human_labels_hash": first["human_labels_hash"],
        "evaluations": summaries,
        "human_baseline": human,
        "counts": counts,
        "usage": usage,
        "report_view": _report_view(manifests, summaries, counts, usage),
        "standard_deviation": "sample (ddof=1); null when fewer than two valid values",
    }


def _statistics(values: list[float | int | None]) -> dict[str, float | int | None]:
    valid = [v for v in values if v is not None and math.isfinite(v)]
    return {
        "mean": mean(valid) if valid else None,
        "sd": stdev(valid) if len(valid) > 1 else None,
        "n": len(valid),
    }


def _report_view(manifests, statistics, counts, usage):
    """Build a mean-value view for the shared renderer; statistics stay in JSON.

    The view determines sections and missing-value notes. Cell means and SDs are
    always formatted from statistics, never derived from rounded report text.
    """
    first = manifests[0]
    view = {
        "report_config": deepcopy(first.get("report_config") or {}),
        "judges": deepcopy(first.get("judges") or []),
        "human_baseline": deepcopy(first.get("human_baseline")),
        "counts": {key: stats["mean"] for key, stats in counts.items()},
        "usage": {key: stats["mean"] for key, stats in usage.items()},
        "evaluations": {},
        "notes": list(dict.fromkeys(note for m in manifests for note in m.get("notes", []))),
    }
    for judge, quantities in statistics.items():

        def average(key, values=quantities):
            return values.get(key, {}).get("mean")

        sources = [m["evaluations"][judge] for m in manifests]
        item = {
            key: average(key)
            for key in (
                "n_items",
                "n_valid",
                "n_failed",
                "n_flip_items",
                "flip_rate",
                "total_tokens",
            )
        }
        item["usage_per_item"] = {
            key: average(f"usage_per_item/{key}")
            for key in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cost_usd",
                "energy_wh",
                "latency_s",
            )
        }
        target_names = list(
            dict.fromkeys(
                t["target"]
                for source in sources
                for t in (source.get("agreement") or {}).get("targets", [])
            )
        )
        targets = []
        for name in target_names:
            originals = [
                next(
                    (
                        t
                        for t in (source.get("agreement") or {}).get("targets", [])
                        if t["target"] == name
                    ),
                    {},
                )
                for source in sources
            ]
            target = {
                "target": name,
                "n_items": average(f"{name}/n_items"),
                "metrics": {},
                "strata": {},
                "stratum_sizes": {},
                "unavailable": {},
                "stratum_unavailable": {},
            }
            groups = [None, *dict.fromkeys(g for t in originals for g in t.get("strata", {}))]
            for group in groups:
                prefix = f"{name}/{group}/" if group else f"{name}/"
                if group:
                    target["stratum_sizes"][group] = average(prefix + "n_items")
                metrics = list(
                    dict.fromkeys(
                        metric
                        for t in originals
                        for metric in (
                            t.get("strata", {}).get(group, {}) if group else t.get("metrics", {})
                        )
                    )
                )
                values, reasons = {}, {}
                for metric in metrics:
                    values[metric] = average(prefix + metric)
                    if values[metric] is None:
                        codes = {
                            (
                                t.get("stratum_unavailable", {}).get(group, {})
                                if group
                                else t.get("unavailable", {})
                            ).get(metric, "undefined")
                            for t in originals
                        }
                        reasons[metric] = next(iter(codes)) if len(codes) == 1 else "undefined"
                if group:
                    target["strata"][group] = values
                    target["stratum_unavailable"][group] = reasons
                else:
                    target["metrics"], target["unavailable"] = values, reasons
            # A shared count is meaningful here; do not present one run's count
            # as though it describes every run if coverage differs.
            ungrouped = [t.get("n_ungrouped", 0) for t in originals]
            if len(set(ungrouped)) == 1:
                target["n_ungrouped"] = ungrouped[0]
            targets.append(target)
        item["agreement"] = {
            "targets": targets,
            "notes": list(
                dict.fromkeys(
                    note
                    for source in sources
                    for note in (source.get("agreement") or {}).get("notes", [])
                )
            ),
        }
        view["evaluations"][judge] = item
    return view


def render_repetitions(
    summary: dict[str, Any], *, width: int | None = None, markdown: bool = False
) -> str:
    if "report_view" not in summary:
        raise RepetitionError(
            "regenerate this summary from its run directories to use the new layout"
        )
    return render_summary(
        summary["report_view"], width=width, markdown=markdown, repetitions=summary
    )


def write_repetitions(summary: dict[str, Any], output: Path | str) -> list[Path]:
    """Save JSON plus readable Markdown and console summaries."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        output,
        output.with_name(output.stem + ".report.md"),
        output.with_name(output.stem + ".console.txt"),
    ]
    # Render everything before replacing any saved artifact.
    contents = [
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        render_repetitions(summary, markdown=True),
        render_repetitions(summary),
    ]
    for path, text in zip(paths, contents, strict=True):
        path.write_text(text, encoding="utf-8")
    return paths
