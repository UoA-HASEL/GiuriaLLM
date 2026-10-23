"""Reporting: Markdown, HTML, JSON, and LaTeX views of a run.

Console and Markdown share one summary layout. JSON retains per-item results
and diagnostics for further analysis.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ..engine.runner import RunResult
from ..spec import RunConfig, config_hash

__all__ = [
    "render_console",
    "render_report",
    "render_html",
    "render_json",
    "render_latex",
    "render_markdown",
    "write_reports",
]


def write_reports(config: RunConfig, result: RunResult, run_dir: Path | str) -> list[Path]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    renderers = {
        "markdown": ("report.md", render_markdown),
        "html": ("report.html", render_html),
        "json": ("results.json", render_json),
        "latex": ("results.tex", render_latex),
    }
    # JSON is the complete result artifact, even when only a human-readable
    # format was requested. Keep accepting "json" in existing configurations.
    for fmt in dict.fromkeys(["json", *config.report.formats]):
        filename, renderer = renderers[fmt]
        path = run_dir / filename
        path.write_text(renderer(config, result))
        written.append(path)
    console_path = run_dir / "console.txt"
    console_path.write_text(render_report(config, result), encoding="utf-8")
    written.append(console_path)
    return written


# --------------------------------------------------------------------------


def render_report(
    config: RunConfig, result: RunResult, *, markdown=False, width=None, verbose=False
) -> str:
    """Complete default report shared by the CLI and integrations."""
    from ..engine.manifest import build_manifest
    from .summary import render_summary

    manifest = build_manifest(config, result)
    text = render_summary(manifest, markdown=markdown, width=width)
    if verbose and not markdown:
        text += _render_detailed(manifest["evaluations"], result.human_baseline)
    return text


def render_markdown(config: RunConfig, result: RunResult) -> str:
    return render_report(config, result, markdown=True)


def render_json(config: RunConfig, result: RunResult) -> str:
    payload = {
        "run_id": result.run_id,
        "config_hash": config_hash(config),
        "flip_rate": result.flip_rate,
        "position_bias": result.position_bias,
        "bradley_terry": result.bradley_terry,
        "usage": result.usage.model_dump(),
        "total_tokens": result.usage.total_tokens,
        "evaluations": {
            name: summary.model_dump(mode="json")
            for name, summary in result.evaluations.items()
        },
        "human_baseline": result.human_baseline,
        "agreement": result.agreement.model_dump(mode="json") if result.agreement else None,
        "items": [
            {
                "item_id": item.item_id,
                "aggregate": item.aggregate.model_dump() if item.aggregate else None,
                "selected_candidate": getattr(item, "selected_candidate", None),
                "flipped": item.flipped,
                "n_judgements": len(item.judgements),
                "n_failed": sum(1 for j in item.judgements if not j.ok),
            }
            for item in result.results
        ],
        "notes": result.notes,
    }
    return json.dumps(payload, indent=2, default=str) + "\n"


def evaluation_values(summary: dict[str, Any]) -> dict[str, float | int | None]:
    """Numeric report fields, shared by console output and repetition summaries."""
    values = {
        key: summary.get(key)
        for key in (
            "n_items",
            "n_valid",
            "n_failed",
            "total_tokens",
            "latency_per_item_mean_s",
            "flip_rate",
            "n_flip_items",
            "n_flipped",
        )
    }
    for key in ("input_tokens", "output_tokens", "cost_usd", "energy_wh", "latency_s"):
        values[f"usage/{key}"] = summary.get("usage", {}).get(key)
    agreement = summary.get("agreement") or {}
    for target in agreement.get("targets", []):
        prefix = target["target"]
        values[f"{prefix}/n_items"] = target["n_items"]
        for metric, value in target.get("metrics", {}).items():
            values[f"{prefix}/{metric}"] = value
        for stratum, metrics in target.get("strata", {}).items():
            values[f"{prefix}/{stratum}/n_items"] = target.get("stratum_sizes", {}).get(stratum)
            for metric, value in metrics.items():
                values[f"{prefix}/{stratum}/{metric}"] = value
    return values


def render_console(
    evaluations: dict[str, dict[str, Any]],
    human_baseline: dict[str, Any] | None = None,
    *,
    markdown: bool = False,
    width: int | None = None,
    verbose: bool = False,
) -> str:
    """Compatibility entry point for callers with summary records only.

    Use render_report(config, result) for complete model and run metadata.
    The default layout is identical; missing metadata is marked unavailable.
    """
    from .summary import render_summary

    primary = evaluations.get("panel") or next(iter(evaluations.values()), {})
    manifest = {
        "evaluations": evaluations,
        "human_baseline": human_baseline,
        "counts": {
            "items": primary.get("n_items", 0),
            "judgements": "n/a",
            "failed_judgements": primary.get("n_failed", 0),
        },
    }
    text = render_summary(manifest, markdown=markdown, width=width)
    if verbose and not markdown:
        text += _render_detailed(evaluations, human_baseline)
    return text


def _render_detailed(evaluations, human_baseline) -> str:
    """Retain the existing --verbose field view; the default stays compact."""
    lines = ["\nDetailed fields (raw totals)\n"]
    for judge, summary in evaluations.items():
        for key, value in evaluation_values(summary).items():
            lines.append(f"{judge} / {key}: {value if value is not None else 'n/a'}")
    for key, value, count in human_values(human_baseline or {}):
        lines.append(f"Human / {key}: {value if value is not None else 'n/a'} (n={count})")
    return "\n".join(lines) + "\n"


def human_values(baseline: dict[str, Any]) -> list[tuple[str, float | None, int]]:
    """Human metric values with metric-specific valid annotator counts."""
    rows = []
    for target in baseline.get("targets", []):
        name = target["target"]
        for metric, value in target.get("metrics", {}).items():
            rows.append(
                (f"{name}/{metric}", value, target.get("metric_annotators", {}).get(metric, 0))
            )
        for stratum, metrics in target.get("strata", {}).items():
            counts = target.get("stratum_annotators", {}).get(stratum, {})
            for metric, value in metrics.items():
                rows.append((f"{name}/{stratum}/{metric}", value, counts.get(metric, 0)))
    return rows


def render_latex(config: RunConfig, result: RunResult) -> str:
    """A booktabs table of the headline diagnostics, ready to \\input into a paper."""
    rows: list[tuple[str, str]] = [
        ("Items", f"{result.n_items}"),
        ("Judgements", f"{result.n_judgements}"),
        ("Failed judgements", f"{result.n_failed}"),
        ("Flip rate", _fmt(result.flip_rate)),
        ("Cost (USD, list)", f"{result.usage.cost_usd:.4f}"),
        ("Energy (Wh, est.)", f"{result.usage.energy_wh:.3f}"),
    ]
    if result.agreement is not None:
        for target in result.agreement.targets:
            for metric, value in target.metrics.items():
                rows.append((f"{target.target}: {metric.replace('_', ' ')}", _fmt(value)))

    lines = [
        "% Generated by giuriallm -- regenerate rather than editing by hand.",
        f"% run_id: {result.run_id}  config_hash: {config_hash(config)}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Quantity & Value \\\\",
        "\\midrule",
    ]
    lines += [f"{_tex(label)} & {_tex(value)} \\\\" for label, value in rows]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def render_html(config: RunConfig, result: RunResult) -> str:
    body = html.escape(render_markdown(config, result))
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        f"<title>GiuriaLLM run {html.escape(result.run_id)}</title>\n"
        "<style>body{font:15px/1.55 system-ui,sans-serif;max-width:60rem;margin:2rem auto;"
        "padding:0 1rem;color:#1a1a1a;background:#fff}pre{white-space:pre-wrap;"
        "font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}"
        "@media(prefers-color-scheme:dark){body{background:#111;color:#eee}}</style>\n"
        "</head>\n<body>\n<pre>" + body + "</pre>\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _tex(text: str) -> str:
    for char, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("$", r"\$"),
    ):
        text = text.replace(char, replacement)
    return text
