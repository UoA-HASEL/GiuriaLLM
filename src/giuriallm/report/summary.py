"""One compact report for live and saved runs, in text or Markdown."""

from __future__ import annotations

import math
import shutil
import textwrap
from urllib.parse import urlsplit, urlunsplit

from .presentation import Table, escape, label, number

REASONS = {
    "no_matched_items": "No items matched the reference labels.",
    "empty_group": "No items in this group.",
    "no_valid_judgements": "No valid judgements are available for this target.",
    "no_reference_labels": "No reference labels are available for this target.",
    "no_defined_annotators": "No annotator has a defined value for this metric.",
    "requires_numeric": "This metric requires numeric scores.",
    "insufficient_pairs": "At least two paired observations are required.",
    "constant_values": "Correlation is undefined for constant values.",
    "kappa_undefined": "Kappa is undefined when expected agreement is one.",
    "invalid_scale": "A positive scoring range is required.",
    "missing_support": "The label or score support is unavailable.",
    "undefined": "The metric is undefined for these observations.",
}


def report_config(config) -> dict:
    """Only report-safe metadata; never include API keys or endpoint credentials."""
    judges = []
    for model in config.judges():
        endpoint = model.options.get("base_url")
        if endpoint:
            parsed = urlsplit(str(endpoint))
            endpoint = urlunsplit(
                (parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", "")
            )
        judges.append(
            {
                "label": model.display,
                "configured_label": model.label,
                "id": model.id,
                "provider": model.provider,
                "endpoint": endpoint,
            }
        )
    return {
        "evaluation_type": config.judge.type.value,
        "output_type": config.judge.outputs.result.value,
        "aggregation": config.panel.aggregation.value if config.panel else None,
        "judges": judges,
        "metrics": [m.value for m in config.meta_evaluation.metrics]
        if config.meta_evaluation
        else [],
        "report_flip_rate": config.debias.report_flip_rate,
        "has_reference_labels": config.meta_evaluation is not None,
        "stratify": bool(config.meta_evaluation and config.meta_evaluation.stratify_by_agreement),
        "energy_wh_per_ktok": config.engine.energy_wh_per_ktok,
        "error_definitions": (
            {
                c.name: (
                    "proportion of incorrect predictions."
                    if c.scale.kind.value in ("binary", "categorical")
                    else "mean absolute error normalized by the score range."
                )
                for c in config.judge.criteria
            }
            if config.judge.outputs.result.value == "graded"
            else {"choice": "proportion of incorrect predictions."}
        ),
    }


def per_item_usage(summary: dict) -> dict:
    """All run items form the denominator, including failed items."""
    n = summary.get("n_items", 0)
    usage = summary.get("usage") or {}
    totals = {
        key: usage.get(key)
        for key in ("input_tokens", "output_tokens", "cost_usd", "energy_wh", "latency_s")
    }
    totals["total_tokens"] = summary.get("total_tokens")
    if totals["total_tokens"] is None and all(
        totals[k] is not None for k in ("input_tokens", "output_tokens")
    ):
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return {key: value / n if n and value is not None else None for key, value in totals.items()}


def _table(table: Table, markdown: bool, width: int, fixed: int = 1) -> list[str]:
    if markdown:
        return [
            "| " + " | ".join(escape(c) for c in table.headers) + " |",
            "| "
            + " | ".join(":---" if i < fixed else "---:" for i in range(len(table.headers)))
            + " |",
            *["| " + " | ".join(escape(c) for c in row) + " |" for row in table.rows],
            "",
        ]
    rows = [table.headers, *table.rows]
    # Wrap full model IDs rather than replacing them with numbered aliases.
    widths = [
        min(34, max(len(row[i]) for row in rows))
        if i < fixed
        else max(
            min(20, len(table.headers[i])), max((len(row[i]) for row in table.rows), default=0)
        )
        for i in range(len(table.headers))
    ]
    minimums = [
        min(widths[i], 16)
        if i < fixed
        else min(widths[i], max(8, max((len(row[i]) for row in table.rows), default=0)))
        for i in range(len(widths))
    ]
    if sum(minimums) + 2 * (len(widths) - 1) <= width:
        while sum(widths) + 2 * (len(widths) - 1) > width:
            i = max(range(len(widths)), key=lambda j: widths[j] - minimums[j])
            widths[i] -= 1
    # Split wide comparisons by judge, repeating the identifying columns.
    if sum(widths) + 2 * (len(widths) - 1) > width:
        if len(widths) > fixed + 1:
            groups, current = [], list(range(fixed))
            for i in range(fixed, len(widths)):
                if (
                    len(current) > fixed
                    and sum(widths[j] for j in [*current, i]) + 2 * len(current) > width
                ):
                    groups.append(current)
                    current = list(range(fixed))
                current.append(i)
            groups.append(current)
            return [
                line
                for group in groups
                for line in _table(
                    Table(
                        table.title,
                        [table.headers[i] for i in group],
                        [[row[i] for i in group] for row in table.rows],
                    ),
                    False,
                    width,
                    fixed,
                )
            ]
        lines = []
        for row in table.rows:
            text = " / ".join(row[:fixed]) + f" — {table.headers[-1]}: {row[-1]}"
            lines.extend(textwrap.wrap(text, width=width))
        return [*lines, ""]
    lines = []
    for index, row in enumerate(rows):
        wrapped = [textwrap.wrap(c, width=widths[i]) or [""] for i, c in enumerate(row)]
        for k in range(max(map(len, wrapped))):
            cells = [col[k] if k < len(col) else "" for col in wrapped]
            lines.append(
                "  ".join(
                    c.ljust(widths[i]) if i < fixed else c.rjust(widths[i])
                    for i, c in enumerate(cells)
                ).rstrip()
            )
        if index == 0:
            lines.append("  ".join("-" * n for n in widths))
    return [*lines, ""]


def render_summary(
    manifest: dict,
    *,
    markdown: bool = False,
    width: int | None = None,
    repetitions: dict | None = None,
) -> str:
    """Render persisted measurements only. No API calls or metric recomputation."""
    width = max(24, width or shutil.get_terminal_size((120, 24)).columns)
    cfg = manifest.get("report_config") or {}
    evaluations = manifest.get("evaluations") or {}
    human = manifest.get("human_baseline") or {}
    names = list(evaluations)
    models = cfg.get("judges") or manifest.get("judges") or []
    model_by_name = {}
    members = [n for n in names if n != "panel"]
    for i, model in enumerate(models):
        name = model.get("label") or (members[i] if i < len(members) else model.get("id"))
        model_by_name[name] = model
    aliases = {n: "Panel" if n == "panel" else model_by_name.get(n, {}).get("id", n) for n in names}
    display_names = list(aliases.values())
    for name, display in list(aliases.items()):
        if display_names.count(display) > 1 or (name != "panel" and display in ("Panel", "Human")):
            aliases[name] = f"{display} ({name})"
    headers = [aliases[n] for n in names]
    lines = ["# GiuriaLLM Report" if markdown else "GiuriaLLM Report", ""]

    def heading(text, level=2):
        lines.extend([("#" * level + " " + escape(text)) if markdown else text, ""])

    def paragraph(text, *, gap=True):
        lines.extend([escape(text)] if markdown else textwrap.wrap(text, width=width))
        if gap:
            lines.append("")

    def cell(judge, key, value, *, count=False):
        if repetitions is None:
            return str(value) if count else number(key, value)
        stats = repetitions["evaluations"][judge].get(key)
        return statistic_cell(key, stats, repetitions["n_runs"], count=count)

    heading("Run summary")
    counts = manifest.get("counts") or {}
    total_usage = manifest.get("usage") or {}
    overview = (
        f"Run: {manifest.get('run_id', 'n/a')}",
        f"Items: {counts.get('items', 'n/a')} | "
        f"Judgements: {counts.get('judgements', 'n/a')} | "
        f"Failed judgements: {counts.get('failed_judgements', 'n/a')}",
        f"Total tokens: {number('total_tokens', total_usage.get('total_tokens'))} | "
        f"Estimated cost: {'$' if total_usage.get('cost_usd') is not None else ''}"
        f"{number('cost_usd', total_usage.get('cost_usd'))}",
    )
    if repetitions is not None:

        def run_stat(section, key, *, count=False):
            return statistic_cell(
                key, repetitions.get(section, {}).get(key), repetitions["n_runs"], count=count
            )

        overview = (
            f"Runs: {repetitions['n_runs']}",
            "Run IDs: " + ", ".join(repetitions["run_ids"]),
            f"Items per run: {run_stat('counts', 'items', count=True)} | "
            f"Judgements per run: {run_stat('counts', 'judgements', count=True)} | "
            f"Failed judgements per run: {run_stat('counts', 'failed_judgements', count=True)}",
            f"Total tokens per run: {run_stat('usage', 'total_tokens')} | "
            f"Estimated cost per run (USD): {run_stat('usage', 'cost_usd')}",
        )
    for text in overview:
        if markdown:
            lines.append(escape(text) + "  ")
        else:
            paragraph(text, gap=False)
    lines.append("")

    heading("Judges")
    if models:
        for i, model in enumerate(models):
            name = model.get("label") or (members[i] if i < len(members) else model.get("id"))
            display = aliases.get(name, model.get("id", "n/a"))
            text = f"{display} | Endpoint: {model.get('endpoint') or 'not recorded'}"
            if markdown:
                lines.append(escape(text) + "  ")
            else:
                paragraph(text, gap=False)
        lines.append("")
    else:
        for name in names:
            if name != "panel":
                paragraph(f"{aliases[name]} | Endpoint: not recorded")
        if not names:
            paragraph("Model metadata is unavailable in this saved run.")

    heading("Evaluation")
    if evaluations:
        heading("Coverage", level=3)
        rows = [
            [title, *[cell(n, key, evaluations[n].get(key, "n/a"), count=True) for n in names]]
            for key, title in (
                ("n_items", "Items"),
                ("n_valid", "Valid items"),
                ("n_failed", "Failed judgements"),
            )
        ]
        lines.extend(_table(Table("", ["Metric", *headers], rows), markdown, width))
    targets = list(
        dict.fromkeys(
            t["target"]
            for v in [*evaluations.values(), {"agreement": human}]
            for t in (v.get("agreement") or {}).get("targets", [])
        )
    )
    has_human = bool(human.get("available", bool(human.get("targets"))))
    notes = []
    shown = False
    group_shown = False
    metric_names = set()
    for target_name in targets:
        selected = {
            n: next(
                (
                    t
                    for t in (evaluations[n].get("agreement") or {}).get("targets", [])
                    if t["target"] == target_name
                ),
                {},
            )
            for n in names
        }
        ht = next((t for t in human.get("targets", []) if t["target"] == target_name), {})
        all_targets = [*selected.values(), ht]
        groups = [None, *dict.fromkeys(g for t in all_targets for g in t.get("strata", {}))]
        for group in groups:
            group_label = {
                "high_agreement": "High human agreement",
                "low_agreement": "Low human agreement",
            }.get(group, label(group) if group else "Overall")
            values = {
                n: selected[n].get("strata", {}).get(group, {})
                if group
                else selected[n].get("metrics", {})
                for n in names
            }
            hv = ht.get("strata", {}).get(group, {}) if group else ht.get("metrics", {})
            metrics = list(dict.fromkeys(k for v in [*values.values(), hv] for k in v))
            if not metrics:
                continue
            metric_names.update(metrics)
            group_shown |= group is not None
            title = group_label
            if not group or len(targets) > 1:
                title += f" — {target_name}"
            heading(title, level=3)
            # Hide a group only when every judge explicitly reports zero items.
            # Missing sizes and Human-only reports must not be mistaken for empty groups.
            if (
                group
                and names
                and all(selected[n].get("stratum_sizes", {}).get(group) == 0 for n in names)
                and not any(v is not None for v in hv.values())
            ):
                paragraph("No items in this group.")
                shown = True
                continue
            prefix = f"{target_name}/{group}/" if group else f"{target_name}/"
            matched = [
                cell(
                    n,
                    prefix + "n_items",
                    selected[n].get("stratum_sizes", {}).get(group, "--")
                    if group
                    else selected[n].get("n_items", "--"),
                    count=True,
                )
                for n in names
            ]
            rows = [["Number of items", *matched, *(["--"] if has_human else [])]]
            for metric in metrics:
                cells = [
                    cell(n, prefix + metric, values[n][metric]) if metric in values[n] else "--"
                    for n in names
                ]
                if has_human:
                    count = (
                        ht.get("stratum_annotators", {}).get(group, {})
                        if group
                        else ht.get("metric_annotators", {})
                    ).get(metric, 0)
                    cells.append(
                        f"{number(metric, hv.get(metric))} (n={count})" if metric in hv else "--"
                    )
                rows.append([label(metric), *cells])
                reasons = {}
                for n in names:
                    t = selected[n]
                    codes = (
                        t.get("stratum_unavailable", {}).get(group, {})
                        if group
                        else t.get("unavailable", {})
                    )
                    if metric in codes:
                        reasons.setdefault(codes[metric], []).append(aliases[n])
                for code, affected in reasons.items():
                    who = "All judges" if len(affected) == len(names) else ", ".join(affected)
                    metric_label = "" if code == "empty_group" else f" / {label(metric)}"
                    notes.append(
                        f"{target_name} / {group_label}{metric_label} ({who}): "
                        + REASONS.get(code, REASONS["undefined"])
                    )
                human_codes = (
                    ht.get("stratum_unavailable", {}).get(group, {})
                    if group
                    else ht.get("unavailable", {})
                )
                if has_human and metric in human_codes:
                    notes.append(
                        f"{target_name} / {group_label} / {label(metric)} (Human): "
                        + REASONS.get(human_codes[metric], REASONS["undefined"])
                    )
            lines.extend(
                _table(
                    Table("", ["Metric", *headers, *(["Human"] if has_human else [])], rows),
                    markdown,
                    width,
                )
            )
            paragraph("Number of items: the number of items included in this table's evaluation.")
            shown = True
        if "error" in selected_metric_names(selected, ht):
            definition = cfg.get("error_definitions", {}).get(target_name)
            if definition:
                notes.append(
                    f"Error{f' ({target_name})' if len(targets) > 1 else ''}: {definition}"
                )
        ungrouped = {}
        for n, target in selected.items():
            if target.get("n_ungrouped"):
                ungrouped.setdefault(target["n_ungrouped"], []).append(aliases[n])
        for count, affected in ungrouped.items():
            who = "All judges" if len(affected) == len(names) else ", ".join(affected)
            notes.append(
                f"{target_name} ({who}): {count} item(s) cannot be grouped by annotator agreement."
            )
    if cfg.get("report_flip_rate", True) and (
        cfg.get("evaluation_type") in ("pairwise", "listwise")
        or any(
            v.get("n_flip_items") or v.get("flip_rate") is not None for v in evaluations.values()
        )
    ):
        heading("Order sensitivity", level=3)
        if any(v.get("n_flip_items") for v in evaluations.values()):
            rows = [
                [
                    title,
                    *[
                        cell(n, key, evaluations[n].get(key), count=key == "n_flip_items")
                        for n in names
                    ],
                ]
                for key, title in (
                    ("n_flip_items", "Number of items"),
                    ("flip_rate", "Flip rate"),
                )
            ]
            lines.extend(_table(Table("", ["Metric", *headers], rows), markdown, width))
            paragraph("Number of items: the number of items included in this table's evaluation.")
        else:
            paragraph("Flip rate is unavailable: no items have comparable presentation orders.")
        shown = True
    if not shown:
        paragraph(
            "No evaluation metrics configured."
            if not cfg.get("metrics")
            else "No evaluation results available for the requested metrics."
        )
    if has_human:
        notes.append(
            "Human: leave-one-annotator-out baseline; n is the number of contributing annotators."
        )
    if group_shown:
        notes.append(
            "High human agreement / Low human agreement: items grouped by agreement "
            "among human annotators, using the configured grouping rules."
        )
    if "jensen_shannon" in metric_names:
        notes.append(
            "Jensen-Shannon divergence: compares judge and human label distributions."
            + (
                " For Panel, the judge distribution uses its members' votes."
                if "panel" in names
                else ""
            )
        )
    missing_ids_note = (
        "Human baseline unavailable: every matched item needs explicit annotator IDs."
    )
    human_note_labels = {
        missing_ids_note: "Human baseline unavailable: annotator identities are missing.",
    }
    notes.extend(human_note_labels.get(note, note) for note in human.get("notes", []))
    for value in evaluations.values():
        notes.extend((value.get("agreement") or {}).get("notes", []))
    heading("Resource use")
    fields = [
        ("input_tokens", "Input tokens (tokens/item)"),
        ("output_tokens", "Output tokens (tokens/item)"),
        ("total_tokens", "Total tokens (tokens/item)"),
        ("cost_usd", "Estimated cost (USD/item)"),
        ("latency_s", "Cumulative API time (s/item)"),
        ("energy_wh", "Estimated energy (Wh/item)"),
    ]
    usage = {
        n: evaluations[n].get("usage_per_item", {})
        if repetitions is not None
        else per_item_usage(evaluations[n])
        for n in names
    }
    rows = []
    for key, title in fields:
        cells = []
        for name in names:
            value = usage[name].get(key)
            cells.append(
                cell(name, f"usage_per_item/{key}", value)
                if repetitions is not None
                else f"{value:,.2f}"
                if key.endswith("tokens") and value is not None and math.isfinite(value)
                else number(key, value)
            )
        rows.append([title, *cells])
    if names:
        lines.extend(_table(Table("", ["Metric", *headers], rows), markdown, width))
        notes.append(
            "Resource use: mean per input item, including items with failed judgements."
            + (" Panel values sum its members." if "panel" in names else "")
        )
        notes.extend(
            [
                "Cumulative API time: summed call durations across presentation orders, "
                "not wall-clock time.",
                "Estimated cost: calculated using configured input/output token list prices.",
                "Estimated energy: calculated from token usage, not measured.",
            ]
        )
        if not any(v.get("n_items") for v in evaluations.values()):
            paragraph("Per-item resources are unavailable because this run contains no items.")
    else:
        paragraph("Per-judge resource records are unavailable in this saved run.")
    notes.extend(manifest.get("notes", []))
    if repetitions is not None:
        notes.insert(
            0,
            "Values are mean ± sample standard deviation across runs. "
            "Unchanged counts are shown once. [runs=n] gives the number of runs with a defined "
            "value; standard deviation requires at least two.",
        )
        if has_human:
            notes.append("Human is a fixed baseline, not an average across runs.")
    missing = []
    if any("n/a" in line for line in lines):
        missing.append("n/a = undefined or unavailable")
    if any("--" in line for line in lines if not set(line.strip()) <= {"-", " ", "|", ":"}):
        missing.append("-- = not reported")
    if missing:
        notes.insert(0, "; ".join(missing) + ".")
    heading("Notes")
    for note in dict.fromkeys(notes):
        if markdown:
            lines.append("- " + escape(note))
        else:
            lines.extend(textwrap.wrap("- " + note, width=width, subsequent_indent="  "))
    if not notes:
        paragraph("No additional notes.")
    return "\n".join(lines).rstrip() + "\n"


def statistic_cell(key: str, stats: dict | None, n_runs: int, *, count: bool = False) -> str:
    """Format repeated measurements without changing their numeric representation."""
    if stats is None:
        return "--"
    average, sd, n = stats["mean"], stats["sd"], stats["n"]
    if count and average is not None and n == n_runs and (sd == 0 or n_runs == 1):
        return number("n_items", average)

    def formatted(value):
        if value is not None and math.isfinite(value) and key.rsplit("/", 1)[-1].endswith("tokens"):
            return f"{value:,.2f}"
        return number(key, value)

    text = f"{formatted(average)} ± {formatted(sd)}"
    return text + (f" [runs={n}]" if n != n_runs else "")


def selected_metric_names(selected: dict, human: dict) -> set[str]:
    return {
        metric
        for target in [*selected.values(), human]
        for values in [target.get("metrics", {}), *target.get("strata", {}).values()]
        for metric in values
    }
