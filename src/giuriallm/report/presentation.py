"""Human-readable views of persisted metrics. No evaluation is performed here."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

LABELS = {
    "cohen_kappa": "Cohen's kappa",
    "spearman": "Spearman's rho",
    "kendall": "Kendall's tau",
    "krippendorff_alpha": "Krippendorff's alpha",
    "icc": "Intraclass correlation",
    "accuracy": "Accuracy",
    "error": "Error",
    "earth_movers_distance": "Earth mover's distance",
    "jensen_shannon": "Jensen-Shannon divergence",
    "high_agreement": "High agreement",
    "low_agreement": "Low agreement",
    "n_items": "Items",
    "n_valid": "Valid items",
    "n_failed": "Failed judgements",
    "n_flip_items": "Items checked for flips",
    "n_flipped": "Items with flips",
    "flip_rate": "Flip rate",
    "input_tokens": "Input tokens",
    "output_tokens": "Output tokens",
    "total_tokens": "Total tokens",
    "cost_usd": "Estimated cost (USD)",
    "energy_wh": "Estimated energy (Wh)",
    "latency_s": "Cumulative API time (s)",
    "latency_per_item_mean_s": "API time per item (s)",
}


def label(key: str) -> str:
    return LABELS.get(key, key.replace("_", " "))


def number(key: str, value: float | int | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    field = key.rsplit("/", 1)[-1]
    if field == "cost_usd":
        return f"{value:.6g}"
    if field.startswith("n_") or field.endswith("tokens"):
        return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"
    if value != 0 and abs(value) < 0.0001:
        return f"{value:.3g}"
    return f"{value:.4f}"


@dataclass
class Table:
    title: str
    headers: list[str]
    rows: list[list[str]]


def escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("\n", " ")
    )
