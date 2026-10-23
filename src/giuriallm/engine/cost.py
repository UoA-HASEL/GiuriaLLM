"""Token, cost, and energy accounting.

Cost uses published list prices per model (overridable per run). Energy is an
*estimate* from a configurable coefficient: hosted-API energy draw is not
observable from the client, so the coefficient and the resulting figure are both
written to the manifest and reported as an estimate, never as a measurement.
"""

from __future__ import annotations

from ..providers.base import RawUsage
from ..providers.pricing import price_for
from ..spec import ModelSpec
from ..types import Usage

#: Grams of CO2-equivalent per kWh. Default is the 2024 world average grid
#: intensity; set a regional figure when the inference region is known.
DEFAULT_GRID_INTENSITY_G_PER_KWH = 480.0


def account(
    raw: RawUsage,
    model: ModelSpec,
    energy_wh_per_ktok: float,
    latency_s: float = 0.0,
    cached: bool = False,
) -> Usage:
    """Turn raw token counts into a costed, energy-estimated usage record."""
    price = price_for(model)
    if cached or price is None:
        cost = 0.0
    else:
        input_price, output_price = price
        cost = (raw.input_tokens * input_price + raw.output_tokens * output_price) / 1_000_000.0

    tokens = raw.input_tokens + raw.output_tokens
    energy = 0.0 if cached else (tokens / 1000.0) * energy_wh_per_ktok

    return Usage(
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        cache_read_tokens=raw.cache_read_tokens,
        cache_write_tokens=raw.cache_write_tokens,
        cost_usd=cost,
        energy_wh=energy,
        latency_s=latency_s,
        cached=cached,
    )


def co2_grams(energy_wh: float, grid_intensity: float = DEFAULT_GRID_INTENSITY_G_PER_KWH) -> float:
    """Estimated CO2-equivalent emissions for a given estimated energy draw."""
    return (energy_wh / 1000.0) * grid_intensity


def has_price(model: ModelSpec) -> bool:
    return price_for(model) is not None
