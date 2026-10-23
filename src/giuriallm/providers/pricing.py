"""Price and locality lookup, shared across adapters.

Only rates the project can state with confidence are hard-coded. For anything
else the run reports a cost of zero *and says so* in the manifest, rather than
printing a plausible-looking number nobody checked. Set
``model.price_per_mtok: [input, output]`` in the config to supply your own.
"""

from __future__ import annotations

from ..spec import ModelSpec

#: USD per million tokens, (input, output), by model-id prefix.
PRICES: dict[str, tuple[float, float]] = {
    # Anthropic list prices.
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

#: Providers whose inference runs on hardware you already pay for. There is no
#: per-token API charge to report; the energy estimate still applies.
LOCAL_PROVIDERS = frozenset({"ollama", "local", "vllm", "mock"})


def price_for(model: ModelSpec) -> tuple[float, float] | None:
    """List price for a model, or ``None`` when no rate is known."""
    if model.price_per_mtok is not None:
        return model.price_per_mtok
    for prefix, price in PRICES.items():
        if model.id.startswith(prefix):
            return price
    return None


def is_local(model: ModelSpec) -> bool:
    return model.provider in LOCAL_PROVIDERS


def price_note(model: ModelSpec) -> str | None:
    """A manifest note when a run's cost figure cannot be trusted."""
    if is_local(model) or price_for(model) is not None:
        return None
    return (
        f"no list price is known for {model.provider}:{model.id}, so its cost is reported "
        "as 0.00 -- set model.price_per_mtok: [input, output] to account for it"
    )
