"""Deterministic offline provider.

Exists so the end-to-end path -- config, prompts, debiasing, aggregation,
meta-evaluation, reporting -- can be exercised and tested without network access
or an API key. Verdicts are derived from a hash of the prompt, so they are
reproducible but carry no evaluative meaning. Never use it for results.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .base import LLMRequest, LLMResponse, Provider, RawUsage


class MockProvider(Provider):
    name = "mock"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        rng = _Deterministic(request.cache_key)
        payload = _synthesise(request.schema, rng)
        # The shared score schema only specifies a numeric type. Binary limits
        # are in the compiler's criteria block, so read them without changing
        # the prompts or schemas sent to real providers.
        binary_criteria = {
            name
            for name, scale in re.findall(
                r"^- ([^\n]+?): .*?^  scale: ([^\n]+)$", request.system, re.MULTILINE | re.DOTALL
            )
            if scale == "1 if the criterion is satisfied, 0 otherwise"
        }
        for entry in payload.get("scores", []):
            if entry["criterion"] in binary_criteria:
                entry["score"] = rng.below(2)
        text = f"[mock verdict for {request.model.id}]"
        usage = RawUsage(
            input_tokens=max(1, len(request.system) + len(request.user)) // 4,
            output_tokens=max(1, len(str(payload))) // 4,
        )
        return LLMResponse(payload=payload, text=text, usage=usage, model=request.model.id)


class _Deterministic:
    """A small deterministic sampler seeded by a string."""

    def __init__(self, seed: str):
        self._state = hashlib.sha256(seed.encode()).digest()
        self._pos = 0

    def _byte(self) -> int:
        if self._pos >= len(self._state):
            self._state = hashlib.sha256(self._state).digest()
            self._pos = 0
        value = self._state[self._pos]
        self._pos += 1
        return value

    def below(self, n: int) -> int:
        return self._byte() % max(1, n)

    def choice(self, options: list[Any]) -> Any:
        return options[self.below(len(options))]

    def shuffled(self, options: list[Any]) -> list[Any]:
        pool = list(options)
        out: list[Any] = []
        while pool:
            out.append(pool.pop(self.below(len(pool))))
        return out


def _synthesise(schema: dict[str, Any], rng: _Deterministic) -> dict[str, Any]:
    out: dict[str, Any] = {}
    properties: dict[str, Any] = schema.get("properties", {})
    for key, spec in properties.items():
        out[key] = _value(key, spec, rng)
    return out


def _enum_key(items: dict[str, Any], count: int) -> tuple[str, list[Any]] | None:
    for field, spec in items.get("properties", {}).items():
        values = spec.get("enum")
        if isinstance(values, list) and len(values) == count:
            return field, list(values)
    return None


def _value(key: str, spec: dict[str, Any], rng: _Deterministic) -> Any:
    kind = spec.get("type")
    if "enum" in spec:
        return rng.choice(list(spec["enum"]))
    if kind == "array":
        items = spec.get("items", {})
        if "enum" in items:
            labels = list(items["enum"])
            if spec.get("minItems") == len(labels):
                return rng.shuffled(labels)
            return [rng.choice(labels)]
        count = int(spec.get("minItems", 1))
        # An object array whose elements are keyed by an enum (e.g. one entry per
        # criterion) must cover each key exactly once, not sample with replacement.
        keyed = _enum_key(items, count)
        rows = [_synthesise(items, rng) for _ in range(count)]
        if keyed is not None:
            field, values = keyed
            for row, value in zip(rows, values, strict=True):
                row[field] = value
        return rows
    if kind == "object":
        return _synthesise(spec, rng)
    if kind == "integer":
        return 1 + rng.below(5)
    if kind == "number":
        return round(rng.below(100) / 20.0, 2)
    if isinstance(kind, list):
        return 1 + rng.below(5)
    return f"Mock {key}: generated offline for pipeline testing; not an evaluation."
