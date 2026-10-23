"""Provider interface.

A provider turns a compiled prompt plus a schema into structured output. It is
responsible for structured output only — parsing, caching, retries, debiasing,
and accounting all live above this layer.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..spec import ModelSpec


class ProviderError(RuntimeError):
    """The provider could not produce a response (transport, auth, or API error)."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class LLMRequest:
    model: ModelSpec
    system: str
    user: str
    schema: dict[str, Any]
    tool_name: str
    sample: int = 0

    @property
    def cache_key(self) -> str:
        payload = "\x1f".join(
            [
                self.model.provider,
                self.model.id,
                str(self.model.temperature),
                self.model.thinking.value,
                str(self.model.effort),
                str(self.model.max_tokens),
                # Provider options change which server answers and how it decodes,
                # so two runs that differ only here must not share a cache entry.
                json.dumps(self.model.options, sort_keys=True, default=str),
                self.system,
                self.user,
                repr(sorted(self.schema.items())),
                str(self.sample),
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class RawUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class LLMResponse:
    payload: dict[str, Any] | None
    text: str = ""
    usage: RawUsage = field(default_factory=RawUsage)
    model: str = ""
    stop_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.payload is not None


class Provider(ABC):
    """Base class for model adapters."""

    name: str

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return structured output for one request."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook, not every provider has one
        """Release any transport resources. Override only if there are any."""
        return None


_REGISTRY: dict[str, Callable[[], Provider]] = {}


def register_provider(name: str, factory: Callable[[], Provider]) -> None:
    _REGISTRY[name] = factory


def get_provider(name: str) -> Provider:
    if name not in _REGISTRY:
        raise ProviderError(
            f"unknown provider {name!r}; available: {sorted(_REGISTRY)}. "
            "Register one with giuriallm.providers.register_provider()."
        )
    return _REGISTRY[name]()


def available_providers() -> list[str]:
    return sorted(_REGISTRY)
