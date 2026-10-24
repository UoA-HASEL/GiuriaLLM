"""Anthropic adapter.

Verdicts come back through forced tool use, so the verdict is structured output
validated against the compiled schema rather than regex over prose.

Two model-specific details are handled here rather than pushed onto the user:

* Sampling parameters (``temperature``, ``top_p``, ``top_k``) are rejected with a
  400 on Claude Opus 5, Opus 4.7/4.8, Sonnet 5, Fable 5, and Mythos 5. A
  ``temperature`` set in the config is dropped for those models and the drop is
  recorded so the manifest reports what was actually sent.
* Thinking is on by default on Claude Opus 5. Judging runs default to
  ``thinking: disabled`` for determinism, which is accepted only at effort
  ``high`` or below.
"""

from __future__ import annotations

import json
from typing import Any

from ..spec import ThinkingMode
from .base import LLMRequest, LLMResponse, Provider, ProviderError, RawUsage
from .pricing import PRICES, price_for  # re-exported for callers of this module

__all__ = ["PRICES", "AnthropicProvider", "price_for", "rejects_sampling_params"]

# Models that reject temperature / top_p / top_k with a 400.
_NO_SAMPLING_PARAMS = (
    "claude-opus-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)

# Effort levels at which thinking cannot be disabled (Claude Opus 5).
_HIGH_EFFORT = ("xhigh", "max")

def rejects_sampling_params(model_id: str) -> bool:
    return any(model_id.startswith(prefix) for prefix in _NO_SAMPLING_PARAMS)


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, client: Any | None = None):
        self._client = client
        self._strict_supported = True
        self.notes: list[str] = []

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ProviderError(
                    "the 'anthropic' package is required for provider=anthropic; "
                    "install it with: pip install anthropic"
                ) from exc
            self._client = AsyncAnthropic()
        return self._client

    def build_params(self, request: LLMRequest, strict: bool) -> dict[str, Any]:
        model = request.model
        tool: dict[str, Any] = {
            "name": request.tool_name,
            "description": "Report the verdict for the candidate(s) under evaluation.",
            "input_schema": request.schema,
        }
        if strict:
            tool["strict"] = True

        params: dict[str, Any] = {
            "model": model.id,
            "max_tokens": model.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            "tools": [tool],
        }

        if model.thinking is ThinkingMode.ADAPTIVE:
            params["thinking"] = {"type": "adaptive"}
            # Forced tool choice and thinking do not combine; instruct instead.
            params["tool_choice"] = {"type": "auto"}
        else:
            if model.effort in _HIGH_EFFORT:
                raise ProviderError(
                    f"model.thinking=disabled is not accepted at model.effort={model.effort} "
                    "on Claude Opus 5. Use effort 'high' or below, or set thinking: adaptive."
                )
            params["thinking"] = {"type": "disabled"}
            params["tool_choice"] = {"type": "tool", "name": request.tool_name}

        if model.effort is not None:
            params["output_config"] = {"effort": model.effort}

        if model.temperature is not None:
            if rejects_sampling_params(model.id):
                self._note(
                    f"temperature was dropped: {model.id} rejects sampling parameters. "
                    "Determinism is controlled by thinking/effort instead."
                )
            else:
                params["temperature"] = model.temperature

        return params

    def _note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        client = self._ensure_client()
        params = self.build_params(request, strict=self._strict_supported)
        try:
            message = await client.messages.create(**params)
        except Exception as exc:  # noqa: BLE001 - normalised below
            if self._strict_supported and "strict" in str(exc).lower():
                self._strict_supported = False
                self._note(f"strict tool use is unavailable on {request.model.id}; disabled it")
                message = await client.messages.create(**self.build_params(request, strict=False))
            else:
                raise _as_provider_error(exc) from exc

        return _to_response(message, request)


def _as_provider_error(exc: Exception) -> ProviderError:
    status = getattr(exc, "status_code", None)
    retryable = status in (408, 409, 429, 500, 502, 503, 504, 529) or status is None
    return ProviderError(f"{type(exc).__name__}: {exc}", retryable=retryable)


def _to_response(message: Any, request: LLMRequest) -> LLMResponse:
    usage = RawUsage(
        input_tokens=getattr(message.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(message.usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(message.usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
    )
    stop_reason = getattr(message, "stop_reason", None)

    if stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None)
        raise ProviderError(f"model refused the request (category={category})", retryable=False)

    payload: dict[str, Any] | None = None
    texts: list[str] = []
    for block in getattr(message, "content", []) or []:
        kind = getattr(block, "type", None)
        if kind == "tool_use" and getattr(block, "name", None) == request.tool_name:
            raw = getattr(block, "input", None)
            payload = raw if isinstance(raw, dict) else _loads(raw)
        elif kind == "text":
            texts.append(getattr(block, "text", "") or "")

    text = "\n".join(t for t in texts if t)
    if payload is None and stop_reason == "max_tokens":
        raise ProviderError(
            f"response hit max_tokens ({request.model.max_tokens}) before the verdict was "
            "returned; raise model.max_tokens",
            retryable=False,
        )
    return LLMResponse(
        payload=payload,
        text=text,
        usage=usage,
        model=getattr(message, "model", request.model.id),
        stop_reason=stop_reason,
    )


def _loads(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return None
