"""OpenAI adapter (and OpenAI-compatible servers).

Verdicts come back as structured output, by one of two routes selected with
``model.options.structured_output``:

``tools`` (default)
    A forced function call whose parameters are the compiled verdict schema.
``json_schema``
    ``response_format={"type": "json_schema", ...}``, for models and servers
    that support response-format schemas but not strict function calling.

Both start in strict mode and fall back to non-strict on a schema-rejection
error, so a server with partial support still produces verdicts.

Point ``model.options.base_url`` at any OpenAI-compatible endpoint (vLLM,
LM Studio, a gateway) to use the same adapter; set ``model.options.api_key_env``
to read the key from a different environment variable.

Use ``model.options.reasoning_effort: none`` to explicitly disable reasoning
on supporting OpenAI models; temperature is then sent rather than omitted.
For Alibaba-compatible DeepSeek, ``model.options.enable_thinking: false`` is
forwarded through the SDK's ``extra_body``. These options are deliberately
explicit: the generic ``model.thinking`` default alone is not an API switch.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .base import LLMRequest, LLMResponse, Provider, ProviderError, RawUsage

#: These model families take `max_completion_tokens`. Sampling support also
#: depends on reasoning_effort; explicitly selecting `none` permits temperature
#: on supported models. An incompatible explicit setting must fail at the API.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")

#: JSON Schema keywords that strict structured output does not accept. They are
#: dropped only from the copy sent to the API -- the parser still enforces the
#: same constraints on the way back, so a violation becomes a recorded parse
#: failure rather than a silent pass.
_UNSUPPORTED_IN_STRICT = (
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "pattern",
    "multipleOf",
)


def is_reasoning_model(model_id: str) -> bool:
    name = model_id.rsplit("/", 1)[-1]
    return name.startswith(_REASONING_PREFIXES)


def sanitise_schema(schema: Any) -> Any:
    """Recursively drop keywords strict structured output rejects."""
    if isinstance(schema, dict):
        return {
            key: sanitise_schema(value)
            for key, value in schema.items()
            if key not in _UNSUPPORTED_IN_STRICT
        }
    if isinstance(schema, list):
        return [sanitise_schema(value) for value in schema]
    return schema


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, client: Any | None = None):
        self._explicit_client = client
        self._clients: dict[str | None, Any] = {}
        # Compatibility is specific to an endpoint, model and output mode.
        self._strict_supported: dict[tuple[str, str, str], bool] = {}
        self.notes: list[str] = []

    # -- request shaping -------------------------------------------------

    def build_params(self, request: LLMRequest, strict: bool) -> dict[str, Any]:
        model = request.model
        options = model.options
        schema = sanitise_schema(request.schema) if strict else request.schema

        params: dict[str, Any] = {
            "model": model.id,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }

        mode = str(options.get("structured_output", "tools"))
        if mode == "tools":
            function: dict[str, Any] = {
                "name": request.tool_name,
                "description": "Report the verdict for the candidate(s) under evaluation.",
                "parameters": schema,
            }
            if strict:
                function["strict"] = True
            params["tools"] = [{"type": "function", "function": function}]
            params["tool_choice"] = {
                "type": "function",
                "function": {"name": request.tool_name},
            }
        elif mode == "json_schema":
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.tool_name,
                    "schema": schema,
                    "strict": strict,
                },
            }
        else:
            raise ProviderError(
                f"model.options.structured_output={mode!r} is not supported; "
                "use 'tools' or 'json_schema'"
            )

        token_param = str(
            options.get(
                "token_param",
                "max_completion_tokens" if is_reasoning_model(model.id) else "max_tokens",
            )
        )
        params[token_param] = model.max_tokens

        effort = options.get("reasoning_effort", model.effort)
        if effort is not None:
            if effort not in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
                raise ProviderError(f"invalid model.options.reasoning_effort: {effort!r}")
            params["reasoning_effort"] = effort

        # Alibaba's OpenAI-compatible DeepSeek endpoint uses this non-standard
        # body field. Keep it explicit in options, rather than guessing from IDs.
        if "enable_thinking" in options:
            enabled = options["enable_thinking"]
            if not isinstance(enabled, bool):
                raise ProviderError("model.options.enable_thinking must be a boolean")
            params["extra_body"] = {"enable_thinking": enabled}

        if model.temperature is not None:
            if is_reasoning_model(model.id) and effort != "none":
                self._note(
                    f"temperature was dropped: {model.id} is a reasoning model and rejects "
                    "sampling parameters unless reasoning_effort='none' is supported and set"
                )
            else:
                params["temperature"] = model.temperature

        # A per-sample seed makes self-consistency samples distinct but each one
        # reproducible, where the server honours it.
        if options.get("seed", True):
            base = options.get("seed") if isinstance(options.get("seed"), int) else 0
            params["seed"] = int(base) + request.sample

        for key in ("top_p", "presence_penalty", "frequency_penalty", "service_tier"):
            if key in options:
                params[key] = options[key]

        return params

    def _note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    # -- transport -------------------------------------------------------

    def _client(self, request: LLMRequest) -> Any:
        if self._explicit_client is not None:
            return self._explicit_client
        base_url = request.model.options.get("base_url")
        key = str(base_url) if base_url else None
        if key not in self._clients:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ProviderError(
                    "the 'openai' package is required for provider=openai; "
                    "install it with: pip install 'giuriallm[openai]'"
                ) from exc
            env_var = str(request.model.options.get("api_key_env", "OPENAI_API_KEY"))
            api_key = os.environ.get(env_var)
            if api_key is None and base_url is None:
                raise ProviderError(
                    f"{env_var} is not set. Export an API key, or point "
                    "model.options.base_url at a local OpenAI-compatible server."
                )
            self._clients[key] = AsyncOpenAI(
                api_key=api_key or "not-needed",
                base_url=str(base_url) if base_url else None,
            )
        return self._clients[key]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        client = self._client(request)
        options = request.model.options
        strict_key = (
            str(options.get("base_url") or ""),
            request.model.id,
            str(options.get("structured_output", "tools")),
        )
        strict = self._strict_supported.get(strict_key, True)
        try:
            completion = await client.chat.completions.create(
                **self.build_params(request, strict=strict)
            )
        except Exception as exc:  # noqa: BLE001 - normalised below
            if strict and _looks_like_schema_rejection(exc):
                self._strict_supported[strict_key] = False
                self._note(
                    f"strict structured output is unavailable on {request.model.id}; "
                    "retried without it"
                )
                completion = await client.chat.completions.create(
                    **self.build_params(request, strict=False)
                )
            else:
                raise _as_provider_error(exc) from exc

        return _to_response(completion, request)

    async def aclose(self) -> None:
        for client in self._clients.values():
            close = getattr(client, "close", None)
            if close is not None:
                await close()
        self._clients.clear()


# --------------------------------------------------------------------------


def _looks_like_schema_rejection(exc: Exception) -> bool:
    message = str(exc).lower()
    if getattr(exc, "status_code", None) not in (400, 422, None):
        return False
    return any(token in message for token in ("strict", "schema", "response_format", "json_schema"))


def _as_provider_error(exc: Exception) -> ProviderError:
    status = getattr(exc, "status_code", None)
    retryable = status in (408, 409, 429, 500, 502, 503, 504) or status is None
    return ProviderError(f"{type(exc).__name__}: {exc}", retryable=retryable)


def _to_response(completion: Any, request: LLMRequest) -> LLMResponse:
    usage_obj = getattr(completion, "usage", None)
    cached = 0
    details = getattr(usage_obj, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    usage = RawUsage(
        input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
        cache_read_tokens=cached,
    )

    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise ProviderError("the response contained no choices", retryable=True)
    choice = choices[0]
    message = getattr(choice, "message", None)
    finish_reason = getattr(choice, "finish_reason", None)

    if finish_reason == "content_filter":
        raise ProviderError("the request was blocked by a content filter", retryable=False)

    text = getattr(message, "content", None) or ""
    payload: dict[str, Any] | None = None

    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        function = getattr(call, "function", None)
        if function is not None and getattr(function, "name", None) == request.tool_name:
            payload = _loads(getattr(function, "arguments", None))
            break
    if payload is None and text:
        payload = _loads(text)

    if payload is None and finish_reason == "length":
        raise ProviderError(
            f"the response hit the token limit ({request.model.max_tokens}) before the "
            "verdict was returned; raise model.max_tokens",
            retryable=False,
        )

    return LLMResponse(
        payload=payload,
        text=text,
        usage=usage,
        model=getattr(completion, "model", request.model.id),
        stop_reason=finish_reason,
    )


def _loads(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
