"""Ollama adapter for locally served open-weight models.

Uses Ollama's native ``/api/chat`` endpoint with its ``format`` parameter, which
constrains decoding to a JSON schema. That is more dependable across
open-weight models than function calling, whose support varies model by model.

Configure with ``model.options``:

``base_url``    where Ollama listens (default ``http://localhost:11434``)
``timeout``     seconds to wait for a response (default 600 -- local generation
                on a cold model can be slow)
``keep_alive``  how long Ollama keeps the model resident, e.g. ``"30m"``
``num_ctx``     context window to allocate for the model
``seed``        base seed; the sample index is added to it

Local inference has no per-token API charge, so cost is reported as zero. The
energy estimate still applies, and on local hardware it is the figure that
matters.
"""

from __future__ import annotations

import json
from typing import Any

from .base import LLMRequest, LLMResponse, Provider, ProviderError, RawUsage

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 600.0


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, client: Any | None = None):
        self._explicit_client = client
        self._clients: dict[str, Any] = {}
        self.notes: list[str] = []

    # -- request shaping -------------------------------------------------

    @staticmethod
    def base_url(request: LLMRequest) -> str:
        return str(request.model.options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        model = request.model
        options = model.options

        runtime: dict[str, Any] = {"num_predict": model.max_tokens}
        if model.temperature is not None:
            runtime["temperature"] = model.temperature
        # A per-sample seed keeps each self-consistency sample reproducible while
        # still making the samples differ from one another.
        base_seed = options.get("seed", 0)
        runtime["seed"] = int(base_seed if isinstance(base_seed, int) else 0) + request.sample
        for key in ("num_ctx", "top_p", "top_k", "repeat_penalty", "num_gpu"):
            if key in options:
                runtime[key] = options[key]

        payload: dict[str, Any] = {
            "model": model.id,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "stream": False,
            "format": request.schema,
            "options": runtime,
        }
        if "keep_alive" in options:
            payload["keep_alive"] = options["keep_alive"]
        return payload

    # -- transport -------------------------------------------------------

    def _client(self, request: LLMRequest) -> Any:
        if self._explicit_client is not None:
            return self._explicit_client
        url = self.base_url(request)
        if url not in self._clients:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - declared dependency
                raise ProviderError(
                    "the 'httpx' package is required for provider=ollama; "
                    "install it with: pip install httpx"
                ) from exc
            timeout = float(request.model.options.get("timeout", DEFAULT_TIMEOUT))
            self._clients[url] = httpx.AsyncClient(base_url=url, timeout=timeout)
        return self._clients[url]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        client = self._client(request)
        try:
            response = await client.post("/api/chat", json=self.build_payload(request))
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise _connection_error(exc, self.base_url(request)) from exc

        status = getattr(response, "status_code", 200)
        if status >= 400:
            raise _http_error(status, _body_text(response), request)

        return _to_response(response.json(), request)

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()


# --------------------------------------------------------------------------


def _body_text(response: Any) -> str:
    try:
        return response.text
    except Exception:  # noqa: BLE001 - diagnostics only
        return ""


def _connection_error(exc: Exception, url: str) -> ProviderError:
    name = type(exc).__name__
    if "Connect" in name or "Timeout" in name:
        return ProviderError(
            f"could not reach Ollama at {url} ({name}: {exc}). "
            "Is 'ollama serve' running, and is model.options.base_url correct?",
            retryable=True,
        )
    return ProviderError(f"{name}: {exc}", retryable=True)


def _http_error(status: int, body: str, request: LLMRequest) -> ProviderError:
    detail = body.strip()[:400]
    if status == 404 and "model" in detail.lower():
        return ProviderError(
            f"Ollama does not have the model {request.model.id!r} "
            f"(pull it with: ollama pull {request.model.id})",
            retryable=False,
        )
    return ProviderError(f"Ollama returned HTTP {status}: {detail}", retryable=status >= 500)


def _to_response(data: dict[str, Any], request: LLMRequest) -> LLMResponse:
    usage = RawUsage(
        input_tokens=int(data.get("prompt_eval_count") or 0),
        output_tokens=int(data.get("eval_count") or 0),
    )
    message = data.get("message") or {}
    text = message.get("content") or ""
    done_reason = data.get("done_reason")

    payload = _loads(text)
    if payload is None and done_reason == "length":
        raise ProviderError(
            f"generation hit the token limit ({request.model.max_tokens}) before the verdict "
            "was returned; raise model.max_tokens",
            retryable=False,
        )

    return LLMResponse(
        payload=payload,
        text=text,
        usage=usage,
        model=data.get("model", request.model.id),
        stop_reason=done_reason,
    )


def _loads(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
