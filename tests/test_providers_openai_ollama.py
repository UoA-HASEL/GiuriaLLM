"""OpenAI and Ollama adapters: request shaping and response handling.

No network. Requests are shaped and inspected directly; responses are fed in
through fake clients that mimic the shape each SDK returns.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from giuriallm.providers import (
    LLMRequest,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    is_local,
    is_reasoning_model,
    price_for,
    price_note,
    sanitise_schema,
)
from giuriallm.spec import ModelSpec

SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string", "enum": ["correctness"]},
                    "score": {"type": "integer"},
                },
                "required": ["criterion", "score"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 1,
        },
    },
    "required": ["explanation", "scores"],
    "additionalProperties": False,
}

VERDICT = {"explanation": "looks right", "scores": [{"criterion": "correctness", "score": 4}]}


def request_for(provider: str, **model_kwargs) -> LLMRequest:
    model_kwargs.setdefault("id", "gpt-test")
    return LLMRequest(
        model=ModelSpec(provider=provider, **model_kwargs),
        system="sys",
        user="user",
        schema=SCHEMA,
        tool_name="submit_verdict",
        sample=model_kwargs.pop("_sample", 0),
    )


# ==========================================================================
# OpenAI
# ==========================================================================


class FakeOpenAIClient:
    """Mimics `client.chat.completions.create`, recording the params it got."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **params):
        self.calls.append(params)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def openai_completion(
    *, arguments=None, content=None, finish_reason="tool_calls", name="submit_verdict"
):
    tool_calls = None
    if arguments is not None:
        tool_calls = [
            SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments), id="call_1")
        ]
    return SimpleNamespace(
        model="gpt-test",
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            prompt_tokens_details=SimpleNamespace(cached_tokens=64),
        ),
    )


class FakeStatusError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def test_default_mode_forces_the_verdict_function():
    params = OpenAIProvider().build_params(request_for("openai"), strict=True)
    assert params["tool_choice"] == {"type": "function", "function": {"name": "submit_verdict"}}
    function = params["tools"][0]["function"]
    assert function["name"] == "submit_verdict"
    assert function["strict"] is True
    assert params["messages"][0]["role"] == "system"


def test_strict_mode_drops_schema_keywords_it_cannot_accept():
    params = OpenAIProvider().build_params(request_for("openai"), strict=True)
    schema = params["tools"][0]["function"]["parameters"]
    assert "minItems" not in schema["properties"]["scores"]
    assert "maxItems" not in schema["properties"]["scores"]
    # Everything load-bearing survives.
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["explanation", "scores"]
    assert schema["properties"]["scores"]["items"]["properties"]["criterion"]["enum"] == [
        "correctness"
    ]


def test_non_strict_mode_keeps_the_full_schema():
    params = OpenAIProvider().build_params(request_for("openai"), strict=False)
    schema = params["tools"][0]["function"]["parameters"]
    assert schema["properties"]["scores"]["minItems"] == 1
    assert "strict" not in params["tools"][0]["function"]


def test_sanitise_schema_is_recursive_and_non_destructive():
    original = json.dumps(SCHEMA, sort_keys=True)
    cleaned = sanitise_schema(SCHEMA)
    assert "minItems" not in json.dumps(cleaned)
    assert json.dumps(SCHEMA, sort_keys=True) == original  # input untouched


def test_json_schema_mode_uses_response_format():
    request = request_for("openai", options={"structured_output": "json_schema"})
    params = OpenAIProvider().build_params(request, strict=True)
    assert params["response_format"]["type"] == "json_schema"
    assert params["response_format"]["json_schema"]["strict"] is True
    assert "tools" not in params


def test_unknown_structured_output_mode_is_rejected():
    request = request_for("openai", options={"structured_output": "freeform"})
    with pytest.raises(ProviderError, match="structured_output"):
        OpenAIProvider().build_params(request, strict=True)


def test_reasoning_models_drop_temperature_and_use_max_completion_tokens():
    provider = OpenAIProvider()
    params = provider.build_params(
        request_for("openai", id="o3-mini", temperature=0.0), strict=True
    )
    assert "temperature" not in params
    assert "max_tokens" not in params
    assert params["max_completion_tokens"] == 4096
    assert any("temperature was dropped" in note for note in provider.notes)


def test_ordinary_models_keep_temperature_and_max_tokens():
    params = OpenAIProvider().build_params(
        request_for("openai", id="gpt-4.1", temperature=0.0), strict=True
    )
    assert params["temperature"] == 0.0
    assert params["max_tokens"] == 4096


@pytest.mark.parametrize("model_id", ["gpt-5.4-mini", "gpt-6-sol"])
def test_explicit_no_reasoning_keeps_temperature(model_id):
    provider = OpenAIProvider()
    request = request_for(
        "openai",
        id=model_id,
        temperature=0.0,
        max_tokens=32768,
        options={"reasoning_effort": "none", "structured_output": "json_schema"},
    )
    params = provider.build_params(request, strict=True)
    assert params["reasoning_effort"] == "none"
    assert params["temperature"] == 0.0
    assert params["max_completion_tokens"] == 32768
    assert params["seed"] == 0
    assert not provider.notes


def test_deepseek_thinking_switch_is_sent_in_extra_body():
    params = OpenAIProvider().build_params(
        request_for(
            "openai",
            id="alicloud-deepseek-v4-flash-0731",
            temperature=0.0,
            max_tokens=32768,
            options={"enable_thinking": False},
        ),
        strict=True,
    )
    assert params["extra_body"] == {"enable_thinking": False}
    assert params["temperature"] == 0.0
    assert params["max_tokens"] == 32768
    assert "reasoning_effort" not in params


@pytest.mark.parametrize(
    "options",
    [
        {"enable_thinking": "false"},
        {"reasoning_effort": "off"},
    ],
)
def test_invalid_reasoning_options_fail_loudly(options):
    with pytest.raises(ProviderError):
        OpenAIProvider().build_params(request_for("openai", options=options), strict=True)


async def test_temperature_rejection_does_not_retry_with_weaker_schema():
    client = FakeOpenAIClient([FakeStatusError("Unsupported value: temperature", 400)])
    provider = OpenAIProvider(client=client)
    with pytest.raises(ProviderError, match="temperature"):
        await provider.complete(
            request_for("openai", id="gpt-5.4-mini", options={"reasoning_effort": "none"})
        )
    assert len(client.calls) == 1
    assert not provider.notes


def test_seed_varies_with_the_sample_index():
    provider = OpenAIProvider()
    first = provider.build_params(request_for("openai"), strict=True)
    request = LLMRequest(
        model=ModelSpec(provider="openai", id="gpt-test"),
        system="sys",
        user="user",
        schema=SCHEMA,
        tool_name="submit_verdict",
        sample=2,
    )
    assert provider.build_params(request, strict=True)["seed"] == first["seed"] + 2


async def test_openai_response_is_parsed_from_the_function_call():
    client = FakeOpenAIClient([openai_completion(arguments=json.dumps(VERDICT))])
    response = await OpenAIProvider(client=client).complete(request_for("openai"))
    assert response.payload == VERDICT
    assert response.usage.input_tokens == 120
    assert response.usage.cache_read_tokens == 64
    assert response.stop_reason == "tool_calls"


async def test_openai_response_is_parsed_from_json_content():
    client = FakeOpenAIClient(
        [openai_completion(content=json.dumps(VERDICT), finish_reason="stop")]
    )
    request = request_for("openai", options={"structured_output": "json_schema"})
    response = await OpenAIProvider(client=client).complete(request)
    assert response.payload == VERDICT


async def test_openai_falls_back_when_strict_is_rejected():
    client = FakeOpenAIClient(
        [
            FakeStatusError("Invalid schema: 'strict' is not supported", 400),
            openai_completion(arguments=json.dumps(VERDICT)),
        ]
    )
    provider = OpenAIProvider(client=client)
    response = await provider.complete(request_for("openai"))
    assert response.payload == VERDICT
    assert len(client.calls) == 2
    assert client.calls[0]["tools"][0]["function"].get("strict") is True
    assert "strict" not in client.calls[1]["tools"][0]["function"]
    assert any("strict structured output is unavailable" in n for n in provider.notes)


async def test_openai_rate_limit_is_retryable():
    client = FakeOpenAIClient([FakeStatusError("rate limited", 429)])
    with pytest.raises(ProviderError) as exc:
        await OpenAIProvider(client=client).complete(request_for("openai"))
    assert exc.value.retryable is True


@pytest.mark.parametrize("changed", ["model", "endpoint", "mode"])
async def test_strict_fallback_is_isolated_by_endpoint_model_and_mode(changed):
    client = FakeOpenAIClient(
        [
            FakeStatusError("'strict' is not supported", 400),
            *[openai_completion(arguments=json.dumps(VERDICT)) for _ in range(3)],
        ]
    )
    provider = OpenAIProvider(client=client)
    options = {"base_url": "https://first.example/v1", "structured_output": "tools"}
    first = request_for("openai", id="model-a", options=options)
    await provider.complete(first)

    other_options = dict(options)
    other_model = "model-a"
    if changed == "model":
        other_model = "model-b"
    elif changed == "endpoint":
        other_options["base_url"] = "https://second.example/v1"
    else:
        other_options["structured_output"] = "json_schema"
    await provider.complete(request_for("openai", id=other_model, options=other_options))
    # The other endpoint/model/mode must still request strict output.
    other_call = client.calls[2]
    if changed == "mode":
        assert other_call["response_format"]["json_schema"]["strict"] is True
    else:
        assert other_call["tools"][0]["function"]["strict"] is True

    # Only the original combination remembers that it needed a fallback.
    await provider.complete(first)
    assert "strict" not in client.calls[3]["tools"][0]["function"]
    assert len(client.calls) == 4


async def test_openai_truncation_is_reported_not_silently_empty():
    client = FakeOpenAIClient([openai_completion(content="", finish_reason="length")])
    with pytest.raises(ProviderError, match="token limit"):
        await OpenAIProvider(client=client).complete(request_for("openai"))


async def test_openai_content_filter_is_not_retried():
    client = FakeOpenAIClient([openai_completion(content="", finish_reason="content_filter")])
    with pytest.raises(ProviderError) as exc:
        await OpenAIProvider(client=client).complete(request_for("openai"))
    assert exc.value.retryable is False


def test_reasoning_model_detection():
    assert is_reasoning_model("o3-mini")
    assert is_reasoning_model("gpt-5")
    assert not is_reasoning_model("gpt-4.1")
    assert not is_reasoning_model("llama3.1:8b")


# ==========================================================================
# Ollama
# ==========================================================================


class FakeHTTPResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or json.dumps(payload)

    def json(self):
        return self._payload


class FakeHTTPClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url, json=None):
        self.calls.append((url, json))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self):
        pass


def ollama_reply(content, *, done_reason="stop"):
    return FakeHTTPResponse(
        {
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": content},
            "done_reason": done_reason,
            "prompt_eval_count": 210,
            "eval_count": 44,
        }
    )


def test_ollama_payload_constrains_decoding_with_the_schema():
    payload = OllamaProvider().build_payload(request_for("ollama", id="llama3.1:8b"))
    assert payload["format"] == SCHEMA  # full schema, no keywords dropped
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == 4096
    assert payload["options"]["temperature"] == 0.0
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


def test_ollama_options_are_passed_through():
    request = request_for(
        "ollama",
        id="qwen2.5-coder:7b",
        options={"num_ctx": 8192, "keep_alive": "30m", "top_k": 20, "seed": 100},
    )
    payload = OllamaProvider().build_payload(request)
    assert payload["options"]["num_ctx"] == 8192
    assert payload["options"]["top_k"] == 20
    assert payload["options"]["seed"] == 100
    assert payload["keep_alive"] == "30m"


def test_ollama_base_url_defaults_to_localhost_and_is_overridable():
    assert OllamaProvider.base_url(request_for("ollama")) == "http://localhost:11434"
    request = request_for("ollama", options={"base_url": "http://gpu-box:11434/"})
    assert OllamaProvider.base_url(request) == "http://gpu-box:11434"


async def test_ollama_response_is_parsed():
    client = FakeHTTPClient([ollama_reply(json.dumps(VERDICT))])
    response = await OllamaProvider(client=client).complete(request_for("ollama"))
    assert response.payload == VERDICT
    assert response.usage.input_tokens == 210
    assert response.usage.output_tokens == 44
    assert client.calls[0][0] == "/api/chat"


async def test_ollama_unparseable_content_yields_no_payload_not_an_exception():
    """The engine records this as a parse failure; the provider does not guess."""
    client = FakeHTTPClient([ollama_reply("I think candidate A is better.")])
    response = await OllamaProvider(client=client).complete(request_for("ollama"))
    assert response.payload is None
    assert response.text.startswith("I think")


async def test_ollama_truncation_is_reported():
    client = FakeHTTPClient([ollama_reply('{"explanation": "tru', done_reason="length")])
    with pytest.raises(ProviderError, match="token limit"):
        await OllamaProvider(client=client).complete(request_for("ollama"))


async def test_ollama_missing_model_names_the_pull_command():
    client = FakeHTTPClient(
        [FakeHTTPResponse({}, status_code=404, text='{"error":"model not found"}')]
    )
    with pytest.raises(ProviderError) as exc:
        await OllamaProvider(client=client).complete(request_for("ollama", id="llama3.1:8b"))
    assert "ollama pull llama3.1:8b" in str(exc.value)
    assert exc.value.retryable is False


async def test_ollama_connection_failure_is_retryable_and_actionable():
    class ConnectError(Exception):
        pass

    client = FakeHTTPClient([ConnectError("connection refused")])
    with pytest.raises(ProviderError) as exc:
        await OllamaProvider(client=client).complete(request_for("ollama"))
    assert "ollama serve" in str(exc.value)
    assert exc.value.retryable is True


# ==========================================================================
# Pricing
# ==========================================================================


def test_local_providers_are_not_billed():
    model = ModelSpec(provider="ollama", id="llama3.1:8b")
    assert is_local(model)
    assert price_note(model) is None


def test_unknown_price_produces_a_note_rather_than_a_silent_zero():
    model = ModelSpec(provider="openai", id="gpt-4.1")
    assert price_for(model) is None
    note = price_note(model)
    assert note is not None and "price_per_mtok" in note


def test_configured_price_silences_the_note():
    model = ModelSpec(provider="openai", id="gpt-4.1", price_per_mtok=(2.0, 8.0))
    assert price_for(model) == (2.0, 8.0)
    assert price_note(model) is None
