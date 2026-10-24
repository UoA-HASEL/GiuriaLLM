"""Model adapters.

Four ship with the framework:

``anthropic``  Claude models, via forced tool use.
``openai``     OpenAI models and any OpenAI-compatible server (vLLM, LM Studio,
               gateways), via strict function calling or response-format schemas.
``ollama``     Locally served open-weight models, via Ollama's schema-constrained
               ``format`` parameter.
``mock``       Deterministic and offline, for tests and dry runs.

Vendor SDKs are imported lazily, so a run that uses one provider does not need
the others installed. Further adapters plug in through :func:`register_provider`
without touching the engine.
"""

from .anthropic_provider import AnthropicProvider, rejects_sampling_params
from .base import (
    LLMRequest,
    LLMResponse,
    Provider,
    ProviderError,
    RawUsage,
    available_providers,
    get_provider,
    register_provider,
)
from .mock import MockProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider, is_reasoning_model, sanitise_schema
from .pricing import LOCAL_PROVIDERS, PRICES, is_local, price_for, price_note

register_provider("anthropic", AnthropicProvider)
register_provider("openai", OpenAIProvider)
register_provider("ollama", OllamaProvider)
register_provider("mock", MockProvider)

__all__ = [
    "LOCAL_PROVIDERS",
    "PRICES",
    "AnthropicProvider",
    "LLMRequest",
    "LLMResponse",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "RawUsage",
    "available_providers",
    "get_provider",
    "is_local",
    "is_reasoning_model",
    "price_for",
    "price_note",
    "register_provider",
    "rejects_sampling_params",
    "sanitise_schema",
]
