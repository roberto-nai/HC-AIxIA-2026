from __future__ import annotations

from src.config_loader import LLMConfiguration
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import LLMProvider, ProviderError
from src.providers.ollama_provider import OllamaProvider
from src.providers.openai_provider import OpenAIProvider


def create_provider(configuration: LLMConfiguration) -> LLMProvider:
    providers: dict[str, type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }

    provider_class = providers.get(configuration.provider)
    if provider_class is None:
        supported = ", ".join(sorted(providers))
        raise ProviderError(
            f"Unsupported provider '{configuration.provider}'. Supported providers: {supported}."
        )

    return provider_class(configuration)
