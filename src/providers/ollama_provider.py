from __future__ import annotations

import os
from typing import Any

import httpx
from ollama import Client, ResponseError

from src.config_loader import LLMConfiguration
from src.providers.base import LLMProvider, ProviderError, ProviderResponse


class OllamaProvider(LLMProvider):
    def __init__(self, configuration: LLMConfiguration) -> None:
        super().__init__(configuration)
        configured_url = configuration.base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.base_url = configured_url.rstrip("/")
        self.client = Client(
            host=self.base_url,
            timeout=configuration.timeout_seconds,
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
    ) -> ProviderResponse:
        try:
            response = self.client.chat(
                model=self.configuration.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=output_schema,
                stream=False,
                options=self.configuration.parameters,
            )
        except (ResponseError, httpx.HTTPError) as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        text = response.message.content
        if not text:
            raise ProviderError("Ollama returned no textual output.")

        usage = {
            "input_tokens": response.prompt_eval_count,
            "output_tokens": response.eval_count,
        }

        metadata = {
            "total_duration": response.total_duration,
            "load_duration": response.load_duration,
            "prompt_eval_duration": response.prompt_eval_duration,
            "eval_duration": response.eval_duration,
            "done_reason": response.done_reason,
        }

        return ProviderResponse(text=text, usage=usage, provider_metadata=metadata)
