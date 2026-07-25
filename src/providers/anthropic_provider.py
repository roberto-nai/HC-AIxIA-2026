from __future__ import annotations

import json
import os
from typing import Any

from anthropic import Anthropic

from src.config_loader import LLMConfiguration
from src.providers.base import LLMProvider, ProviderError, ProviderResponse


class AnthropicProvider(LLMProvider):
    def __init__(self, configuration: LLMConfiguration) -> None:
        super().__init__(configuration)

        api_key = None
        if configuration.api_key_env:
            api_key = os.getenv(configuration.api_key_env)
            if not api_key:
                raise ProviderError(
                    f"Environment variable {configuration.api_key_env} is not set."
                )

        client_options: dict[str, Any] = {
            "api_key": api_key,
            "timeout": configuration.timeout_seconds,
        }
        if configuration.base_url:
            client_options["base_url"] = configuration.base_url

        self.client = Anthropic(**client_options)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
    ) -> ProviderResponse:
        parameters = dict(self.configuration.parameters)
        max_tokens = int(parameters.pop("max_tokens", 16000))

        schema_instruction = (
            "\n\nReturn one JSON object only. It must conform exactly to this JSON Schema:\n"
            + json.dumps(output_schema, ensure_ascii=False)
        )

        try:
            message = self.client.messages.create(
                model=self.configuration.model,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt + schema_instruction,
                    }
                ],
                max_tokens=max_tokens,
                **parameters,
            )
        except Exception as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        text_parts = [
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ]
        text = "\n".join(text_parts).strip()
        if not text:
            raise ProviderError("Anthropic returned no textual output.")

        usage = None
        if getattr(message, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(message.usage, "input_tokens", None),
                "output_tokens": getattr(message.usage, "output_tokens", None),
            }

        return ProviderResponse(
            text=text,
            usage=usage,
            provider_metadata={
                "message_id": getattr(message, "id", None),
                "stop_reason": getattr(message, "stop_reason", None),
            },
        )
