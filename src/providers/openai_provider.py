from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from openai import OpenAI

from src.config_loader import LLMConfiguration
from src.providers.base import LLMProvider, ProviderError, ProviderResponse


def _make_openai_compatible_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy restricted to OpenAI's supported JSON Schema subset."""
    compatible_schema = deepcopy(schema)

    def transform(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("uniqueItems", None)
            if "oneOf" in value:
                value["anyOf"] = value.pop("oneOf")
            for child in value.values():
                transform(child)
        elif isinstance(value, list):
            for child in value:
                transform(child)

    transform(compatible_schema)
    return compatible_schema


class OpenAIProvider(LLMProvider):
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

        self.client = OpenAI(**client_options)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
    ) -> ProviderResponse:
        parameters = dict(self.configuration.parameters)
        max_output_tokens = parameters.pop("max_output_tokens", 16000)

        try:
            response = self.client.responses.create(
                model=self.configuration.model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "pdta_extraction",
                        "strict": True,
                        "schema": _make_openai_compatible_schema(output_schema),
                    }
                },
                **parameters,
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        text = response.output_text
        if not text:
            raise ProviderError("OpenAI returned no textual output.")

        usage = None
        if getattr(response, "usage", None) is not None:
            usage_obj = response.usage
            usage = {
                "input_tokens": getattr(usage_obj, "input_tokens", None),
                "output_tokens": getattr(usage_obj, "output_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }

        return ProviderResponse(
            text=text,
            usage=usage,
            provider_metadata={"response_id": getattr(response, "id", None)},
        )
