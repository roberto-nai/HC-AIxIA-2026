from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.config_loader import LLMConfiguration


class ProviderError(RuntimeError):
    """Raised when a provider call fails."""


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    usage: dict[str, Any] | None = None
    provider_metadata: dict[str, Any] | None = None


class LLMProvider(ABC):
    def __init__(self, configuration: LLMConfiguration) -> None:
        self.configuration = configuration

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
    ) -> ProviderResponse:
        raise NotImplementedError
