from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("output")


class ConfigurationError(ValueError):
    """Raised when a prompt or LLM configuration file is invalid."""


@dataclass(frozen=True)
class PromptConfiguration:
    name: str
    system_prompt: str
    user_prompt_template: str
    output_schema: dict[str, Any]


@dataclass(frozen=True)
class LLMConfiguration:
    provider: str
    model: str
    api_key_env: str | None
    base_url: str | None
    timeout_seconds: int
    parameters: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError(f"The root element in {path} must be a JSON object.")

    return data


def load_prompt_configuration(path: Path) -> PromptConfiguration:
    data = _load_json(path)
    required = ("name", "system_prompt", "user_prompt_template", "output_schema")
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigurationError(f"Missing prompt fields in {path}: {', '.join(missing)}")

    if not isinstance(data["output_schema"], dict):
        raise ConfigurationError("'output_schema' must be a JSON object.")

    return PromptConfiguration(
        name=str(data["name"]),
        system_prompt=str(data["system_prompt"]),
        user_prompt_template=str(data["user_prompt_template"]),
        output_schema=data["output_schema"],
    )


def load_llm_configuration(path: Path) -> LLMConfiguration:
    data = _load_json(path)
    required = ("provider", "model", "parameters")
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigurationError(f"Missing LLM fields in {path}: {', '.join(missing)}")

    parameters = data["parameters"]
    if not isinstance(parameters, dict):
        raise ConfigurationError("'parameters' must be a JSON object.")

    return LLMConfiguration(
        provider=str(data["provider"]).lower(),
        model=str(data["model"]),
        api_key_env=str(data["api_key_env"]) if data.get("api_key_env") else None,
        base_url=str(data["base_url"]) if data.get("base_url") else None,
        timeout_seconds=int(data.get("timeout_seconds", 180)),
        parameters=parameters,
    )
