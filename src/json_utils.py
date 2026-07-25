from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


class JSONOutputError(ValueError):
    """Raised when an LLM response is not valid or does not match the schema."""


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise JSONOutputError("The LLM returned an empty response.")

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
        else:
            first = text.find("{")
            last = text.rfind("}")
            if first == -1 or last == -1 or last <= first:
                raise JSONOutputError("No JSON object was found in the LLM response.")
            candidate = text[first : last + 1]

        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise JSONOutputError(f"The LLM response contains invalid JSON: {exc}") from exc

    if not isinstance(value, dict):
        raise JSONOutputError("The LLM response root must be a JSON object.")

    return value


def validate_json(data: dict[str, Any], schema: dict[str, Any]) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.absolute_path))
    if not errors:
        return

    messages = []
    for error in errors[:10]:
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        messages.append(f"{path}: {error.message}")

    suffix = "" if len(errors) <= 10 else f" (+{len(errors) - 10} more errors)"
    raise JSONOutputError("Schema validation failed: " + "; ".join(messages) + suffix)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
