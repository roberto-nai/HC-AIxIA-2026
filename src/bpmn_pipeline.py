from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.config_loader import (
    ConfigurationError,
    LLMConfiguration,
    PromptConfiguration,
    load_llm_configuration,
    load_prompt_configuration,
)
from src.json_utils import (
    JSONOutputError,
    extract_json_object,
    save_json,
    save_text,
    validate_json,
)
from src.providers.base import ProviderError
from src.providers.factory import create_provider


class BPMNKnowledgeError(RuntimeError):
    """Raised when BPMN-oriented knowledge extraction fails."""


class BPMNKnowledgePipeline:
    """Extract BPMN-oriented knowledge and create detailed and compact outputs."""

    def __init__(
        self,
        *,
        source_json_path: Path,
        prompt_path: Path,
        config_path: Path,
        output_path: Path | None,
        summary_output_path: Path | None,
        output_dir: Path,
    ) -> None:
        self.source_json_path = source_json_path
        self.prompt_path = prompt_path
        self.config_path = config_path
        self.output_path = output_path
        self.summary_output_path = summary_output_path
        self.output_dir = output_dir

    def run(self) -> tuple[Path, Path, Path]:
        """Run Stage 2 and return detailed, summary, and metadata paths."""
        load_dotenv()

        try:
            source = self._load_source_json(self.source_json_path)
            prompt_config = load_prompt_configuration(self.prompt_path)
            llm_config = load_llm_configuration(self.config_path)
            provider = create_provider(llm_config)

            user_prompt = self._build_user_prompt(prompt_config, source)
            started_at = datetime.now(timezone.utc)

            detailed_path = self.output_path or self._default_detailed_output_path(
                llm_config, started_at
            )
            detailed_path.parent.mkdir(parents=True, exist_ok=True)

            prompt_debug_path = detailed_path.with_name(
                f"bpmn_prompt_{detailed_path.stem.removeprefix('bpmn_')}.txt"
            )
            save_text(
                prompt_debug_path,
                f"SYSTEM PROMPT\n=============\n{prompt_config.system_prompt}\n\n"
                f"USER PROMPT\n===========\n{user_prompt}\n",
            )

            response = provider.generate(
                system_prompt=prompt_config.system_prompt,
                user_prompt=user_prompt,
                output_schema=prompt_config.output_schema,
            )
            completed_at = datetime.now(timezone.utc)

            raw_path = detailed_path.with_name(
                f"bpmn_raw_{detailed_path.stem.removeprefix('bpmn_')}.txt"
            )
            save_text(raw_path, response.text)

            detailed = extract_json_object(response.text)
            validate_json(detailed, prompt_config.output_schema)
            self._validate_semantics(detailed)
            save_json(detailed_path, detailed)

            summary = self._build_summary(detailed)
            summary_path = self.summary_output_path or self._default_summary_output_path(
                detailed_path
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(summary_path, summary)

            metadata_path = detailed_path.with_name(
                f"bpmn_metadata_{detailed_path.stem.removeprefix('bpmn_')}.json"
            )
            metadata = self._build_metadata(
                detailed_path=detailed_path,
                summary_path=summary_path,
                prompt_debug_path=prompt_debug_path,
                raw_path=raw_path,
                prompt_config=prompt_config,
                llm_config=llm_config,
                started_at=started_at,
                completed_at=completed_at,
                usage=response.usage,
                provider_metadata=response.provider_metadata,
            )
            save_json(metadata_path, metadata)
            return detailed_path, summary_path, metadata_path

        except (
            ConfigurationError,
            ProviderError,
            JSONOutputError,
            OSError,
            ValueError,
        ) as exc:
            raise BPMNKnowledgeError(str(exc)) from exc

    @staticmethod
    def _load_source_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ValueError(f"Structured source JSON not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as stream:
                source = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

        if not isinstance(source, dict):
            raise ValueError("The structured source JSON root must be an object.")
        records = source.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(
                "The structured source JSON must contain a non-empty 'records' array."
            )
        return source

    @staticmethod
    def _build_user_prompt(
        prompt_config: PromptConfiguration,
        source: dict[str, Any],
    ) -> str:
        source_text = json.dumps(source, ensure_ascii=False, indent=2)
        try:
            return prompt_config.user_prompt_template.format(source_json=source_text)
        except KeyError as exc:
            raise ValueError(
                f"Unknown placeholder in BPMN prompt template: {exc.args[0]}"
            ) from exc

    @staticmethod
    def _build_summary(detailed: dict[str, Any]) -> dict[str, Any]:
        """Create a source-independent compact representation for BPMN modelling."""

        def labels(key: str) -> list[str]:
            items = detailed.get(key, [])
            return [
                item["label"]
                for item in items
                if isinstance(item, dict)
                and isinstance(item.get("label"), str)
                and item["label"].strip()
            ]

        lanes = labels("lanes")
        activities = labels("activities")
        resources = labels("resources")

        return {
            "summary": {
                "lanes": len(lanes),
                "activities": len(activities),
                "resources": len(resources),
            },
            "lanes": lanes,
            "activities": activities,
            "resources": resources,
        }

    @staticmethod
    def _validate_semantics(result: dict[str, Any]) -> None:
        lanes = result.get("lanes")
        activities = result.get("activities")
        resources = result.get("resources")
        assignments = result.get("resource_activity_assignments")

        if not isinstance(lanes, list):
            raise JSONOutputError("'lanes' must be an array.")
        if not isinstance(activities, list) or not activities:
            raise JSONOutputError("The model returned no BPMN-oriented activities.")
        if not isinstance(resources, list):
            raise JSONOutputError("'resources' must be an array.")
        if not isinstance(assignments, list):
            raise JSONOutputError("'resource_activity_assignments' must be an array.")

        BPMNKnowledgePipeline._ensure_unique_ids(lanes, "lane")
        BPMNKnowledgePipeline._ensure_unique_ids(activities, "activity")
        BPMNKnowledgePipeline._ensure_unique_ids(resources, "resource")

        activity_ids = {item["id"] for item in activities}
        resource_ids = {item["id"] for item in resources}

        for index, assignment in enumerate(assignments, start=1):
            if assignment.get("activity_id") not in activity_ids:
                raise JSONOutputError(
                    f"Assignment {index} references an unknown activity_id: "
                    f"{assignment.get('activity_id')}"
                )
            if assignment.get("resource_id") not in resource_ids:
                raise JSONOutputError(
                    f"Assignment {index} references an unknown resource_id: "
                    f"{assignment.get('resource_id')}"
                )

    @staticmethod
    def _ensure_unique_ids(items: list[dict[str, Any]], element_name: str) -> None:
        identifiers = [item.get("id") for item in items]
        duplicates = sorted(
            identifier
            for identifier in set(identifiers)
            if identifier is not None and identifiers.count(identifier) > 1
        )
        if duplicates:
            raise JSONOutputError(
                f"Duplicate {element_name} IDs: {', '.join(duplicates)}"
            )

    def _default_detailed_output_path(
        self,
        llm_config: LLMConfiguration,
        started_at: datetime,
    ) -> Path:
        source_stem = self._slug(self.source_json_path.stem)
        model_slug = self._slug(llm_config.model)
        timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
        filename = (
            f"bpmn_knowledge_detailed_{source_stem}_{llm_config.provider}_"
            f"{model_slug}_{timestamp}.json"
        )
        return self.output_dir / filename

    @staticmethod
    def _default_summary_output_path(detailed_path: Path) -> Path:
        detailed_name = detailed_path.name
        if detailed_name.startswith("bpmn_knowledge_detailed_"):
            summary_name = detailed_name.replace(
                "bpmn_knowledge_detailed_", "bpmn_knowledge_summary_", 1
            )
        elif detailed_name.startswith("bpmn_"):
            summary_name = f"bpmn_summary_{detailed_name[len('bpmn_'):]}"
        else:
            summary_name = f"bpmn_summary_{detailed_name}"
        return detailed_path.with_name(summary_name)

    def _build_metadata(
        self,
        *,
        detailed_path: Path,
        summary_path: Path,
        prompt_debug_path: Path,
        raw_path: Path,
        prompt_config: PromptConfiguration,
        llm_config: LLMConfiguration,
        started_at: datetime,
        completed_at: datetime,
        usage: dict[str, Any] | None,
        provider_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "task": "bpmn_oriented_knowledge_extraction",
            "provider": llm_config.provider,
            "model": llm_config.model,
            "timestamp": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round(
                (completed_at - started_at).total_seconds(), 3
            ),
            "source_json": str(self.source_json_path),
            "source_sha256": self._sha256(self.source_json_path),
            "prompt_name": prompt_config.name,
            "prompt_file": str(self.prompt_path),
            "configuration_file": str(self.config_path),
            "parameters": llm_config.parameters,
            "usage": usage,
            "provider_metadata": provider_metadata,
            "outputs": {
                "bpmn_knowledge_detailed": str(detailed_path),
                "bpmn_knowledge_summary": str(summary_path),
                "bpmn_prompt": str(prompt_debug_path),
                "bpmn_raw": str(raw_path),
            },
        }

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
        return slug.strip("-").lower() or "unknown"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
