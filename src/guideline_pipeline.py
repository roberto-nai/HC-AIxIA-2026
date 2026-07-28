from __future__ import annotations

import hashlib
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
from src.json_utils import JSONOutputError, extract_json_object, save_json, validate_json
from src.pdf_loader import PDFExtractionError, ExtractedDocument, extract_pdf_text
from src.providers.base import ProviderError
from src.providers.factory import create_provider


class GuidelinePipelineError(RuntimeError):
    """Raised when clinical-guideline extraction fails."""


class GuidelineExtractionPipeline:
    """Independent extraction stage for guideline references in a PDTA PDF."""

    def __init__(
        self,
        *,
        pdf_path: Path,
        prompt_path: Path,
        config_path: Path,
        source_page_start: int,
        source_page_end: int | None,
        output_path: Path | None,
        output_dir: Path,
    ) -> None:
        self.pdf_path = pdf_path
        self.prompt_path = prompt_path
        self.config_path = config_path
        self.source_page_start = source_page_start
        self.source_page_end = source_page_end
        self.output_path = output_path
        self.output_dir = output_dir

    def run(self) -> tuple[Path, Path]:
        load_dotenv()

        try:
            prompt_config = load_prompt_configuration(self.prompt_path)
            llm_config = load_llm_configuration(self.config_path)
            document = extract_pdf_text(
                self.pdf_path,
                self.source_page_start,
                self.source_page_end,
            )
            provider = create_provider(llm_config)
            user_prompt = self._build_user_prompt(prompt_config, document)
            started_at = datetime.now(timezone.utc)

            response = provider.generate(
                system_prompt=prompt_config.system_prompt,
                user_prompt=user_prompt,
                output_schema=prompt_config.output_schema,
            )
            completed_at = datetime.now(timezone.utc)

            base_output_path = self.output_path or self._default_output_path(
                llm_config, document, started_at
            )
            data_path, metadata_path = self._output_paths(base_output_path)

            try:
                extracted = extract_json_object(response.text)
                validate_json(extracted, prompt_config.output_schema)
            except JSONOutputError:
                error_path = base_output_path.with_name(
                    self._base_output_stem(base_output_path) + "_error.json"
                )
                save_json(
                    error_path,
                    {"source_file": document.filename, "raw_response": response.text},
                )
                raise

            data_output = {
                "source_document": document.filename,
                "source_pages": document.source_pages,
                "task": "clinical_guideline_extraction",
                "guidelines": extracted["guidelines"],
            }
            metadata_output = self._build_metadata_output(
                document=document,
                prompt_config=prompt_config,
                llm_config=llm_config,
                started_at=started_at,
                completed_at=completed_at,
                usage=response.usage,
                provider_metadata=response.provider_metadata,
                data_path=data_path,
            )

            save_json(data_path, data_output)
            save_json(metadata_path, metadata_output)
            return data_path, metadata_path

        except (
            ConfigurationError,
            PDFExtractionError,
            ProviderError,
            JSONOutputError,
            OSError,
        ) as exc:
            raise GuidelinePipelineError(str(exc)) from exc

    @staticmethod
    def _build_user_prompt(
        prompt_config: PromptConfiguration,
        document: ExtractedDocument,
    ) -> str:
        replacements = {
            "source_filename": document.filename,
            "first_source_page": document.source_pages[0],
            "last_source_page": document.source_pages[-1],
            "document_text": document.text,
        }
        try:
            return prompt_config.user_prompt_template.format(**replacements)
        except KeyError as exc:
            raise GuidelinePipelineError(
                f"Unknown placeholder in guideline prompt template: {exc.args[0]}"
            ) from exc

    def _build_metadata_output(
        self,
        *,
        document: ExtractedDocument,
        prompt_config: PromptConfiguration,
        llm_config: LLMConfiguration,
        started_at: datetime,
        completed_at: datetime,
        usage: dict[str, Any] | None,
        provider_metadata: dict[str, Any] | None,
        data_path: Path,
    ) -> dict[str, Any]:
        return {
            "source_file": document.filename,
            "source_pages": document.source_pages,
            "task": "clinical_guideline_extraction",
            "prompt_name": prompt_config.name,
            "data_file": data_path.name,
            "llm_execution": {
                "run_id": f"{self._slug(llm_config.model)}_{started_at.strftime('%Y%m%dT%H%M%SZ')}",
                "provider": llm_config.provider,
                "model": llm_config.model,
                "timestamp": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
                "duration_minutes": round((completed_at - started_at).total_seconds() / 60, 3),
                "prompt_name": prompt_config.name,
                "configuration_file": self.config_path.name,
                "prompt_file": self.prompt_path.name,
                "parameters": llm_config.parameters,
                "usage": usage,
                "provider_metadata": provider_metadata,
            },
            "document": {
                "filename": document.filename,
                "pages": document.source_pages,
                "text_length": document.character_count,
                "sha256": self._sha256(self.pdf_path),
            },
        }

    @staticmethod
    def _base_output_stem(path: Path) -> str:
        stem = path.stem
        for suffix in ("_data", "_metadata"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    @classmethod
    def _output_paths(cls, base_path: Path) -> tuple[Path, Path]:
        base_stem = cls._base_output_stem(base_path)
        return (
            base_path.with_name(base_stem + "_data.json"),
            base_path.with_name(base_stem + "_metadata.json"),
        )

    def _default_output_path(
        self,
        llm_config: LLMConfiguration,
        document: ExtractedDocument,
        started_at: datetime,
    ) -> Path:
        filename = (
            f"guidelines_{self._slug(Path(document.filename).stem)}_"
            f"{llm_config.provider}_{self._slug(llm_config.model)}_"
            f"{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        return self.output_dir / filename

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
