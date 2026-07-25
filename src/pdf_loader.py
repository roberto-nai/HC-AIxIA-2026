from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


class PDFExtractionError(RuntimeError):
    """Raised when PDF text cannot be extracted."""


@dataclass(frozen=True)
class ExtractedDocument:
    filename: str
    source_pages: list[int]
    text: str
    character_count: int


def extract_pdf_text(
    pdf_path: Path,
    source_page_start: int,
    source_page_end: int | None = None,
) -> ExtractedDocument:
    if not pdf_path.exists():
        raise PDFExtractionError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise PDFExtractionError(f"Expected a PDF file, received: {pdf_path}")
    if source_page_start < 1:
        raise PDFExtractionError("Source page start must be at least 1.")
    if source_page_end is not None and source_page_end < source_page_start:
        raise PDFExtractionError(
            "Source page end must be greater than or equal to source page start."
        )

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise PDFExtractionError(f"Unable to open PDF {pdf_path}: {exc}") from exc

    total_pages = len(reader.pages)
    if source_page_start > total_pages:
        raise PDFExtractionError(
            f"Source page start {source_page_start} exceeds the PDF page count "
            f"({total_pages})."
        )

    selected_page_end = source_page_end or total_pages
    if selected_page_end > total_pages:
        raise PDFExtractionError(
            f"Source page end {selected_page_end} exceeds the PDF page count "
            f"({total_pages})."
        )

    page_blocks: list[str] = []
    source_pages: list[int] = []

    selected_pages = reader.pages[source_page_start - 1 : selected_page_end]
    for index, page in enumerate(selected_pages):
        source_page = source_page_start + index
        source_pages.append(source_page)

        try:
            page_text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:
            # Compatibility fallback for older pypdf versions.
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise PDFExtractionError(
                f"Unable to extract text from PDF page {source_page}: {exc}"
            ) from exc

        page_blocks.append(
            f"===== SOURCE PAGE {source_page} =====\n{page_text.strip()}"
        )

    document_text = "\n\n".join(page_blocks).strip()
    if not document_text:
        raise PDFExtractionError(
            "No text was extracted from the PDF. The document may require OCR."
        )

    return ExtractedDocument(
        filename=pdf_path.name,
        source_pages=source_pages,
        text=document_text,
        character_count=len(document_text),
    )
