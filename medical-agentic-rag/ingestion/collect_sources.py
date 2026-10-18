
"""Load and normalize source documents for the CKD Agentic RAG pipeline.

Supports TXT, Markdown, and PDF files.
Unknown provenance defaults to 'unknown', matching agent.state.DataStatus.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.DOTALL)

ALLOWED_DATA_STATUSES = {
    "demo",
    "guideline",
    "regulatory",
    "reference",
    "literature",
    "licensed",
    "live",
    "unknown",
}


@dataclass(slots=True)
class RawDocument:
    """A loaded source document before chunking."""

    document_id: str
    text: str
    path: Path
    title: str | None = None
    source: str | None = None
    url: str | None = None
    publication_date: str | None = None
    document_type: str | None = None
    evidence_level: str | None = None
    data_status: str = "unknown"
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(char for char in text if char in "\n\t" or char >= " ")
    text = re.sub(r"[ \t]+(\n)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text

    metadata: dict[str, Any] = {}

    for line in match.group(1).splitlines():
        line = line.strip()

        if not line or line.startswith("#") or ":" not in line:
            continue

        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        lowered = value.lower()

        if lowered in {"null", "none", "~", ""}:
            metadata[key] = None
        elif lowered in {"true", "false"}:
            metadata[key] = lowered == "true"
        else:
            metadata[key] = value

    return metadata, text[match.end():]


def document_id_for(path: Path, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}-{digest}"


class BaseLoader(ABC):
    extensions: tuple[str, ...] = ()

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    @abstractmethod
    def load_text(self, path: Path) -> str:
        """Read a file and return its raw text."""


class TextLoader(BaseLoader):
    extensions = (".txt",)

    def load_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")


class MarkdownLoader(TextLoader):
    extensions = (".md", ".markdown")


class PdfLoader(BaseLoader):
    extensions = (".pdf",)

    def load_text(self, path: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: list[str] = []

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                logger.warning(
                    "Could not extract text from page %s of %s",
                    page_number,
                    path.name,
                    exc_info=True,
                )

        return "\n\n".join(pages)


DEFAULT_LOADERS: tuple[BaseLoader, ...] = (
    TextLoader(),
    MarkdownLoader(),
    PdfLoader(),
)


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None

        return None

    return None


def _safe_data_status(value: Any, path: Path) -> str:
    if not isinstance(value, str):
        return "unknown"

    status = value.strip().lower()

    if status not in ALLOWED_DATA_STATUSES:
        logger.warning(
            "Unknown data_status %r in %s; using 'unknown'",
            value,
            path.name,
        )
        return "unknown"

    return status


def load_document(
    path: Path,
    loaders: Iterable[BaseLoader] = DEFAULT_LOADERS,
) -> RawDocument | None:
    loader = next((item for item in loaders if item.can_load(path)), None)

    if loader is None:
        logger.debug("No loader available for %s", path.name)
        return None

    try:
        raw_text = loader.load_text(path)
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return None

    metadata, body = parse_front_matter(raw_text)
    body = normalize_text(body)

    if not body:
        logger.warning("Skipping empty document: %s", path.name)
        return None

    known_metadata = {
        "title",
        "source",
        "url",
        "publication_date",
        "document_type",
        "evidence_level",
        "data_status",
    }

    title = metadata.get("title") or _first_heading(body) or path.stem
    source = metadata.get("source") or "local corpus"

    return RawDocument(
        document_id=document_id_for(path, body),
        text=body,
        path=path,
        title=str(title),
        source=str(source),
        url=metadata.get("url"),
        publication_date=metadata.get("publication_date"),
        document_type=metadata.get("document_type"),
        evidence_level=metadata.get("evidence_level"),
        data_status=_safe_data_status(
            metadata.get("data_status", "unknown"),
            path,
        ),
        extra={
            key: value
            for key, value in metadata.items()
            if key not in known_metadata
        },
    )


def collect_documents(
    directory: Path,
    loaders: Iterable[BaseLoader] = DEFAULT_LOADERS,
) -> list[RawDocument]:
    if not directory.exists() or not directory.is_dir():
        logger.warning("Source directory is missing or invalid: %s", directory)
        return []

    documents: list[RawDocument] = []

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        document = load_document(path, loaders)

        if document is not None:
            documents.append(document)

    logger.info(
        "Collected %d document(s) from %s",
        len(documents),
        directory,
    )
    return documents
