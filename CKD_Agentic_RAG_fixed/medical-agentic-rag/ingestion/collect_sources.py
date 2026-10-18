"""Source collection: load raw documents from disk into normalised records.

Supports TXT, Markdown and PDF through a small loader abstraction. Front-matter
metadata (``---`` YAML-style block at the top of a Markdown file) is parsed with
a minimal scalar parser so no extra dependency is required; absent fields stay
``None`` and are never invented.
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

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(slots=True)
class RawDocument:
    """A loaded source document prior to chunking."""

    document_id: str
    text: str
    path: Path
    title: str | None = None
    source: str | None = None
    url: str | None = None
    publication_date: str | None = None
    document_type: str | None = None
    evidence_level: str | None = None
    data_status: str = "demo"
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Unicode-normalise, strip control chars, collapse blank-line runs."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch >= " ")
    text = re.sub(r"[ \t]+(\n)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Extract a simple ``key: value`` front-matter block.

    Returns ``(metadata, body)``. Values ``null``/``none``/``~`` become ``None``.
    """
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    metadata: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip('"').strip("'")
        lowered = value.lower()
        if lowered in {"null", "none", "~", ""}:
            metadata[key.strip()] = None
        elif lowered in {"true", "false"}:
            metadata[key.strip()] = lowered == "true"
        else:
            metadata[key.strip()] = value
    return metadata, text[match.end() :]


def document_id_for(path: Path, text: str) -> str:
    """Stable id derived from filename + content hash (re-ingestion is idempotent)."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}-{digest}"


class BaseLoader(ABC):
    """Loader interface: claim a file extension, return raw text."""

    extensions: tuple[str, ...] = ()

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    @abstractmethod
    def load_text(self, path: Path) -> str:
        """Return the raw text of ``path``."""


class TextLoader(BaseLoader):
    extensions = (".txt",)

    def load_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")


class MarkdownLoader(TextLoader):
    extensions = (".md", ".markdown")


class PdfLoader(BaseLoader):
    extensions = (".pdf",)

    def load_text(self, path: Path) -> str:
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError("pypdf is required to ingest PDF files") from exc
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - one bad page must not kill ingestion
                logger.warning("Failed to extract a page from %s", path.name, exc_info=True)
        return "\n\n".join(pages)


DEFAULT_LOADERS: tuple[BaseLoader, ...] = (TextLoader(), MarkdownLoader(), PdfLoader())


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
        if stripped:
            return None
    return None


def load_document(path: Path, loaders: Iterable[BaseLoader] = DEFAULT_LOADERS) -> RawDocument | None:
    """Load a single file into a :class:`RawDocument` (``None`` if unsupported/empty)."""
    loader = next((ld for ld in loaders if ld.can_load(path)), None)
    if loader is None:
        logger.debug("No loader for %s", path.name)
        return None
    try:
        raw = loader.load_text(path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load %s: %s", path, exc)
        return None

    metadata, body = parse_front_matter(raw)
    body = normalize_text(body)
    if not body:
        logger.warning("Skipping empty document: %s", path.name)
        return None

    return RawDocument(
        document_id=document_id_for(path, body),
        text=body,
        path=path,
        title=metadata.get("title") or _first_heading(body) or path.stem,
        source=metadata.get("source") or "local corpus",
        url=metadata.get("url"),
        publication_date=metadata.get("publication_date"),
        document_type=metadata.get("document_type"),
        evidence_level=metadata.get("evidence_level"),
        data_status=metadata.get("data_status") or "demo",
        extra={k: v for k, v in metadata.items() if k not in {
            "title", "source", "url", "publication_date", "document_type",
            "evidence_level", "data_status",
        }},
    )


def collect_documents(
    directory: Path, loaders: Iterable[BaseLoader] = DEFAULT_LOADERS
) -> list[RawDocument]:
    """Recursively load every supported document under ``directory``."""
    if not directory.exists():
        logger.warning("Source directory does not exist: %s", directory)
        return []
    documents: list[RawDocument] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        document = load_document(path, loaders)
        if document is not None:
            documents.append(document)
    logger.info("Collected %d document(s) from %s", len(documents), directory)
    return documents
