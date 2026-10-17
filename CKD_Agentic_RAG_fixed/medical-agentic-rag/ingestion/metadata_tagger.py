"""Attach provenance metadata to chunks.

Missing values are recorded as ``None``. Nothing here guesses a title, a date or
an evidence level that the source document did not provide.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ingestion.chunker import Chunk
from ingestion.collect_sources import RawDocument
from safety.source_trust import classify_url, evidence_level_for_tier

def build_embed_text(title: str | None, section: str | None, text: str) -> str:
    """Prefix a chunk with its document title and section ("contextual header").

    A chunk that reads "It is commonly assessed using a urine albumin-to-creatinine
    ratio" carries no indication that it is about CKD staging. Prefixing the title
    and heading path gives both the embedding model and the lexical index the
    context that the surrounding document provided, which is the single cheapest
    retrieval improvement available here.
    """
    header_parts = [part for part in (title, section) if part]
    header = " \u2014 ".join(header_parts)
    return f"{header}\n\n{text}" if header else text


METADATA_FIELDS = (
    "document_id", "chunk_id", "title", "source", "url", "publication_date",
    "retrieved_at", "section", "evidence_level", "document_type", "trust_tier",
    "data_status", "block_kinds", "char_count", "chunk_index", "file_name",
)


def tag_chunk(chunk: Chunk, document: RawDocument) -> dict[str, Any]:
    """Build the payload stored alongside a chunk vector."""
    tier = classify_url(document.url)
    payload: dict[str, Any] = {
        "document_id": document.document_id,
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        # Indexed representation; "text" stays the clean text shown to the user.
        "embed_text": build_embed_text(document.title, chunk.section, chunk.text),
        "title": document.title,
        "source": document.source,
        "url": document.url,
        "publication_date": document.publication_date,
        "retrieved_at": document.retrieved_at or datetime.now(timezone.utc).isoformat(),
        "section": chunk.section,
        # Prefer the document's declared level; only fall back to the tier map.
        "evidence_level": document.evidence_level or evidence_level_for_tier(tier),
        "document_type": document.document_type,
        "trust_tier": tier.value,
        "data_status": document.data_status,
        "block_kinds": list(chunk.block_kinds),
        "char_count": chunk.char_count,
        "chunk_index": chunk.index,
        "file_name": document.path.name,
    }
    for key, value in (document.extra or {}).items():
        payload.setdefault(f"extra_{key}", value)
    return payload


def tag_document(document: RawDocument, chunks: list[Chunk]) -> list[dict[str, Any]]:
    """Tag every chunk of one document."""
    return [tag_chunk(chunk, document) for chunk in chunks]


def validate_payload(payload: dict[str, Any]) -> bool:
    """A payload is usable only if it can be traced back to a source."""
    return bool(payload.get("text")) and bool(payload.get("chunk_id")) and bool(
        payload.get("document_id")
    )
