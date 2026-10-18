"""Indexing: embed tagged chunks and write them into the vector store."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from ingestion.embedder import BaseEmbedder
from stores.vector_store import BaseVectorStore, VectorRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexReport:
    """Summary emitted at the end of an ingestion run."""

    documents: int = 0
    chunks_indexed: int = 0
    chunks_skipped: int = 0
    vectors_in_store: int = 0
    embedder: dict[str, Any] = field(default_factory=dict)
    vector_backend: str = ""
    per_document: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks_indexed": self.chunks_indexed,
            "chunks_skipped": self.chunks_skipped,
            "vectors_in_store": self.vectors_in_store,
            "embedder": self.embedder,
            "vector_backend": self.vector_backend,
            "per_document": self.per_document,
            "errors": self.errors,
        }

    def render(self) -> str:
        lines = [
            "Ingestion summary",
            "-----------------",
            f"documents        : {self.documents}",
            f"chunks indexed   : {self.chunks_indexed}",
            f"chunks skipped   : {self.chunks_skipped}",
            f"vectors in store : {self.vectors_in_store}",
            f"vector backend   : {self.vector_backend}",
            f"embedder         : {self.embedder}",
        ]
        for entry in self.per_document:
            lines.append(
                f"  - {entry['file_name']}: {entry['chunks']} chunk(s) "
                f"[{entry.get('title') or 'untitled'}]"
            )
        for err in self.errors:
            lines.append(f"  ! {err}")
        return "\n".join(lines)


class Indexer:
    """Embeds payloads in batches and upserts them into a vector store."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        batch_size: int = 16,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.batch_size = max(1, batch_size)

    def index_payloads(self, payloads: Sequence[dict[str, Any]]) -> int:
        """Embed and store payloads; returns the number of vectors written."""
        if not payloads:
            return 0
        self.vector_store.ensure_collection(self.embedder.dimension)
        written = 0
        for start in range(0, len(payloads), self.batch_size):
            batch = payloads[start : start + self.batch_size]
            vectors = self.embedder.embed_documents(
                [p.get("embed_text") or p["text"] for p in batch]
            )
            records = [
                VectorRecord(id=payload["chunk_id"], vector=vector, payload=payload)
                for payload, vector in zip(batch, vectors, strict=True)
            ]
            written += self.vector_store.upsert(records)
            logger.debug("Indexed batch of %d chunk(s)", len(records))
        return written
