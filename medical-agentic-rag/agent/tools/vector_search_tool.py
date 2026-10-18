
"""Semantic and lexical search over the curated local medical knowledge base."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from agent.state import Evidence, TrustTier, utc_now_iso
from agent.tools.base import BaseTool
from config.settings import Settings, get_settings
from ingestion.embedder import BaseEmbedder, build_embedder
from stores.lexical_index import BM25Index, reciprocal_rank_fusion
from stores.vector_store import BaseVectorStore, build_vector_store

logger = logging.getLogger(__name__)

# These are the allowed values in agent.state.DataStatus.
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

# Convert legacy or incompatible values into the accepted vocabulary.
DATA_STATUS_ALIASES = {
    "unverified": "unknown",
    "verified": "unknown",
    "": "unknown",
    "none": "unknown",
    "null": "unknown",
}


def normalize_data_status(value: Any) -> str:
    """Return a data status accepted by the Evidence model."""
    if value is None:
        return "unknown"

    status = str(value).strip().lower()
    status = DATA_STATUS_ALIASES.get(status, status)

    if status not in ALLOWED_DATA_STATUSES:
        logger.warning(
            "Unknown data_status %r in retrieved metadata; using 'unknown'",
            value,
        )
        return "unknown"

    return status


def normalize_trust_tier(value: Any) -> TrustTier:
    """Convert stored trust-tier metadata to the TrustTier enum safely."""
    if isinstance(value, TrustTier):
        return value

    if value is None:
        return TrustTier("UNKNOWN")

    status = str(value).strip().upper()

    try:
        return TrustTier(status)
    except ValueError:
        logger.warning(
            "Unknown trust_tier %r in retrieved metadata; using UNKNOWN",
            value,
        )
        return TrustTier("UNKNOWN")


class VectorSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=2,
        max_length=512,
        description="Search query text.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to return.",
    )


class VectorSearchTool(BaseTool):
    """Retrieve relevant chunks from the curated local medical corpus."""

    name: ClassVar[str] = "vector_search"
    description: ClassVar[str] = (
        "Search the curated local medical knowledge base for relevant passages. "
        "Use for general medical background, mechanisms, indications, cautions, "
        "and questions the local corpus may cover. Returns text chunks with "
        "source provenance. Retrieved content is evidence, not a substitute "
        "for clinical judgement."
    )
    args_model: ClassVar[type[BaseModel]] = VectorSearchArgs

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        embedder: BaseEmbedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = vector_store
        self._embedder = embedder
        self._lexical: BM25Index | None = None
        self._lexical_size: int = -1

    @property
    def store(self) -> BaseVectorStore:
        if self._store is None:
            self._store = build_vector_store(self._settings)
        return self._store

    @property
    def embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder = build_embedder(self._settings)
        return self._embedder

    @property
    def lexical(self) -> BM25Index:
        """Build BM25 over stored chunks; rebuild when the store size changes."""
        count = self.store.count()

        if self._lexical is None or self._lexical_size != count:
            self._lexical = BM25Index(self.store.list_payloads())
            self._lexical_size = count

        return self._lexical

    def run(
        self,
        args: BaseModel,
    ) -> tuple[list[Evidence], dict[str, Any]]:
        if not isinstance(args, VectorSearchArgs):
            args = VectorSearchArgs.model_validate(args.model_dump())

        top_k = min(args.top_k, self._settings.top_k * 2)
        store_count = self.store.count()

        if store_count == 0:
            return [], {
                "hits": 0,
                "index_empty": True,
                "message": (
                    "The vector index is empty. Build the index using the "
                    "project's ingestion pipeline before querying."
                ),
            }

        # Retrieve candidates from dense search and BM25.
        candidate_pool = max(top_k * 3, 10)

        query_vector = self.embedder.embed_query(args.query)
        dense_hits = self.store.search(
            query_vector,
            top_k=candidate_pool,
        )

        dense_by_id = {
            hit.payload.get("chunk_id") or hit.id: hit
            for hit in dense_hits
        }
        dense_ranking = [
            hit.payload.get("chunk_id") or hit.id
            for hit in dense_hits
        ]

        lexical_hits = self.lexical.search(
            args.query,
            top_k=candidate_pool,
        )

        lexical_by_id: dict[str, dict[str, Any]] = {}
        lexical_ranking: list[str] = []
        lexical_scores: dict[str, float] = {}

        for payload, score in lexical_hits:
            key = payload.get("chunk_id") or payload.get("_record_id")

            if key is None:
                continue

            key = str(key)
            lexical_by_id[key] = payload
            lexical_ranking.append(key)
            lexical_scores[key] = float(score)

        fused = reciprocal_rank_fusion(
            [dense_ranking, lexical_ranking]
        )
        ordered = sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        evidence: list[Evidence] = []

        for key, fused_score in ordered:
            if len(evidence) >= top_k:
                break

            hit = dense_by_id.get(key)
            payload = (
                hit.payload
                if hit is not None
                else lexical_by_id.get(key)
            )

            if not payload:
                continue

            dense_score = float(hit.score) if hit is not None else 0.0
            lexical_score = lexical_scores.get(key, 0.0)

            # Keep a result if either retrieval method finds a usable match.
            if (
                dense_score < self._settings.min_similarity
                and lexical_score <= 0.0
            ):
                continue

            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            data_status = normalize_data_status(
                payload.get("data_status", "unknown")
            )
            trust_tier = normalize_trust_tier(
                payload.get("trust_tier", "UNKNOWN")
            )

            try:
                item = Evidence(
                    text=text,
                    source=payload.get("source") or "local corpus",
                    title=payload.get("title"),
                    url=payload.get("url"),
                    publication_date=payload.get("publication_date"),
                    retrieved_at=(
                        payload.get("retrieved_at") or utc_now_iso()
                    ),
                    evidence_level=payload.get("evidence_level"),
                    document_type=payload.get("document_type"),
                    section=payload.get("section"),
                    document_id=payload.get("document_id"),
                    chunk_id=payload.get("chunk_id"),
                    # Preserve the dense score's existing scale.
                    score=round(dense_score, 4),
                    trust_tier=trust_tier,
                    data_status=data_status,
                    extra={
                        "retrieval": "hybrid_vector_bm25",
                        "file_name": payload.get("file_name"),
                        "fused_score": round(float(fused_score), 5),
                        "lexical_score": round(lexical_score, 4),
                        "semantic_embeddings": bool(
                            self.embedder.describe().get("semantic")
                        ),
                        "matched_by": (
                            "both"
                            if hit is not None and lexical_score > 0
                            else "dense"
                            if hit is not None
                            else "lexical"
                        ),
                    },
                )
            except Exception:
                logger.exception(
                    "Could not create Evidence for retrieved chunk %s",
                    key,
                )
                continue

            evidence.append(item)

        return evidence, {
            "hits": len(evidence),
            "candidates": len(fused),
            "index_empty": False,
            "retrieval_mode": "hybrid (dense + BM25, reciprocal rank fusion)",
            "dense_candidates": len(dense_hits),
            "lexical_candidates": len(lexical_hits),
            "min_similarity": self._settings.min_similarity,
            "embedder": self.embedder.describe(),
        }
