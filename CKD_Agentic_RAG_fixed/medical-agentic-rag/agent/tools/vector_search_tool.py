"""Semantic search over the curated local medical knowledge base."""

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


class VectorSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=512, description="Search query text.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to return.")


class VectorSearchTool(BaseTool):
    """Retrieve the most similar chunks from the curated local corpus."""

    name: ClassVar[str] = "vector_search"
    description: ClassVar[str] = (
        "Search the curated local medical knowledge base for passages relevant to a "
        "query. Use for general medical/clinical background, mechanisms, indications, "
        "cautions and any question the local corpus may cover. Returns text chunks "
        "with full source provenance."
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
        """BM25 index over the same chunks, rebuilt when the store changes size."""
        count = self.store.count()
        if self._lexical is None or self._lexical_size != count:
            self._lexical = BM25Index(self.store.list_payloads())
            self._lexical_size = count
        return self._lexical

    def run(self, args: BaseModel) -> tuple[list[Evidence], dict[str, Any]]:
        assert isinstance(args, VectorSearchArgs)
        top_k = min(args.top_k, self._settings.top_k * 2)

        if self.store.count() == 0:
            return [], {
                "hits": 0,
                "index_empty": True,
                "message": (
                    "The vector index is empty. Run `python -m ingestion.run_ingestion` "
                    "before querying."
                ),
            }

        # ---- hybrid retrieval: dense similarity fused with BM25 ------------
        # Dense-only retrieval failed on short keyword questions (a "CKD staging"
        # query returned the medicines document), so both retrievers run and their
        # RANKS are fused. Ranks fuse safely; raw cosine and BM25 scores do not.
        candidate_pool = max(top_k * 3, 10)

        vector = self.embedder.embed_query(args.query)
        dense_hits = self.store.search(vector, top_k=candidate_pool)
        dense_by_id = {h.payload.get("chunk_id") or h.id: h for h in dense_hits}
        dense_ranking = [h.payload.get("chunk_id") or h.id for h in dense_hits]

        lexical_hits = self.lexical.search(args.query, top_k=candidate_pool)
        lexical_by_id: dict[str, dict[str, Any]] = {}
        lexical_ranking: list[str] = []
        lexical_scores: dict[str, float] = {}
        for payload, score in lexical_hits:
            key = payload.get("chunk_id") or payload.get("_record_id")
            if not key:
                continue
            lexical_by_id[key] = payload
            lexical_ranking.append(key)
            lexical_scores[key] = score

        fused = reciprocal_rank_fusion([dense_ranking, lexical_ranking])

        # A chunk qualifies if either retriever found it credible: it cleared the
        # dense similarity floor, or BM25 matched real query terms in it. Requiring
        # both would reinstate the dense-only failure this fusion exists to fix.
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        evidence: list[Evidence] = []
        for key, fused_score in ordered:
            if len(evidence) >= top_k:
                break
            hit = dense_by_id.get(key)
            payload = hit.payload if hit is not None else lexical_by_id.get(key)
            if not payload:
                continue
            dense_score = float(hit.score) if hit is not None else 0.0
            lexical_score = lexical_scores.get(key, 0.0)
            if dense_score < self._settings.min_similarity and lexical_score <= 0.0:
                continue
            text = payload.get("text")
            if not text:
                continue
            evidence.append(
                Evidence(
                    text=text,
                    source=payload.get("source") or "local corpus",
                    title=payload.get("title"),
                    url=payload.get("url"),
                    publication_date=payload.get("publication_date"),
                    retrieved_at=payload.get("retrieved_at") or utc_now_iso(),
                    evidence_level=payload.get("evidence_level"),
                    document_type=payload.get("document_type"),
                    section=payload.get("section"),
                    document_id=payload.get("document_id"),
                    chunk_id=payload.get("chunk_id"),
                    # Reported score stays the dense similarity so the confidence
                    # heuristic keeps interpreting it on its original scale.
                    score=round(dense_score, 4),
                    trust_tier=TrustTier(payload.get("trust_tier", "UNKNOWN")),
                    data_status=payload.get("data_status", "unknown"),
                    extra={
                        "retrieval": "hybrid_vector_bm25",
                        "file_name": payload.get("file_name"),
                        "fused_score": round(fused_score, 5),
                        "lexical_score": round(lexical_score, 4),
                        # Whether `score` is a semantic cosine or a lexical-hash
                        # cosine. The two live on different scales, so anything
                        # interpreting `score` must know which it is holding.
                        "semantic_embeddings": bool(self.embedder.describe().get("semantic")),
                        "matched_by": (
                            "both" if (hit is not None and lexical_score > 0)
                            else ("dense" if hit is not None else "lexical")
                        ),
                    },
                )
            )

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
