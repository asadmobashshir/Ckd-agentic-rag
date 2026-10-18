"""Vector store and retrieval tests."""

from __future__ import annotations

import pytest

from agent.state import ToolCallRequest
from agent.tools.vector_search_tool import VectorSearchTool
from ingestion.indexer import Indexer
from stores.vector_store import NumpyVectorStore, VectorRecord, VectorStoreError

PAYLOADS = [
    {
        "chunk_id": "doc1::c0000",
        "document_id": "doc1",
        "text": "Metformin is used in the management of type 2 diabetes mellitus.",
        "title": "DEMO: Metformin",
        "source": "DEMO corpus",
        "url": None,
        "publication_date": None,
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "section": "Clinical use",
        "evidence_level": "demo_material",
        "document_type": "demo_reference_note",
        "trust_tier": "UNKNOWN",
        "data_status": "demo",
        "file_name": "demo_metformin.md",
    },
    {
        "chunk_id": "doc2::c0000",
        "document_id": "doc2",
        "text": "Warfarin is an anticoagulant monitored using the international normalised ratio.",
        "title": "DEMO: Warfarin",
        "source": "DEMO corpus",
        "url": None,
        "publication_date": None,
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "section": "Monitoring",
        "evidence_level": "demo_material",
        "document_type": "demo_reference_note",
        "trust_tier": "UNKNOWN",
        "data_status": "demo",
        "file_name": "demo_warfarin.md",
    },
]


def test_upsert_search_and_count(vector_store, embedder):
    indexer = Indexer(embedder, vector_store)
    assert indexer.index_payloads(PAYLOADS) == 2
    assert vector_store.count() == 2

    hits = vector_store.search(embedder.embed_query("type 2 diabetes metformin"), top_k=2)
    assert hits
    assert hits[0].payload["document_id"] == "doc1"
    assert hits[0].score >= hits[-1].score


def test_upsert_is_idempotent(vector_store, embedder):
    indexer = Indexer(embedder, vector_store)
    indexer.index_payloads(PAYLOADS)
    indexer.index_payloads(PAYLOADS)
    assert vector_store.count() == 2, "re-indexing the same chunk ids must not duplicate"


def test_index_persists_across_instances(settings, embedder):
    store = NumpyVectorStore(settings.vector_db_path, "persist_test")
    Indexer(embedder, store).index_payloads(PAYLOADS)
    reopened = NumpyVectorStore(settings.vector_db_path, "persist_test")
    assert reopened.count() == 2


def test_dimension_mismatch_is_rejected(vector_store):
    vector_store.ensure_collection(128)
    with pytest.raises(VectorStoreError):
        vector_store.ensure_collection(384)


def test_empty_store_returns_no_hits(vector_store, embedder):
    assert vector_store.search(embedder.embed_query("anything"), top_k=5) == []
    assert vector_store.count() == 0


def test_reset_clears_the_index(vector_store, embedder):
    Indexer(embedder, vector_store).index_payloads(PAYLOADS)
    vector_store.reset()
    assert vector_store.count() == 0


def test_search_results_carry_full_provenance(vector_store, embedder, settings):
    Indexer(embedder, vector_store).index_payloads(PAYLOADS)
    tool = VectorSearchTool(vector_store=vector_store, embedder=embedder, settings=settings)
    evidence, structured = tool.run(tool.validate_arguments({"query": "metformin diabetes"}))

    assert evidence, f"expected retrieval hits, structured={structured}"
    top = evidence[0]
    for field in ("document_id", "chunk_id", "source", "retrieved_at"):
        assert getattr(top, field), f"{field} must survive retrieval"
    assert top.score is not None
    assert top.data_status == "demo"
    # Metadata absent in the source must remain None, never invented.
    assert top.url is None
    assert top.publication_date is None


def test_tool_reports_empty_index_rather_than_guessing(settings, embedder):
    empty = NumpyVectorStore(settings.vector_db_path, "empty_col")
    tool = VectorSearchTool(vector_store=empty, embedder=embedder, settings=settings)
    evidence, structured = tool.run(tool.validate_arguments({"query": "metformin"}))
    assert evidence == []
    assert structured["index_empty"] is True
    assert "ingestion" in structured["message"]
