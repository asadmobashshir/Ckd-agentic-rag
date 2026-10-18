"""Regression tests for the fixes to retrieval, index integrity and LLM reporting.

Each test here pins one of the bugs that made the deployed app misbehave:

* dense-only retrieval returning the wrong document for keyword questions
* an index built by one embedder being reused silently by another
* every LLM failure collapsing into "No language model is configured"

No Gemini key and no network are required.
"""

from __future__ import annotations

import json

import pytest

from agent.tools.vector_search_tool import VectorSearchTool
from ingestion.indexer import Indexer
from ingestion.metadata_tagger import build_embed_text
from llm.base import (
    LLM_STATUS_AUTH,
    LLM_STATUS_MISSING_KEY,
    LLM_STATUS_MODEL,
    LLM_STATUS_NETWORK,
    LLM_STATUS_RATE_LIMIT,
    LLM_STATUS_SDK_MISSING,
    UnavailableLLMClient,
    classify_llm_error,
)
from llm.factory import build_llm_client
from stores.index_meta import (
    IndexFingerprint,
    check_compatibility,
    fingerprint_for,
    read_fingerprint,
    write_fingerprint,
)
from stores.lexical_index import BM25Index, reciprocal_rank_fusion, tokenize

# Two documents that are lexically similar but topically distinct - the exact
# situation that broke dense-only retrieval in the deployed app.
CORPUS = [
    {
        "chunk_id": "sym::c0",
        "document_id": "sym",
        "title": "CKD Symptoms, Causes and Kidney Function Tests",
        "section": "Common symptoms of chronic kidney disease",
        "text": (
            "Chronic kidney disease is frequently asymptomatic in its early stages. "
            "When symptoms appear they develop gradually and commonly include tiredness, "
            "swelling of the ankles and feet, changes in urination, poor appetite, nausea, "
            "itchy skin, muscle cramps and shortness of breath."
        ),
        "source": "DEMO corpus", "url": None, "publication_date": None,
        "retrieved_at": "2026-01-01T00:00:00+00:00", "evidence_level": "demo_material",
        "document_type": "demo_reference_note", "trust_tier": "UNKNOWN",
        "data_status": "demo", "file_name": "ckd_symptoms_diagnosis.md",
    },
    {
        "chunk_id": "med::c0",
        "document_id": "med",
        "title": "CKD Medication Safety and Interaction Concepts",
        "section": "The triple whammy",
        "text": (
            "General references use the informal term triple whammy for the combination "
            "of an ACE inhibitor or an ARB, plus a diuretic, plus an NSAID. The risk of "
            "acute kidney injury is described as higher with all three together."
        ),
        "source": "DEMO corpus", "url": None, "publication_date": None,
        "retrieved_at": "2026-01-01T00:00:00+00:00", "evidence_level": "demo_material",
        "document_type": "demo_reference_note", "trust_tier": "UNKNOWN",
        "data_status": "demo", "file_name": "ckd_medication_safety.md",
    },
]


def _indexed(payloads):
    """Attach the contextual header the ingestion pipeline adds."""
    return [
        {**p, "embed_text": build_embed_text(p["title"], p["section"], p["text"])}
        for p in payloads
    ]


# ------------------------------------------------------------ contextual headers
def test_embed_text_prefixes_title_and_section():
    text = build_embed_text("CKD Overview", "eGFR and staging", "It is calculated from creatinine.")
    assert text.startswith("CKD Overview")
    assert "eGFR and staging" in text
    assert text.endswith("It is calculated from creatinine.")


def test_embed_text_without_header_is_unchanged():
    assert build_embed_text(None, None, "Body only.") == "Body only."


# --------------------------------------------------------------------- BM25
def test_tokenize_drops_stopwords_keeps_medical_terms():
    tokens = tokenize("What are the common symptoms of CKD?")
    assert "symptoms" in tokens and "ckd" in tokens
    assert "what" not in tokens and "the" not in tokens


def test_bm25_ranks_the_topically_correct_document_first():
    index = BM25Index(_indexed(CORPUS))
    top = index.search("What are the common symptoms of CKD?", top_k=2)
    assert top, "BM25 returned nothing"
    assert top[0][0]["document_id"] == "sym"


def test_bm25_separates_two_similar_ckd_documents():
    index = BM25Index(_indexed(CORPUS))
    assert index.search("What is the triple whammy?")[0][0]["document_id"] == "med"
    assert index.search("symptoms of kidney disease")[0][0]["document_id"] == "sym"


def test_bm25_returns_nothing_for_unmatched_query():
    index = BM25Index(_indexed(CORPUS))
    assert index.search("photosynthesis chlorophyll") == []


def test_bm25_handles_empty_index():
    assert BM25Index([]).search("anything") == []
    assert len(BM25Index([])) == 0


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]])
    # 'b' is 2nd and 1st; 'a' is 1st and 2nd - equal. 'c' is last in both.
    assert fused["c"] < fused["a"]
    assert pytest.approx(fused["a"], rel=1e-6) == fused["b"]


def test_fusion_surfaces_a_document_only_one_retriever_found():
    fused = reciprocal_rank_fusion([["a"], ["z"]])
    assert set(fused) == {"a", "z"}


# ------------------------------------------------------------ hybrid retrieval
@pytest.fixture
def hybrid_tool(settings, vector_store, embedder):
    Indexer(embedder, vector_store).index_payloads(_indexed(CORPUS))
    return VectorSearchTool(vector_store=vector_store, embedder=embedder, settings=settings)


def test_hybrid_retrieval_returns_symptoms_doc_for_symptoms_query(hybrid_tool):
    """The exact failure reported: a symptoms question returning medication content."""
    evidence, structured = hybrid_tool.run(
        hybrid_tool.validate_arguments({"query": "What are the common symptoms of CKD?", "top_k": 1})
    )
    assert evidence, f"no evidence returned ({structured})"
    assert evidence[0].document_id == "sym"
    assert "hybrid" in structured["retrieval_mode"]


def test_hybrid_retrieval_still_finds_medication_content(hybrid_tool):
    evidence, _ = hybrid_tool.run(
        hybrid_tool.validate_arguments({"query": "What is the triple whammy?", "top_k": 1})
    )
    assert evidence and evidence[0].document_id == "med"


def test_hybrid_results_record_which_retriever_matched(hybrid_tool):
    evidence, _ = hybrid_tool.run(
        hybrid_tool.validate_arguments({"query": "symptoms of chronic kidney disease", "top_k": 2})
    )
    assert evidence
    for item in evidence:
        assert item.extra["matched_by"] in {"dense", "lexical", "both"}
        assert item.extra["retrieval"] == "hybrid_vector_bm25"


def test_hybrid_preserves_full_provenance(hybrid_tool):
    evidence, _ = hybrid_tool.run(
        hybrid_tool.validate_arguments({"query": "CKD symptoms", "top_k": 1})
    )
    top = evidence[0]
    for field in ("document_id", "chunk_id", "source", "retrieved_at", "title"):
        assert getattr(top, field), f"{field} was lost in hybrid retrieval"
    assert top.data_status == "demo"
    assert top.url is None  # absent metadata must stay absent


def test_empty_index_reported_not_guessed(settings, embedder):
    from stores.vector_store import NumpyVectorStore

    empty = NumpyVectorStore(settings.vector_db_path, "empty_hybrid")
    tool = VectorSearchTool(vector_store=empty, embedder=embedder, settings=settings)
    evidence, structured = tool.run(tool.validate_arguments({"query": "CKD symptoms"}))
    assert evidence == []
    assert structured["index_empty"] is True


# ------------------------------------------------------------ index fingerprint
def test_fingerprint_round_trip(tmp_path, embedder):
    fp = fingerprint_for(embedder)
    write_fingerprint(tmp_path, fp)
    assert read_fingerprint(tmp_path) == fp


def test_missing_fingerprint_is_incompatible(tmp_path, embedder):
    compatible, message = check_compatibility(tmp_path, embedder)
    assert compatible is False
    assert "not been built" in message


def test_matching_fingerprint_is_compatible(tmp_path, embedder):
    write_fingerprint(tmp_path, fingerprint_for(embedder))
    compatible, _ = check_compatibility(tmp_path, embedder)
    assert compatible is True


def test_same_dimension_different_backend_is_rejected(tmp_path, embedder):
    """The dangerous case: dimensions match, vector spaces do not."""
    stored = IndexFingerprint(
        backend="sentence_transformers",
        model="pritamdeka/S-PubMedBert-MS-MARCO",
        dimension=embedder.dimension,  # identical dimension
    )
    write_fingerprint(tmp_path, stored)
    compatible, message = check_compatibility(tmp_path, embedder)
    assert compatible is False
    assert "mismatch" in message.lower()
    assert "--reset" in message


def test_corrupt_fingerprint_is_treated_as_missing(tmp_path, embedder):
    (tmp_path / "index_fingerprint.json").write_text("{not json", encoding="utf-8")
    assert read_fingerprint(tmp_path) is None


def test_fingerprint_contains_no_secrets(tmp_path, embedder):
    write_fingerprint(tmp_path, fingerprint_for(embedder))
    data = json.loads((tmp_path / "index_fingerprint.json").read_text())
    assert set(data) == {"backend", "model", "dimension", "embed_text_version"}


# ------------------------------------------------------------ LLM status
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Error code: 401 - invalid api key", LLM_STATUS_AUTH),
        ("authentication failed", LLM_STATUS_AUTH),
        ("Rate limit reached for model", LLM_STATUS_RATE_LIMIT),
        ("429 Too Many Requests", LLM_STATUS_RATE_LIMIT),
        ("model_not_found: it has been decommissioned", LLM_STATUS_MODEL),
        ("Connection error: dns failure", LLM_STATUS_NETWORK),
        ("Request timed out", LLM_STATUS_NETWORK),
        ("groq SDK not installed: No module named 'groq'", LLM_STATUS_SDK_MISSING),
    ],
)
def test_llm_errors_are_classified(message, expected):
    assert classify_llm_error(message) == expected


def test_classifier_never_echoes_the_error_text():
    """A classified status must not leak a key fragment from the raw error."""
    leaked = "Invalid api key gsk_SUPERSECRETVALUE123 rejected by 401 authentication"
    status = classify_llm_error(leaked)
    assert "gsk_" not in status
    assert "SUPERSECRET" not in status


def test_missing_key_produces_specific_reason(settings):
    """With no key at all, the reason must name the missing-key condition.

    The message also names which env var to set, so it is checked by prefix
    rather than exact equality.
    """
    settings.gemini_api_key = None
    settings.groq_api_key = None
    client = build_llm_client(settings)
    assert isinstance(client, UnavailableLLMClient)
    reason = client.health()["reason"]
    assert reason.startswith(LLM_STATUS_MISSING_KEY)
    assert "API_KEY" in reason


def test_unavailable_reason_reaches_the_user(settings, structured_store, vector_store, embedder):
    """The deployed app showed a generic message; the real reason must surface."""
    from agent.orchestrator import Orchestrator
    from agent.tools.tool_registry import ToolRegistry

    settings.gemini_api_key = None
    registry = ToolRegistry(settings=settings)
    Indexer(embedder, vector_store).index_payloads(_indexed(CORPUS))
    registry.register(
        VectorSearchTool(vector_store=vector_store, embedder=embedder, settings=settings)
    )
    response = Orchestrator(settings=settings, registry=registry).run(
        "What are the common symptoms of CKD?"
    )
    joined = " ".join(response.warnings)
    assert LLM_STATUS_MISSING_KEY in joined
    assert "No language model is configured" not in response.answer


# ------------------------------------------------------------ diagnostics
def test_diagnostics_never_reveal_the_key(settings):
    from scripts.diagnostics import check_api_key

    settings.gemini_api_key = "gsk_this_must_never_be_printed_1234567890"
    result = check_api_key(settings)
    assert result.ok
    assert "gsk_" not in result.detail
    assert "must_never" not in result.detail
    assert result.data["configured"] is True


def test_diagnostics_report_missing_key(settings):
    from scripts.diagnostics import check_api_key

    settings.gemini_api_key = None
    result = check_api_key(settings)
    assert result.status == "FAIL"
    assert result.data["configured"] is False


def test_diagnostics_skip_live_call_without_key(settings):
    from scripts.diagnostics import check_llm_response

    settings.gemini_api_key = None
    assert check_llm_response(settings).status == "SKIP"


def test_diagnostics_run_without_network_or_key(settings):
    """Offline diagnostics must cover every layer and name the active key var."""
    from scripts.diagnostics import run_diagnostics

    settings.gemini_api_key = None
    settings.groq_api_key = None
    results = run_diagnostics(settings, include_llm_call=False)
    names = {r.name for r in results}
    # The key check is named after whichever provider is active.
    assert settings.active_key_name in names
    assert {"LLM client", "Embeddings", "Vector store"} <= names


# ------------------------------------------------- extractive excerpt rendering
def test_markdown_headings_do_not_leak_into_excerpts():
    """A chunk starting with '## ...' rendered as a giant H2 in the chat UI."""
    from agent.synthesizer import _plain_excerpt

    excerpt = _plain_excerpt("## eGFR and staging concepts\n\nCKD staging is based on eGFR.")
    assert not excerpt.startswith("#")
    assert excerpt.startswith("eGFR and staging concepts")


def test_excerpt_strips_list_quote_and_emphasis_markers():
    from agent.synthesizer import _plain_excerpt

    excerpt = _plain_excerpt("> **Note:** important\n- first item\n1. second item")
    for marker in ("**", ">", "- ", "1."):
        assert not excerpt.lstrip().startswith(marker)
    assert "important" in excerpt and "first item" in excerpt


def test_excerpt_truncates_on_a_word_boundary():
    from agent.synthesizer import _plain_excerpt

    excerpt = _plain_excerpt("word " * 300, limit=100)
    assert len(excerpt) <= 104 and excerpt.endswith("...")


def test_extractive_answer_renders_passages_as_blockquotes(sample_evidence):
    from agent.synthesizer import Synthesizer

    answer = Synthesizer.extractive_summary("CKD", sample_evidence, [], reason="test")
    body = answer.split("Most relevant retrieved passages:")[1]
    for line in body.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("#"), f"heading leaked into answer: {line!r}"
