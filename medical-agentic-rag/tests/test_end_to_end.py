"""End-to-end tests: orchestrator wiring, application startup, live integration.

The orchestrator tests drive the *whole* pipeline with a scripted provider, so
they assert the behaviour a user actually sees: a synthesised answer, real tool
execution, honest confidence and validated citations.

The live test at the bottom is opt-in and is skipped unless an API key is
present. It is the only test that proves the configured model genuinely works.
"""

from __future__ import annotations

import os

import pytest

from agent.orchestrator import Orchestrator
from agent.state import RiskCategory
from agent.tools.dosage_lookup_tool import DosageLookupTool
from agent.tools.drug_interaction_tool import DrugInteractionTool
from agent.tools.tool_registry import ToolRegistry
from agent.tools.vector_search_tool import VectorSearchTool
from ingestion.indexer import Indexer
from ingestion.metadata_tagger import build_embed_text
from tests.test_agent_tool_loop import CORPUS, ScriptedToolLLM, _indexed


@pytest.fixture
def orchestrator_factory(settings, vector_store, embedder, structured_store):
    """Build a fully wired orchestrator around a scripted provider."""
    Indexer(embedder, vector_store).index_payloads(_indexed(CORPUS))

    def _build(llm):
        registry = ToolRegistry(settings=settings)
        registry.register(VectorSearchTool(vector_store=vector_store,
                                           embedder=embedder, settings=settings))
        registry.register(DrugInteractionTool(store=structured_store, settings=settings))
        registry.register(DosageLookupTool(store=structured_store, settings=settings))
        return Orchestrator(settings=settings, llm=llm, registry=registry)

    return _build


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_pipeline_returns_synthesised_answer_not_raw_passages(orchestrator_factory):
    """The reported symptom was raw passages shown instead of an answer."""
    llm = ScriptedToolLLM([
        [("vector_search", {"query": "CKD staging", "top_k": 3})],
        "**Answer**\n\nCKD staging is based on eGFR categories G1 to G5 {CITE}.",
    ])
    response = orchestrator_factory(llm).run("What is CKD staging based on?")

    assert response.answer.startswith("**Answer**")
    assert response.answer_source == "agent_loop"
    assert "Answer generation is unavailable" not in response.answer
    assert "Most relevant retrieved passages" not in response.answer
    assert response.agent_iterations >= 2


def test_pipeline_reports_only_executed_tools(orchestrator_factory):
    llm = ScriptedToolLLM([
        [("drug_interaction", {"drug_a": "lisinopril", "drug_b": "ibuprofen"})],
        [("vector_search", {"query": "NSAID kidney", "top_k": 3})],
        "**Answer**\n\nCombining them reduces kidney perfusion {CITE}.",
    ])
    response = orchestrator_factory(llm).run("Does lisinopril interact with ibuprofen?")

    assert response.tools_used == ["drug_interaction", "vector_search"]
    assert len(response.tools_used) == len(set(response.tools_used)), "no duplicates"


def test_pipeline_citations_resolve_to_real_evidence(orchestrator_factory):
    llm = ScriptedToolLLM([
        [("vector_search", {"query": "CKD staging", "top_k": 3})],
        "**Answer**\n\nStaging uses eGFR {CITE}.",
    ])
    response = orchestrator_factory(llm).run("What is CKD staging based on?")

    assert response.citations, "a grounded answer must carry citations"
    for citation in response.citations:
        assert citation.evidence_id.startswith("S")
        assert citation.data_status == "demo", "demo provenance must be preserved"
    assert "Sources:" in response.answer


def test_out_of_domain_question_runs_no_tools(orchestrator_factory):
    llm = ScriptedToolLLM([[("vector_search", {"query": "migraine", "top_k": 3})]])
    response = orchestrator_factory(llm).run("How is migraine treated?")

    assert response.tools_used == []
    assert response.status == "out_of_scope"
    assert llm.turns == [], "no model call for an out-of-domain question"


def test_emergency_short_circuits_before_any_model_call(orchestrator_factory):
    llm = ScriptedToolLLM([[("vector_search", {"query": "chest pain", "top_k": 3})]])
    response = orchestrator_factory(llm).run(
        "I have crushing chest pain radiating down my left arm"
    )

    assert response.risk_category is RiskCategory.EMERGENCY
    assert response.tools_used == []
    assert llm.turns == []
    assert "emergency" in response.answer.lower()


def test_personalised_dosing_yields_no_numeric_dose(orchestrator_factory):
    llm = ScriptedToolLLM([
        [("dosage_lookup", {"drug": "metformin"})],
        "**Answer**\n\nYou should take 500 mg of metformin twice daily {CITE}.",
    ])
    response = orchestrator_factory(llm).run(
        "I have kidney disease. What dose of metformin should I take?"
    )

    assert response.risk_category is RiskCategory.PERSONALIZED_CLINICAL_DECISION
    assert "500 mg" not in response.answer, "guardrail must redact a personalised dose"
    assert any(word in response.answer.lower() for word in ("clinician", "prescriber"))


def test_model_failure_degrades_without_fabricating(orchestrator_factory):
    """A dead model must not be papered over with raw passages presented as an answer."""
    from llm.base import LLMError

    llm = ScriptedToolLLM(
        [[("vector_search", {"query": "CKD staging", "top_k": 3})], "unused"],
        raise_on_turn={1: LLMError("Groq request failed: model_not_found")},
    )
    response = orchestrator_factory(llm).run("What is CKD staging based on?")

    assert response.answer, "the user still gets a response"
    assert any("Agent loop stopped" in w for w in response.warnings)
    assert response.answer_source != "agent_loop"


def test_no_llm_still_retrieves_and_labels_the_answer(orchestrator_factory):
    """Legacy path is preserved: retrieval runs, answer is labelled extractive."""
    llm = ScriptedToolLLM([], available=False)
    response = orchestrator_factory(llm).run("What is CKD staging based on?")

    assert response.tools_used, "retrieval must still run without a model"
    assert response.answer_source == "extractive"
    assert any("synthesis is disabled" in w.lower() for w in response.warnings)


def test_weak_evidence_is_reported_honestly(orchestrator_factory):
    llm = ScriptedToolLLM([
        [("vector_search", {"query": "CKD staging", "top_k": 3})],
        "**Answer**\n\nStaging uses eGFR {CITE}.",
    ])
    response = orchestrator_factory(llm).run("What is CKD staging based on?")

    assert response.confidence.value in {"HIGH", "MODERATE", "LOW", "INSUFFICIENT"}
    assert 0.0 <= response.confidence_score <= 1.0
    # Demo provenance must always be surfaced, whatever the score.
    assert any("DEMO" in w or "demo" in w for w in response.warnings) or \
        any(c.data_status == "demo" for c in response.citations)


# ---------------------------------------------------------------------------
# Application startup
# ---------------------------------------------------------------------------

def test_core_modules_import():
    """Catches the import-time breakage that stops the app booting at all."""
    import agent.orchestrator  # noqa: F401
    import agent.tool_loop  # noqa: F401
    import llm.factory  # noqa: F401
    import llm.groq_client  # noqa: F401
    import llm.gemini_client  # noqa: F401
    import scripts.diagnostics  # noqa: F401


def test_streamlit_entrypoint_compiles():
    """The Streamlit script must at least parse; a syntax error is a dead app."""
    import pathlib
    import ast

    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("streamlit_app.py", "app.py", "main.py"):
        source = (root / name).read_text(encoding="utf-8")
        ast.parse(source)  # raises SyntaxError on failure


def test_orchestrator_builds_without_any_api_key(settings):
    """Startup must not require credentials."""
    settings.groq_api_key = None
    settings.gemini_api_key = None
    orchestrator = Orchestrator(settings=settings)
    health = orchestrator.health()
    assert health["llm"]["available"] is False
    assert "tools" in health and health["tools"]


def test_settings_never_leak_the_key(settings):
    settings.groq_api_key = "gsk_super_secret_value_1234567890"
    dumped = settings.safe_dump()
    assert dumped["groq_api_key"] == "***redacted***"
    assert "super_secret" not in str(dumped)


# ---------------------------------------------------------------------------
# Optional live integration test
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.skipif(
    not (os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="No API key configured; set GROQ_API_KEY or GEMINI_API_KEY to run.",
)
def test_live_model_round_trip():
    """Proves the configured model genuinely answers. Requires a real key.

    Run with:  pytest -m live
    """
    from config.settings import reload_settings
    from llm.factory import build_llm_client

    settings = reload_settings()
    client = build_llm_client(settings)
    assert client.available, f"client unavailable: {client.health()}"

    ok, detail = client.ping()
    assert ok, f"live ping failed: {detail}"

    models = client.list_models()
    if models:
        assert getattr(client, "model", "") in models, (
            f"configured model {getattr(client, 'model', '')!r} not in {models[:10]}"
        )


@pytest.mark.live
@pytest.mark.skipif(
    not (os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="No API key configured; set GROQ_API_KEY or GEMINI_API_KEY to run.",
)
def test_live_agent_executes_tools_and_answers():
    """Full live agentic round trip against the real provider."""
    from agent.orchestrator import build_orchestrator
    from config.settings import reload_settings

    response = build_orchestrator(reload_settings()).run(
        "Does lisinopril interact with ibuprofen?"
    )
    assert response.tools_used, "the live model must actually call a tool"
    assert response.answer.strip()
    assert "Answer generation is unavailable" not in response.answer
