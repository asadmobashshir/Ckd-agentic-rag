"""Orchestrator behaviour with a mocked LLM (no Gemini key required)."""

from __future__ import annotations

import asyncio

import pytest

from agent.orchestrator import Orchestrator
from agent.planner import Planner
from agent.state import ConfidenceLevel, Evidence, RiskCategory, ToolCallRequest
from agent.synthesizer import Synthesizer
from agent.tools.base import BaseTool
from agent.tools.dosage_lookup_tool import DosageLookupTool
from agent.tools.drug_interaction_tool import DrugInteractionTool
from agent.tools.tool_registry import ToolRegistry
from agent.tools.vector_search_tool import VectorSearchTool
from audit.audit_logger import AuditLogger
from ingestion.indexer import Indexer
from llm.base import LLMError, LLMToolCall
from safety.guardrails import GuardrailEngine
from safety.verifier import ClaimVerifier
from tests.conftest import FakeLLM

CORPUS = [
    {
        "chunk_id": "doc1::c0000",
        "document_id": "doc1",
        "text": (
            "Metformin is an oral biguanide used in the management of type 2 diabetes "
            "mellitus and is commonly described as a first-line pharmacological option."
        ),
        "title": "DEMO: Metformin overview",
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
        "text": (
            "Combining an anticoagulant such as warfarin with an NSAID such as ibuprofen "
            "is widely described as increasing bleeding risk relative to either alone."
        ),
        "title": "DEMO: Anticoagulants and NSAIDs",
        "source": "DEMO corpus",
        "url": None,
        "publication_date": None,
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "section": "Combined use",
        "evidence_level": "demo_material",
        "document_type": "demo_reference_note",
        "trust_tier": "UNKNOWN",
        "data_status": "demo",
        "file_name": "demo_anticoag.md",
    },
]


def make_orchestrator(settings, structured_store, vector_store, embedder, llm) -> Orchestrator:
    Indexer(embedder, vector_store).index_payloads(CORPUS)
    registry = ToolRegistry(settings=settings)
    registry.register(VectorSearchTool(vector_store=vector_store, embedder=embedder, settings=settings))
    registry.register(DrugInteractionTool(store=structured_store, settings=settings))
    registry.register(DosageLookupTool(store=structured_store, settings=settings))
    return Orchestrator(
        settings=settings,
        llm=llm,
        registry=registry,
        planner=Planner(llm, registry, settings, known_drugs=["warfarin", "ibuprofen", "metformin"]),
        synthesizer=Synthesizer(llm, settings),
        verifier=ClaimVerifier(llm),
        guardrails=GuardrailEngine(settings),
        audit_logger=AuditLogger(settings.audit_log_path, settings),
    )


# ------------------------------------------------------------- agent behaviour
def test_planner_selected_tools_are_executed(settings, structured_store, vector_store, embedder):
    llm = FakeLLM(
        tool_calls=[
            LLMToolCall(name="drug_interaction", arguments={"drug_a": "warfarin", "drug_b": "ibuprofen"}),
            LLMToolCall(name="vector_search", arguments={"query": "warfarin ibuprofen bleeding"}),
        ],
        text="Bleeding risk is increased when these are combined [S1][S2].",
    )
    orch = make_orchestrator(settings, structured_store, vector_store, embedder, llm)
    response = orch.run("Does warfarin interact with ibuprofen?")

    assert set(response.tools_used) == {"drug_interaction", "vector_search"}
    assert response.status == "ok"
    assert response.citations


def test_zero_tool_selection_for_greeting(settings, structured_store, vector_store, embedder):
    llm = FakeLLM(tool_calls=[], text="No retrieval needed.")
    orch = make_orchestrator(settings, structured_store, vector_store, embedder, llm)
    response = orch.run("Hello")
    assert response.tools_used == []
    assert response.confidence is ConfidenceLevel.INSUFFICIENT


def test_agent_loop_is_bounded(settings, structured_store, vector_store, embedder):
    """Even if the planner keeps proposing tools, rounds are capped."""

    class LoopingLLM(FakeLLM):
        def chat(self, messages, **kwargs):
            self.calls.append({"messages": list(messages), **kwargs})
            from llm.base import LLMResponse

            return LLMResponse(
                text="keep going",
                tool_calls=[LLMToolCall(name="vector_search", arguments={"query": f"round {len(self.calls)}"})],
            )

    llm = LoopingLLM()
    orch = make_orchestrator(settings, structured_store, vector_store, embedder, llm)
    response = orch.run("Tell me about diabetes management options")
    # max_tool_rounds is 2 in the test settings.
    assert len(response.tools_used) <= settings.max_tool_rounds * 3


def test_unregistered_tool_from_llm_is_never_executed(settings, structured_store, vector_store, embedder):
    llm = FakeLLM(
        tool_calls=[LLMToolCall(name="delete_database", arguments={"path": "/"})],
        text="done",
    )
    orch = make_orchestrator(settings, structured_store, vector_store, embedder, llm)
    response = orch.run("What is metformin used for?")
    assert "delete_database" not in response.tools_used


def test_invalid_arguments_from_llm_are_dropped(settings, structured_store, vector_store, embedder):
    llm = FakeLLM(
        tool_calls=[LLMToolCall(name="drug_interaction", arguments={"drug_a": "warfarin"})],
        text="done",
    )
    orch = make_orchestrator(settings, structured_store, vector_store, embedder, llm)
    response = orch.run("Does warfarin interact with something?")
    assert "drug_interaction" not in response.tools_used


# ------------------------------------------------------------------- failures
def test_llm_failure_degrades_to_evidence_only(settings, structured_store, vector_store, embedder):
    llm = FakeLLM(raise_error=LLMError("groq is down"))
    orch = make_orchestrator(settings, structured_store, vector_store, embedder, llm)
    response = orch.run("What is metformin used for?")
    # Heuristic planner still retrieves, synthesis falls back to extraction.
    assert "vector_search" in response.tools_used
    assert response.answer.strip()
    assert "No language model" in response.answer or response.citations


def test_tool_failure_is_reported_not_fabricated(settings, structured_store, vector_store, embedder):
    class BrokenSearch(BaseTool):
        name = "vector_search"
        description = "broken"
        args_model = VectorSearchTool.args_model

        def run(self, args):
            raise RuntimeError("index corrupted")

    llm = FakeLLM(tool_calls=[LLMToolCall(name="vector_search", arguments={"query": "metformin"})], text="")
    orch = make_orchestrator(settings, structured_store, vector_store, embedder, llm)
    orch.registry.register(BrokenSearch(), override=True)

    response = orch.run("What is metformin used for?")
    assert any("failed" in w.lower() for w in response.warnings)
    assert response.citations == []


def test_no_evidence_produces_abstention(settings, structured_store, embedder):
    from stores.vector_store import NumpyVectorStore

    empty = NumpyVectorStore(settings.vector_db_path, "empty")
    registry = ToolRegistry(settings=settings)
    registry.register(VectorSearchTool(vector_store=empty, embedder=embedder, settings=settings))
    llm = FakeLLM(tool_calls=[LLMToolCall(name="vector_search", arguments={"query": "obscure topic"})], text="")
    orch = Orchestrator(
        settings=settings, llm=llm, registry=registry,
        planner=Planner(llm, registry, settings), synthesizer=Synthesizer(llm, settings),
        verifier=ClaimVerifier(None), guardrails=GuardrailEngine(settings),
        audit_logger=AuditLogger(settings.audit_log_path, settings),
    )
    response = orch.run("What does the corpus say about an entirely unknown topic?")
    assert response.confidence is ConfidenceLevel.INSUFFICIENT
    assert "No reliable evidence" in response.answer
    assert response.citations == []


# --------------------------------------------------------------------- safety
def test_emergency_blocks_before_any_tool_runs(settings, structured_store, vector_store, embedder):
    llm = FakeLLM(tool_calls=[LLMToolCall(name="vector_search", arguments={"query": "chest pain"})])
    orch = make_orchestrator(settings, structured_store, vector_store, embedder, llm)
    response = orch.run("I have crushing chest pain radiating down my left arm")

    assert response.risk_category is RiskCategory.EMERGENCY
    assert response.tools_used == []
    assert llm.calls == [], "no LLM call should be made for an emergency"
    assert "emergency services" in response.answer.lower()


def test_personalised_dosing_answer_carries_no_dose(settings, structured_store, vector_store, embedder):
    llm = FakeLLM(
        tool_calls=[LLMToolCall(name="dosage_lookup", arguments={"drug": "ibuprofen"})],
        text="You should take 400 mg of ibuprofen every 6 hours for your pain.",
    )
    orch = make_orchestrator(settings, structured_store, vector_store, embedder, llm)
    response = orch.run("I take warfarin. How much ibuprofen should I take for my pain?")

    assert response.risk_category is RiskCategory.PERSONALIZED_CLINICAL_DECISION
    assert "400 mg" not in response.answer
    assert "clinician" in response.answer.lower() or "prescriber" in response.answer.lower()


def test_fabricated_citation_from_llm_is_stripped(settings, structured_store, vector_store, embedder):
    llm = FakeLLM(
        tool_calls=[LLMToolCall(name="vector_search", arguments={"query": "metformin"})],
        text="Metformin is used in type 2 diabetes [S1]. It also cures baldness [S99].",
    )
    orch = make_orchestrator(settings, structured_store, vector_store, embedder, llm)
    response = orch.run("What is metformin used for?")
    assert "[S99]" not in response.answer
    assert all(c.evidence_id != "S99" for c in response.citations)


# ---------------------------------------------------------------------- audit
def test_audit_record_is_written_and_redacted(settings, structured_store, vector_store, embedder):
    llm = FakeLLM(tool_calls=[LLMToolCall(name="vector_search", arguments={"query": "metformin"})], text="ok [S1]")
    orch = make_orchestrator(settings, structured_store, vector_store, embedder, llm)
    response = orch.run("My name is Jane Doe, email jane@example.com. What is metformin used for?")

    records = orch.audit.read_all()
    assert records
    record = records[-1]
    assert record["request_id"] == response.request_id
    assert "jane@example.com" not in record["query"]
    assert "[EMAIL]" in record["query"]
    assert record["tool_calls"]
    assert "confidence" in record and record["confidence"]["level"]


def test_async_and_sync_entrypoints_agree(settings, structured_store, vector_store, embedder):
    llm = FakeLLM(tool_calls=[], text="no tools")
    orch = make_orchestrator(settings, structured_store, vector_store, embedder, llm)
    sync = orch.run("Hello")
    async_result = asyncio.run(orch.arun("Hello"))
    assert sync.answer == async_result.answer
    assert sync.request_id != async_result.request_id
