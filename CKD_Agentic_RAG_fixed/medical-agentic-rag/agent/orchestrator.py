"""Orchestration of the online query pipeline.

    safety pre-check -> plan -> validate -> execute tools -> observe
      -> (optional re-plan, bounded by MAX_TOOL_ROUNDS)
      -> synthesise -> verify claims -> score confidence
      -> guardrail post-check -> format citations -> audit log

This module coordinates; it does not implement. Every stage lives in its own
module and is injected here, so each can be tested and replaced independently.

The loop is a genuine agent loop: the planner may select zero, one or several
tools, results are fed back as observations, and the loop stops early once the
evidence is sufficient rather than always running a fixed pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from agent.domain import OUT_OF_SCOPE_INTENT
from agent.planner import Planner
from agent.state import (
    AgentState,
    ConfidenceLevel,
    Evidence,
    FinalResponse,
    Plan,
    RiskCategory,
    SafetyAssessment,
    ToolResult,
    VerificationResult,
)
from agent.synthesizer import Synthesizer
from agent.tool_loop import ToolCallingLoop
from agent.tools.tool_registry import ToolRegistry, build_default_registry
from audit.audit_logger import AuditLogger
from config.settings import Settings, get_settings
from llm.base import BaseLLMClient
from llm.factory import build_llm_client
from safety.citation_formatter import assign_evidence_ids, format_answer
from safety.confidence import compute_confidence
from safety.guardrails import GuardrailEngine
from safety.verifier import ClaimVerifier
from stores.structured_store import build_structured_store

logger = logging.getLogger(__name__)

MAX_EVIDENCE_ITEMS = 12


class Orchestrator:
    """Coordinates the full online pipeline for one query at a time."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseLLMClient | None = None,
        registry: ToolRegistry | None = None,
        planner: Planner | None = None,
        synthesizer: Synthesizer | None = None,
        verifier: ClaimVerifier | None = None,
        guardrails: GuardrailEngine | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.llm = llm or build_llm_client(self.settings)
        self.registry = registry or build_default_registry(self.settings)
        self.guardrails = guardrails or GuardrailEngine(self.settings)
        self.planner = planner or Planner(
            self.llm, self.registry, self.settings, known_drugs=self._known_drugs()
        )
        self.synthesizer = synthesizer or Synthesizer(self.llm, self.settings)
        self.verifier = verifier or ClaimVerifier(self.llm)
        self.audit = audit_logger or AuditLogger(settings=self.settings)
        self.tool_loop = ToolCallingLoop(self.llm, self.registry, self.settings)

    def _known_drugs(self) -> list[str]:
        """Drug vocabulary for the offline heuristic planner."""
        try:
            store = build_structured_store(self.settings)
            with store._connect() as conn:  # noqa: SLF001 - internal read-only helper
                return [row[0] for row in conn.execute("SELECT name FROM drugs").fetchall()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load drug vocabulary: %s", exc)
            return []

    # ------------------------------------------------------------- entrypoints
    def run(self, query: str) -> FinalResponse:
        """Blocking entrypoint shared by the CLI, tests and scripts."""
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> FinalResponse:
        """Async entrypoint used by FastAPI."""
        state = AgentState(query=(query or "").strip(), llm_available=self.llm.available)

        safety = self.guardrails.pre_check(state.query)
        state.safety = safety
        for warning in safety.warnings:
            state.add_warning(warning)

        if safety.block_pipeline:
            return self._finalise_blocked(state, safety)

        # Domain scope is enforced deterministically, before any model call, so
        # an out-of-CKD question is refused identically with and without a key.
        out_of_scope = self._check_scope(state)

        if not out_of_scope:
            try:
                await self._agent_loop(state, safety)
            except Exception as exc:  # noqa: BLE001 - never surface a stack trace
                logger.exception("Agent loop failed for %s", state.request_id)
                state.add_warning(f"Retrieval stage failed: {exc}")

        state.evidence = assign_evidence_ids(_dedupe(state.evidence)[:MAX_EVIDENCE_ITEMS])

        # ---------------------------------------------------------- synthesis
        if not self.llm.available:
            reason = str(self.llm.health().get("reason") or "no LLM backend configured")
            state.add_warning(
                f"Answer synthesis is disabled: {reason}. Evidence retrieval still ran; "
                "the answer below is extracted, not synthesised."
            )

        if state.loop_answer and not out_of_scope:
            # The agent already wrote this answer while observing real tool
            # output. Re-synthesising would discard that reasoning.
            state.draft = state.loop_answer
            state.answer_source = "agent_loop"
        else:
            state.draft = self.synthesizer.synthesize(
                state.query,
                state.evidence,
                state.tool_results,
                self.guardrails.synthesis_constraints(safety),
                out_of_scope=out_of_scope,
            )
            state.answer_source = "synthesizer" if self.llm.available else "extractive"

        # ------------------------------------------------------- verification
        if out_of_scope:
            # A scope refusal is fixed text with no claims and no evidence.
            # Running claim extraction over it costs a model call and can only
            # produce noise.
            state.verification = VerificationResult(
                revised_answer=state.draft,
                notes=["Out-of-scope refusal; no claims to verify."],
            )
        else:
            # The no-LLM path quotes evidence verbatim, so revision is disabled there.
            state.verification = self.verifier.verify(
                state.draft, state.evidence, extractive=not self.llm.available
            )
        answer = state.verification.revised_answer or state.draft
        # An out-of-scope reply is a fixed message, not a failed retrieval; the
        # "no evidence" notes would be misleading noise in the UI. They stay in the
        # audit record, which still carries the full verification result.
        if not out_of_scope:
            for note in state.verification.notes:
                state.add_warning(note)

        # --------------------------------------------------------- confidence
        state.confidence = compute_confidence(state.evidence, state.verification)
        if not out_of_scope:
            for reason in state.confidence.rationale:
                if state.confidence.level in (ConfidenceLevel.LOW, ConfidenceLevel.INSUFFICIENT):
                    state.add_warning(reason)

        # --------------------------------------------------- guardrail output
        answer, post_warnings = self.guardrails.post_check(answer, safety)
        for warning in post_warnings:
            state.add_warning(warning)

        # ---------------------------------------------------------- citations
        answer, citations, citation_warnings = format_answer(answer, state.evidence)
        state.citations = citations
        for warning in citation_warnings:
            state.add_warning(warning)

        if not out_of_scope and state.confidence.level is ConfidenceLevel.INSUFFICIENT and state.evidence:
            state.add_warning(
                "Evidence confidence is INSUFFICIENT; treat this answer as unverified."
            )

        state.final_answer = answer
        if out_of_scope:
            state.status = "out_of_scope"
        else:
            state.status = "ok" if state.evidence or not state.tool_results else "no_evidence"

        response = FinalResponse(
            request_id=state.request_id,
            answer=state.final_answer,
            confidence=state.confidence.level,
            confidence_score=state.confidence.score,
            citations=state.citations,
            warnings=state.warnings,
            tools_used=state.tools_used,
            risk_category=safety.risk_category,
            status=state.status,
            agent_iterations=state.agent_iterations,
            answer_source=state.answer_source,
        )
        self.audit.log(state, response)
        return response

    # ------------------------------------------------------------------ scope
    def _check_scope(self, state: AgentState) -> bool:
        """Refuse clearly non-CKD questions before any model call."""
        try:
            if not self.planner.is_out_of_scope(state.query):
                return False
        except Exception as exc:  # noqa: BLE001 - scope check must never crash the run
            logger.warning("Domain scope check failed: %s", exc)
            return False
        state.intent = OUT_OF_SCOPE_INTENT
        state.plans.append(Plan(
            intent=OUT_OF_SCOPE_INTENT,
            tools=[],
            reason="Query is outside the CKD domain; no retrieval performed.",
        ))
        return True

    # -------------------------------------------------------------- agent loop
    async def _agent_loop(self, state: AgentState, safety: SafetyAssessment) -> None:
        """Run the native tool-calling loop, falling back to plan-then-execute.

        The native loop is the genuine agentic path: the model sees each tool's
        real output and decides what to do next. The legacy planner path is kept
        intact for when no LLM is reachable, the provider cannot do tool calling,
        or the feature is switched off in configuration.
        """
        if self.tool_loop.usable():
            outcome = await self.tool_loop.run(state.query, safety)
            state.agent_iterations = outcome.iterations
            state.tool_results.extend(outcome.tool_results)
            state.evidence.extend(outcome.evidence)
            if state.intent is None:
                state.intent = "agentic_tool_use"

            for warning in outcome.warnings:
                state.add_warning(warning)
            for rejected in outcome.rejected_calls:
                state.add_warning(f"Model requested an invalid tool call - {rejected}")

            if outcome.llm_failed:
                state.add_warning(
                    f"Agent loop stopped after {outcome.iterations} iteration(s): "
                    f"{outcome.error}."
                )
            elif outcome.final_text:
                state.loop_answer = outcome.final_text
                if outcome.stop_reason == "iteration_limit":
                    state.add_warning(
                        f"Agent reached its {self.settings.max_agent_iterations}-iteration "
                        "limit; the answer uses the evidence gathered so far."
                    )
                return

            # Loop produced evidence but no usable answer: keep the evidence and
            # let the synthesizer write from it rather than re-running retrieval.
            if outcome.tool_results:
                return

        await self._plan_execute_loop(state, safety)

    async def _plan_execute_loop(self, state: AgentState, safety: SafetyAssessment) -> None:
        """Legacy plan / execute / observe path, bounded by ``max_tool_rounds``."""
        # Signatures of calls already executed, so a re-plan that proposes the
        # same retrieval does not run it again. Without this the planner could
        # execute an identical vector_search once per round.
        executed: set[tuple[str, str]] = set()

        for round_index in range(self.settings.max_tool_rounds):
            plan = self.planner.plan(
                state.query,
                safety=safety,
                round_index=round_index,
                observations=state.tool_results,
            )
            state.plans.append(plan)
            if state.intent is None:
                state.intent = plan.intent

            if not plan.tools:
                logger.info(
                    "Round %d: planner selected no tools (%s)", round_index, plan.reason[:120]
                )
                break

            fresh = []
            for call in plan.tools:
                signature = (call.name, repr(sorted(call.arguments.items())))
                if signature in executed:
                    continue
                executed.add(signature)
                fresh.append(call)

            if not fresh:
                logger.info(
                    "Round %d: planner repeated already-executed calls; stopping.", round_index
                )
                break

            results = await self.registry.execute_many(fresh)
            state.tool_results.extend(results)
            for result in results:
                state.evidence.extend(result.evidence)
                if not result.ok:
                    state.add_warning(f"Tool '{result.tool}' failed: {result.error}")

            if self._evidence_is_sufficient(state, results):
                logger.info("Round %d: stopping early, evidence is sufficient.", round_index)
                break

        if not state.evidence and state.tool_results:
            state.add_warning(
                "Retrieval returned no usable evidence for this query."
            )

    def _evidence_is_sufficient(self, state: AgentState, results: Sequence[ToolResult]) -> bool:
        """Stopping rule for the agent loop.

        Stops when a structured lookup answered the question outright, or when
        enough distinct passages were retrieved. Continues (allowing a re-plan)
        when everything failed or nothing came back.
        """
        if any(r.ok and r.structured.get("found") is True for r in results):
            return True
        distinct = {e.document_id or e.url or e.text[:60] for e in state.evidence}
        if len(distinct) >= self.settings.top_k:
            return True
        # Every tool in this round failed: a re-plan may pick a different route.
        if results and all(not r.ok for r in results):
            return False
        return bool(state.evidence) and not self.llm.available

    # ---------------------------------------------------------------- blocked
    def _finalise_blocked(self, state: AgentState, safety: SafetyAssessment) -> FinalResponse:
        """Short-circuit path for emergencies, self-harm and invalid input."""
        state.final_answer = safety.direct_response or "This request cannot be processed."
        state.status = f"blocked:{safety.risk_category.value.lower()}"
        state.confidence = compute_confidence([], None)
        state.verification = None
        response = FinalResponse(
            request_id=state.request_id,
            answer=state.final_answer,
            confidence=ConfidenceLevel.INSUFFICIENT,
            confidence_score=0.0,
            citations=[],
            warnings=state.warnings,
            tools_used=[],
            risk_category=safety.risk_category,
            status=state.status,
        )
        self.audit.log(state, response)
        return response

    # ----------------------------------------------------------------- health
    def health(self) -> dict[str, object]:
        """Component status for ``GET /health`` and the smoke test."""
        from stores.vector_store import build_vector_store  # noqa: PLC0415

        try:
            with build_vector_store(self.settings) as store:
                vector_health = store.health()
        except Exception as exc:  # noqa: BLE001
            vector_health = {"ok": False, "error": str(exc)}

        try:
            structured_health = {"ok": True, **build_structured_store(self.settings).stats()}
        except Exception as exc:  # noqa: BLE001
            structured_health = {"ok": False, "error": str(exc)}

        return {
            "llm": self.llm.health(),
            "vector_store": vector_health,
            "structured_store": structured_health,
            "tools": self.registry.names,
            "max_tool_rounds": self.settings.max_tool_rounds,
        }


def _dedupe(evidence: Sequence[Evidence]) -> list[Evidence]:
    """Drop duplicate passages, keeping the highest-scoring copy."""
    best: dict[str, Evidence] = {}
    order: list[str] = []
    for item in evidence:
        key = (item.url or "") + "|" + item.text.strip()[:200]
        existing = best.get(key)
        if existing is None:
            best[key] = item
            order.append(key)
        elif (item.score or 0) > (existing.score or 0):
            best[key] = item
    return [best[key] for key in order]


def build_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """Factory used by the CLI and the API."""
    return Orchestrator(settings=settings)
