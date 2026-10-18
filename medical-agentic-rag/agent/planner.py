"""Planning: decide which registered tools (if any) should run.

Three layers, tried in order:

1. **Native Gemini tool calling** - the model picks tools through the function-calling
   API, which keeps argument shapes aligned with the Pydantic schemas.
2. **Structured JSON output** - used when the model returns no tool calls or the
   SDK/model does not support tool calling.
3. **Deterministic heuristics** - used when no LLM is reachable at all, so the
   system still retrieves evidence instead of inventing an answer.

Every produced plan is filtered against the registry before it is returned;
unregistered names and invalid arguments are dropped with a recorded reason.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.domain import (
    OUT_OF_SCOPE_INTENT,
    DomainVerdict,
    classify_domain,
    matched_out_of_scope_terms,
)
from agent.prompts import (
    DOMAIN_SCOPE_NOTICE,
    INJECTION_NOTICE,
    PLANNER_JSON_INSTRUCTION,
    PLANNER_SYSTEM,
    REPLAN_INSTRUCTION,
)
from agent.state import Plan, SafetyAssessment, ToolCallRequest, ToolResult
from agent.tools.tool_registry import ToolRegistry
from config.settings import Settings, get_settings
from llm.base import BaseLLMClient, LLMError

logger = logging.getLogger(__name__)

GREETING_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|good (morning|afternoon|evening)|thanks|thank you|"
    r"how are you|who are you|what can you do|help)\b[\s!.?]*$",
    re.IGNORECASE,
)
LITERATURE_RE = re.compile(
    r"\b(literature|stud(y|ies)|trial|trials|research|evidence|meta-?analys[ei]s|"
    r"systematic review|paper|papers|publication|rct)\b",
    re.IGNORECASE,
)
RECENCY_RE = re.compile(
    r"\b(recent|latest|current|today|this year|2024|2025|2026|news|updated?|"
    r"new guidance|outbreak)\b",
    re.IGNORECASE,
)
DOSAGE_RE = re.compile(r"\b(dose|dosage|dosing|how much|mg\b|posology)\b", re.IGNORECASE)
INTERACTION_RE = re.compile(
    r"\b(interact\w*|combin\w*|together|with|contraindicat\w*|co-?administ\w*)\b",
    re.IGNORECASE,
)


class Planner:
    """Chooses tools for a query, with graceful degradation."""

    def __init__(
        self,
        llm: BaseLLMClient,
        registry: ToolRegistry,
        settings: Settings | None = None,
        known_drugs: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.settings = settings or get_settings()
        self._known_drugs = [d.lower() for d in (known_drugs or [])]

    # --------------------------------------------------------------- public
    def plan(
        self,
        query: str,
        safety: SafetyAssessment | None = None,
        round_index: int = 0,
        observations: list[ToolResult] | None = None,
    ) -> Plan:
        """Produce a validated plan for this round."""
        if GREETING_RE.match(query) and round_index == 0:
            return Plan(
                intent="conversational",
                tools=[],
                reason="Greeting or meta question; no retrieval required.",
                round_index=round_index,
            )

        # Domain scope is enforced deterministically, before any LLM call, so the
        # behaviour holds identically with and without a Gemini key.
        if round_index == 0 and self.is_out_of_scope(query):
            terms = matched_out_of_scope_terms(query)
            return Plan(
                intent=OUT_OF_SCOPE_INTENT,
                tools=[],
                reason=(
                    "Query is outside the CKD domain"
                    + (f" (matched: {', '.join(terms)})" if terms else "")
                    + "; no retrieval performed."
                ),
                round_index=round_index,
            )

        if self.llm.available:
            for strategy in (self._plan_with_tool_calling, self._plan_with_json):
                try:
                    plan = strategy(query, safety, round_index, observations or [])
                except LLMError as exc:
                    logger.warning("Planner strategy %s failed: %s", strategy.__name__, exc)
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Planner strategy %s raised: %s", strategy.__name__, exc)
                    continue
                if plan is not None:
                    plan.round_index = round_index
                    return self._filter(plan)
            logger.warning("All LLM planning strategies failed; using heuristics.")

        plan = self.heuristic_plan(query, round_index, observations or [])
        return self._filter(plan)

    def is_out_of_scope(self, query: str) -> bool:
        """Whether ``query`` is clearly unrelated to CKD.

        Only a confident out-of-scope verdict returns ``True``; ambiguous queries
        keep the existing conservative behaviour and are allowed to retrieve.
        """
        verdict = classify_domain(query, extra_in_scope_terms=self._known_drugs)
        return verdict is DomainVerdict.OUT_OF_SCOPE

    # ------------------------------------------------------- LLM strategies
    def _system_prompt(self, safety: SafetyAssessment | None) -> str:
        constraints = "SAFETY CONTEXT: standard general-information handling."
        if safety is not None:
            constraints = (
                f"SAFETY CONTEXT: risk_category={safety.risk_category.value}. "
                + (
                    "The user is asking about their own care. Retrieve GENERAL reference "
                    "evidence only; never plan a tool call intended to produce a "
                    "personalised dose or diagnosis."
                    if safety.risk_category.value == "PERSONALIZED_CLINICAL_DECISION"
                    else "Handle as a general information request."
                )
            )
        return PLANNER_SYSTEM.format(
            domain_scope_notice=DOMAIN_SCOPE_NOTICE,
            catalog="AVAILABLE TOOLS:\n" + self.registry.planner_catalog(),
            safety_context=constraints,
            injection_notice=INJECTION_NOTICE,
        )

    def _user_prompt(
        self, query: str, round_index: int, observations: list[ToolResult]
    ) -> str:
        prompt = f"USER QUESTION:\n{query}"
        if round_index > 0:
            prompt += "\n\n" + REPLAN_INSTRUCTION.format(
                round_index=round_index + 1,
                max_rounds=self.settings.max_tool_rounds,
                observations=self._render_observations(observations),
            )
        return prompt

    @staticmethod
    def _render_observations(observations: list[ToolResult]) -> str:
        if not observations:
            return "(none)"
        lines = []
        for result in observations:
            status = "ok" if result.ok else f"FAILED ({result.error})"
            lines.append(
                f"- {result.tool}{result.arguments}: {status}; "
                f"{len(result.evidence)} evidence item(s); "
                f"structured={_truncate(result.structured, 220)}"
            )
        return "\n".join(lines)

    def _plan_with_tool_calling(
        self,
        query: str,
        safety: SafetyAssessment | None,
        round_index: int,
        observations: list[ToolResult],
    ) -> Plan | None:
        if not self.llm.supports_tool_calling:
            return None
        response = self.llm.chat(
            [
                {"role": "system", "content": self._system_prompt(safety)},
                {"role": "user", "content": self._user_prompt(query, round_index, observations)},
            ],
            tools=self.registry.function_specs(),
            tool_choice="auto",
            temperature=0.0,
        )
        if not response.tool_calls:
            # An empty selection is a legitimate outcome, but only trust it when the
            # model actually explained itself; otherwise let the JSON path retry.
            if response.text:
                return Plan(
                    intent=_guess_intent(query),
                    tools=[],
                    reason=response.text[:400],
                )
            return None
        return Plan(
            intent=_guess_intent(query),
            tools=[
                ToolCallRequest(name=call.name, arguments=call.arguments)
                for call in response.tool_calls
            ],
            reason=(response.text or "Tools selected via native function calling.")[:400],
        )

    def _plan_with_json(
        self,
        query: str,
        safety: SafetyAssessment | None,
        round_index: int,
        observations: list[ToolResult],
    ) -> Plan | None:
        payload = self.llm.chat_json(
            [
                {
                    "role": "system",
                    "content": self._system_prompt(safety) + "\n\n" + PLANNER_JSON_INSTRUCTION,
                },
                {"role": "user", "content": self._user_prompt(query, round_index, observations)},
            ],
            temperature=0.0,
        )
        return _plan_from_payload(payload)

    # -------------------------------------------------------- deterministic
    def heuristic_plan(
        self, query: str, round_index: int = 0, observations: list[ToolResult] | None = None
    ) -> Plan:
        """Rule-based plan used when no LLM is available.

        Deliberately simple and conservative: it exists so the system still
        retrieves real evidence offline, not to imitate model reasoning. It
        enforces the same CKD domain scope as the LLM planner.

        Known limitation: interaction routing requires BOTH drug names to appear
        in the structured store's vocabulary, so a question about an unknown drug
        pair falls back to vector search alone. The LLM planner has no such limit.
        This is why ``eval`` case ``interact-002`` scores 0 on tool selection in
        offline mode - it is a real gap, not a mislabelled test.
        """
        if round_index > 0:
            return Plan(
                intent="sufficient",
                tools=[],
                reason="Heuristic planner does not re-plan; one retrieval round only.",
                round_index=round_index,
            )

        # Repeated here so the heuristic planner respects CKD scope even when it is
        # invoked directly, not only through plan()'s gate.
        if self.is_out_of_scope(query):
            return Plan(
                intent=OUT_OF_SCOPE_INTENT,
                tools=[],
                reason="Query is outside the CKD domain; no retrieval performed.",
                round_index=round_index,
            )

        lowered = query.lower()
        mentioned = [drug for drug in self._known_drugs if drug and drug in lowered]
        tools: list[ToolCallRequest] = []
        intent = "general_medical_information"

        if len(mentioned) >= 2 and INTERACTION_RE.search(query) and self.registry.has("drug_interaction"):
            intent = "drug_interaction"
            tools.append(
                ToolCallRequest(
                    name="drug_interaction",
                    arguments={"drug_a": mentioned[0], "drug_b": mentioned[1]},
                )
            )
        elif DOSAGE_RE.search(query) and mentioned and self.registry.has("dosage_lookup"):
            intent = "dosage_reference"
            tools.append(ToolCallRequest(name="dosage_lookup", arguments={"drug": mentioned[0]}))

        if self.registry.has("vector_search"):
            tools.append(
                ToolCallRequest(
                    name="vector_search",
                    arguments={"query": query[:512], "top_k": self.settings.top_k},
                )
            )

        if LITERATURE_RE.search(query) and self.registry.has("literature_search"):
            intent = "literature_review" if intent == "general_medical_information" else intent
            tools.append(
                ToolCallRequest(name="literature_search", arguments={"query": query[:256]})
            )

        if RECENCY_RE.search(query) and self.registry.has("web_search"):
            tools.append(ToolCallRequest(name="web_search", arguments={"query": query[:256]}))

        return Plan(
            intent=intent,
            tools=tools,
            reason="Deterministic keyword-based plan (LLM planner unavailable).",
            round_index=round_index,
        )

    # -------------------------------------------------------------- filtering
    def _filter(self, plan: Plan) -> Plan:
        """Drop unregistered tools and invalid arguments; de-duplicate calls."""
        kept: list[ToolCallRequest] = []
        seen: set[tuple[str, str]] = set()
        rejected: list[str] = []

        for call in plan.tools:
            error = self.registry.validate_call(call)
            if error:
                rejected.append(error)
                continue
            signature = (call.name, repr(sorted(call.arguments.items())))
            if signature in seen:
                continue
            seen.add(signature)
            kept.append(call)

        if rejected:
            plan.reason = (plan.reason + " | rejected: " + "; ".join(rejected))[:800]
            logger.warning("Planner produced %d invalid call(s): %s", len(rejected), rejected)

        plan.tools = kept[: self.settings.max_tool_rounds * 3]
        return plan


def _plan_from_payload(payload: dict[str, Any]) -> Plan | None:
    """Coerce a loosely shaped JSON payload into a :class:`Plan`."""
    if not isinstance(payload, dict):
        return None
    raw_tools = payload.get("tools") or payload.get("tool_calls") or []
    calls: list[ToolCallRequest] = []
    if isinstance(raw_tools, list):
        for item in raw_tools:
            if isinstance(item, str):
                calls.append(ToolCallRequest(name=item, arguments={}))
            elif isinstance(item, dict):
                name = item.get("name") or item.get("tool")
                if not name:
                    continue
                arguments = item.get("arguments") or item.get("args") or {}
                if not isinstance(arguments, dict):
                    arguments = {}
                calls.append(ToolCallRequest(name=str(name), arguments=arguments))
    return Plan(
        intent=str(payload.get("intent") or "general_medical_information")[:64],
        tools=calls,
        reason=str(payload.get("reason") or "")[:400],
    )


def _guess_intent(query: str) -> str:
    if INTERACTION_RE.search(query) and DOSAGE_RE.search(query):
        return "medication_safety"
    if LITERATURE_RE.search(query):
        return "literature_review"
    if DOSAGE_RE.search(query):
        return "dosage_reference"
    if INTERACTION_RE.search(query):
        return "drug_interaction"
    return "general_medical_information"


def _truncate(value: Any, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
