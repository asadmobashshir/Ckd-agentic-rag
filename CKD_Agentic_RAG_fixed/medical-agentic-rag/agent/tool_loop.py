"""Native tool-calling agent loop.

This is the component that makes the system genuinely agentic. The previous
design was plan-then-execute: a planner emitted JSON, the application ran the
tools, and a *separate* synthesis call received a flat evidence block. The model
never saw a tool result, so it could not react to one.

The loop implemented here is the real protocol:

    1. send the question + tool specifications to the model
    2. parse the tool calls the model requests
    3. validate names and arguments against the registry (Pydantic schemas)
    4. execute the tools in Python
    5. send each tool's ACTUAL output back as a ``role="tool"`` message
    6. let the model call more tools because of what it just read
    7. stop when the model answers, or at ``max_agent_iterations``

Guarantees worth stating explicitly:

* A tool appears in ``tool_results`` only if it actually reached execution.
  Calls rejected at validation are reported separately, never as "used".
* An identical (name, arguments) pair is executed once; a repeat is answered
  from cache with a note, which stops the model looping on one tool.
* The loop always terminates: the iteration ceiling is enforced, and the final
  turn is made with tools withheld so the model must produce prose.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from agent.prompts import (
    AGENT_FINAL_NUDGE,
    AGENT_SYSTEM,
    DOMAIN_SCOPE_NOTICE,
    INJECTION_NOTICE,
)
from agent.state import Evidence, SafetyAssessment, ToolCallRequest, ToolResult
from agent.tools.tool_registry import ToolRegistry
from config.settings import Settings, get_settings
from llm.base import BaseLLMClient, LLMError, classify_llm_error
from safety.citation_formatter import assign_evidence_ids

logger = logging.getLogger(__name__)

#: Caps applied to tool output before it is sent back to the model.
MAX_EVIDENCE_PER_MESSAGE = 6
MAX_EVIDENCE_CHARS = 600
MAX_STRUCTURED_CHARS = 900


@dataclass
class ToolLoopResult:
    """Everything the orchestrator needs from one agent run."""

    final_text: str = ""
    tool_results: list[ToolResult] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    iterations: int = 0
    rejected_calls: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stop_reason: str = "completed"
    llm_failed: bool = False
    error: str | None = None

    @property
    def executed_tools(self) -> list[str]:
        """Tools that genuinely reached execution, in order."""
        return [r.tool for r in self.tool_results]


class ToolCallingLoop:
    """Runs the model/tool conversation for a single query."""

    def __init__(
        self,
        llm: BaseLLMClient,
        registry: ToolRegistry,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ API
    def usable(self) -> bool:
        """Whether a real tool-calling conversation is possible right now."""
        return bool(
            self.settings.enable_native_tool_calling
            and self.llm.available
            and getattr(self.llm, "supports_tool_calling", False)
            and len(self.registry) > 0
        )

    async def run(self, query: str, safety: SafetyAssessment | None = None) -> ToolLoopResult:
        """Execute the loop and return the model's final answer plus evidence."""
        result = ToolLoopResult()
        tool_specs = self.registry.function_specs()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(safety)},
            {"role": "user", "content": query},
        ]

        # (name, canonical-args) -> ToolResult, so a repeated request is answered
        # from cache instead of running the same retrieval again.
        executed: dict[tuple[str, str], ToolResult] = {}
        max_iterations = self.settings.max_agent_iterations

        for iteration in range(1, max_iterations + 1):
            result.iterations = iteration
            final_turn = iteration == max_iterations

            try:
                response = await asyncio.to_thread(
                    self.llm.chat,
                    messages,
                    tools=None if final_turn else tool_specs,
                    tool_choice=None if final_turn else "auto",
                    temperature=self.settings.llm_temperature,
                )
            except LLMError as exc:
                reason = classify_llm_error(exc)
                logger.warning("Agent loop LLM call failed at iteration %d: %s", iteration, reason)
                result.llm_failed = True
                result.error = reason
                result.stop_reason = "llm_error"
                return result
            except Exception as exc:  # noqa: BLE001
                reason = classify_llm_error(exc)
                logger.exception("Agent loop raised at iteration %d", iteration)
                result.llm_failed = True
                result.error = reason
                result.stop_reason = "llm_error"
                return result

            # ---- model produced an answer instead of tool calls -> done -----
            if not response.tool_calls:
                text = (response.text or "").strip()
                if text:
                    result.final_text = text
                    result.stop_reason = "answered" if iteration > 1 else "answered_without_tools"
                    return result
                if final_turn:
                    result.stop_reason = "empty_final_response"
                    return result
                # Empty, non-final turn: ask once for prose, then stop.
                messages.append({"role": "user", "content": AGENT_FINAL_NUDGE})
                continue

            if final_turn:
                # Tools were withheld this turn; a tool call here is a protocol
                # violation, so take whatever prose came with it.
                result.final_text = (response.text or "").strip()
                result.stop_reason = "iteration_limit"
                continue

            # ---- record the assistant turn verbatim -------------------------
            raw_calls = response.usage.get("_raw_tool_calls") or self._rebuild_raw_calls(response)
            messages.append({
                "role": "assistant",
                "content": response.text or "",
                "tool_calls": raw_calls,
            })

            # ---- validate + execute ----------------------------------------
            to_run: list[tuple[str, ToolCallRequest]] = []
            for call in response.tool_calls:
                call_id = call.id or f"call_{iteration}_{call.name}"
                request = ToolCallRequest(name=call.name, arguments=dict(call.arguments or {}))

                error = self.registry.validate_call(request)
                if error:
                    # Tell the model what was wrong so it can correct itself.
                    result.rejected_calls.append(f"{call.name}: {error}")
                    messages.append(self._tool_message(
                        call_id, call.name,
                        {"ok": False, "error": error,
                         "hint": "Fix the arguments and call again, or call a different tool."},
                    ))
                    continue

                signature = (request.name, _canonical(request.arguments))
                cached = executed.get(signature)
                if cached is not None:
                    messages.append(self._tool_message(
                        call_id, call.name,
                        {"ok": cached.ok,
                         "note": "Duplicate call - previous result reused, not re-executed.",
                         "summary": _clip_json(cached.structured, MAX_STRUCTURED_CHARS)},
                    ))
                    continue

                to_run.append((call_id, request))

            if to_run:
                outcomes = await self.registry.execute_many([req for _, req in to_run])
                for (call_id, request), outcome in zip(to_run, outcomes):
                    executed[(request.name, _canonical(request.arguments))] = outcome
                    result.tool_results.append(outcome)
                    result.evidence.extend(outcome.evidence)

                    # Assign stable citation ids now, so the model cites the same
                    # ids the UI will later render.
                    assign_evidence_ids(result.evidence)
                    messages.append(self._tool_message(
                        call_id, request.name, self._render_outcome(outcome)
                    ))

                    if not outcome.ok:
                        result.warnings.append(
                            f"Tool '{outcome.tool}' failed: {outcome.error}"
                        )

        result.stop_reason = result.stop_reason or "iteration_limit"
        return result

    # -------------------------------------------------------------- helpers
    def _system_prompt(self, safety: SafetyAssessment | None) -> str:
        constraints = "SAFETY CONTEXT: standard general-information handling."
        if safety is not None:
            value = safety.risk_category.value
            if value == "PERSONALIZED_CLINICAL_DECISION":
                constraints = (
                    "SAFETY CONTEXT: risk_category=PERSONALIZED_CLINICAL_DECISION. "
                    "The user is asking about their own care. Retrieve GENERAL "
                    "reference evidence only. Never produce a personalised dose, "
                    "a titration schedule or a diagnosis. State explicitly that "
                    "the decision belongs to their prescriber or pharmacist."
                )
            else:
                constraints = f"SAFETY CONTEXT: risk_category={value}."
        return AGENT_SYSTEM.format(
            domain_scope_notice=DOMAIN_SCOPE_NOTICE,
            safety_context=constraints,
            injection_notice=INJECTION_NOTICE,
        )

    @staticmethod
    def _rebuild_raw_calls(response: Any) -> list[dict[str, Any]]:
        """Reconstruct provider-shaped tool_calls when the client did not supply them."""
        return [
            {
                "id": call.id or f"call_{i}",
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments or {})},
            }
            for i, call in enumerate(response.tool_calls)
        ]

    @staticmethod
    def _tool_message(call_id: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": json.dumps(payload, ensure_ascii=False, default=str),
        }

    @staticmethod
    def _render_outcome(outcome: ToolResult) -> dict[str, Any]:
        """Compact, citable view of a tool's real output for the model."""
        payload: dict[str, Any] = {
            "tool": outcome.tool,
            "ok": outcome.ok,
            "summary": _clip_json(outcome.structured, MAX_STRUCTURED_CHARS),
        }
        if outcome.error:
            payload["error"] = outcome.error

        items = []
        for item in outcome.evidence[:MAX_EVIDENCE_PER_MESSAGE]:
            text = " ".join((item.text or "").split())
            if len(text) > MAX_EVIDENCE_CHARS:
                text = text[:MAX_EVIDENCE_CHARS].rsplit(" ", 1)[0] + " ..."
            items.append({
                "id": item.evidence_id,          # cite this exact id
                "title": item.title,
                "section": item.section,
                "data_status": item.data_status,
                "text": text,
            })
        payload["evidence"] = items
        if not items and outcome.ok:
            payload["note"] = (
                "This tool returned no evidence. That is an absence of data, not "
                "proof of a negative."
            )
        return payload


def _canonical(arguments: dict[str, Any]) -> str:
    """Stable key for an argument dict, so reordering is not a new call."""
    try:
        return json.dumps(arguments, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        return repr(sorted(arguments.items()))


def _clip_json(value: Any, limit: int) -> Any:
    """Keep structured payloads small without destroying their shape."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        text = str(value)
    if len(text) <= limit:
        return value
    return {"_truncated": text[:limit] + " ..."}
