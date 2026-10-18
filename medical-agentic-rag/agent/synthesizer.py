"""Answer synthesis from retrieved evidence.

The synthesizer sees only the question, the evidence block, the structured tool
outputs and the safety constraints - never the raw conversation or the model's
own prior knowledge framing. When no LLM is available it falls back to an
*extractive* summary that quotes retrieved passages with their ids rather than
generating unsupported prose.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence

from agent.domain import OUT_OF_SCOPE_ANSWER
from agent.prompts import (
    DOMAIN_SCOPE_NOTICE,
    INJECTION_NOTICE,
    SYNTHESIZER_SYSTEM,
    SYNTHESIZER_USER,
)
from agent.state import Evidence, ToolResult
from config.settings import Settings, get_settings
from llm.base import BaseLLMClient, LLMError, classify_llm_error
from safety.citation_formatter import build_evidence_block

logger = logging.getLogger(__name__)

NO_EVIDENCE_ANSWER = (
    "No reliable evidence was retrieved for this request, so I cannot give a "
    "grounded answer. This system answers only from its curated corpus, structured "
    "medical store and the retrieval tools it can reach; it does not answer from "
    "model memory.\n\n"
    "You could try rephrasing the question, ingesting relevant source documents into "
    "the local corpus, or consulting a clinician or an authoritative source such as a "
    "national health service or regulator."
)


class Synthesizer:
    """Turns evidence into a grounded draft answer."""

    def __init__(self, llm: BaseLLMClient, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    def synthesize(
        self,
        query: str,
        evidence: Sequence[Evidence],
        tool_results: Sequence[ToolResult],
        safety_constraints: str = "",
        out_of_scope: bool = False,
    ) -> str:
        """Produce a draft answer grounded in ``evidence``.

        When ``out_of_scope`` is set the planner has already refused the query as
        unrelated to CKD. With nothing retrieved there is nothing to synthesise, so
        a fixed scope message is returned rather than invoking the model - which
        removes any opportunity to answer from memory.
        """
        if out_of_scope and not evidence:
            return OUT_OF_SCOPE_ANSWER

        if not evidence and not _has_useful_structured_output(tool_results):
            return NO_EVIDENCE_ANSWER

        evidence_block = build_evidence_block(evidence)
        structured_block = _render_structured(tool_results)

        llm_reason = self.llm_unavailable_reason()
        if self.llm.available:
            try:
                response = self.llm.chat(
                    [
                        {
                            "role": "system",
                            "content": SYNTHESIZER_SYSTEM.format(
                                domain_scope_notice=DOMAIN_SCOPE_NOTICE,
                                injection_notice=INJECTION_NOTICE,
                            ),
                        },
                        {
                            "role": "user",
                            "content": SYNTHESIZER_USER.format(
                                query=query,
                                safety_constraints=safety_constraints or "- (none)",
                                evidence_block=evidence_block,
                                structured_block=structured_block,
                            ),
                        },
                    ],
                    temperature=self.settings.gemini_temperature,
                )
                if response.text.strip():
                    return response.text.strip()
                logger.warning("Synthesizer returned empty text; using extractive fallback.")
                llm_reason = "the model returned an empty response"
            except LLMError as exc:
                llm_reason = classify_llm_error(exc)
                logger.warning("Synthesis via LLM failed (%s); using extractive fallback.", llm_reason)
            except Exception as exc:  # noqa: BLE001
                llm_reason = classify_llm_error(exc)
                logger.exception("Unexpected synthesis failure: %s", exc)

        return self.extractive_summary(query, evidence, tool_results, reason=llm_reason)

    def llm_unavailable_reason(self) -> str | None:
        """Why generation is unavailable, or ``None`` when it is available."""
        if self.llm.available:
            return None
        health = self.llm.health()
        return str(health.get("reason") or "no LLM backend configured")

    # ------------------------------------------------------------- fallback
    @staticmethod
    def extractive_summary(
        query: str,
        evidence: Sequence[Evidence],
        tool_results: Sequence[ToolResult],
        reason: str | None = None,
    ) -> str:
        """Deterministic, non-generative answer built from retrieved text.

        Used when no LLM is reachable. It quotes retrieved passages verbatim with
        their evidence ids so nothing is asserted that was not retrieved.
        """
        if not evidence:
            structured_lines = _structured_findings(tool_results)
            if structured_lines:
                return (
                    "No text passages were retrieved, but the structured lookups returned:\n"
                    + "\n".join(f"- {line}" for line in structured_lines)
                    + "\n\n(Generated without a language model: this is a direct report of "
                    "database results, not a synthesised explanation.)"
                )
            return NO_EVIDENCE_ANSWER

        query_terms = {w for w in re.findall(r"[a-z]{4,}", query.lower())}
        ranked = sorted(
            evidence,
            key=lambda e: (
                len(query_terms & set(re.findall(r"[a-z]{4,}", e.text.lower()))),
                e.score or 0.0,
            ),
            reverse=True,
        )[:3]

        parts = [
            f"**Answer generation is unavailable** ({reason or 'reason unknown'}), so the "
            "system is returning the retrieved evidence directly instead of a "
            "synthesised answer. The evidence below is real and traceable; only the "
            "summarisation step is missing.",
            "",
            "Most relevant retrieved passages:",
        ]
        for item in ranked:
            excerpt = _plain_excerpt(item.text, limit=420)
            # Rendered as a blockquote so the quoted passage is visually distinct
            # from the assistant's own words.
            parts.append(f"\n**[{item.evidence_id}] {item.short_label()}**\n\n> {excerpt}")

        structured_lines = _structured_findings(tool_results)
        if structured_lines:
            parts.append("\nStructured lookups:")
            parts.extend(f"- {line}" for line in structured_lines)

        return "\n".join(parts).strip()


#: Markdown structure that must not survive into a rendered excerpt. Corpus
#: chunks are Markdown, so a passage beginning "## eGFR and staging concepts"
#: would otherwise render as a giant H2 heading in the chat UI.
_MD_PREFIX_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s+|>\s?|[-*+]\s+|\d+[.)]\s+)", re.MULTILINE)
_MD_INLINE_RE = re.compile(r"[*_`]{1,3}")


def _plain_excerpt(text: str, limit: int = 420) -> str:
    """Flatten a Markdown chunk into one safe, quotable line.

    Strips heading/list/quote markers and inline emphasis before collapsing
    whitespace, so the excerpt renders as text rather than as a heading.
    """
    cleaned = _MD_PREFIX_RE.sub("", text or "")
    cleaned = _MD_INLINE_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rsplit(" ", 1)[0] + " ..."
    return cleaned


def _has_useful_structured_output(tool_results: Sequence[ToolResult]) -> bool:
    return any(
        result.ok and result.structured and result.structured.get("found") is not None
        for result in tool_results
    )


def _structured_findings(tool_results: Sequence[ToolResult]) -> list[str]:
    """One-line summaries of structured tool outcomes, hits and misses alike."""
    lines: list[str] = []
    for result in tool_results:
        data = result.structured or {}
        if not result.ok:
            lines.append(f"{result.tool}: FAILED ({result.error})")
            continue
        if result.tool == "drug_interaction":
            if data.get("found"):
                lines.append(
                    f"drug_interaction: record found for {data.get('drug_a')} + "
                    f"{data.get('drug_b')} (severity: {data.get('severity') or 'not recorded'}, "
                    f"data_status: {data.get('data_status')})"
                )
            else:
                lines.append(
                    f"drug_interaction: NO record in the database for "
                    f"{data.get('drug_a')} + {data.get('drug_b')}. Absence of a record is "
                    "not evidence that the combination is safe."
                )
        elif result.tool == "dosage_lookup":
            if data.get("found"):
                lines.append(
                    f"dosage_lookup: reference text retrieved for {data.get('drug')} "
                    "(reference information only, not a recommendation)"
                )
            else:
                lines.append(f"dosage_lookup: no reference record for {data.get('drug')}")
        elif data.get("error"):
            lines.append(f"{result.tool}: unavailable ({data['error']})")
    return lines


def _render_structured(tool_results: Sequence[ToolResult]) -> str:
    """Compact JSON view of structured outputs for the synthesis prompt."""
    payload: list[dict[str, Any]] = []
    for result in tool_results:
        payload.append(
            {
                "tool": result.tool,
                "ok": result.ok,
                "error": result.error,
                "structured": result.structured,
            }
        )
    if not payload:
        return "(no structured tool output)"
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)[:4000]
