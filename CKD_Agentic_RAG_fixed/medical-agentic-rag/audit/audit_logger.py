"""JSONL audit logging with redaction.

One JSON object per query, appended to ``data/processed/audit.jsonl``.

Two things are deliberately *not* recorded: secrets and free-text personal
health detail. Queries are sanitised (emails, phone numbers, long digit strings,
dates of birth, explicit names/ages), and tool arguments are filtered so only
non-sensitive keys survive.

.. note::
   This package is named ``audit`` rather than ``logging`` on purpose. A
   top-level directory called ``logging`` shadows the standard library module of
   the same name for every absolute import in the project, which breaks any
   dependency that does ``import logging``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from agent.state import AgentState, FinalResponse, utc_now_iso
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"), "[PHONE]"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),
    (re.compile(r"\b\d{6,}\b"), "[ID]"),
    (re.compile(r"\b(?:mrn|nhs|ssn|aadhaar|patient(?:\s+id)?)\s*[:#]?\s*\S+", re.IGNORECASE), "[PATIENT_ID]"),
    (re.compile(r"\bmy name is\s+[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?", re.IGNORECASE), "[NAME]"),
    (re.compile(r"\bi am\s+\d{1,3}\s*(?:years old|yo|y/o)\b", re.IGNORECASE), "[AGE]"),
    (re.compile(r"\b\d{1,3}\s*(?:years old|yo|y/o)\b", re.IGNORECASE), "[AGE]"),
)

#: Argument keys safe to persist; anything else is dropped.
_SAFE_ARG_KEYS = frozenset(
    {"query", "drug", "drug_a", "drug_b", "top_k", "max_results", "population",
     "indication", "include_abstract"}
)

_SECRET_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|authorization)", re.IGNORECASE)

MAX_LOGGED_QUERY_CHARS = 300


def sanitize_text(text: str) -> str:
    """Redact obvious identifiers from free text."""
    cleaned = text or ""
    for pattern, replacement in _REDACTIONS:
        cleaned = pattern.sub(replacement, cleaned)
    if len(cleaned) > MAX_LOGGED_QUERY_CHARS:
        cleaned = cleaned[:MAX_LOGGED_QUERY_CHARS] + "…[truncated]"
    return cleaned


def sanitize_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only whitelisted, non-secret argument keys, with values sanitised."""
    safe: dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        if _SECRET_KEY_RE.search(key) or key not in _SAFE_ARG_KEYS:
            continue
        safe[key] = sanitize_text(value) if isinstance(value, str) else value
    return safe


class AuditLogger:
    """Append-only JSONL audit trail."""

    def __init__(self, path: Path | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.path = path or self._settings.audit_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def build_record(self, state: AgentState, response: FinalResponse) -> dict[str, Any]:
        """Assemble the audit record for a completed request."""
        total_ms = round(sum(r.duration_ms for r in state.tool_results), 2)
        return {
            "request_id": state.request_id,
            "timestamp": utc_now_iso(),
            "query": sanitize_text(state.query),
            "query_chars": len(state.query or ""),
            "intent": state.intent,
            "llm_available": state.llm_available,
            "planner": [
                {
                    "round": plan.round_index,
                    "intent": plan.intent,
                    "reason": sanitize_text(plan.reason)[:300],
                    "tools": [call.name for call in plan.tools],
                }
                for plan in state.plans
            ],
            "tool_calls": [
                {
                    "tool": result.tool,
                    "ok": result.ok,
                    "arguments": sanitize_arguments(result.arguments),
                    "duration_ms": result.duration_ms,
                    "evidence_count": len(result.evidence),
                    "error": result.error,
                }
                for result in state.tool_results
            ],
            "tool_duration_ms_total": total_ms,
            "evidence_sources": [
                {
                    "evidence_id": item.evidence_id,
                    "document_id": item.document_id,
                    "chunk_id": item.chunk_id,
                    "url": item.url,
                    "trust_tier": item.trust_tier.value,
                    "data_status": item.data_status,
                    "score": item.score,
                }
                for item in state.evidence
            ],
            "verification": (
                {
                    "grounded_claim_ratio": state.verification.grounded_claim_ratio,
                    "claims": [
                        {"status": c.status.value, "overlap": c.overlap_score, "high_risk": c.high_risk}
                        for c in state.verification.claims
                    ],
                    "removed_claims": len(state.verification.removed_claims),
                    "notes": state.verification.notes,
                }
                if state.verification
                else None
            ),
            "confidence": (
                {
                    "level": state.confidence.level.value,
                    "score": state.confidence.score,
                    "factors": state.confidence.factors,
                }
                if state.confidence
                else None
            ),
            "safety": (
                {
                    "risk_category": state.safety.risk_category.value,
                    "blocked": state.safety.block_pipeline,
                    "matched_rules": state.safety.matched_rules,
                }
                if state.safety
                else None
            ),
            "warnings": state.warnings,
            "citation_count": len(response.citations),
            "answer_chars": len(response.answer or ""),
            "status": response.status,
        }

    def log(self, state: AgentState, response: FinalResponse) -> dict[str, Any]:
        """Write one audit record. Logging failures never break a response."""
        record = self.build_record(state, response)
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to write audit record %s: %s", state.request_id, exc)
        return record

    def read_all(self) -> list[dict[str, Any]]:
        """Read back every record (used by the evaluation harness)."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed audit line")
        return records
