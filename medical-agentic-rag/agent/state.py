"""Structured state and data models shared across the agent pipeline.

Everything that crosses a module boundary is a Pydantic model so provenance
fields cannot be silently dropped and malformed tool output fails loudly.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    """Current UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat()


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


#: Provenance of a piece of evidence. This records WHERE the text came from; it
#: is not a quality score and never implies the content is correct.
#:
#: ``demo``        synthetic demonstration text written for this repository
#: ``guideline``   curated clinical practice guideline (e.g. KDIGO, NICE)
#: ``regulatory``  regulatory product labelling (e.g. FDA SPL / DailyMed)
#: ``reference``   public-health reference material (e.g. NIDDK, MedlinePlus)
#: ``literature``  peer-reviewed literature (e.g. PubMed / PMC)
#: ``licensed``    licensed third-party dataset
#: ``live``        fetched from the network at query time
#: ``unknown``     unclassified
DataStatus = Literal[
    "demo",
    "guideline",
    "regulatory",
    "reference",
    "literature",
    "licensed",
    "live",
    "unknown",
]

#: Statuses whose text was *authored for demonstration* rather than drawn from a
#: real source. Confidence scoring and the citation footer key off this set, so
#: replacing the demo corpus with real documents lifts the synthetic-data penalty
#: because the evidence genuinely stopped being synthetic - not because a
#: threshold was lowered.
SYNTHETIC_DATA_STATUSES: frozenset[str] = frozenset({"demo"})


def is_synthetic(data_status: str) -> bool:
    """Whether this provenance marks text as demonstration-only content."""
    return data_status in SYNTHETIC_DATA_STATUSES


class TrustTier(str, Enum):
    """Configurable retrieval-prioritisation heuristic - NOT a truth guarantee."""

    A = "A"  # government / public-health / regulatory / clinical guidelines
    B = "B"  # peer-reviewed scientific literature
    C = "C"  # established medical institutions
    D = "D"  # general web sources
    UNKNOWN = "UNKNOWN"


class RiskCategory(str, Enum):
    GENERAL_INFORMATION = "GENERAL_INFORMATION"
    PERSONALIZED_CLINICAL_DECISION = "PERSONALIZED_CLINICAL_DECISION"
    EMERGENCY = "EMERGENCY"
    SELF_HARM = "SELF_HARM"
    NON_MEDICAL = "NON_MEDICAL"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONFLICTING = "conflicting"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class Evidence(BaseModel):
    """A single retrieved evidence unit with full provenance.

    Unknown metadata stays ``None``; it is never back-filled with guesses.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(default="", description="Assigned late, e.g. 'S1'.")
    text: str
    source: str = Field(description="Origin system, e.g. 'vector_search'.")
    title: str | None = None
    url: str | None = None
    publication_date: str | None = None
    retrieved_at: str = Field(default_factory=utc_now_iso)
    evidence_level: str | None = None
    document_type: str | None = None
    section: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    trust_tier: TrustTier = TrustTier.UNKNOWN
    data_status: DataStatus = "unknown"
    extra: dict[str, Any] = Field(default_factory=dict)

    def short_label(self) -> str:
        return self.title or self.url or f"{self.source} chunk {self.chunk_id or '?'}"


class ToolCallRequest(BaseModel):
    """A tool invocation requested by the planner."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """Validated planner output."""

    model_config = ConfigDict(extra="forbid")

    intent: str = "general_medical_information"
    tools: list[ToolCallRequest] = Field(default_factory=list)
    reason: str = ""
    round_index: int = 0


class ToolResult(BaseModel):
    """Normalised result envelope returned by every tool."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    ok: bool
    arguments: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    structured: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0
    started_at: str = Field(default_factory=utc_now_iso)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    status: ClaimStatus = ClaimStatus.UNSUPPORTED
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    overlap_score: float = 0.0
    high_risk: bool = False
    note: str | None = None


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[Claim] = Field(default_factory=list)
    grounded_claim_ratio: float = 0.0
    removed_claims: list[str] = Field(default_factory=list)
    revised_answer: str = ""
    notes: list[str] = Field(default_factory=list)


class ConfidenceReport(BaseModel):
    """Evidence confidence - NOT a probability that a medical claim is true."""

    model_config = ConfigDict(extra="forbid")

    level: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    score: float = 0.0
    factors: dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Evidence confidence reflects retrieval quality and claim grounding only. "
        "It is not a probability that any medical conclusion is correct."
    )


class SafetyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_category: RiskCategory = RiskCategory.GENERAL_INFORMATION
    block_pipeline: bool = False
    matched_rules: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    safety_note: str | None = None
    direct_response: str | None = None


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    title: str | None = None
    source: str
    url: str | None = None
    publication_date: str | None = None
    trust_tier: TrustTier = TrustTier.UNKNOWN
    data_status: str = "unknown"

    def render(self) -> str:
        parts = [self.title or "Untitled source", self.source]
        if self.publication_date:
            parts.append(self.publication_date)
        if self.url:
            parts.append(self.url)
        return f"[{self.evidence_id}] " + " — ".join(parts)


class AgentState(BaseModel):
    """Mutable state carried through the whole online pipeline."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=new_request_id)
    created_at: str = Field(default_factory=utc_now_iso)
    query: str
    intent: str | None = None
    plans: list[Plan] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    draft: str = ""
    verification: VerificationResult | None = None
    confidence: ConfidenceReport | None = None
    safety: SafetyAssessment | None = None
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    final_answer: str = ""
    status: str = "pending"
    llm_available: bool = False
    #: Iterations consumed by the native tool-calling loop (0 = loop not used).
    agent_iterations: int = 0
    #: Answer produced by the agent loop itself, from observed tool output.
    loop_answer: str = ""
    #: Which component produced the draft: "agent_loop", "synthesizer" or "none".
    answer_source: str = "none"

    @property
    def tools_used(self) -> list[str]:
        """Tools that actually executed, in order, without duplicates."""
        ordered: list[str] = []
        for result in self.tool_results:
            if result.tool not in ordered:
                ordered.append(result.tool)
        return ordered

    def add_warning(self, message: str) -> None:
        if message and message not in self.warnings:
            self.warnings.append(message)


class FinalResponse(BaseModel):
    """Transport-agnostic response returned by the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    answer: str
    confidence: ConfidenceLevel
    confidence_score: float = 0.0
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    risk_category: RiskCategory = RiskCategory.GENERAL_INFORMATION
    status: str = "ok"
    #: Iterations the native tool-calling loop consumed (0 = loop not used).
    agent_iterations: int = 0
    #: "agent_loop" when the model answered from observed tool output,
    #: "synthesizer" for the legacy path, "extractive" when no LLM was reachable.
    answer_source: str = "none"
    disclaimer: str = (
        "Research prototype. Not medical advice and not a diagnostic device. "
        "Consult a qualified healthcare professional for individual care."
    )
