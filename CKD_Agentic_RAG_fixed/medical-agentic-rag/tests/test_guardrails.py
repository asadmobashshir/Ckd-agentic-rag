"""Guardrail classification, post-check redaction, and confidence scoring."""

from __future__ import annotations

import pytest

from agent.state import ClaimStatus, ConfidenceLevel, RiskCategory, VerificationResult
from safety.citation_formatter import assign_evidence_ids, format_answer, strip_invalid_markers
from safety.confidence import compute_confidence
from safety.guardrails import GuardrailEngine
from safety.source_trust import classify_url, tier_weight
from safety.verifier import ClaimVerifier


@pytest.fixture
def engine(settings) -> GuardrailEngine:
    return GuardrailEngine(settings)


# ---------------------------------------------------------------- pre-check
@pytest.mark.parametrize(
    "query",
    [
        "What are the general clinical uses of metformin?",
        "How does warfarin work?",
        "What is the difference between type 1 and type 2 diabetes?",
    ],
)
def test_general_medical_questions_are_not_blocked(engine, query):
    assessment = engine.pre_check(query)
    assert assessment.risk_category is RiskCategory.GENERAL_INFORMATION
    assert assessment.block_pipeline is False


@pytest.mark.parametrize(
    "query",
    [
        "I have crushing chest pain radiating down my left arm",
        "My father is unresponsive and won't wake up",
        "I can't breathe and my throat is closing",
        "She is bleeding heavily and it won't stop",
    ],
)
def test_emergencies_short_circuit_the_pipeline(engine, query):
    assessment = engine.pre_check(query)
    assert assessment.risk_category is RiskCategory.EMERGENCY
    assert assessment.block_pipeline is True
    assert "emergency" in (assessment.direct_response or "").lower()


@pytest.mark.parametrize(
    "query",
    [
        "How much ibuprofen should I take for my back pain?",
        "Should I stop taking my metformin?",
        "Can I double my dose tonight?",
        "What's wrong with me, do I have diabetes?",
    ],
)
def test_personalised_requests_are_flagged_but_still_answered(engine, query):
    assessment = engine.pre_check(query)
    assert assessment.risk_category is RiskCategory.PERSONALIZED_CLINICAL_DECISION
    assert assessment.block_pipeline is False, "must still retrieve general information"
    assert assessment.safety_note


def test_self_harm_routes_to_support_without_retrieval(engine):
    assessment = engine.pre_check("I want to kill myself, what is a lethal dose")
    assert assessment.risk_category is RiskCategory.SELF_HARM
    assert assessment.block_pipeline is True
    assert "116 123" in assessment.direct_response or "988" in assessment.direct_response


def test_input_limits(engine, settings):
    assert engine.pre_check("").block_pipeline is True
    assert engine.pre_check("x" * (settings.max_query_chars + 1)).block_pipeline is True


# --------------------------------------------------------------- post-check
def test_post_check_removes_individualised_dosing(engine):
    answer = "You should take 400 mg of ibuprofen every six hours for your pain."
    cleaned, warnings = engine.post_check(answer)
    assert "400 mg" not in cleaned
    assert "Removed" in cleaned
    assert warnings


def test_post_check_removes_diagnostic_assertions(engine):
    cleaned, warnings = engine.post_check("Based on this, you have type 2 diabetes.")
    assert "you have type 2 diabetes" not in cleaned.lower()
    assert warnings


def test_post_check_leaves_general_information_alone(engine):
    answer = "Metformin is commonly used in the management of type 2 diabetes [S1]."
    cleaned, warnings = engine.post_check(answer)
    assert cleaned == answer
    assert warnings == []


# -------------------------------------------------------------- source trust
def test_trust_tiers():
    assert classify_url("https://www.cdc.gov/page").value == "A"
    assert classify_url("https://pubmed.ncbi.nlm.nih.gov/123/").value == "B"
    assert classify_url("https://www.mayoclinic.org/x").value == "C"
    assert classify_url("https://someblog.example/post").value == "D"
    assert classify_url(None).value == "UNKNOWN"
    assert tier_weight(classify_url("https://www.cdc.gov")) > tier_weight(
        classify_url("https://someblog.example")
    )


# --------------------------------------------------------------- confidence
def test_confidence_is_insufficient_without_evidence():
    report = compute_confidence([], None)
    assert report.level is ConfidenceLevel.INSUFFICIENT
    assert report.score == 0.0
    assert "not a probability" in report.disclaimer


def test_confidence_rises_with_better_evidence(sample_evidence):
    verification = VerificationResult(grounded_claim_ratio=1.0)
    verification.claims = []
    weak = compute_confidence(sample_evidence[:1], None)
    strong = compute_confidence(sample_evidence, verification)
    assert strong.score >= weak.score
    assert set(strong.factors) == {
        "retrieval_strength", "source_count", "source_agreement",
        "source_trust", "recency", "verification",
    }


def test_demo_data_reduces_confidence(sample_evidence):
    demo = compute_confidence(sample_evidence, None)
    for item in sample_evidence:
        item.data_status = "live"
    live = compute_confidence(sample_evidence, None)
    assert demo.score < live.score


def test_confidence_never_comes_from_the_model(sample_evidence):
    """The score must be reproducible from inputs alone."""
    a = compute_confidence(sample_evidence, None)
    b = compute_confidence(sample_evidence, None)
    assert a.score == b.score


# --------------------------------------------------------------- verification
def test_supported_claim_is_recognised(sample_evidence):
    verifier = ClaimVerifier(llm=None)
    draft = "Metformin is widely used in the management of type 2 diabetes mellitus [S1]."
    result = verifier.verify(draft, sample_evidence)
    assert result.claims
    assert result.claims[0].status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED)
    assert result.grounded_claim_ratio > 0


def test_unsupported_high_risk_claim_is_removed(sample_evidence):
    verifier = ClaimVerifier(llm=None)
    draft = "Chewing raw ginger root permanently cures pancreatic cancer and prevents all relapse."
    result = verifier.verify(draft, sample_evidence)
    assert result.claims[0].status is ClaimStatus.UNSUPPORTED
    assert result.claims[0].high_risk is True
    assert result.removed_claims
    assert "Removed" in result.revised_answer or "disregarded" in result.revised_answer


def test_claim_citing_unknown_evidence_id_is_unsupported(sample_evidence):
    verifier = ClaimVerifier(llm=None)
    result = verifier.verify("Metformin is used in type 2 diabetes mellitus [S99].", sample_evidence)
    assert result.claims[0].status is ClaimStatus.UNSUPPORTED
    assert "unknown evidence id" in (result.claims[0].note or "").lower()


def test_verification_with_no_evidence_grounds_nothing():
    verifier = ClaimVerifier(llm=None)
    result = verifier.verify("Some drug treats some condition effectively.", [])
    assert result.grounded_claim_ratio == 0.0
    assert all(c.status is ClaimStatus.UNSUPPORTED for c in result.claims)


def test_extractive_mode_does_not_revise(sample_evidence):
    verifier = ClaimVerifier(llm=None)
    draft = sample_evidence[0].text
    result = verifier.verify(draft, sample_evidence, extractive=True)
    assert result.revised_answer == draft
    assert result.removed_claims == []


# ---------------------------------------------------------------- citations
def test_evidence_ids_are_assigned_in_order(sample_evidence):
    for item in sample_evidence:
        item.evidence_id = ""
    assigned = assign_evidence_ids(sample_evidence)
    assert [e.evidence_id for e in assigned] == ["S1", "S2"]


def test_fabricated_citation_markers_are_stripped(sample_evidence):
    answer = "Metformin treats diabetes [S1]. Aspirin cures everything [S7]."
    cleaned, invalid = strip_invalid_markers(answer, {"S1", "S2"})
    assert "[S7]" not in cleaned
    assert invalid == ["S7"]
    assert "[S1]" in cleaned


def test_format_answer_lists_only_cited_sources(sample_evidence):
    answer = "Metformin is used in type 2 diabetes [S1]."
    formatted, citations, warnings = format_answer(answer, sample_evidence)
    assert len(citations) == 1
    assert citations[0].evidence_id == "S1"
    assert "Sources:" in formatted
    assert "[S2]" not in formatted.split("Sources:")[1]
    assert warnings == []


def test_format_answer_warns_when_nothing_is_cited(sample_evidence):
    formatted, citations, warnings = format_answer("Metformin treats diabetes.", sample_evidence)
    assert citations == []
    assert any("no citations" in w for w in warnings)


def test_citation_for_fabricated_id_never_appears(sample_evidence):
    formatted, citations, warnings = format_answer("Claim [S42].", sample_evidence)
    assert citations == []
    assert "[S42]" not in formatted
    assert any("did not correspond" in w for w in warnings)


def test_demo_sources_are_labelled(sample_evidence):
    formatted, _, _ = format_answer("Metformin is used in type 2 diabetes [S1].", sample_evidence)
    assert "DEMO" in formatted
