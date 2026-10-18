"""Provenance handling for REAL (non-demo) source documents.

The bundled corpus is synthetic, but the whole point of the pipeline is that it
can be pointed at real guidance (KDIGO, NICE, NIDDK, FDA labelling, PubMed).
These tests pin the behaviour that makes that swap safe:

* every real-source provenance value is accepted by the ``Evidence`` model
* provenance set in a document's front matter survives ingestion unchanged
* the synthetic-data penalty and the demo footer apply to demo content *only*,
  and lift when the evidence genuinely stops being synthetic

The last point matters for honesty in both directions: a real corpus should not
be labelled "demo", and a demo corpus must never be able to shed the label.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.state import (
    SYNTHETIC_DATA_STATUSES,
    Citation,
    Evidence,
    TrustTier,
    is_synthetic,
)
from ingestion.collect_sources import load_document
from safety.citation_formatter import assign_evidence_ids, format_answer
from safety.confidence import compute_confidence

REAL_STATUSES = ["guideline", "regulatory", "reference", "literature", "licensed", "live"]


def _evidence(status: str, **kw) -> Evidence:
    return Evidence(
        text=kw.pop("text", "Blood pressure control slows CKD progression."),
        source=kw.pop("source", "KDIGO 2024 CKD Guideline"),
        data_status=status,
        **kw,
    )


# --------------------------------------------------------------- model accepts
@pytest.mark.parametrize("status", REAL_STATUSES + ["demo", "unknown"])
def test_every_documented_provenance_is_accepted(status):
    """A real-source status must not be rejected by the Evidence model.

    Regression: ``data_status`` was ``Literal["demo","live","unknown"]`` while
    ``stores/structured_store.py`` instructed operators to set ``'licensed'``
    for real data. Following that instruction raised ValidationError at
    Evidence construction, so real data could not reach the UI at all.
    """
    assert _evidence(status).data_status == status


def test_unrecognised_provenance_is_still_rejected():
    """The field stays a closed vocabulary - typos must not pass silently."""
    with pytest.raises(Exception):
        _evidence("probably_fine")


# ------------------------------------------------------- ingestion round-trip
def test_front_matter_provenance_survives_ingestion(tmp_path: Path):
    """A real document declaring its provenance keeps it through loading."""
    doc = tmp_path / "kdigo_bp.md"
    doc.write_text(
        '---\n'
        'title: "CKD Blood Pressure Management"\n'
        'source: "KDIGO 2024 Clinical Practice Guideline"\n'
        'url: "https://kdigo.org/guidelines/ckd-evaluation-and-management/"\n'
        'publication_date: "2024-03-01"\n'
        'document_type: "clinical_guideline"\n'
        'evidence_level: "guideline"\n'
        'data_status: "guideline"\n'
        '---\n\n'
        '# Blood Pressure\n\nLowering systolic blood pressure slows progression.\n',
        encoding="utf-8",
    )

    loaded = load_document(doc)

    assert loaded is not None
    assert loaded.data_status == "guideline"
    assert loaded.source == "KDIGO 2024 Clinical Practice Guideline"
    assert loaded.url == "https://kdigo.org/guidelines/ckd-evaluation-and-management/"
    assert not is_synthetic(loaded.data_status)


def test_document_without_declared_provenance_defaults_to_demo(tmp_path: Path):
    """Unlabelled content is treated as demo, never silently promoted."""
    doc = tmp_path / "unlabelled.md"
    doc.write_text("# Notes\n\nSome text about kidneys.\n", encoding="utf-8")

    loaded = load_document(doc)

    assert loaded is not None
    assert loaded.data_status == "demo"
    assert is_synthetic(loaded.data_status)


# ------------------------------------------------------------ demo labelling
def _answer_with(statuses: list[str]) -> str:
    """Render a cited answer over evidence carrying the given provenance."""
    evidence = [_evidence(s, source=f"Source {i}") for i, s in enumerate(statuses)]
    assign_evidence_ids(evidence)
    markers = " ".join(f"[{e.evidence_id}]" for e in evidence)
    formatted, _, _ = format_answer(f"Control blood pressure. {markers}", evidence)
    return formatted


def test_demo_footer_shown_for_demo_evidence():
    assert "demonstration" in _answer_with(["demo"]).lower()


def test_demo_footer_absent_for_real_evidence():
    """Real sources must not carry a disclaimer saying they are demo data."""
    assert "demonstration" not in _answer_with(["guideline", "reference"]).lower()


def test_one_demo_citation_still_triggers_the_warning():
    """Mixing real and demo evidence must NOT hide the demo disclosure."""
    assert "demonstration" in _answer_with(["guideline", "demo"]).lower()


def test_only_demo_is_classified_synthetic():
    """Real provenance values must never be counted as synthetic."""
    assert SYNTHETIC_DATA_STATUSES == frozenset({"demo"})
    for status in REAL_STATUSES:
        assert not is_synthetic(status), status


# ----------------------------------------------------------------- confidence
def test_real_evidence_does_not_take_the_synthetic_penalty():
    """Swapping in real sources lifts the demo penalty on its own merits.

    This is a provenance change, not a threshold change: the same scorer and
    the same weights, applied to evidence that genuinely is no longer
    synthetic. Demo evidence must still be penalised.
    """
    demo = [_evidence("demo", source="DEMO corpus") for _ in range(3)]
    real = [_evidence("guideline", source="KDIGO 2024") for _ in range(3)]

    demo_report = compute_confidence(demo)
    real_report = compute_confidence(real)

    assert real_report.score >= demo_report.score
    assert any("demo" in note.lower() for note in demo_report.rationale)


def test_demo_penalty_cannot_be_bypassed_by_mislabelling_trust():
    """A high trust tier must not launder synthetic content into real data."""
    mislabelled = [
        _evidence("demo", source="DEMO corpus", trust_tier=TrustTier.A) for _ in range(3)
    ]
    report = compute_confidence(mislabelled)
    assert any("demo" in note.lower() for note in report.rationale)
