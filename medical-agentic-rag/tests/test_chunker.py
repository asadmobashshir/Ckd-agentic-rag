"""Chunking and metadata-preservation tests."""

from __future__ import annotations

from pathlib import Path

from ingestion.chunker import chunk_text, split_blocks
from ingestion.collect_sources import RawDocument, normalize_text, parse_front_matter
from ingestion.metadata_tagger import tag_document, validate_payload

DOC = """# Metformin

Metformin is an oral agent used in type 2 diabetes.

## Dosing table

| Population | Reference |
|---|---|
| Adult | See product label |
| Paediatric | Specialist guidance |

## Cautions

- Renal impairment
- Metabolic acidosis
- Gastrointestinal effects

Dosing is individualised by a prescriber and titrated to response.
"""


def test_headings_are_tracked_into_sections():
    chunks = chunk_text(DOC, "doc1", chunk_size=400, chunk_overlap=50)
    sections = {c.section for c in chunks if c.section}
    assert any("Dosing table" in s for s in sections)
    assert any("Cautions" in s for s in sections)


def test_tables_are_not_split_across_chunks():
    chunks = chunk_text(DOC, "doc1", chunk_size=400, chunk_overlap=50)
    table_chunks = [c for c in chunks if "| Population |" in c.text]
    assert len(table_chunks) == 1, "the table must live in exactly one chunk"
    assert "| Paediatric |" in table_chunks[0].text, "table rows must stay together"


def test_lists_stay_intact():
    chunks = chunk_text(DOC, "doc1", chunk_size=400, chunk_overlap=50)
    list_chunks = [c for c in chunks if "- Renal impairment" in c.text]
    assert len(list_chunks) == 1
    assert "- Metabolic acidosis" in list_chunks[0].text


def test_dosage_blocks_are_classified():
    blocks = split_blocks(DOC)
    kinds = {b.kind for b in blocks}
    assert "table" in kinds
    assert "list" in kinds
    assert "dosage" in kinds, "dosage-like prose should be recognised"


def test_chunk_ids_are_stable_and_traceable():
    chunks = chunk_text(DOC, "doc1", chunk_size=300, chunk_overlap=40)
    assert all(c.chunk_id.startswith("doc1::c") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_oversized_block_is_hard_split_not_dropped():
    huge = "word " * 800
    chunks = chunk_text(huge, "big", chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert sum(len(c.text) for c in chunks) > 1000


def test_overlap_carries_context_forward():
    text = "\n\n".join(f"Paragraph {i} about anticoagulation therapy and monitoring." for i in range(12))
    chunks = chunk_text(text, "ov", chunk_size=200, chunk_overlap=80)
    assert len(chunks) > 1


def test_front_matter_missing_values_become_none():
    metadata, body = parse_front_matter("---\ntitle: A\nurl: null\n---\nBody text")
    assert metadata["title"] == "A"
    assert metadata["url"] is None
    assert body.strip() == "Body text"


def test_metadata_is_preserved_and_never_invented():
    document = RawDocument(
        document_id="d1",
        text=DOC,
        path=Path("demo.md"),
        title="DEMO: Metformin",
        source="DEMO corpus",
        url=None,
        publication_date=None,
        document_type="demo_reference_note",
        evidence_level="demo_material",
    )
    payloads = tag_document(document, chunk_text(DOC, "d1", 400, 50))
    assert payloads
    for payload in payloads:
        assert validate_payload(payload)
        assert payload["document_id"] == "d1"
        assert payload["title"] == "DEMO: Metformin"
        # Absent metadata must stay absent, not be filled in with a guess.
        assert payload["url"] is None
        assert payload["publication_date"] is None
        assert payload["trust_tier"] == "UNKNOWN"
        assert "retrieved_at" in payload


def test_normalize_text_collapses_blank_runs():
    assert normalize_text("a\r\n\n\n\n\nb") == "a\n\nb"
