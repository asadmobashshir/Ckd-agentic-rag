"""Tests for the real-corpus fetcher.

The network calls are not exercised here. What is pinned is the part that can
silently corrupt a corpus: HTML extraction, the front matter it emits, and the
fact that the result ingests with its declared provenance intact.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from agent.state import is_synthetic
from ingestion.collect_sources import load_document

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "fetch_real_corpus", SCRIPTS / "fetch_real_corpus.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_real_corpus"] = mod
    spec.loader.exec_module(mod)
    return mod


frc = _load_module()


PAGE = """
<html><head><style>.a{color:red}</style><script>var tracker=1;</script></head>
<body>
<nav><ul><li>Home</li><li>Kidney Disease</li></ul></nav>
<header><p>Skip to main content</p></header>
<main>
  <h1>Chronic Kidney Disease (CKD)</h1>
  <p>Chronic kidney disease means the kidneys are damaged &amp; cannot filter blood well.</p>
  <h2>What are the stages of CKD?</h2>
  <p>Professionals <strong>use</strong> the eGFR to determine the stage.</p>
  <ul><li>Stage 1: eGFR 90 or higher</li><li>Stage 2: eGFR 60 to 89</li></ul>
</main>
<footer><p>NIDDK footer links</p></footer>
</body></html>
"""


# ------------------------------------------------------------- extraction
def test_navigation_and_scripts_are_excluded():
    md = frc.html_to_markdown(PAGE)
    for noise in ("var tracker", "Skip to main", "footer links", "color:red"):
        assert noise not in md
    assert "- Home" not in md, "navigation list leaked into the corpus"


def test_headings_and_lists_survive_as_markdown():
    md = frc.html_to_markdown(PAGE)
    assert "# Chronic Kidney Disease (CKD)" in md
    assert "## What are the stages of CKD?" in md
    assert md.count("- Stage") == 2


def test_inline_markup_does_not_break_a_sentence():
    assert "use the eGFR" in frc.html_to_markdown(PAGE)


def test_entities_are_decoded():
    md = frc.html_to_markdown(PAGE)
    assert "&amp;" not in md and "damaged & cannot" in md


def test_empty_page_yields_no_blocks():
    assert frc.html_to_markdown("<html><body><nav>x</nav></body></html>").strip() == ""


# ------------------------------------------------------------ front matter
def test_front_matter_declares_real_provenance():
    fm = frc.front_matter(
        "CKD Overview", "NIDDK, National Institutes of Health",
        "https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd",
        "public_health_reference", "authoritative_guidance", "reference",
    )
    assert 'data_status: "reference"' in fm
    assert 'source: "NIDDK, National Institutes of Health"' in fm
    assert "niddk.nih.gov" in fm


def test_fetched_document_ingests_with_provenance_preserved(tmp_path: Path):
    """The emitted file must load as a real source, not as demo content."""
    url = "https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd"
    doc = tmp_path / "niddk_ckd_overview.md"
    doc.write_text(
        frc.front_matter("Chronic Kidney Disease (CKD) - Overview",
                         "NIDDK, National Institutes of Health", url,
                         "public_health_reference", "authoritative_guidance", "reference")
        + frc.html_to_markdown(PAGE),
        encoding="utf-8",
    )

    loaded = load_document(doc)

    assert loaded is not None
    assert loaded.data_status == "reference"
    assert not is_synthetic(loaded.data_status)
    assert loaded.url == url
    assert "eGFR" in loaded.text


# ------------------------------------------------------------------ config
def test_every_configured_page_is_a_niddk_https_url():
    assert frc.NIDDK_PAGES, "no pages configured"
    slugs = [s for s, _, _ in frc.NIDDK_PAGES]
    assert len(slugs) == len(set(slugs)), "duplicate output filenames"
    for slug, title, url in frc.NIDDK_PAGES:
        assert url.startswith("https://www.niddk.nih.gov/"), url
        assert title and slug.startswith("niddk")


def test_openfda_fields_exclude_dosage_sections():
    """Dosage text must not be imported - the project never emits a dose."""
    joined = " ".join(frc.OPENFDA_FIELDS)
    assert "dosage" not in joined.lower()
    assert "administration" not in joined.lower()


@pytest.mark.parametrize("name,klass", frc.OPENFDA_DRUGS)
def test_openfda_drug_list_matches_the_modelled_medicines(name, klass):
    assert name.islower() and klass
