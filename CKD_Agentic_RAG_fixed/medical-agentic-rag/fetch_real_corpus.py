"""Download a real CKD corpus and write it in this project's ingestion format.

Run this on a machine with internet access:

    python scripts/fetch_real_corpus.py --niddk      # narrative corpus
    python scripts/fetch_real_corpus.py --openfda    # structured drug labelling
    python scripts/fetch_real_corpus.py --all

What it writes
--------------
``--niddk``    data/raw/niddk_*.md      front-matter Markdown, data_status: reference
``--openfda``  data/structured/openfda_drug_data.json   data_status: regulatory

Both sources are works of the United States federal government and are
therefore not subject to domestic copyright. openFDA additionally carries
FDA's own disclaimer: the agency has neither altered nor verified the label
content, it may differ from the currently approved labelling, and it must not
be relied on for medical care decisions. It is ingested here as *regulatory
labelling*, never as clinical guidance.

After running, rebuild the index:

    python -m ingestion.run_ingestion --reset

IMPORTANT - read before using --openfda
---------------------------------------
Real drug labels contain numeric doses. The test
``tests/test_tools.py::test_no_dosage_reference_contains_a_numeric_dose``
asserts that no dosage row does, and it will fail. That test is a guarantee
about the *demo fixture*; re-scope it rather than delete it. This script
deliberately does NOT write dosage_reference rows for that reason - it writes
drug records and renal-relevant label sections only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
STRUCTURED_DIR = ROOT / "data" / "structured"

USER_AGENT = "CKD-Agentic-RAG/1.0 (academic project; contact: see repository)"
TIMEOUT = 30

# --------------------------------------------------------------------------
# NIDDK - National Institute of Diabetes and Digestive and Kidney Diseases
# URLs confirmed against the NIDDK kidney-disease section index.
# --------------------------------------------------------------------------
NIDDK_PAGES: list[tuple[str, str, str]] = [
    ("niddk_ckd_overview",
     "Chronic Kidney Disease (CKD) - Overview",
     "https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd"),
    ("niddk_ckd_causes",
     "Causes of Chronic Kidney Disease",
     "https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/causes"),
    ("niddk_ckd_tests_diagnosis",
     "Chronic Kidney Disease Tests and Diagnosis",
     "https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/tests-diagnosis"),
    ("niddk_ckd_managing",
     "Managing Chronic Kidney Disease",
     "https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/managing"),
    ("niddk_ckd_prevention",
     "Preventing Chronic Kidney Disease",
     "https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/prevention"),
    ("niddk_ckd_healthy_eating",
     "Healthy Eating for Adults with Chronic Kidney Disease",
     "https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/healthy-eating-adults-chronic-kidney-disease"),
    ("niddk_ckd_anemia",
     "Anemia in Chronic Kidney Disease",
     "https://www.niddk.nih.gov/health-information/kidney-disease/anemia"),
    ("niddk_ckd_mineral_bone",
     "Mineral and Bone Disorder in Chronic Kidney Disease",
     "https://www.niddk.nih.gov/health-information/kidney-disease/mineral-bone-disorder"),
    ("niddk_ckd_blood_pressure",
     "High Blood Pressure and Kidney Disease",
     "https://www.niddk.nih.gov/health-information/kidney-disease/high-blood-pressure"),
    ("niddk_diabetic_kidney_disease",
     "Diabetic Kidney Disease",
     "https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/diabetic-kidney-disease"),
    ("niddk_uacr_gfr_quick_reference",
     "Quick Reference on UACR and GFR",
     "https://www.niddk.nih.gov/health-information/professionals/advanced-search/quick-reference-uacr-gfr"),
]

# --------------------------------------------------------------------------
# openFDA - drug labelling for the medicines this project already models
# --------------------------------------------------------------------------
OPENFDA_ENDPOINT = "https://api.fda.gov/drug/label.json"
OPENFDA_DRUGS = [
    ("lisinopril", "ACE inhibitor"),
    ("losartan", "Angiotensin receptor blocker"),
    ("dapagliflozin", "SGLT2 inhibitor"),
    ("furosemide", "Loop diuretic"),
    ("ibuprofen", "NSAID"),
    ("metformin", "Biguanide"),
    ("warfarin", "Anticoagulant"),
]
# Label sections that matter for kidney impairment. Dosage sections are
# deliberately excluded - see the module docstring.
OPENFDA_FIELDS = [
    "warnings_and_cautions", "warnings", "contraindications",
    "drug_interactions", "use_in_specific_populations", "precautions",
    "boxed_warning", "clinical_pharmacology",
]


# ======================================================================
# HTML -> Markdown
# ======================================================================
BLOCK_SKIP = {"script", "style", "nav", "header", "footer", "form", "button",
              "aside", "noscript", "svg", "iframe"}
HEADINGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####"}


class ContentExtractor(HTMLParser):
    """Pull headings, paragraphs and list items out of a page as Markdown.

    Deliberately conservative: anything it does not recognise is dropped,
    which is preferable to importing navigation chrome into the corpus.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._skip_depth = 0
        self._current: str | None = None       # tag name of the open block
        self._buf: list[str] = []

    # -- tag handling --------------------------------------------------
    def handle_starttag(self, tag, attrs):
        if tag in BLOCK_SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in HEADINGS or tag in ("p", "li"):
            self._flush()
            self._current = tag
        elif tag == "br" and self._current:
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag in BLOCK_SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in HEADINGS or tag in ("p", "li"):
            self._flush()

    def handle_data(self, data):
        if self._skip_depth or self._current is None:
            return
        self._buf.append(data)

    # -- emit ----------------------------------------------------------
    def _flush(self):
        if self._current is None:
            return
        text = " ".join("".join(self._buf).split())
        self._buf = []
        tag, self._current = self._current, None
        if not text:
            return
        if tag in HEADINGS:
            self.blocks.append(f"{HEADINGS[tag]} {text}")
        elif tag == "li":
            self.blocks.append(f"- {text}")
        else:
            self.blocks.append(text)

    def markdown(self) -> str:
        self._flush()
        out: list[str] = []
        for b in self.blocks:
            if out and b.startswith("-") and out[-1].startswith("-"):
                out.append(b)
            else:
                out.append("")
                out.append(b)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(out).strip()) + "\n"


def html_to_markdown(html: str) -> str:
    parser = ContentExtractor()
    parser.feed(html)
    return parser.markdown()


# ======================================================================
# fetching
# ======================================================================
def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def front_matter(title: str, source: str, url: str, doc_type: str,
                 evidence_level: str, data_status: str) -> str:
    return (
        "---\n"
        f'title: "{title}"\n'
        f'source: "{source}"\n'
        f'url: "{url}"\n'
        "publication_date: null\n"
        f'document_type: "{doc_type}"\n'
        f'evidence_level: "{evidence_level}"\n'
        f'data_status: "{data_status}"\n'
        "---\n\n"
    )


MIN_BODY_CHARS = 400


def run_niddk(limit: int | None = None) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    pages = NIDDK_PAGES[:limit] if limit else NIDDK_PAGES
    for slug, title, url in pages:
        try:
            html = fetch(url)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  FAILED  {slug}: {exc}")
            continue
        body = html_to_markdown(html)
        if len(body) < MIN_BODY_CHARS:
            print(f"  SKIPPED {slug}: extracted only {len(body)} chars "
                  f"- the page layout may have changed; inspect it manually")
            continue
        target = RAW_DIR / f"{slug}.md"
        target.write_text(
            front_matter(title, "NIDDK, National Institutes of Health", url,
                         "public_health_reference", "authoritative_guidance",
                         "reference") + body,
            encoding="utf-8",
        )
        print(f"  wrote   {target.relative_to(ROOT)}  ({len(body):,} chars)")
        written += 1
        time.sleep(1.0)          # be polite to a public service
    return written


def run_openfda() -> int:
    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    drugs, sections = [], []
    for name, drug_class in OPENFDA_DRUGS:
        query = urllib.parse.urlencode({
            "search": f'openfda.generic_name:"{name}"',
            "limit": "1",
        })
        try:
            raw = fetch(f"{OPENFDA_ENDPOINT}?{query}")
            payload = json.loads(raw)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  FAILED  {name}: {exc}")
            continue
        results = payload.get("results") or []
        if not results:
            print(f"  no label found for {name}")
            continue
        label = results[0]
        openfda = label.get("openfda", {})
        spl_id = (openfda.get("spl_set_id") or [None])[0]
        label_url = (f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={spl_id}"
                     if spl_id else "https://open.fda.gov/apis/drug/label/")
        drugs.append({
            "name": name,
            "drug_class": drug_class,
            "brand_names": openfda.get("brand_name", [])[:5],
            "source": "FDA Structured Product Labeling via openFDA",
            "source_url": label_url,
            "data_status": "regulatory",
            "notes": "Regulatory labelling. FDA has neither altered nor verified this "
                     "content and it may differ from current approved labelling.",
        })
        for field_name in OPENFDA_FIELDS:
            for text in label.get(field_name, []) or []:
                cleaned = " ".join(text.split())
                if len(cleaned) < 40:
                    continue
                sections.append({
                    "drug": name,
                    "section": field_name,
                    "text": cleaned[:8000],
                    "source": "FDA SPL via openFDA",
                    "source_url": label_url,
                    "data_status": "regulatory",
                })
        print(f"  {name}: {len([s for s in sections if s['drug'] == name])} label section(s)")
        time.sleep(0.5)

    out = STRUCTURED_DIR / "openfda_drug_data.json"
    out.write_text(json.dumps({
        "_DATA_STATUS": "REGULATORY LABELLING - NOT CLINICAL GUIDANCE",
        "_SOURCE": "FDA Structured Product Labeling, retrieved via the openFDA API",
        "_DISCLAIMER": (
            "FDA states it has neither altered nor verified this label content, that it "
            "may differ from the currently approved labelling, and that it must not be "
            "relied upon for medical care decisions."
        ),
        "_NO_DOSAGE_ROWS": (
            "Dosage sections are deliberately not imported. Real labels contain numeric "
            "doses, and this project does not emit individualised dosing."
        ),
        "drugs": drugs,
        "label_sections": sections,
    }, indent=2), encoding="utf-8")
    print(f"  wrote   {out.relative_to(ROOT)}  "
          f"({len(drugs)} drugs, {len(sections)} sections)")
    return len(drugs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--niddk", action="store_true", help="fetch the NIDDK CKD corpus")
    ap.add_argument("--openfda", action="store_true", help="fetch openFDA drug labelling")
    ap.add_argument("--all", action="store_true", help="fetch everything")
    ap.add_argument("--limit", type=int, default=None, help="only the first N NIDDK pages")
    args = ap.parse_args()

    if not (args.niddk or args.openfda or args.all):
        ap.print_help()
        return 1

    if args.niddk or args.all:
        print("NIDDK (public domain, US federal government work)")
        n = run_niddk(args.limit)
        print(f"  -> {n} document(s)\n")

    if args.openfda or args.all:
        print("openFDA drug labelling (public domain, US federal government work)")
        run_openfda()
        print()

    print("Next:  python -m ingestion.run_ingestion --reset")
    print("Then:  pytest -q      # expect the demo-fixture dose test to need re-scoping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
