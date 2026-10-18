# Replacing the demo corpus with real CKD data

The bundled corpus in `data/raw/` is synthetic. It exists so the ingestion,
chunking, embedding, indexing and retrieval code has something to operate on.
It is **not** a medical knowledge base and the UI says so on every citation.

This document lists real sources that can replace it, what each one is good
for, and what the licence actually permits. Licensing is the part people skip;
it is the part that decides whether you may redistribute a corpus or only link
to it.

> Nothing here is legal advice. Check the current terms yourself before
> redistributing anything — terms change, and several of these changed recently.

---

## 1. Public domain — safest, redistributable

Works of the US federal government are not subject to domestic copyright. You
can ingest, store and redistribute these. They are the right backbone for this
project.

| Source | What it gives you | Use it for |
|---|---|---|
| **NIDDK** (`niddk.nih.gov`) | NIH's kidney institute. Patient and clinician CKD content: staging, causes, progression, diet, anaemia, mineral and bone disorder | The main narrative corpus — this is the closest real equivalent to the demo files |
| **MedlinePlus** (`medlineplus.gov`) | NLM plain-language health topics, with a web service API and bulk XML | Plain-language definitions and patient-facing explanations |
| **CDC CKD** (`cdc.gov/kidney-disease`) | Surveillance data, prevalence, risk factors | Epidemiology and risk-factor statements |
| **openFDA drug labelling** (`api.fda.gov/drug/label.json`) | Real FDA Structured Product Labelling as JSON, updated weekly, coverage from ~June 2009 | Replacing `demo_drug_data.json` — contraindications, warnings, renal-impairment sections |
| **DailyMed** (`dailymed.nlm.nih.gov`) | The SPL source behind openFDA, with bulk downloads | Same as above when you want the full label rather than API slices |
| **PMC Open Access Subset** (`pmc.ncbi.nlm.nih.gov`) | Full-text articles under CC licences | Literature grounding you are allowed to store |

Two cautions on the last two rows:

- **openFDA carries its own disclaimer.** FDA states it has "neither altered
  nor verified" the label content, that it may differ from the currently
  approved labelling, and that it should not be relied on for medical care
  decisions. Ingest it as *regulatory labelling*, not as clinical guidance.
- **Only the PMC Open Access Subset is redistributable.** The rest of PMC is
  free to *read*, not free to *republish*. The existing `literature_search`
  tool queries PubMed live and stores nothing, which stays on the right side of
  this. If you start caching full text, restrict it to the OA subset.

## 2. Free to read, redistribution restricted

Ingest these for a private prototype; do not republish the text.

| Source | Notes |
|---|---|
| **KDIGO 2024 CKD Guideline** (`kdigo.org`) | *The* global CKD guideline. Executive summary PDF is public; full guideline published in Kidney International (PubMed `38490803`). Free to download, but redistribution and derivative works are not automatically granted |
| **NICE NG203** (`nice.org.uk/guidance/ng203`) | UK CKD assessment and management. Free to access under the NICE UK Open Content Licence, which has conditions and is generally non-commercial |
| **NHS** (`nhs.uk`) | Usually Open Government Licence — among the more permissive of this group |
| **WHO** (`who.int`) | Typically CC BY-NC-SA |

## 3. Do not scrape

Proprietary clinical references prohibit automated extraction in their terms:
UpToDate, Medscape, DynaMed, Lexicomp, Micromedex, and the commercial DrugBank
dataset. Mayo Clinic, Cleveland Clinic and Drugs.com appear in
`config/source_trust.json` as **ranking** tiers for links the web-search tool
returns — that is not permission to copy their text into your corpus.

## 4. The RxNav caveat — verify before you build on it

NLM's RxNav family (`RxNorm`, `RxClass`, `RxTerms`) is the standard way to
normalise drug names and map ingredients, and is worth using for that.

**The Drug Interaction API is a different matter.** It has been widely reported
as discontinued by NIH, and NLM's own API-update page does not state its current
status either way. Confirm it responds before you design around it; if it does
not, drug–drug interaction data is the one piece of this project with no good
free replacement, and a licensed source is the honest answer.

Interaction data is also the highest-risk content in the system. Leaving
`drug_interaction` backed by a clearly-labelled demo fixture is more defensible
than silently wiring it to an unvalidated scrape.

---

## 5. How to actually load it

### The short version: run the fetcher

`scripts/fetch_real_corpus.py` downloads the public-domain sources and writes
them in this project's ingestion format. Run it on a machine with internet
access:

```bash
python scripts/fetch_real_corpus.py --niddk      # 11 NIDDK CKD pages -> data/raw/
python scripts/fetch_real_corpus.py --openfda    # FDA labelling -> data/structured/
python -m ingestion.run_ingestion --reset
```

It strips navigation, scripts and footers, keeps headings and lists as
Markdown, writes the correct front matter, and pauses between requests. A page
whose layout has changed enough to yield almost no text is skipped with a
warning rather than written as an empty document.

The sections below describe the format it produces, for adding sources by hand.

### Narrative documents → `data/raw/*.md`

Front matter drives provenance. Copy this template:

```markdown
---
title: "CKD Blood Pressure Management"
source: "KDIGO 2024 Clinical Practice Guideline"
url: "https://kdigo.org/guidelines/ckd-evaluation-and-management/"
publication_date: "2024-03-01"
document_type: "clinical_guideline"
evidence_level: "guideline"
data_status: "guideline"
---

# Blood Pressure Targets

Body text here. Keep the source's own wording where accuracy matters.
```

Then rebuild the index:

```bash
python -m ingestion.run_ingestion --reset
```

### `data_status` vocabulary

The field is a closed set, so typos fail loudly instead of passing silently:

| Value | Meaning |
|---|---|
| `demo` | Synthetic content written for this repository |
| `guideline` | Clinical practice guideline (KDIGO, NICE) |
| `regulatory` | Regulatory product labelling (FDA SPL / DailyMed) |
| `reference` | Public-health reference material (NIDDK, MedlinePlus, CDC) |
| `literature` | Peer-reviewed literature (PubMed / PMC) |
| `licensed` | Licensed third-party dataset |
| `live` | Fetched from the network at query time |
| `unknown` | Unclassified |

A document with no `data_status` defaults to `demo`. Unlabelled content is
never silently promoted to real.

### Structured drug data → `data/structured/*.json`

`stores/structured_store.py` loads drugs, interactions and dosage references.
Set `data_status` per row — `regulatory` for FDA labelling, `licensed` for a
commercial dataset.

---

## 6. Three things that will bite you

**1. The no-numeric-dose test is scoped to the demo fixture.**
`tests/test_tools.py::test_no_dosage_reference_contains_a_numeric_dose` asserts
no `dosage_reference` row contains a numeric dose and that every row says
"consult clinician". Real FDA labelling contains doses, so this test will fail
the moment you load it.

Do not delete it. Either keep real doses out of `dosage_reference` and let the
narrative corpus carry them (the output guardrail redacts numeric doses from
answers regardless of what the store holds), or re-scope the test to the demo
fixture explicitly so the guarantee still means something.

**2. Confidence will rise, and that is legitimate.**
`compute_confidence` penalises evidence that is entirely `demo`. Real
provenance lifts that penalty. This is a change in the evidence, not a change
in the thresholds — the scorer, weights and calibration are untouched. The
penalty cannot be bypassed by mislabelling trust tier; there is a test for that.

**3. Trust tiers decide retrieval ranking.**
`kdigo.org` was previously unlisted and classified Tier D — below a Mayo Clinic
page. It is now Tier A alongside NICE, and the nephrology journals are Tier B.
If you add sources on other domains, add them to `config/source_trust.json` or
they will be ranked as general web content.

---

## 7. A reasonable first real corpus

If you want the smallest change that makes the system defensible:

1. **NIDDK CKD pages** → `data/raw/`, `data_status: reference` — replaces the
   five synthetic narrative files
2. **KDIGO 2024 executive summary** → `data/raw/`, `data_status: guideline` —
   gives real staging and management statements
3. **Leave `drug_interaction` on the demo fixture**, clearly labelled, until
   you have a licensed interaction source
4. **Leave `literature_search` as-is** — it already queries PubMed live and
   stores nothing

That gets you real retrieval and real citations without a licence problem and
without pretending the riskiest component is validated.
