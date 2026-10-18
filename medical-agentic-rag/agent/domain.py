"""CKD domain-scope classification.

This prototype is specialised for **Chronic Kidney Disease**. Its curated corpus
and structured store cover CKD only, so answering a question about an unrelated
condition would mean answering from model memory - exactly what the rest of the
architecture exists to prevent.

Scope is enforced *deterministically* here rather than relying on the planner
prompt, because the system is expected to run without a Gemini key. The prompt
carries the same instruction, but this module is what guarantees the behaviour.

The classifier is intentionally crude and three-valued:

``IN_SCOPE``
    A CKD concept, a kidney term, or a drug this system actually holds data for.
``OUT_OF_SCOPE``
    A clearly unrelated condition or body system, with no CKD signal present.
``AMBIGUOUS``
    Everything else. Existing conservative behaviour is preserved - retrieval is
    allowed, and the usual no-evidence abstention handles it if nothing matches.

Only a confident ``OUT_OF_SCOPE`` verdict suppresses retrieval. A term appearing
in *both* lists (diabetes and hypertension are CKD causes, for instance) resolves
to in-scope, since the CKD signal is checked first.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Iterable, Sequence

DOMAIN_NAME = "Chronic Kidney Disease (CKD)"


class DomainVerdict(str, Enum):
    """Three-valued scope decision."""

    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"


#: Core CKD / nephrology vocabulary.
CKD_TERMS: tuple[str, ...] = (
    r"ckd", r"chronic kidney", r"kidney disease", r"kidney failure", r"kidney function",
    r"kidneys?", r"renal", r"nephro\w*", r"nephritis", r"nephropathy", r"nephrotic",
    r"glomerul\w*", r"egfr", r"gfr", r"creatinine", r"albuminuria", r"proteinuria",
    r"albumin[- ]to[- ]creatinine", r"acr\b", r"uacr", r"dialysis", r"haemodialysis",
    r"hemodialysis", r"peritoneal dialysis", r"kidney transplant", r"renal replacement",
    r"esrd", r"esrf", r"end[- ]stage (kidney|renal)", r"uraemi\w*", r"uremi\w*",
    r"hyperkal(a)?emi\w*", r"hypokal(a)?emi\w*", r"potassium", r"acute kidney injury",
    r"\baki\b", r"nephrotoxic\w*", r"triple whammy", r"ras blockade",
    r"renin[- ]angiotensin", r"fluid overload", r"oedema", r"edema",
    r"urine albumin", r"kidney health", r"kidney care", r"kidney damage",
)

#: Drug and drug-class vocabulary this system holds CKD-relevant data for.
CKD_DRUG_TERMS: tuple[str, ...] = (
    r"ace inhibitor", r"ace-?i\b", r"\barbs?\b", r"angiotensin",
    r"sglt2", r"sglt-2", r"gliflozin", r"loop diuretic", r"diuretics?",
    r"nsaids?", r"non[- ]steroidal", r"anti[- ]inflammator\w*",
    r"lisinopril", r"ramipril", r"enalapril", r"perindopril",
    r"losartan", r"valsartan", r"candesartan", r"irbesartan",
    r"dapagliflozin", r"empagliflozin", r"canagliflozin",
    r"furosemide", r"frusemide", r"bumetanide", r"torasemide",
    r"ibuprofen", r"naproxen", r"diclofenac", r"celecoxib",
    r"metformin", r"warfarin",
)

#: Clearly unrelated conditions, specialties and drug classes.
#: Kept narrow on purpose - a false OUT_OF_SCOPE silently refuses a real question.
OUT_OF_SCOPE_TERMS: tuple[str, ...] = (
    # neurology / psychiatry
    r"migraine", r"headaches?", r"epilep\w*", r"seizures?", r"parkinson\w*",
    r"alzheimer\w*", r"dementia", r"multiple sclerosis", r"depression",
    r"antidepressant\w*", r"ssri\b", r"bipolar", r"schizophreni\w*",
    r"antipsychotic\w*", r"adhd", r"autism", r"anxiety disorder",
    # respiratory
    r"asthma", r"copd", r"bronchit\w*", r"pneumoni\w*", r"cystic fibrosis",
    r"inhaler\w*", r"tuberculosis",
    # dermatology
    r"skin cancer", r"melanoma", r"eczema", r"psoriasis", r"acne", r"dermatit\w*",
    r"rosacea",
    # oncology (non-renal)
    r"breast cancer", r"lung cancer", r"prostate cancer", r"colon cancer",
    r"colorectal cancer", r"leukaemi\w*", r"leukemi\w*", r"lymphoma",
    r"chemotherap\w*",
    # gastro / hepatic / musculoskeletal / other
    r"irritable bowel", r"\bibs\b", r"crohn\w*", r"ulcerative colitis",
    r"hepatitis", r"cirrhosis", r"fatty liver",
    r"osteoporos\w*", r"rheumatoid arthritis", r"gout", r"fibromyalgia",
    # ID / ophtho / ENT / obstetrics / derm-adjacent
    r"malaria", r"\bhiv\b", r"covid", r"influenza", r"measles", r"dengue",
    r"glaucoma", r"cataract", r"conjunctivit\w*", r"tinnitus",
    r"pregnan\w*", r"contracept\w*", r"menopaus\w*", r"fertilit\w*",
    # cardiology topics not framed through the kidney
    r"myocardial infarction", r"heart attack", r"atrial fibrillation",
    r"heart failure", r"cholesterol", r"statins?",
)


def _compile(patterns: Iterable[str]) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z])(?:" + "|".join(patterns) + r")(?![a-z])", re.IGNORECASE)


_CKD_RE = _compile(CKD_TERMS)
_CKD_DRUG_RE = _compile(CKD_DRUG_TERMS)
_OUT_RE = _compile(OUT_OF_SCOPE_TERMS)


def classify_domain(query: str, extra_in_scope_terms: Sequence[str] | None = None) -> DomainVerdict:
    """Classify a query against the CKD domain.

    ``extra_in_scope_terms`` lets the caller widen the in-scope vocabulary with
    the drug names actually present in the structured store, so the scope check
    tracks the data rather than a hard-coded list.
    """
    text = (query or "").strip()
    if not text:
        return DomainVerdict.AMBIGUOUS

    lowered = text.lower()
    for term in extra_in_scope_terms or ():
        if term and term.lower() in lowered:
            return DomainVerdict.IN_SCOPE

    # CKD signal wins over an out-of-scope hit: "diabetes and kidney disease" and
    # "does my heart failure medication affect my kidneys" are both in scope.
    if _CKD_RE.search(text) or _CKD_DRUG_RE.search(text):
        return DomainVerdict.IN_SCOPE

    if _OUT_RE.search(text):
        return DomainVerdict.OUT_OF_SCOPE

    return DomainVerdict.AMBIGUOUS


def matched_out_of_scope_terms(query: str) -> list[str]:
    """Terms that triggered an out-of-scope verdict (for audit and debugging)."""
    return sorted({m.group(0).lower() for m in _OUT_RE.finditer(query or "")})


#: Intent recorded on the plan when a query is refused as out of domain.
OUT_OF_SCOPE_INTENT = "out_of_scope"

OUT_OF_SCOPE_ANSWER = (
    f"That question is outside this assistant's supported scope.\n\n"
    f"This is a demonstration assistant specialised for {DOMAIN_NAME}. Its curated "
    "corpus and structured drug data cover CKD concepts, CKD staging, and the "
    "medicines commonly discussed in kidney care. It has no evidence to retrieve for "
    "other conditions, and it does not answer from model memory - so rather than "
    "produce an ungrounded answer, it declines.\n\n"
    "You can ask about CKD staging and eGFR, CKD medicines such as ACE inhibitors, "
    "ARBs, SGLT2 inhibitors, loop diuretics and NSAIDs, or interactions between those "
    "medicines. For anything else, please consult a clinician or an authoritative "
    "source."
)
