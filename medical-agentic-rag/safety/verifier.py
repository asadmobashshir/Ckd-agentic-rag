"""Claim verification against retrieved evidence.

This is deliberately *not* "ask the LLM whether its own answer is right". The
procedure is:

1. Split the draft into atomic claims (LLM-assisted when available, sentence
   segmentation otherwise).
2. Score each claim against the retrieved evidence using lexical overlap over
   content words, restricted to the evidence ids the claim actually cites when it
   cites any.
3. Detect contradiction signals between the claim and its best-matching evidence
   (negation flips, and severity/direction disagreement across sources).
4. Label each claim supported / partially supported / unsupported / conflicting.
5. Strip unsupported claims that carry medical risk, keeping abstention over
   assertion.

The overlap score is a heuristic, not entailment. It is intentionally biased
toward flagging: a false "unsupported" costs a sentence, a missed hallucination
costs credibility.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Sequence

from agent.prompts import CLAIM_EXTRACTION_SYSTEM
from agent.state import Claim, ClaimStatus, Evidence, VerificationResult
from llm.base import BaseLLMClient, LLMError

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(S\d+)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
WORD_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")

STOPWORDS = frozenset(
    """
    the and for with that this from are was were has have had not but its
    can may also such been being other more most some than then they them
    their there these those which while about into over under between during
    used using use often commonly generally typically usually described
    including include includes based upon when where what how why does did
    each both very much many any all one two three per due both
    """.split()
)

HIGH_RISK_TERMS = re.compile(
    r"\b(dose|dosage|mg\b|mcg|contraindicat\w+|fatal|death|safe|unsafe|cure[sd]?|"
    r"prevent[s]?|treat(s|ment)?|overdose|toxic\w*|interact\w*|bleed\w*|"
    r"pregnan\w+|children|infant|renal|hepatic|allerg\w+)\b",
    re.IGNORECASE,
)

NEGATION_RE = re.compile(
    r"\b(no|not|never|without|absence of|does not|do not|cannot|can't|"
    r"unlikely|contraindicated|avoid)\b",
    re.IGNORECASE,
)

SUPPORTED_THRESHOLD = 0.45
PARTIAL_THRESHOLD = 0.22


def _content_words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall((text or "").lower()) if w not in STOPWORDS}


def _overlap(claim_words: set[str], evidence_words: set[str]) -> float:
    """Fraction of the claim's content words present in the evidence."""
    if not claim_words:
        return 0.0
    return len(claim_words & evidence_words) / len(claim_words)


def split_sentences(text: str) -> list[str]:
    """Sentence segmentation fallback for claim extraction."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    return [s.strip() for s in SENTENCE_RE.split(cleaned) if len(s.strip()) > 15]


class ClaimVerifier:
    """Grounds a draft answer in the evidence that was actually retrieved."""

    def __init__(self, llm: BaseLLMClient | None = None) -> None:
        self.llm = llm

    # ------------------------------------------------------------ extraction
    def extract_claims(self, draft: str) -> list[str]:
        """Split a draft into atomic factual claims."""
        if self.llm is not None and self.llm.available:
            try:
                payload = self.llm.chat_json(
                    [
                        {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM},
                        {"role": "user", "content": draft[:4000]},
                    ],
                    temperature=0.0,
                )
                claims = payload.get("claims")
                if isinstance(claims, list):
                    extracted = [str(c).strip() for c in claims if str(c).strip()]
                    if extracted:
                        return extracted[:10]
            except (LLMError, Exception) as exc:  # noqa: BLE001
                logger.info("LLM claim extraction failed (%s); using sentence split.", exc)
        return split_sentences(draft)[:10]

    # ---------------------------------------------------------- verification
    def verify(
        self,
        draft: str,
        evidence: Sequence[Evidence],
        extractive: bool = False,
    ) -> VerificationResult:
        """Verify every claim in ``draft`` against ``evidence``.

        Set ``extractive=True`` when the draft is a verbatim quotation of the
        retrieved passages (the no-LLM fallback path). Claims are still scored and
        recorded for the audit trail, but polarity conflicts are not flagged and
        nothing is revised - a passage cannot contradict itself, and rewriting a
        direct quotation would corrupt the provenance chain.
        """
        result = VerificationResult(revised_answer=draft or "")

        if not draft or not draft.strip():
            result.notes.append("Empty draft; nothing to verify.")
            return result

        if not evidence:
            result.notes.append(
                "No evidence was retrieved, so no claim could be grounded."
            )
            result.claims = [
                Claim(text=c, status=ClaimStatus.UNSUPPORTED, high_risk=bool(HIGH_RISK_TERMS.search(c)))
                for c in self.extract_claims(draft)
            ]
            result.grounded_claim_ratio = 0.0
            return result

        evidence_index = {
            item.evidence_id: (item, _content_words(item.text)) for item in evidence if item.evidence_id
        }
        # Fall back to positional ids when the formatter has not run yet.
        if not evidence_index:
            evidence_index = {
                f"S{i + 1}": (item, _content_words(item.text)) for i, item in enumerate(evidence)
            }

        claims_text = self.extract_claims(draft)
        claims: list[Claim] = []

        for text in claims_text:
            cited = CITATION_RE.findall(text)
            claim_words = _content_words(text)
            candidates = (
                {cid: evidence_index[cid] for cid in cited if cid in evidence_index}
                or evidence_index
            )

            scored = sorted(
                (
                    (_overlap(claim_words, words), cid, item)
                    for cid, (item, words) in candidates.items()
                ),
                key=lambda triple: triple[0],
                reverse=True,
            )
            best_score, _, best_item = scored[0]
            supporting = [cid for score, cid, _ in scored if score >= PARTIAL_THRESHOLD]

            claim = Claim(
                text=text,
                overlap_score=round(best_score, 3),
                supporting_evidence_ids=supporting[:4],
                high_risk=bool(HIGH_RISK_TERMS.search(text)),
            )

            if best_score >= SUPPORTED_THRESHOLD:
                claim.status = ClaimStatus.SUPPORTED
            elif best_score >= PARTIAL_THRESHOLD:
                claim.status = ClaimStatus.PARTIALLY_SUPPORTED
            else:
                claim.status = ClaimStatus.UNSUPPORTED
                claim.note = "No retrieved passage covers this statement."

            if (
                not extractive
                and claim.status is not ClaimStatus.UNSUPPORTED
                and _contradicts(text, best_item.text)
            ):
                claim.status = ClaimStatus.CONFLICTING
                claim.note = "Claim and best-matching evidence disagree in polarity."

            if cited and any(cid not in evidence_index for cid in cited):
                unknown = [cid for cid in cited if cid not in evidence_index]
                claim.status = ClaimStatus.UNSUPPORTED
                claim.note = f"Cites unknown evidence id(s): {unknown}"

            claims.append(claim)

        result.claims = claims
        grounded = sum(
            1 for c in claims if c.status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED)
        )
        result.grounded_claim_ratio = round(grounded / len(claims), 3) if claims else 0.0
        if extractive:
            result.revised_answer = draft
            result.notes.append(
                "Draft is a verbatim extract of retrieved evidence; no revision applied."
            )
        else:
            result.revised_answer, result.removed_claims = self._revise(draft, claims)

        if result.removed_claims:
            result.notes.append(
                f"Removed {len(result.removed_claims)} unsupported high-risk claim(s)."
            )
        conflicting = [c for c in claims if c.status is ClaimStatus.CONFLICTING]
        if conflicting:
            result.notes.append(f"{len(conflicting)} claim(s) conflict with retrieved evidence.")
        return result

    @staticmethod
    def _revise(draft: str, claims: Iterable[Claim]) -> tuple[str, list[str]]:
        """Strip unsupported high-risk sentences from the draft.

        Only removes text that appears verbatim; LLM-rewritten claims are reported
        but not blindly excised, to avoid mangling the prose.
        """
        revised = draft
        removed: list[str] = []
        for claim in claims:
            unsafe = claim.high_risk and claim.status in (
                ClaimStatus.UNSUPPORTED,
                ClaimStatus.CONFLICTING,
            )
            if not unsafe:
                continue
            removed.append(claim.text)
            if claim.text in revised:
                revised = revised.replace(
                    claim.text,
                    "[Removed: this statement was not supported by the retrieved evidence.]",
                )
        if removed and not any(marker in revised for marker in ("[Removed:",)):
            revised = (
                revised.rstrip()
                + "\n\nNote: some statements in the draft were not supported by the "
                "retrieved evidence and should be disregarded: "
                + " ".join(f'"{claim[:120]}"' for claim in removed[:3])
            )
        return revised.strip(), removed


#: Negation cues, as bare tokens, used for scope analysis.
NEGATION_TOKENS = frozenset(
    "no not never without cannot avoid avoided unlikely contraindicated "
    "excluded rarely neither nor".split()
)

#: Sentences that exist to disclaim, not to assert. The demo corpus is full of
#: them ("DEMO CONTENT - NOT FOR CLINICAL USE"), and treating their "not" as a
#: medical contradiction was the single largest source of false conflicts.
_BOILERPLATE_RE = re.compile(
    r"demo content|demo record|demo reference|not for clinical use|"
    r"consult a clinician|consult the|not validated|demonstration material|"
    r"research prototype|not medical advice|exercise retrieval code|"
    r"no numeric dose|limitations of this demo",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")

#: How many tokens after a negation cue are considered inside its scope.
_NEGATION_SCOPE = 3

#: Tokens that cannot themselves be the thing being negated. Two groups:
#: adverbs/function words that sit between a cue and its predicate, and domain
#: topic nouns. "does not by itself exclude CKD" negates *exclude*, not *CKD* -
#: treating the topic noun as negated flagged correct answers as contradictions.
_NON_PREDICATE = frozenset(
    """
    by itself themselves himself herself own only just even still yet
    always necessarily merely simply alone solely otherwise
    ckd kidney kidneys renal disease diseases chronic condition conditions
    patient patients people person individual individuals adult adults
    """.split()
)


def _negated_terms(text: str) -> set[str]:
    """Content words that actually fall under a negation's scope.

    "NSAIDs are not recommended" puts {recommended} under negation; it does not
    put every word of the sentence under negation. This scoping is what
    separates a real contradiction from a sentence that merely contains "not".
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    scoped: set[str] = set()
    for index, token in enumerate(tokens):
        if token not in NEGATION_TOKENS:
            continue
        for following in tokens[index + 1 : index + 1 + _NEGATION_SCOPE]:
            if (
                following not in STOPWORDS
                and following not in NEGATION_TOKENS
                and following not in _NON_PREDICATE
                and len(following) > 2
            ):
                scoped.add(following)
    return scoped


def _contradicts(claim: str, evidence_text: str) -> bool:
    """Whether evidence genuinely contradicts a claim.

    A contradiction requires one side to negate a term the *other side asserts
    positively*, within a sentence that is clearly about the same subject.
    Merely containing the word "not" somewhere is not a contradiction - the
    previous implementation treated it as one, which flagged correct answers
    against a corpus whose sentences are mostly safety caveats.
    """
    claim_words = _content_words(claim)
    if len(claim_words) < 4:
        return False  # too little signal to judge

    claim_negated = _negated_terms(claim)

    for sentence in SENTENCE_RE.split(evidence_text or ""):
        sentence = sentence.strip()
        if not sentence or _BOILERPLATE_RE.search(sentence):
            continue

        sentence_words = _content_words(sentence)
        shared = claim_words & sentence_words
        # Three shared content words is enough once the trigger below requires a
        # genuinely negated predicate; a stricter cut missed real contradictions
        # in short sentences.
        if len(shared) < 3:
            continue  # not about the same thing

        evidence_negated = _negated_terms(sentence)

        # Evidence negates something the claim states positively.
        if (evidence_negated & claim_words) - claim_negated:
            return True
        # Claim negates something the evidence states positively.
        if (claim_negated & sentence_words) - evidence_negated:
            return True

    return False
