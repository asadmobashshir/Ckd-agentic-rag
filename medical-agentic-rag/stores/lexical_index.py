"""Lexical (BM25) index over the indexed chunks.

Dense retrieval alone was routinely returning the wrong document for short,
keyword-shaped questions: "What is CKD staging based on?" retrieved the
*medicines* document because a short query against long passages is dominated by
passage length rather than topic. BM25 handles exactly that case well - it scores
rare query terms ("staging", "symptoms", "eGFR") highly and normalises for
document length.

Neither retriever is reliable alone, so :mod:`agent.tools.vector_search_tool`
fuses the two with reciprocal rank fusion. This index is built in memory from the
payloads already stored in the vector store, so it needs no extra persistence and
cannot drift out of sync with what was indexed.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Query words that carry no retrieval signal. Deliberately small - dropping a
#: medically meaningful word is far more costly than keeping a common one.
STOPWORDS = frozenset(
    """
    a an the and or but if of for to in on at by with from as is are was were be
    been being do does did what which who whom whose when where why how this that
    these those it its i you we they them there here can could should would may
    might will shall about into than then so such not no nor only own same too
    very just also more most other some any all both each few own s t
    """.split()
)

# BM25 parameters. k1 controls term-frequency saturation, b the length
# normalisation strength; these are the conventional defaults.
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in STOPWORDS]


class BM25Index:
    """In-memory BM25 index over chunk payloads."""

    def __init__(self, payloads: Sequence[dict[str, Any]] | None = None) -> None:
        self._payloads: list[dict[str, Any]] = []
        self._tokens: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_len: float = 0.0
        if payloads:
            self.build(payloads)

    def build(self, payloads: Iterable[dict[str, Any]]) -> "BM25Index":
        """(Re)build the index from chunk payloads."""
        self._payloads, self._tokens, self._lengths = [], [], []
        self._doc_freq = Counter()

        for payload in payloads:
            # Index the contextual representation so the document title and
            # section heading are searchable alongside the chunk body.
            source_text = payload.get("embed_text") or payload.get("text") or ""
            tokens = tokenize(source_text)
            if not tokens:
                continue
            counts = Counter(tokens)
            self._payloads.append(payload)
            self._tokens.append(counts)
            self._lengths.append(len(tokens))
            self._doc_freq.update(counts.keys())

        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        logger.debug("BM25 index built over %d chunk(s)", len(self._payloads))
        return self

    def __len__(self) -> int:
        return len(self._payloads)

    def _idf(self, term: str) -> float:
        n = len(self._payloads)
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        # BM25+ style idf; the 0.5 offsets keep it positive for very common terms.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict[str, Any], float]]:
        """Return ``(payload, score)`` ranked by BM25, best first."""
        if not self._payloads:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[dict[str, Any], float]] = []
        for idx, counts in enumerate(self._tokens):
            length = self._lengths[idx]
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = tf + K1 * (1 - B + B * (length / self._avg_len if self._avg_len else 1.0))
                score += self._idf(term) * (tf * (K1 + 1)) / denom
            if score > 0:
                scored.append((self._payloads[idx], score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> dict[str, float]:
    """Fuse several ranked id lists into one score map.

    RRF is used rather than a weighted sum of raw scores because BM25 scores and
    cosine similarities live on incompatible scales; ranks are comparable, raw
    scores are not.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused
