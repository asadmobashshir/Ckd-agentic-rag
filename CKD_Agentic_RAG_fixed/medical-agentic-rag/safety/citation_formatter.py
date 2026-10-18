"""Citation assignment, validation and rendering.

Invariants enforced here:

* An evidence id (``[S1]``, ``[S2]``, ...) is assigned only to an evidence object
  that a tool actually returned.
* Any citation marker in the answer that does not resolve to an assigned id is
  stripped - a dangling marker is a fabricated citation.
* The rendered source list contains only ids that survive in the final text.

No component downstream may add a citation; this module is the only place ids
are minted.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from agent.state import Citation, Evidence

logger = logging.getLogger(__name__)

CITATION_MARKER_RE = re.compile(r"\[(S\d+)\]")


def assign_evidence_ids(evidence: Sequence[Evidence], start: int = 1) -> list[Evidence]:
    """Assign stable ``S<n>`` ids in retrieval order, de-duplicating identical text."""
    assigned: list[Evidence] = []
    seen: dict[str, str] = {}
    counter = start
    for item in evidence:
        key = (item.url or "") + "|" + item.text.strip()[:200]
        if key in seen:
            item.evidence_id = seen[key]
            continue
        item.evidence_id = f"S{counter}"
        seen[key] = item.evidence_id
        counter += 1
        assigned.append(item)
    return assigned


def build_evidence_block(evidence: Sequence[Evidence], max_chars: int = 1400) -> str:
    """Render evidence for the synthesizer prompt, provenance included."""
    if not evidence:
        return "(no evidence retrieved)"
    blocks: list[str] = []
    for item in evidence:
        header_parts = [f"[{item.evidence_id}]"]
        if item.title:
            header_parts.append(item.title)
        header_parts.append(f"source={item.source}")
        if item.publication_date:
            header_parts.append(f"date={item.publication_date}")
        header_parts.append(f"trust_tier={item.trust_tier.value}")
        header_parts.append(f"data_status={item.data_status}")
        if item.score is not None:
            header_parts.append(f"similarity={item.score:.3f}")
        if item.url:
            header_parts.append(f"url={item.url}")
        body = item.text.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + " ..."
        blocks.append(" | ".join(header_parts) + "\n" + body)
    return "\n\n---\n\n".join(blocks)


def to_citation(item: Evidence) -> Citation:
    return Citation(
        evidence_id=item.evidence_id,
        title=item.title,
        source=item.source,
        url=item.url,
        publication_date=item.publication_date,
        trust_tier=item.trust_tier,
        data_status=item.data_status,
    )


def strip_invalid_markers(text: str, valid_ids: set[str]) -> tuple[str, list[str]]:
    """Remove citation markers that do not resolve to retrieved evidence."""
    invalid: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        marker_id = match.group(1)
        if marker_id in valid_ids:
            return match.group(0)
        invalid.append(marker_id)
        return ""

    cleaned = CITATION_MARKER_RE.sub(_replace, text)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip(), sorted(set(invalid))


def format_answer(
    answer: str, evidence: Sequence[Evidence], include_source_list: bool = True
) -> tuple[str, list[Citation], list[str]]:
    """Validate markers and append the source list.

    Returns ``(formatted_answer, citations, warnings)``. Only evidence actually
    cited in the final text is listed as a source.
    """
    warnings: list[str] = []
    by_id = {item.evidence_id: item for item in evidence if item.evidence_id}

    cleaned, invalid = strip_invalid_markers(answer or "", set(by_id))
    if invalid:
        warnings.append(
            "Removed citation marker(s) that did not correspond to any retrieved "
            f"source: {', '.join('[' + i + ']' for i in invalid)}."
        )
        logger.warning("Fabricated citation markers removed: %s", invalid)

    used_ids = [cid for cid in dict.fromkeys(CITATION_MARKER_RE.findall(cleaned)) if cid in by_id]
    citations = [to_citation(by_id[cid]) for cid in used_ids]

    if by_id and not citations:
        warnings.append(
            "The answer contains no citations even though evidence was retrieved; "
            "treat its factual content as ungrounded."
        )

    if citations and include_source_list:
        lines = "\n".join(citation.render() for citation in citations)
        cleaned = f"{cleaned.rstrip()}\n\nSources:\n{lines}"

        if any(c.data_status == "demo" for c in citations):
            cleaned += (
                "\n\nNote: sources marked DEMO come from the bundled demonstration "
                "dataset and are not a validated medical knowledge source."
            )

    return cleaned.strip(), citations, warnings
