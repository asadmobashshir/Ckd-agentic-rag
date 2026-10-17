"""Free web search via DuckDuckGo (``ddgs``), with source classification.

Web results are **discovery material**, not verified medical evidence. Every
result keeps its title, URL, snippet, domain, retrieval timestamp and trust tier,
and low-tier results are marked so the confidence heuristic can discount them.

Retrieved snippets are treated strictly as data: any instruction-like text inside
them is neutralised before it reaches the model (see :func:`sanitize_snippet`).
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.state import Evidence, TrustTier, utc_now_iso
from agent.tools.base import BaseTool
from config.settings import Settings, get_settings
from safety.source_trust import classify_url, evidence_level_for_tier, extract_domain

logger = logging.getLogger(__name__)

#: Patterns that look like an attempt to steer the model from inside a document.
INJECTION_PATTERNS = (
    r"ignore (all |any |the )?(previous|prior|above) instructions?",
    r"disregard (all |any |the )?(previous|prior|above)",
    r"you are now\b",
    r"system prompt",
    r"</?(system|assistant|user)>",
    r"new instructions?:",
    r"act as (an?|the)\b",
    r"do not follow",
    r"reveal your (prompt|instructions|system)",
)
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def sanitize_snippet(text: str, max_len: int = 1200) -> str:
    """Neutralise instruction-like content in retrieved text and clamp length."""
    if not text:
        return ""
    cleaned = _INJECTION_RE.sub("[redacted-instruction-like-text]", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_len]


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3, max_length=256)
    max_results: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def _clean(cls, value: str) -> str:
        return " ".join(value.strip().split())


class WebSearchTool(BaseTool):
    """General web search for discovery of current, non-curated information."""

    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Search the open web (DuckDuckGo) for current or general information not in "
        "the curated corpus, e.g. recent public-health guidance or news. Results are "
        "DISCOVERY material with a trust tier attached; general web pages are not "
        "authoritative medical evidence. Prefer vector_search or literature_search "
        "for clinical questions."
    )
    args_model: ClassVar[type[BaseModel]] = WebSearchArgs
    requires_network: ClassVar[bool] = True

    def __init__(self, settings: Settings | None = None, search_fn: Any = None) -> None:
        self._settings = settings or get_settings()
        self._search_fn = search_fn  # injectable for tests

    def _search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if self._search_fn is not None:
            return list(self._search_fn(query, max_results))
        try:
            from ddgs import DDGS  # noqa: PLC0415
        except ImportError:  # pragma: no cover - fall back to the older package name
            try:
                from duckduckgo_search import DDGS  # type: ignore # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "Install `ddgs` (or `duckduckgo-search`) to enable web search."
                ) from exc
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    @staticmethod
    def _field(raw: dict[str, Any], *names: str) -> str | None:
        for name in names:
            value = raw.get(name)
            if value:
                return str(value)
        return None

    def run(self, args: BaseModel) -> tuple[list[Evidence], dict[str, Any]]:
        assert isinstance(args, WebSearchArgs)
        if not self._settings.enable_network_tools:
            return [], {
                "results": 0,
                "skipped": True,
                "message": "Network tools are disabled (ENABLE_NETWORK_TOOLS=false).",
            }

        limit = min(args.max_results, self._settings.web_search_max_results)
        try:
            raw_results = self._search(args.query, limit)
        except Exception as exc:  # noqa: BLE001 - a search outage is not an answer
            logger.warning("Web search failed: %s", exc)
            return [], {
                "results": 0,
                "error": f"web search unavailable: {exc}",
                "message": "No web results retrieved. Do not substitute recalled facts.",
            }

        timestamp = utc_now_iso()
        evidence: list[Evidence] = []
        tier_counts: dict[str, int] = {}

        for raw in raw_results:
            url = self._field(raw, "href", "url", "link")
            title = self._field(raw, "title", "heading")
            snippet = sanitize_snippet(self._field(raw, "body", "snippet", "description") or "")
            if not snippet and not title:
                continue
            domain = extract_domain(url)
            tier = classify_url(url)
            tier_counts[tier.value] = tier_counts.get(tier.value, 0) + 1
            evidence.append(
                Evidence(
                    text=snippet or (title or ""),
                    source=domain or "web",
                    title=title,
                    url=url,
                    publication_date=self._field(raw, "date", "published"),
                    retrieved_at=timestamp,
                    evidence_level=evidence_level_for_tier(tier),
                    document_type="web_page",
                    score=None,
                    trust_tier=tier,
                    data_status="live",
                    extra={
                        "retrieval": "web_search",
                        "domain": domain,
                        "authoritative": tier in (TrustTier.A, TrustTier.B),
                    },
                )
            )

        return evidence, {
            "results": len(evidence),
            "query": args.query,
            "tier_counts": tier_counts,
            "note": (
                "Web results are discovery material. Tier D/UNKNOWN sources are not "
                "authoritative medical evidence."
            ),
        }
