"""Source trust classification.

The tiers are a *retrieval prioritisation heuristic*. A high tier means the
publisher is generally accountable and editorially reviewed; it does not mean a
specific statement is correct, current, or applicable to any individual.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from urllib.parse import urlparse

from agent.state import TrustTier

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "source_trust.json"

_TIER_WEIGHTS: dict[TrustTier, float] = {
    TrustTier.A: 1.0,
    TrustTier.B: 0.85,
    TrustTier.C: 0.6,
    TrustTier.D: 0.3,
    TrustTier.UNKNOWN: 0.25,
}


@functools.lru_cache(maxsize=1)
def _load_config() -> dict:
    if not CONFIG_PATH.exists():  # pragma: no cover - packaging safety net
        return {"tiers": {}, "evidence_level_by_tier": {}}
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=1)
def _domain_index() -> dict[str, TrustTier]:
    index: dict[str, TrustTier] = {}
    for tier_name, block in _load_config().get("tiers", {}).items():
        try:
            tier = TrustTier(tier_name)
        except ValueError:  # pragma: no cover - defensive
            continue
        for domain in block.get("domains", []):
            index[domain.lower().lstrip(".")] = tier
    return index


def extract_domain(url: str | None) -> str | None:
    """Return the lowercase host of ``url`` without a leading ``www.``."""
    if not url:
        return None
    candidate = url if "://" in url else f"https://{url}"
    host = (urlparse(candidate).hostname or "").lower()
    if not host:
        return None
    return host[4:] if host.startswith("www.") else host


def classify_domain(domain: str | None) -> TrustTier:
    """Classify a bare domain into a trust tier (suffix match on registrable parts)."""
    if not domain:
        return TrustTier.UNKNOWN
    domain = domain.lower().lstrip(".")
    index = _domain_index()
    if domain in index:
        return index[domain]
    for known, tier in index.items():
        if domain == known or domain.endswith("." + known):
            return tier
    # Generic institutional heuristics (still only a heuristic).
    if domain.endswith(".gov") or domain.endswith(".gov.uk") or domain.endswith(".gov.in"):
        return TrustTier.A
    if domain.endswith(".edu") or domain.endswith(".ac.uk"):
        return TrustTier.C
    return TrustTier.D


def classify_url(url: str | None) -> TrustTier:
    return classify_domain(extract_domain(url))


def tier_weight(tier: TrustTier) -> float:
    """Numeric weight used by the confidence heuristic."""
    return _TIER_WEIGHTS.get(tier, 0.25)


def evidence_level_for_tier(tier: TrustTier) -> str:
    mapping = _load_config().get("evidence_level_by_tier", {})
    return mapping.get(tier.value, "unknown")


def tier_label(tier: TrustTier) -> str:
    block = _load_config().get("tiers", {}).get(tier.value, {})
    return block.get("label", "Unclassified source")
