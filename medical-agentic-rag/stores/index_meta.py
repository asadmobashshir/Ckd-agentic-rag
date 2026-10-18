"""Index fingerprinting.

The single most dangerous failure mode when switching embedding backends is a
*silent* one: an index built with the lexical hashing embedder is still 384
dimensions, so a semantic model queries it without any error and returns
confident nonsense. Dimension matching is not enough to prove compatibility.

Every ingestion run therefore writes a fingerprint of the embedder that produced
the vectors. Retrieval compares the live embedder against it and reports a
mismatch, and ingestion rebuilds automatically rather than appending
incompatible vectors to an existing collection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FINGERPRINT_FILENAME = "index_fingerprint.json"

#: Bump when the indexed text representation changes in a way that invalidates
#: existing vectors (e.g. adding contextual title/section headers to chunks).
EMBED_TEXT_VERSION = 2


@dataclass(frozen=True, slots=True)
class IndexFingerprint:
    """Identity of the embedder that produced an index."""

    backend: str
    model: str
    dimension: int
    embed_text_version: int = EMBED_TEXT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "dimension": self.dimension,
            "embed_text_version": self.embed_text_version,
        }

    def matches(self, other: "IndexFingerprint | None") -> bool:
        return other is not None and self.as_dict() == other.as_dict()

    def describe(self) -> str:
        return f"{self.backend}:{self.model} (dim={self.dimension}, v{self.embed_text_version})"


def fingerprint_for(embedder: Any) -> IndexFingerprint:
    """Build a fingerprint from a live embedder instance."""
    info = embedder.describe()
    return IndexFingerprint(
        backend=str(info.get("backend", "unknown")),
        model=str(info.get("model", info.get("backend", "unknown"))),
        dimension=int(info.get("dimension", 0)),
    )


def _path(vector_db_path: Path) -> Path:
    return vector_db_path / FINGERPRINT_FILENAME


def read_fingerprint(vector_db_path: Path) -> IndexFingerprint | None:
    """Read the stored fingerprint, or ``None`` when absent/unreadable."""
    path = _path(vector_db_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return IndexFingerprint(
            backend=str(data["backend"]),
            model=str(data["model"]),
            dimension=int(data["dimension"]),
            embed_text_version=int(data.get("embed_text_version", 1)),
        )
    except Exception:  # noqa: BLE001 - a corrupt fingerprint means "unknown"
        logger.warning("Could not read index fingerprint at %s", path, exc_info=True)
        return None


def write_fingerprint(vector_db_path: Path, fingerprint: IndexFingerprint) -> None:
    """Persist the fingerprint alongside the index."""
    vector_db_path.mkdir(parents=True, exist_ok=True)
    _path(vector_db_path).write_text(
        json.dumps(fingerprint.as_dict(), indent=2), encoding="utf-8"
    )


def clear_fingerprint(vector_db_path: Path) -> None:
    _path(vector_db_path).unlink(missing_ok=True)


def check_compatibility(vector_db_path: Path, embedder: Any) -> tuple[bool, str]:
    """Compare the live embedder with the stored fingerprint.

    Returns ``(compatible, message)``. An index that exists but was built by a
    different embedder is reported as incompatible so the caller can rebuild.
    """
    current = fingerprint_for(embedder)
    stored = read_fingerprint(vector_db_path)

    if stored is None:
        return False, "No index fingerprint found - the index has not been built yet."
    if current.matches(stored):
        return True, f"Index matches the active embedder: {current.describe()}"
    return False, (
        f"Index/embedder mismatch. Index was built with {stored.describe()}, "
        f"but the active embedder is {current.describe()}. "
        "Re-run: python -m ingestion.run_ingestion --reset"
    )
