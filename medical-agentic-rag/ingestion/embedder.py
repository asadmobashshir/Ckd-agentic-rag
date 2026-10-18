"""Embedding backends.

Two interchangeable implementations sit behind :class:`BaseEmbedder`:

``SentenceTransformerEmbedder``
    The production path. Loads a sentence-transformers-compatible *biomedical
    retrieval* model (default ``pritamdeka/S-PubMedBert-MS-MARCO``, a PubMedBERT
    checkpoint that has actually been fine-tuned into a sentence-embedding
    pipeline - a raw ``microsoft/PubMedBERT`` masked-LM checkpoint is **not** a
    retrieval model and is deliberately not used here).

``HashingEmbedder``
    A deterministic, dependency-free character n-gram hashing vectoriser used
    when sentence-transformers (or its model cache) is unavailable, e.g. in CI
    or an air-gapped container. It makes the pipeline runnable and testable but
    it is lexical, not semantic. **Never ship it as a retrieval backend.**
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Sequence

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BaseEmbedder(ABC):
    """Interface every embedding backend must implement."""

    name: str = "base"
    is_semantic: bool = False

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality produced by this backend."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (defaults to the document path)."""
        return self.embed_documents([text])[0]

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "dimension": self.dimension,
            "semantic": self.is_semantic,
        }


class HashingEmbedder(BaseEmbedder):
    """Deterministic lexical hashing embedder (offline fallback)."""

    name = "hashing"
    is_semantic = False

    def __init__(self, dimension: int = 384, ngram_range: tuple[int, int] = (1, 2)) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._ngram_range = ngram_range

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _bucket(token: str, dimension: int) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % dimension

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        features = list(tokens)
        low, high = self._ngram_range
        for n in range(max(low, 2), high + 1):
            features.extend(
                " ".join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))
            )
        return features

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self._dimension
            for feature in self._features(text or ""):
                idx = self._bucket(feature, self._dimension)
                # Signed hashing keeps collisions from systematically inflating norms.
                sign = 1.0 if self._bucket("s:" + feature, 2) == 0 else -1.0
                vector[idx] += sign
            norm = math.sqrt(sum(v * v for v in vector))
            vectors.append([v / norm for v in vector] if norm else vector)
        return vectors


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wrapper around a sentence-transformers biomedical retrieval model."""

    name = "sentence_transformers"
    is_semantic = True

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "sentence-transformers is not installed. Install it or set "
                "EMBEDDING_BACKEND=hashing for offline development."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._batch_size = batch_size
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(x) for x in row] for row in vectors]

    def describe(self) -> dict[str, object]:
        info = super().describe()
        info["model"] = self._model_name
        return info


def build_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """Instantiate the configured embedder, degrading gracefully when offline."""
    settings = settings or get_settings()
    backend = settings.embedding_backend

    if backend == "hashing":
        return HashingEmbedder(dimension=settings.embedding_dim)

    try:
        return SentenceTransformerEmbedder(
            settings.embedding_model, batch_size=settings.embedding_batch_size
        )
    except Exception as exc:  # noqa: BLE001 - fallback must never crash startup
        if backend == "sentence_transformers":
            raise
        logger.warning(
            "Falling back to the offline hashing embedder (%s). Retrieval will be "
            "lexical, not semantic. Reason: %s",
            settings.embedding_model,
            exc,
        )
        return HashingEmbedder(dimension=settings.embedding_dim)
