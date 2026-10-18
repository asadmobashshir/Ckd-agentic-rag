"""Vector store abstraction.

Default backend is **Qdrant local mode** (embedded, on-disk, no server and no
cloud account). A dependency-free NumPy/JSON backend is provided as a fallback
so the project still runs where ``qdrant-client`` cannot be installed.

Adding another backend means implementing :class:`BaseVectorStore`; nothing
else in the codebase needs to change.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Sequence

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot satisfy a request."""


class VectorRecord:
    """A vector plus its payload, ready for indexing."""

    __slots__ = ("id", "vector", "payload")

    def __init__(self, id: str, vector: Sequence[float], payload: dict[str, Any]) -> None:
        self.id = id
        self.vector = list(vector)
        self.payload = payload


class SearchHit:
    """A scored payload returned from a similarity search."""

    __slots__ = ("id", "score", "payload")

    def __init__(self, id: str, score: float, payload: dict[str, Any]) -> None:
        self.id = id
        self.score = score
        self.payload = payload

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SearchHit(id={self.id!r}, score={self.score:.4f})"


class BaseVectorStore(ABC):
    """Minimal interface required by the retrieval tools."""

    backend: str = "base"

    @abstractmethod
    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if it does not exist (idempotent)."""

    @abstractmethod
    def upsert(self, records: Iterable[VectorRecord]) -> int:
        """Insert or update records; returns the number written."""

    @abstractmethod
    def search(self, vector: Sequence[float], top_k: int = 5) -> list[SearchHit]:
        """Cosine-similarity search."""

    @abstractmethod
    def count(self) -> int:
        """Number of indexed vectors (0 when the collection is missing)."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all indexed data."""

    @abstractmethod
    def list_payloads(self) -> list[dict[str, Any]]:
        """Return every stored payload (used to build the lexical index)."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release backend resources."""

    def __enter__(self) -> "BaseVectorStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        try:
            return {"backend": self.backend, "ok": True, "count": self.count()}
        except Exception as exc:  # noqa: BLE001
            return {"backend": self.backend, "ok": False, "error": str(exc)}


# --------------------------------------------------------------------- Qdrant
class QdrantVectorStore(BaseVectorStore):
    """Embedded Qdrant (local mode) - persists to a directory, no server."""

    backend = "qdrant"

    def __init__(self, path: Path, collection: str) -> None:
        try:
            from qdrant_client import QdrantClient  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on env
            raise VectorStoreError("qdrant-client is not installed") from exc
        path.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._collection = collection
        self._client = QdrantClient(path=str(path))

    def ensure_collection(self, dimension: int) -> None:
        from qdrant_client import models  # noqa: PLC0415

        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            ),
        )

    @staticmethod
    def _point_id(record_id: str) -> str:
        """Qdrant point ids must be UUIDs or unsigned ints; map deterministically."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, record_id))

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        from qdrant_client import models  # noqa: PLC0415

        points = [
            models.PointStruct(
                id=self._point_id(rec.id),
                vector=rec.vector,
                payload={**rec.payload, "_record_id": rec.id},
            )
            for rec in records
        ]
        if not points:
            return 0
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(self, vector: Sequence[float], top_k: int = 5) -> list[SearchHit]:
        if not self._client.collection_exists(self._collection):
            return []
        response = self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=top_k,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = dict(point.payload or {})
            hits.append(
                SearchHit(str(payload.pop("_record_id", point.id)), float(point.score), payload)
            )
        return hits

    def count(self) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        return int(self._client.count(self._collection, exact=True).count)

    def reset(self) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)

    def list_payloads(self) -> list[dict[str, Any]]:
        if not self._client.collection_exists(self._collection):
            return []
        payloads: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                payload.setdefault("_record_id", str(point.id))
                payloads.append(payload)
            if offset is None:
                break
        return payloads

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - closing must never raise
            logger.debug("Qdrant client close failed", exc_info=True)


# ---------------------------------------------------------------------- NumPy
class NumpyVectorStore(BaseVectorStore):
    """Flat cosine-similarity index persisted as ``.npy`` + JSON metadata."""

    backend = "numpy"

    def __init__(self, path: Path, collection: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._vectors_path = path / f"{collection}.vectors.npy"
        self._meta_path = path / f"{collection}.meta.json"
        self._ids: list[str] = []
        self._payloads: list[dict[str, Any]] = []
        self._matrix = None
        self._dimension: int | None = None
        self._load()

    def _load(self) -> None:
        import numpy as np  # noqa: PLC0415

        if self._meta_path.exists():
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._ids = meta.get("ids", [])
            self._payloads = meta.get("payloads", [])
            self._dimension = meta.get("dimension")
        if self._vectors_path.exists():
            self._matrix = np.load(self._vectors_path)

    def _persist(self) -> None:
        import numpy as np  # noqa: PLC0415

        if self._matrix is not None:
            np.save(self._vectors_path, self._matrix)
        self._meta_path.write_text(
            json.dumps(
                {"ids": self._ids, "payloads": self._payloads, "dimension": self._dimension},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def ensure_collection(self, dimension: int) -> None:
        if self._dimension is not None and self._dimension != dimension:
            raise VectorStoreError(
                f"Existing index has dimension {self._dimension}, got {dimension}. "
                "Re-run ingestion with --reset after changing the embedding model."
            )
        self._dimension = dimension

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        import numpy as np  # noqa: PLC0415

        records = list(records)
        if not records:
            return 0
        index = {rid: i for i, rid in enumerate(self._ids)}
        new_vectors: list[list[float]] = []
        for rec in records:
            if rec.id in index:
                self._payloads[index[rec.id]] = rec.payload
                if self._matrix is not None:
                    self._matrix[index[rec.id]] = np.asarray(rec.vector, dtype="float32")
            else:
                self._ids.append(rec.id)
                self._payloads.append(rec.payload)
                new_vectors.append(rec.vector)
        if new_vectors:
            block = np.asarray(new_vectors, dtype="float32")
            self._matrix = block if self._matrix is None else np.vstack([self._matrix, block])
        self._persist()
        return len(records)

    def search(self, vector: Sequence[float], top_k: int = 5) -> list[SearchHit]:
        import numpy as np  # noqa: PLC0415

        if self._matrix is None or not len(self._ids):
            return []
        query = np.asarray(vector, dtype="float32")
        q_norm = float(np.linalg.norm(query)) or 1.0
        matrix_norms = np.linalg.norm(self._matrix, axis=1)
        matrix_norms[matrix_norms == 0] = 1.0
        scores = (self._matrix @ query) / (matrix_norms * q_norm)
        order = np.argsort(-scores)[:top_k]
        return [
            SearchHit(self._ids[i], float(scores[i]), dict(self._payloads[i])) for i in order
        ]

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._ids, self._payloads, self._matrix, self._dimension = [], [], None, None
        self._vectors_path.unlink(missing_ok=True)
        self._meta_path.unlink(missing_ok=True)

    def list_payloads(self) -> list[dict[str, Any]]:
        return [
            {**dict(payload), "_record_id": record_id}
            for record_id, payload in zip(self._ids, self._payloads)
        ]


def build_vector_store(settings: Settings | None = None) -> BaseVectorStore:
    """Instantiate the configured vector store with a safe fallback."""
    settings = settings or get_settings()
    path, collection = settings.vector_db_path, settings.vector_collection

    if settings.vector_backend == "numpy":
        return NumpyVectorStore(path, collection)
    try:
        return QdrantVectorStore(path, collection)
    except Exception as exc:  # noqa: BLE001
        if settings.vector_backend == "qdrant":
            raise
        logger.warning("Qdrant unavailable (%s); using the NumPy vector store.", exc)
        return NumpyVectorStore(path, collection)
