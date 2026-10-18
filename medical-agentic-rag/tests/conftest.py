"""Shared pytest fixtures.

Every fixture is hermetic: temporary directories, an isolated SQLite database and
an in-memory vector store. No test requires a Gemini API key or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.state import Evidence, TrustTier  # noqa: E402
from config.settings import Settings  # noqa: E402
from ingestion.embedder import HashingEmbedder  # noqa: E402
from llm.base import BaseLLMClient, LLMResponse, LLMToolCall  # noqa: E402
from stores.structured_store import StructuredStore  # noqa: E402
from stores.vector_store import NumpyVectorStore  # noqa: E402

SEED_FILE = Path(__file__).resolve().parent.parent / "data" / "structured" / "demo_drug_data.json"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings pointing at a temporary data directory."""
    return Settings(
        gemini_api_key=None,
        data_dir=tmp_path / "data",
        vector_db_path=tmp_path / "data" / "vectors",
        sqlite_db_path=tmp_path / "data" / "medical.db",
        audit_log_path=tmp_path / "data" / "audit.jsonl",
        embedding_backend="hashing",
        vector_backend="numpy",
        enable_network_tools=False,
        top_k=5,
        max_tool_rounds=2,
        # The lexical hashing embedder produces low absolute scores; thresholding
        # is exercised in its own test rather than in every pipeline fixture.
        min_similarity=0.0,
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=128)


@pytest.fixture
def vector_store(settings: Settings) -> NumpyVectorStore:
    settings.ensure_directories()
    return NumpyVectorStore(settings.vector_db_path, settings.vector_collection)


@pytest.fixture
def structured_store(settings: Settings) -> StructuredStore:
    settings.ensure_directories()
    store = StructuredStore(settings.sqlite_db_path)
    store.seed_from_file(SEED_FILE, replace=True)
    return store


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    return [
        Evidence(
            evidence_id="S1",
            text=(
                "Metformin is an oral biguanide widely used in the management of type 2 "
                "diabetes mellitus and is commonly described as a first-line option."
            ),
            source="DEMO corpus",
            title="DEMO: Metformin overview",
            document_id="demo_metformin-abc123",
            chunk_id="demo_metformin-abc123::c0000",
            score=0.82,
            trust_tier=TrustTier.A,
            data_status="demo",
        ),
        Evidence(
            evidence_id="S2",
            text=(
                "Combining an anticoagulant with an NSAID is widely described as increasing "
                "the risk of bleeding compared with either agent used alone."
            ),
            source="DEMO corpus",
            title="DEMO: Anticoagulants and NSAIDs",
            document_id="demo_anticoag-def456",
            chunk_id="demo_anticoag-def456::c0001",
            score=0.74,
            trust_tier=TrustTier.B,
            data_status="demo",
        ),
    ]


class FakeLLM(BaseLLMClient):
    """Scripted LLM used to test agent behaviour without a live provider."""

    provider = "fake"
    supports_tool_calling = True

    def __init__(
        self,
        tool_calls: Sequence[LLMToolCall] | None = None,
        text: str = "",
        json_payload: dict[str, Any] | None = None,
        available: bool = True,
        raise_error: Exception | None = None,
    ) -> None:
        self._tool_calls = list(tool_calls or [])
        self._text = text
        self._json_payload = json_payload
        self._available = available
        self._raise = raise_error
        self.calls: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return self._available

    def chat(self, messages, **kwargs) -> LLMResponse:  # noqa: ANN001
        self.calls.append({"messages": list(messages), **kwargs})
        if self._raise is not None:
            raise self._raise
        # First round proposes tools; later rounds return none so the loop terminates.
        tool_calls = self._tool_calls if len(self.calls) == 1 else []
        if kwargs.get("json_mode") and self._json_payload is not None:
            import json as _json

            return LLMResponse(text=_json.dumps(self._json_payload))
        return LLMResponse(text=self._text, tool_calls=tool_calls)


@pytest.fixture
def fake_llm_factory():
    return FakeLLM
