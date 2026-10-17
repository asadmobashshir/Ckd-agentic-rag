"""Environment-driven configuration for the Medical Agentic RAG system.

All paths are resolved with :mod:`pathlib` so the project runs unchanged on
Windows, Linux and WSL. Secrets are read from the environment only and are
never written to logs.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # ------------------------------------------------------------------ LLM
    # Two providers are supported. 'auto' resolves to whichever key is present,
    # preferring Groq because that is the key most deployments of this project
    # already hold. Neither provider is removed: both implement native tool
    # calling through the same BaseLLMClient contract.
    llm_provider: Literal["auto", "groq", "gemini"] = "auto"
    llm_max_retries: int = Field(default=3, ge=0, le=6)

    # --- Groq ---------------------------------------------------------------
    groq_api_key: str | None = Field(default=None, description="Groq API key.")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description=(
            "Groq chat model id. Must support tool calling. Verify against "
            "https://console.groq.com/docs/models - model ids are retired "
            "regularly, which is why this is configuration, not code."
        ),
    )

    # --- Gemini -------------------------------------------------------------
    gemini_api_key: str | None = Field(default=None, description="Google Gemini API key.")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description=(
            "Google Gemini API model id. Verify availability for your key with "
            "`python scripts/diagnostics.py --list-models`."
        ),
    )

    # --- Shared generation parameters ---------------------------------------
    # Legacy ``gemini_*`` names are kept as aliases so existing .env files and
    # Streamlit secrets keep working unchanged.
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0, alias="gemini_temperature")
    llm_max_tokens: int = Field(default=1400, gt=0, alias="gemini_max_tokens")

    #: Candidate models tried, in order, when the configured model is rejected by
    #: the provider. Keeps the app usable when a model id is retired upstream.
    groq_model_fallbacks: list[str] = Field(
        default_factory=lambda: [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-20b",
        ]
    )
    gemini_model_fallbacks: list[str] = Field(
        default_factory=lambda: [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
        ]
    )

    # ------------------------------------------------------------ Embeddings
    embedding_backend: Literal["auto", "sentence_transformers", "hashing"] = Field(
        default="auto",
        description=(
            "'auto' uses sentence-transformers when installed and the model is "
            "cached, otherwise falls back to the deterministic offline hashing "
            "backend (development/CI only - NOT semantically meaningful)."
        ),
    )
    embedding_model: str = Field(
        default="pritamdeka/S-PubMedBert-MS-MARCO",
        description="sentence-transformers-compatible biomedical retrieval model.",
    )
    embedding_dim: int = Field(default=384, gt=0, description="Hashing-backend dim.")
    embedding_batch_size: int = Field(default=16, gt=0)

    # ---------------------------------------------------------------- Stores
    vector_backend: Literal["auto", "qdrant", "numpy"] = "auto"
    vector_collection: str = "medical_chunks"
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="medical_agent_data_dir")
    vector_db_path: Path = Field(default=PROJECT_ROOT / "data" / "processed" / "qdrant")
    sqlite_db_path: Path = Field(default=PROJECT_ROOT / "data" / "structured" / "medical.db")
    audit_log_path: Path = Field(default=PROJECT_ROOT / "data" / "processed" / "audit.jsonl")

    # ------------------------------------------------------------- Retrieval
    top_k: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.15, ge=0.0, le=1.0)
    chunk_size: int = Field(default=900, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)

    # ----------------------------------------------------------------- Agent
    max_tool_rounds: int = Field(default=3, ge=1, le=6)
    #: Hard ceiling on iterations of the native tool-calling loop. Each iteration
    #: is one model turn that may request tools; the loop always terminates.
    max_agent_iterations: int = Field(default=5, ge=1, le=10)
    #: When false, the agent skips the native tool-calling loop and uses the
    #: legacy plan-then-execute path. Kept so the older behaviour stays reachable.
    enable_native_tool_calling: bool = True
    max_query_chars: int = Field(default=1500, ge=32)
    request_timeout: float = Field(default=30.0, gt=0)
    tool_timeout: float = Field(default=20.0, gt=0)

    # --------------------------------------------------------------- Runtime
    log_level: str = Field(default="INFO")
    web_search_max_results: int = Field(default=5, ge=1, le=20)
    literature_max_results: int = Field(default=5, ge=1, le=20)
    enable_network_tools: bool = Field(
        default=True,
        description="Set false in offline/CI environments to skip network tools.",
    )
    ncbi_email: str | None = None
    ncbi_api_key: str | None = None

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 900)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        value = v.upper()
        if value not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return value

    # -------------------------------------------------------------- Helpers
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def structured_dir(self) -> Path:
        return self.data_dir / "structured"

    # --------------------------------------------------- provider resolution
    @property
    def active_provider(self) -> str:
        """Resolve ``llm_provider``, honouring 'auto'.

        'auto' prefers Groq when a Groq key is present, then Gemini. When no key
        is configured it still returns a concrete provider so error messages can
        name the key the deployment is expected to set.
        """
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.groq_api_key and self.groq_api_key.strip():
            return "groq"
        if self.gemini_api_key and self.gemini_api_key.strip():
            return "gemini"
        return "groq"

    @property
    def active_api_key(self) -> str | None:
        return self.groq_api_key if self.active_provider == "groq" else self.gemini_api_key

    @property
    def active_model(self) -> str:
        return self.groq_model if self.active_provider == "groq" else self.gemini_model

    @property
    def active_model_fallbacks(self) -> list[str]:
        if self.active_provider == "groq":
            return list(self.groq_model_fallbacks)
        return list(self.gemini_model_fallbacks)

    @property
    def active_key_name(self) -> str:
        return "GROQ_API_KEY" if self.active_provider == "groq" else "GEMINI_API_KEY"

    @property
    def has_llm_credentials(self) -> bool:
        key = self.active_api_key
        return bool(key and key.strip())

    # Backward-compatible aliases: older modules read these names directly.
    @property
    def gemini_temperature(self) -> float:
        return self.llm_temperature

    @property
    def gemini_max_tokens(self) -> int:
        return self.llm_max_tokens

    def ensure_directories(self) -> None:
        """Create every directory the runtime writes into."""
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.structured_dir,
            self.vector_db_path.parent,
            self.sqlite_db_path.parent,
            self.audit_log_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def safe_dump(self) -> dict[str, object]:
        """Configuration snapshot with every secret removed (safe to log)."""
        data = self.model_dump(mode="json")
        for secret_key in ("gemini_api_key", "groq_api_key", "ncbi_api_key", "ncbi_email"):
            if data.get(secret_key):
                data[secret_key] = "***redacted***"
        return data


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and re-read the environment (used by tests)."""
    get_settings.cache_clear()
    return get_settings()
