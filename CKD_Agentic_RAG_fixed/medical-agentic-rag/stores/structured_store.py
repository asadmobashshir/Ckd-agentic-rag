"""SQLite-backed structured medical store.

Holds drugs, pairwise interactions and dosage *reference* rows. The shipped
dataset is a tiny DEMO fixture whose only purpose is exercising software paths.
Every row carries ``data_status``; the demo fixture sets it to ``"demo"`` and
that value is propagated all the way to the user-visible answer.

A real deployment must replace ``data/structured/demo_drug_data.json`` with a
properly licensed and validated source and set ``data_status='licensed'``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_SEED_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "structured" / "demo_drug_data.json"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS drugs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    drug_class      TEXT,
    summary         TEXT,
    contraindications TEXT,
    source          TEXT NOT NULL,
    source_url      TEXT,
    data_status     TEXT NOT NULL DEFAULT 'demo',
    last_reviewed   TEXT
);

CREATE TABLE IF NOT EXISTS drug_interactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_a          TEXT NOT NULL,
    drug_b          TEXT NOT NULL,
    interaction     TEXT NOT NULL,
    severity        TEXT,
    mechanism       TEXT,
    management      TEXT,
    source          TEXT NOT NULL,
    source_url      TEXT,
    data_status     TEXT NOT NULL DEFAULT 'demo',
    last_reviewed   TEXT,
    UNIQUE (drug_a, drug_b)
);

CREATE TABLE IF NOT EXISTS dosage_reference (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    drug            TEXT NOT NULL,
    population      TEXT NOT NULL,
    indication      TEXT,
    reference_text  TEXT NOT NULL,
    route           TEXT,
    source          TEXT NOT NULL,
    source_url      TEXT,
    data_status     TEXT NOT NULL DEFAULT 'demo',
    last_reviewed   TEXT,
    UNIQUE (drug, population, indication)
);

CREATE INDEX IF NOT EXISTS idx_interactions_a ON drug_interactions (drug_a);
CREATE INDEX IF NOT EXISTS idx_interactions_b ON drug_interactions (drug_b);
CREATE INDEX IF NOT EXISTS idx_dosage_drug ON dosage_reference (drug);
"""


def normalize_drug_name(name: str) -> str:
    """Lowercase/trim a drug name for lookup (no clinical normalisation)."""
    return " ".join((name or "").strip().lower().split())


class StructuredStore:
    """Thin, typed wrapper around the SQLite medical reference database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------- lifecycle
    def initialize(self) -> None:
        """Create tables if they do not exist."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def seed_from_file(self, seed_file: Path | None = None, *, replace: bool = False) -> dict[str, int]:
        """Load the JSON fixture into SQLite. Existing rows are kept unless ``replace``."""
        seed_file = seed_file or DEFAULT_SEED_FILE
        if not seed_file.exists():
            raise FileNotFoundError(f"Seed file not found: {seed_file}")
        payload = json.loads(seed_file.read_text(encoding="utf-8"))
        self.initialize()
        counts = {"drugs": 0, "interactions": 0, "dosage": 0}

        with self._connect() as conn:
            if replace:
                conn.executescript(
                    "DELETE FROM drugs; DELETE FROM drug_interactions; DELETE FROM dosage_reference;"
                )
            for drug in payload.get("drugs", []):
                conn.execute(
                    """INSERT OR REPLACE INTO drugs
                       (name, drug_class, summary, contraindications, source, source_url,
                        data_status, last_reviewed)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        normalize_drug_name(drug["name"]),
                        drug.get("drug_class"),
                        drug.get("summary"),
                        json.dumps(drug.get("contraindications", []), ensure_ascii=False),
                        drug.get("source", "DEMO fixture"),
                        drug.get("source_url"),
                        drug.get("data_status", "demo"),
                        drug.get("last_reviewed"),
                    ),
                )
                counts["drugs"] += 1

            for item in payload.get("interactions", []):
                a, b = sorted([normalize_drug_name(item["drug_a"]), normalize_drug_name(item["drug_b"])])
                conn.execute(
                    """INSERT OR REPLACE INTO drug_interactions
                       (drug_a, drug_b, interaction, severity, mechanism, management,
                        source, source_url, data_status, last_reviewed)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        a,
                        b,
                        item["interaction"],
                        item.get("severity"),
                        item.get("mechanism"),
                        item.get("management"),
                        item.get("source", "DEMO fixture"),
                        item.get("source_url"),
                        item.get("data_status", "demo"),
                        item.get("last_reviewed"),
                    ),
                )
                counts["interactions"] += 1

            for item in payload.get("dosage_reference", []):
                conn.execute(
                    """INSERT OR REPLACE INTO dosage_reference
                       (drug, population, indication, reference_text, route, source,
                        source_url, data_status, last_reviewed)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        normalize_drug_name(item["drug"]),
                        item.get("population", "adult"),
                        item.get("indication"),
                        item["reference_text"],
                        item.get("route"),
                        item.get("source", "DEMO fixture"),
                        item.get("source_url"),
                        item.get("data_status", "demo"),
                        item.get("last_reviewed"),
                    ),
                )
                counts["dosage"] += 1
        logger.info("Seeded structured store from %s: %s", seed_file.name, counts)
        return counts

    # --------------------------------------------------------------- queries
    def get_drug(self, name: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM drugs WHERE name = ?", (normalize_drug_name(name),)
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        try:
            record["contraindications"] = json.loads(record.get("contraindications") or "[]")
        except json.JSONDecodeError:  # pragma: no cover - defensive
            record["contraindications"] = []
        return record

    def find_interaction(self, drug_a: str, drug_b: str) -> dict[str, Any] | None:
        """Return the stored interaction row, or ``None`` when absent.

        ``None`` means *not in this database*. It must never be presented as
        evidence that no interaction exists.
        """
        a, b = sorted([normalize_drug_name(drug_a), normalize_drug_name(drug_b)])
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM drug_interactions WHERE drug_a = ? AND drug_b = ?", (a, b)
            ).fetchone()
        return dict(row) if row else None

    def list_interactions_for(self, drug: str) -> list[dict[str, Any]]:
        target = normalize_drug_name(drug)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM drug_interactions WHERE drug_a = ? OR drug_b = ?", (target, target)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_dosage_reference(
        self, drug: str, population: str | None = None, indication: str | None = None
    ) -> list[dict[str, Any]]:
        """Return stored dosage *reference* rows (never a personalised recommendation)."""
        sql = "SELECT * FROM dosage_reference WHERE drug = ?"
        params: list[Any] = [normalize_drug_name(drug)]
        if population:
            sql += " AND lower(population) = ?"
            params.append(population.strip().lower())
        if indication:
            sql += " AND lower(indication) LIKE ?"
            params.append(f"%{indication.strip().lower()}%")
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            self._ensure_tables(conn)
            return {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("drugs", "drug_interactions", "dosage_reference")
            }

    @staticmethod
    def _ensure_tables(conn: sqlite3.Connection) -> None:
        conn.executescript(SCHEMA)


def build_structured_store(settings: Settings | None = None) -> StructuredStore:
    """Create a store instance from settings (tables guaranteed to exist)."""
    settings = settings or get_settings()
    store = StructuredStore(settings.sqlite_db_path)
    store.initialize()
    return store
