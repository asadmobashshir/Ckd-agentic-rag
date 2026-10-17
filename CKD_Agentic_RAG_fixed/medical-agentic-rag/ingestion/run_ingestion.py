"""Offline ingestion pipeline entrypoint.

    python -m ingestion.run_ingestion [--source DIR] [--reset] [--json]

Runs collect -> chunk -> tag -> embed -> index and prints a summary. This is a
build step; it must never run inside the online query path.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config.settings import Settings, get_settings
from ingestion.chunker import chunk_text
from ingestion.collect_sources import collect_documents
from ingestion.embedder import build_embedder
from ingestion.indexer import Indexer, IndexReport
from ingestion.metadata_tagger import tag_document, validate_payload
from stores.index_meta import (
    check_compatibility,
    fingerprint_for,
    write_fingerprint,
)
from stores.structured_store import build_structured_store
from stores.vector_store import build_vector_store

logger = logging.getLogger(__name__)


def run_ingestion(
    source_dir: Path | None = None,
    settings: Settings | None = None,
    reset: bool = False,
    seed_structured: bool = True,
) -> IndexReport:
    """Execute the full offline pipeline and return a summary report."""
    settings = settings or get_settings()
    settings.ensure_directories()
    source_dir = source_dir or settings.raw_dir

    report = IndexReport()
    documents = collect_documents(source_dir)
    report.documents = len(documents)
    if not documents:
        report.errors.append(f"No supported documents found in {source_dir}")

    embedder = build_embedder(settings)
    report.embedder = embedder.describe()

    # Never append to an index built by a different embedder: the dimensions can
    # match while the vector spaces are unrelated, which fails silently at query
    # time. Detect it here and rebuild instead.
    compatible, compat_message = check_compatibility(settings.vector_db_path, embedder)
    force_reset = reset
    if not compatible and "not been built" not in compat_message:
        logger.warning("%s Rebuilding the index from scratch.", compat_message)
        report.errors.append(f"index rebuilt: {compat_message}")
        force_reset = True

    with build_vector_store(settings) as store:
        report.vector_backend = store.backend
        if force_reset:
            store.reset()
        indexer = Indexer(embedder, store, batch_size=settings.embedding_batch_size)

        for document in documents:
            try:
                chunks = chunk_text(
                    document.text,
                    document.document_id,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
                payloads = tag_document(document, chunks)
                valid = [p for p in payloads if validate_payload(p)]
                report.chunks_skipped += len(payloads) - len(valid)
                report.chunks_indexed += indexer.index_payloads(valid)
                report.per_document.append(
                    {
                        "file_name": document.path.name,
                        "document_id": document.document_id,
                        "title": document.title,
                        "chunks": len(valid),
                        "data_status": document.data_status,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - one bad doc must not abort the run
                logger.exception("Failed to ingest %s", document.path)
                report.errors.append(f"{document.path.name}: {exc}")

        report.vectors_in_store = store.count()

    if report.chunks_indexed:
        write_fingerprint(settings.vector_db_path, fingerprint_for(embedder))

    if seed_structured:
        try:
            counts = build_structured_store(settings).seed_from_file(replace=reset)
            logger.info("Structured store seeded: %s", counts)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"structured store seeding failed: {exc}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline ingestion pipeline.")
    parser.add_argument("--source", type=Path, default=None, help="Source directory.")
    parser.add_argument("--reset", action="store_true", help="Drop existing index first.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--no-structured", action="store_true", help="Skip SQLite seeding.")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    report = run_ingestion(
        source_dir=args.source,
        settings=settings,
        reset=args.reset,
        seed_structured=not args.no_structured,
    )
    print(json.dumps(report.as_dict(), indent=2) if args.json else report.render())
    return 0 if report.chunks_indexed or not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
