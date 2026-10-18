"""Build or rebuild the vector index.

    python scripts/build_index.py [--reset] [--source DIR]

Thin wrapper around :func:`ingestion.run_ingestion.run_ingestion` for people who
prefer a script to ``python -m``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from ingestion.run_ingestion import run_ingestion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the vector index.")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    report = run_ingestion(source_dir=args.source, settings=settings, reset=args.reset)
    print(report.render())
    return 0 if report.chunks_indexed else 1


if __name__ == "__main__":
    raise SystemExit(main())
