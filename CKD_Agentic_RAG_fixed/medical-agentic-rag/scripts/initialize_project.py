"""One-shot project initialisation.

    python scripts/initialize_project.py

Creates the runtime directories, copies ``.env.example`` to ``.env`` when absent,
initialises and seeds the SQLite store, and exports the tool schemas. Idempotent:
existing files are left alone.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.tools.tool_registry import build_default_registry  # noqa: E402
from config.settings import PROJECT_ROOT, get_settings  # noqa: E402
from stores.structured_store import build_structured_store  # noqa: E402


def main() -> int:
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    settings = get_settings()

    settings.ensure_directories()
    print(f"[ok] directories ready under {settings.data_dir}")

    env_path, example_path = PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example"
    if env_path.exists():
        print("[ok] .env already exists (left unchanged)")
    elif example_path.exists():
        shutil.copyfile(example_path, env_path)
        print(f"[ok] created .env from .env.example - add your {settings.active_key_name}")
    else:
        print("[!!] .env.example not found; create .env manually")

    counts = build_structured_store(settings).seed_from_file()
    print(f"[ok] structured store seeded (DEMO data): {counts}")

    path = build_default_registry(settings).export_schemas()
    print(f"[ok] tool schemas exported to {path.relative_to(PROJECT_ROOT)}")

    print("\nNext: python -m ingestion.run_ingestion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
