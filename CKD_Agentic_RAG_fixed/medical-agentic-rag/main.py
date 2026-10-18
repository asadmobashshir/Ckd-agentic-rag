"""Interactive CLI for the Medical Agentic RAG prototype.

    python main.py                    # interactive REPL
    python main.py -q "your question" # single question, then exit
    python main.py --json -q "..."    # machine-readable output

Uses exactly the same :class:`~agent.orchestrator.Orchestrator` as the HTTP API;
no query logic is duplicated here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from agent.orchestrator import build_orchestrator
from agent.state import FinalResponse
from config.settings import get_settings

BANNER = """
=========================================================
  Medical Agentic RAG  -  research prototype
  Evidence-grounded medical information retrieval.
  NOT medical advice. NOT for diagnosis or treatment.
=========================================================
Type a question, or /help for commands, /quit to exit.
"""

HELP = """
Commands:
  /help     show this message
  /health   show component status
  /tools    list registered tools
  /quit     exit
"""


def render(response: FinalResponse, show_warnings: bool = True) -> str:
    """Format a response for the terminal."""
    lines = ["", response.answer, ""]
    lines.append(f"Evidence confidence: {response.confidence.value} ({response.confidence_score:.2f})")
    lines.append("  (how well-supported by retrieved evidence - not a probability of correctness)")
    lines.append(f"Tools used: {', '.join(response.tools_used) or 'none'}")
    lines.append(f"Risk category: {response.risk_category.value}")
    if show_warnings and response.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in response.warnings)
    lines.append(f"Request id: {response.request_id}")
    return "\n".join(lines)


def run_once(orchestrator, query: str, as_json: bool) -> int:
    response = orchestrator.run(query)
    if as_json:
        print(json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        print(render(response))
    return 0


def repl(orchestrator) -> int:
    print(BANNER)
    health = orchestrator.health()
    if not health["llm"].get("available"):
        print(
            "! No LLM backend is configured "
            f"(set {settings.active_key_name} in .env).\n"
            "  Running in evidence-only mode: retrieval works, answers are extractive.\n"
        )
    indexed = health.get("vector_store", {}).get("count", 0)
    if not indexed:
        print("! The vector index is empty. Run: python -m ingestion.run_ingestion\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if not query:
            continue
        command = query.lower()
        if command in {"/quit", "/exit", "/q"}:
            print("Bye.")
            return 0
        if command == "/help":
            print(HELP)
            continue
        if command == "/health":
            print(json.dumps(orchestrator.health(), indent=2, default=str))
            continue
        if command == "/tools":
            for tool in orchestrator.registry.describe_all():
                print(f"  {tool['name']}: {tool['description'][:110]}")
            continue
        try:
            print(render(orchestrator.run(query)))
        except Exception as exc:  # noqa: BLE001 - the REPL must survive any failure
            print(f"! Query failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Medical Agentic RAG CLI")
    parser.add_argument("-q", "--query", help="Run a single query and exit.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument("--log-level", default=None, help="Override LOG_LEVEL.")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=args.log_level or settings.log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    orchestrator = build_orchestrator(settings)
    if args.query:
        return run_once(orchestrator, args.query, args.json)
    return repl(orchestrator)


if __name__ == "__main__":
    sys.exit(main())
