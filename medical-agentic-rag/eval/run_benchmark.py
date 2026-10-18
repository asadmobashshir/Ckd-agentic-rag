"""Run the starter evaluation suite.

    python -m eval.run_benchmark [--offline] [--json out.json] [--case ID]

``--offline`` skips network-dependent cases and forces a failing web_search so
the tool-failure-recovery path is exercised deterministically.

The report measures software behaviour: tool routing, citation grounding,
abstention and safety escalation. It is not clinical validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from agent.orchestrator import Orchestrator, build_orchestrator
from agent.state import Evidence
from agent.tools.base import BaseTool
from config.settings import get_settings
from eval.metrics import CaseMetrics, aggregate, evaluate_case

logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).resolve().parent / "sample_questions.json"


class AlwaysFailingTool(BaseTool):
    """Stand-in that always fails, used to test failure recovery deterministically."""

    def __init__(self, wrapped: BaseTool) -> None:
        self._wrapped = wrapped
        self.name = wrapped.name  # type: ignore[misc]
        self.description = wrapped.description  # type: ignore[misc]
        self.args_model = wrapped.args_model  # type: ignore[misc]
        self.requires_network = False  # type: ignore[misc]

    def run(self, args) -> tuple[list[Evidence], dict[str, Any]]:  # noqa: ANN001
        raise RuntimeError("simulated tool outage")


def load_cases(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("cases", [])


def run_case(orchestrator: Orchestrator, case: dict[str, Any]) -> CaseMetrics:
    """Execute one case and score it."""
    simulate = case.get("simulate_tool_failure")
    original: BaseTool | None = None
    if simulate and orchestrator.registry.has(simulate):
        original = orchestrator.registry.get(simulate)
        orchestrator.registry.register(AlwaysFailingTool(original), override=True)

    try:
        response = orchestrator.run(case["query"])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Case %s crashed", case["id"])
        return CaseMetrics(
            case_id=case["id"], category=case.get("category", "?"), error=f"{type(exc).__name__}: {exc}"
        )
    finally:
        if original is not None:
            orchestrator.registry.register(original, override=True)

    # Re-derive evidence and grounding from the last audited record.
    records = orchestrator.audit.read_all()
    record = next((r for r in reversed(records) if r.get("request_id") == response.request_id), {})
    evidence = [
        Evidence(
            text="",
            source="audit",
            document_id=item.get("document_id"),
            chunk_id=item.get("chunk_id"),
            url=item.get("url"),
        )
        for item in record.get("evidence_sources", [])
    ]
    grounded = float((record.get("verification") or {}).get("grounded_claim_ratio", 0.0) or 0.0)
    had_failure = bool(simulate) or any(
        not call.get("ok") for call in record.get("tool_calls", [])
    )
    return evaluate_case(case, response, evidence, grounded, had_failure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation suite.")
    parser.add_argument("--offline", action="store_true", help="Skip network-dependent cases.")
    parser.add_argument("--json", type=Path, default=None, help="Write the report to a file.")
    parser.add_argument("--case", default=None, help="Run a single case by id.")
    args = parser.parse_args()

    settings = get_settings()
    # CRITICAL: the suite deliberately triggers tool outages; their tracebacks are
    # expected output, and the per-case report already records the failures.
    logging.basicConfig(level="CRITICAL", format="%(levelname)s %(name)s: %(message)s")

    cases = load_cases()
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"No case with id {args.case!r}")
            return 1
    if args.offline or not settings.enable_network_tools:
        cases = [c for c in cases if not c.get("requires_network")]

    orchestrator = build_orchestrator(settings)
    print("=" * 84)
    print("  Medical Agentic RAG - evaluation")
    print(f"  LLM: {'available' if orchestrator.llm.available else 'UNAVAILABLE (extractive mode)'}")
    print(f"  Cases: {len(cases)}")
    print("=" * 84)

    metrics: list[CaseMetrics] = []
    for case in cases:
        result = run_case(orchestrator, case)
        metrics.append(result)
        flag = "ERROR" if result.error else ("UNSAFE" if result.unsafe else "ok")
        print(
            f"[{flag:>6}] {result.case_id:<14} {result.category:<32} "
            f"tools={result.tools_used or '[]'} conf={result.confidence_level} "
            f"tool_sel={result.tool_selection:.1f} cite_valid={result.citation_validity:.1f}"
        )
        if result.error:
            print(f"          {result.error}")

    report = {
        "summary": aggregate(metrics),
        "cases": [m.as_dict() for m in metrics],
        "llm_available": orchestrator.llm.available,
    }

    print("-" * 84)
    print(json.dumps(report["summary"], indent=2))
    print("=" * 84)
    print("These are software-behaviour metrics. They are NOT clinical validation.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport written to {args.json}")

    unsafe = report["summary"]["safety"]["unsafe_answer_rate"] or 0.0
    return 1 if unsafe > 0 or report["summary"]["errored"] else 0


if __name__ == "__main__":
    sys.exit(main())
