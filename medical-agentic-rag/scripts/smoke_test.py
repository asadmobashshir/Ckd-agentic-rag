"""Offline smoke test.

    python scripts/smoke_test.py

Checks each layer independently and prints a PASS/FAIL summary. Everything
except the final LLM connectivity probe runs without network access or an API
key; the provider check is SKIPPED (not failed) when no API key is present.

Exit code 0 when no check FAILED.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Allow `python scripts/smoke_test.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    duration_ms: float = 0.0


def run_check(name: str, fn: Callable[[], tuple[str, str]]) -> CheckResult:
    started = time.perf_counter()
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a failing check is data, not a crash
        status, detail = FAIL, f"{type(exc).__name__}: {exc}"
        logging.debug("Check %s raised\n%s", name, traceback.format_exc())
    return CheckResult(name, status, detail, round((time.perf_counter() - started) * 1000, 1))


# --------------------------------------------------------------------- checks
def check_imports() -> tuple[str, str]:
    import agent.orchestrator  # noqa: F401
    import agent.planner  # noqa: F401
    import agent.synthesizer  # noqa: F401
    import agent.tool_loop  # noqa: F401
    import audit.audit_logger  # noqa: F401
    import ingestion.run_ingestion  # noqa: F401
    import llm.factory  # noqa: F401
    import safety.guardrails  # noqa: F401
    import stores.vector_store  # noqa: F401

    # The FastAPI server is an optional local extra (requirements-full.txt); the
    # Streamlit deployment deliberately does not install it, so a missing
    # fastapi must not read as a broken build.
    try:
        import api.server  # noqa: F401
    except ModuleNotFoundError as exc:
        return PASS, f"core modules import cleanly (optional API skipped: {exc.name})"

    return PASS, "all core modules import cleanly"


def check_configuration() -> tuple[str, str]:
    from config.settings import get_settings

    settings = get_settings()
    settings.ensure_directories()
    missing = [p for p in (settings.raw_dir, settings.processed_dir) if not p.exists()]
    if missing:
        return FAIL, f"directories missing: {missing}"
    return PASS, (
        f"provider={settings.active_provider}, model={settings.active_model}, "
        f"top_k={settings.top_k}, "
        f"max_tool_rounds={settings.max_tool_rounds}, "
        f"api_key={'set' if settings.has_llm_credentials else 'NOT set'}"
    )


def check_structured_store() -> tuple[str, str]:
    from config.settings import get_settings
    from stores.structured_store import build_structured_store

    store = build_structured_store(get_settings())
    stats = store.stats()
    if not any(stats.values()):
        try:
            stats = store.seed_from_file()
        except Exception as exc:  # noqa: BLE001
            return FAIL, f"database empty and seeding failed: {exc}"
    row = store.find_interaction("lisinopril", "ibuprofen")
    if row is None:
        return FAIL, f"demo interaction row missing (stats={stats})"
    return PASS, f"{stats}; CKD demo lookup (lisinopril+ibuprofen) severity={row.get('severity')}"


def check_embedder() -> tuple[str, str]:
    from config.settings import get_settings
    from ingestion.embedder import build_embedder

    embedder = build_embedder(get_settings())
    vector = embedder.embed_query("chronic kidney disease staging eGFR")
    if len(vector) != embedder.dimension:
        return FAIL, f"dimension mismatch: {len(vector)} != {embedder.dimension}"
    info = embedder.describe()
    if not info.get("semantic"):
        return PASS, (
            f"{info} - WARNING: offline lexical fallback in use. "
            "Install requirements-embeddings.txt for semantic retrieval."
        )
    return PASS, str(info)


def check_vector_store() -> tuple[str, str]:
    from config.settings import get_settings
    from stores.vector_store import build_vector_store

    with build_vector_store(get_settings()) as store:
        count = store.count()
    if count == 0:
        return FAIL, "vector index is empty - run: python -m ingestion.run_ingestion"
    return PASS, f"backend={store.backend}, vectors={count}"


def check_tool_registry() -> tuple[str, str]:
    from agent.state import ToolCallRequest
    from agent.tools.tool_registry import build_default_registry

    registry = build_default_registry()
    expected = {"vector_search", "drug_interaction", "dosage_lookup", "literature_search", "web_search"}
    missing = expected - set(registry.names)
    if missing:
        return FAIL, f"missing tools: {sorted(missing)}"
    # An unregistered tool must be rejected rather than executed.
    if registry.validate_call(ToolCallRequest(name="rm_rf", arguments={})) is None:
        return FAIL, "registry accepted an unregistered tool name"
    # Invalid arguments must be rejected before execution.
    if registry.validate_call(ToolCallRequest(name="drug_interaction", arguments={"drug_a": "x"})) is None:
        return FAIL, "registry accepted invalid arguments"
    return PASS, f"{len(registry)} tools registered; unknown names and bad args rejected"


def check_sample_retrieval() -> tuple[str, str]:
    from agent.state import ToolCallRequest
    from agent.tools.tool_registry import build_default_registry

    registry = build_default_registry()
    results = registry.execute_sync(
        [ToolCallRequest(name="vector_search", arguments={"query": "chronic kidney disease staging eGFR"})]
    )
    result = results[0]
    if not result.ok:
        return FAIL, f"vector_search failed: {result.error}"
    if not result.evidence:
        return FAIL, f"no evidence retrieved (structured={result.structured})"
    top = result.evidence[0]
    if not top.chunk_id or not top.document_id:
        return FAIL, "retrieved evidence is missing provenance identifiers"
    return PASS, f"{len(result.evidence)} chunk(s); top score={top.score}, source={top.source!r}"


def check_guardrails() -> tuple[str, str]:
    from agent.state import RiskCategory
    from safety.guardrails import GuardrailEngine

    engine = GuardrailEngine()
    cases = {
        "What is CKD staging based on?": RiskCategory.GENERAL_INFORMATION,
        "I have kidney disease, how much ibuprofen should I take?": RiskCategory.PERSONALIZED_CLINICAL_DECISION,
        "I have crushing chest pain going down my left arm": RiskCategory.EMERGENCY,
    }
    for query, expected in cases.items():
        actual = engine.pre_check(query).risk_category
        if actual is not expected:
            return FAIL, f"{query!r} -> {actual.value}, expected {expected.value}"
    return PASS, "general / personalised / emergency classification correct"


def check_pipeline_offline() -> tuple[str, str]:
    from agent.orchestrator import build_orchestrator

    orchestrator = build_orchestrator()
    response = orchestrator.run("Does lisinopril interact with ibuprofen?")
    if not response.answer.strip():
        return FAIL, "pipeline returned an empty answer"
    if "drug_interaction" not in response.tools_used:
        return FAIL, f"expected drug_interaction to run, got {response.tools_used}"
    cited = {c.evidence_id for c in response.citations}
    if cited and not cited.issubset({f"S{i}" for i in range(1, 40)}):
        return FAIL, f"malformed citation ids: {cited}"
    return PASS, (
        f"tools={response.tools_used}, confidence={response.confidence.value}, "
        f"citations={len(response.citations)}"
    )


def check_audit_log() -> tuple[str, str]:
    from audit.audit_logger import sanitize_arguments, sanitize_text

    redacted = sanitize_text("Contact me at jane.doe@example.com or +1 415 555 0199, I am 47 years old")
    if "@" in redacted or "555" in redacted:
        return FAIL, f"redaction incomplete: {redacted}"
    args = sanitize_arguments({"query": "lisinopril", "api_key": "sk-secret", "note": "x"})
    if "api_key" in args or "note" in args:
        return FAIL, f"argument filtering leaked keys: {list(args)}"
    return PASS, "PII redaction and argument whitelisting behave correctly"


def check_domain_scope() -> tuple[str, str]:
    from agent.domain import DomainVerdict, classify_domain

    in_scope = ["What is CKD staging based on?", "Does lisinopril interact with ibuprofen?"]
    out_of_scope = ["How is migraine treated?", "What are the symptoms of asthma?"]
    for query in in_scope:
        if classify_domain(query) is DomainVerdict.OUT_OF_SCOPE:
            return FAIL, f"CKD query wrongly refused: {query!r}"
    for query in out_of_scope:
        if classify_domain(query) is not DomainVerdict.OUT_OF_SCOPE:
            return FAIL, f"unrelated query not refused: {query!r}"
    return PASS, "CKD queries admitted; unrelated queries refused"


def check_out_of_scope_pipeline() -> tuple[str, str]:
    from agent.orchestrator import build_orchestrator

    response = build_orchestrator().run("How is migraine treated?")
    if response.tools_used:
        return FAIL, f"out-of-scope query triggered tools: {response.tools_used}"
    if response.status != "out_of_scope":
        return FAIL, f"unexpected status: {response.status}"
    if "outside" not in response.answer.lower():
        return FAIL, "answer did not state that the query is out of scope"
    return PASS, "zero tools, no citations, explicit CKD scope message"


def check_gemini_connectivity() -> tuple[str, str]:
    from config.settings import get_settings
    from llm.gemini_client import GeminiClient

    settings = get_settings()
    if not settings.has_llm_credentials:
        return SKIP, f"{settings.active_key_name} not set - offline mode (extractive answers only)"
    client = GeminiClient(settings)
    ok, detail = client.ping()
    return (PASS if ok else FAIL), f"model={client.model}: {detail}"


CHECKS: list[tuple[str, Callable[[], tuple[str, str]]]] = [
    ("1. imports", check_imports),
    ("2. configuration", check_configuration),
    ("3. structured store (SQLite)", check_structured_store),
    ("4. embedding model", check_embedder),
    ("5. vector store", check_vector_store),
    ("6. tool registry", check_tool_registry),
    ("7. sample retrieval", check_sample_retrieval),
    ("8. guardrails", check_guardrails),
    ("9. end-to-end pipeline", check_pipeline_offline),
    ("10. audit redaction", check_audit_log),
    ("11. CKD domain scope", check_domain_scope),
    ("12. out-of-scope pipeline", check_out_of_scope_pipeline),
    ("13. LLM connectivity", check_gemini_connectivity),
]


def main() -> int:
    logging.basicConfig(level="ERROR", format="%(levelname)s %(name)s: %(message)s")
    print("=" * 78)
    print("  Medical Agentic RAG - smoke test")
    print("=" * 78)

    results = []
    for name, fn in CHECKS:
        result = run_check(name, fn)
        results.append(result)
        marker = {PASS: "[PASS]", FAIL: "[FAIL]", SKIP: "[SKIP]"}[result.status]
        print(f"{marker} {name:<32} ({result.duration_ms:>7.1f} ms)")
        if result.detail:
            print(f"        {result.detail}")

    passed = sum(1 for r in results if r.status == PASS)
    failed = sum(1 for r in results if r.status == FAIL)
    skipped = sum(1 for r in results if r.status == SKIP)

    print("-" * 78)
    print(f"  {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 78)
    if failed:
        print("\nFailing checks:")
        for r in results:
            if r.status == FAIL:
                print(f"  - {r.name}: {r.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
