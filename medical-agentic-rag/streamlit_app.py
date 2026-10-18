"""Streamlit entry point for the CKD Medical Agentic RAG prototype.

    streamlit run medical-agentic-rag/streamlit_app.py

This is a **view layer only**. It calls the existing
:class:`~agent.orchestrator.Orchestrator` and renders what comes back; no
planning, retrieval, synthesis, verification, confidence scoring or citation
logic lives here.

Initialisation order matters and is enforced below:

1. import streamlit
2. read ``st.secrets``
3. export them as environment variables
4. clear the cached ``Settings`` (``reload_settings()``)
5. import the backend
6. build the orchestrator

Building ``Settings`` before step 3 produces an object with no Gemini key that is
then cached for the life of the process - the original cause of the
"No language model is configured" message.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# --------------------------------------------------------------------------
# 1. Project path
# --------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 2-3. Streamlit secrets -> environment, BEFORE any Settings object exists
# --------------------------------------------------------------------------
SECRET_KEYS = (
    # Provider selection: "auto" (default), "groq" or "gemini".
    "LLM_PROVIDER",
    # Groq
    "GROQ_API_KEY", "GROQ_MODEL",
    # Gemini
    "GEMINI_API_KEY", "GEMINI_MODEL",
    # Shared generation parameters (legacy GEMINI_* names still accepted)
    "LLM_TEMPERATURE", "LLM_MAX_TOKENS",
    "GEMINI_TEMPERATURE", "GEMINI_MAX_TOKENS",
    # Retrieval / agent
    "EMBEDDING_BACKEND", "EMBEDDING_MODEL", "TOP_K", "MIN_SIMILARITY",
    "MAX_TOOL_ROUNDS", "MAX_AGENT_ITERATIONS", "ENABLE_NATIVE_TOOL_CALLING",
    "LOG_LEVEL", "ENABLE_NETWORK_TOOLS",
    "NCBI_EMAIL", "NCBI_API_KEY",
)


def load_secrets_into_env() -> list[str]:
    """Copy known secrets into ``os.environ``. Returns the names that were set.

    Only names are returned and logged - never values. ``st.secrets`` raises when
    no secrets file exists (normal for local runs), so that case is handled
    explicitly rather than swallowed.
    """
    loaded: list[str] = []
    try:
        available = set(st.secrets.keys())
    except Exception:  # noqa: BLE001 - no secrets.toml locally; not an error
        logger.info("No Streamlit secrets found; falling back to .env / environment.")
        return loaded

    for key in SECRET_KEYS:
        if key in available:
            value = st.secrets[key]
            if value is not None and str(value).strip():
                os.environ[key] = str(value)
                loaded.append(key)
    return loaded


LOADED_SECRETS = load_secrets_into_env()

# --------------------------------------------------------------------------
# 4-5. Clear any cached Settings, then import the backend
# --------------------------------------------------------------------------
from config.settings import Settings, reload_settings  # noqa: E402

SETTINGS: Settings = reload_settings()

from agent.orchestrator import build_orchestrator  # noqa: E402
from scripts.diagnostics import (  # noqa: E402
    check_api_key,
    check_embedder,
    check_llm_client,
    check_llm_response,
    check_vector_store,
)
from stores.vector_store import build_vector_store  # noqa: E402

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="CKD Medical Agentic RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 40px; font-weight: 700; margin-bottom: 4px; }
    .subtitle { font-size: 17px; color: #666; margin-bottom: 20px; }
    .warning-box {
        padding: 15px; border-radius: 10px; background-color: #fff4e5;
        border-left: 5px solid #ff9800; margin-bottom: 20px;
    }
    .source-box {
        padding: 10px 12px; border-radius: 8px; background-color: #f5f7fa;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BANNER = """
<div class="warning-box">
<b>🩺 Chronic Kidney Disease Medical Agentic RAG</b>
<br><br>
<strong>⚠️ Research prototype — NOT medical advice.</strong>
<br><br>
This system provides evidence-grounded information from a small CKD demo
knowledge base. It does not diagnose disease, interpret test results, or
prescribe treatment. Always consult a qualified healthcare professional.
</div>
"""

EXAMPLES = [
    "What are the common symptoms of CKD?",
    "What is CKD staging based on?",
    "What is eGFR?",
    "What tests are used to evaluate kidney function?",
    "Why are NSAIDs a concern in CKD?",
    "Does lisinopril interact with ibuprofen?",
    "What is the triple whammy?",
    "How are SGLT2 inhibitors relevant to CKD?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------------------------------
# 6-9. Index, then orchestrator
# --------------------------------------------------------------------------
def ensure_index(settings: Settings) -> int:
    """Build the index on first run, and rebuild it if the embedder changed.

    ``data/processed/`` is gitignored, so a fresh deployment has no index. An
    index built by a *different* embedder is worse than none - dimensions can
    match while the vector spaces are unrelated - so that case forces a rebuild.
    """
    from ingestion.embedder import build_embedder
    from stores.index_meta import check_compatibility

    count = 0
    try:
        with build_vector_store(settings) as store:
            count = store.count()
    except Exception:  # noqa: BLE001
        logger.exception("Could not open the vector store; will attempt ingestion.")

    needs_rebuild = False
    if count:
        try:
            compatible, message = check_compatibility(
                settings.vector_db_path, build_embedder(settings)
            )
            if not compatible:
                logger.warning("%s Rebuilding.", message)
                needs_rebuild = True
        except Exception:  # noqa: BLE001
            logger.exception("Index compatibility check failed; rebuilding.")
            needs_rebuild = True
        if not needs_rebuild:
            return count

    try:
        from ingestion.run_ingestion import run_ingestion

        report = run_ingestion(settings=settings, reset=needs_rebuild)
        return report.vectors_in_store
    except Exception:  # noqa: BLE001
        logger.exception("Startup ingestion failed.")
        return 0


@st.cache_resource(show_spinner="Initializing CKD Agentic RAG…")
def get_orchestrator() -> tuple[Any, int]:
    settings = reload_settings()
    settings.ensure_directories()
    vector_count = ensure_index(settings)
    return build_orchestrator(settings), vector_count


@st.cache_data(ttl=300, show_spinner=False)
def get_system_status() -> list[dict[str, Any]]:
    """Component status for the sidebar. Cached so Gemini is not pinged per rerun."""
    settings = reload_settings()
    checks = [
        check_api_key(settings),
        check_llm_client(settings),
        check_llm_response(settings),
        check_embedder(settings),
        check_vector_store(settings),
    ]
    return [c.as_dict() for c in checks]


def run_query(question: str):
    orchestrator, _ = get_orchestrator()
    return orchestrator.run(question)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="main-title">🩺 CKD Medical Agentic RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Evidence-grounded Chronic Kidney Disease information assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(BANNER, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar: examples + system status
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Example Questions")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.selected_question = example

    st.divider()
    st.header("⚙️ System Status")

    try:
        _, vector_count = get_orchestrator()
        st.success("Agentic RAG: OK")
    except Exception as exc:  # noqa: BLE001
        vector_count = 0
        st.error("Agentic RAG: initialisation failed")
        st.caption(f"{type(exc).__name__}: {exc}")

    try:
        status = get_system_status()
        icons = {"OK": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "➖"}

        llm_ok = any(c["name"] == "LLM response" and c["status"] == "OK" for c in status)
        llm_client = next((c for c in status if c["name"] == "LLM client"), {})

        provider_label = {"groq": "Groq", "gemini": "Google Gemini"}.get(
            SETTINGS.active_provider, SETTINGS.active_provider
        )
        st.write(f"**LLM provider:** {provider_label}")
        st.write(f"**LLM model:** `{SETTINGS.active_model}`")
        st.write(f"**LLM configured:** {'YES' if llm_ok else 'NO'}")
        st.write(
            f"**Native tool calling:** "
            f"{'ON' if llm_client.get('tool_calling') else 'OFF'}"
        )

        embed = next((c for c in status if c["name"] == "Embeddings"), {})
        st.write(f"**Embedding backend:** {embed.get('backend', 'unknown')}")
        if embed.get("model"):
            st.write(f"**Embedding model:** `{embed['model']}`")

        vec = next((c for c in status if c["name"] == "Vector store"), {})
        st.write(f"**Vector store:** {vec.get('backend', 'unknown')}")
        st.write(f"**Indexed vectors:** {vector_count or vec.get('vectors', 0)}")

        with st.expander("Diagnostics detail"):
            for check in status:
                st.markdown(
                    f"{icons.get(check['status'], '•')} **{check['name']}** — {check['detail']}"
                )
            if LOADED_SECRETS:
                # Names only. Values are never rendered or logged.
                st.caption("Loaded from Streamlit secrets: " + ", ".join(LOADED_SECRETS))

        if not llm_ok:
            reason = llm_client.get("detail") or "see Diagnostics detail"
            st.warning(f"Answer synthesis disabled — {reason}")
    except Exception as exc:  # noqa: BLE001
        st.error("Status check failed.")
        st.caption(f"{type(exc).__name__}: {exc}")

    st.divider()
    st.caption("Research prototype. Demo CKD corpus only. Not medical advice.")


# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("💬 Ask the CKD Agent")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    selected_question = st.session_state.pop("selected_question", None)
    question = st.chat_input("Ask a question about Chronic Kidney Disease…")
    if selected_question:
        question = selected_question

    if question and question.strip():
        question = question.strip()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Agent is planning, retrieving and verifying evidence…"):
                try:
                    response = run_query(question)
                    answer = response.answer
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_response = response
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Agentic RAG query failed")
                    # Type only - never the message, which could echo a payload.
                    error_message = (
                        "Sorry, something went wrong while processing your question. "
                        f"(`{type(exc).__name__}` — see the Diagnostics panel.)"
                    )
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )


with right:
    st.subheader("📊 Response Details")

    if "last_response" not in st.session_state:
        st.info("Ask a question to see confidence, risk category, tools and sources.")
    else:
        response = st.session_state.last_response

        col_a, col_b = st.columns(2)
        col_a.metric("Confidence", response.confidence.value)
        col_b.metric("Evidence Score", f"{response.confidence_score:.2f}")
        st.caption(
            "Evidence confidence reflects retrieval quality and claim grounding — "
            "not the probability that a medical statement is correct."
        )

        st.write("**Risk Category**")
        st.info(response.risk_category.value)

        st.write("**Tools Used**")
        if response.tools_used:
            st.write(" · ".join(f"`{tool}`" for tool in response.tools_used))
            st.caption("Tools that actually executed, not tools merely suggested.")
        else:
            st.write("None (no retrieval was required)")

        if response.agent_iterations:
            st.write("**Agent Iterations**")
            st.write(
                f"{response.agent_iterations} model turn(s) in the tool-calling loop"
            )
        source_label = {
            "agent_loop": "Agent loop (model answered from observed tool output)",
            "synthesizer": "Synthesizer (plan-then-execute path)",
            "extractive": "Extractive fallback (no LLM reachable)",
        }.get(response.answer_source)
        if source_label:
            st.caption(f"Answer source: {source_label}")

        st.write("### 📚 Sources")
        if response.citations:
            for citation in response.citations:
                title = citation.title or citation.source or "Untitled source"
                st.markdown(
                    f'<div class="source-box"><b>[{citation.evidence_id}] {title}</b><br>'
                    f"<small>data_status: {citation.data_status}</small></div>",
                    unsafe_allow_html=True,
                )
                if citation.url:
                    st.markdown(f"[View source]({citation.url})")
        else:
            st.write("No retrieved sources.")

        if response.warnings:
            st.write("### ⚠️ Warnings")
            for warning in response.warnings:
                st.warning(warning)


st.divider()
st.caption(
    "CKD Medical Agentic RAG — research prototype. This system does not diagnose or "
    "prescribe treatment. Always consult a qualified healthcare professional."
)
