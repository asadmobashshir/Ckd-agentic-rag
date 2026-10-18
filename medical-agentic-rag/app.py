"""Streamlit frontend for the CKD Agentic RAG demo.

This file is only the UI layer.
The existing Agentic RAG backend handles:
- planning
- retrieval
- synthesis
- verification
- confidence scoring
- safety/guardrails
- citations
"""

from __future__ import annotations

import logging

import streamlit as st

from agent.orchestrator import Orchestrator, build_orchestrator
from agent.state import FinalResponse
from config.settings import Settings, get_settings
from stores.vector_store import build_vector_store


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CKD Agentic RAG",
    page_icon="🩺",
    layout="wide",
)


logger = logging.getLogger(__name__)


# ============================================================
# TEXT
# ============================================================

BANNER = """
# 🩺 Chronic Kidney Disease Agentic RAG

**Evidence-grounded CKD research prototype**

> ⚠️ **Not medical advice. Demo data only. Always consult a qualified clinician.**

This assistant answers questions using the bundled synthetic CKD
knowledge corpus and structured drug data.
"""

EXAMPLES = [
    "What is CKD staging based on?",
    "Does lisinopril interact with ibuprofen?",
    "What are ACE inhibitors used for in CKD?",
    "How are SGLT2 inhibitors relevant to CKD?",
    "Why are NSAIDs a concern in CKD?",
    "What should I know about metformin and CKD?",
]

FRIENDLY_ERROR = (
    "Sorry, something went wrong while processing your question. "
    "Please try again."
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "metadata" not in st.session_state:
    st.session_state.metadata = (
        "Ask a question to see confidence, risk category and sources."
    )


# ============================================================
# VECTOR INDEX
# ============================================================

def ensure_index(settings: Settings) -> int:
    """Build the vector index if it is empty."""

    try:
        with build_vector_store(settings) as store:
            count = store.count()

        if count:
            logger.info(
                "Vector index already populated: %d vectors",
                count,
            )
            return count

    except Exception:
        logger.warning(
            "Could not read vector store. "
            "Attempting ingestion.",
            exc_info=True,
        )

    logger.info(
        "Vector index is empty. Running CKD corpus ingestion."
    )

    try:
        from ingestion.run_ingestion import run_ingestion

        report = run_ingestion(
            settings=settings,
            reset=False,
        )

        logger.info(
            "Ingestion complete: %d documents, %d chunks.",
            report.documents,
            report.chunks_indexed,
        )

        return report.vectors_in_store

    except Exception:
        logger.exception(
            "Startup ingestion failed."
        )

        return 0


# ============================================================
# ORCHESTRATOR
# ============================================================

@st.cache_resource
def get_app_orchestrator() -> Orchestrator:
    """Create the CKD Agentic RAG orchestrator once."""

    settings = get_settings()

    settings.ensure_directories()

    ensure_index(settings)

    return build_orchestrator(settings)


# ============================================================
# METADATA
# ============================================================

def format_metadata(response: FinalResponse) -> str:
    """Format response metadata for the sidebar."""

    lines = [
        "### Response Details",
        "",
        f"**Confidence:** {response.confidence.value}",
        "",
        f"**Score:** {response.confidence_score:.2f}",
        "",
        f"**Risk category:** {response.risk_category.value}",
        "",
    ]

    if response.tools_used:
        lines.extend(
            [
                "**Tools used:**",
                "",
                ", ".join(response.tools_used),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "**Tools used:** none",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "*Evidence confidence reflects retrieval quality "
            "and claim grounding only. It is not a probability "
            "that a medical statement is correct.*",
            "",
            "### Sources",
            "",
        ]
    )

    if not response.citations:

        lines.append(
            "No retrieved sources."
        )

    else:

        for citation in response.citations:

            label = (
                citation.title
                or citation.source
                or "Untitled source"
            )

            lines.append(
                f"- **[{citation.evidence_id}]** {label}"
            )

            lines.append(
                f"  - `data_status: {citation.data_status}`"
            )

            if citation.url:
                lines.append(
                    f"  - {citation.url}"
                )

    if response.warnings:

        lines.extend(
            [
                "",
                "### Warnings",
                "",
            ]
        )

        for warning in response.warnings:
            lines.append(
                f"- {warning}"
            )

    return "\n".join(lines)


# ============================================================
# ASK QUESTION
# ============================================================

def answer_query(query: str) -> tuple[str, str]:
    """Run a question through the CKD Agentic RAG pipeline."""

    query = (query or "").strip()

    if not query:
        return (
            "Please enter a question about chronic kidney disease.",
            "No question entered.",
        )

    try:
        orchestrator = get_app_orchestrator()

        logger.info("Starting orchestrator for query: %s", query)

        response = orchestrator.run(query)

        logger.info("Orchestrator completed successfully.")

        return (
            response.answer,
            format_metadata(response),
        )

    except Exception as exc:
        logger.exception("Orchestrator failed.")

        return (
            f"⚠️ The CKD pipeline could not complete the request.\n\n"
            f"**Error:** `{type(exc).__name__}: {exc}`",
            "The request failed before response metadata was generated.",
        )
# ============================================================
# HEADER
# ============================================================

st.markdown(BANNER)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🩺 CKD Agentic RAG")

    st.markdown(
        """
        **Pipeline**

        1. Query analysis
        2. Agent planning
        3. Evidence retrieval
        4. Evidence synthesis
        5. Verification
        6. Safety checks
        7. Final answer
        """
    )

    st.divider()

    st.subheader("Example Questions")

    for example in EXAMPLES:

        if st.button(
            example,
            use_container_width=True,
        ):

            st.session_state.selected_question = example

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.metadata = (
            "Ask a question to see confidence, "
            "risk category and sources."
        )

        st.rerun()


# ============================================================
# MAIN LAYOUT
# ============================================================

chat_column, details_column = st.columns(
    [3, 2]
)


# ============================================================
# CHAT
# ============================================================

with chat_column:

    st.subheader("Conversation")

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # Example question selected from sidebar
    selected_question = st.session_state.pop(
        "selected_question",
        None,
    )

    # Chat input
    user_question = st.chat_input(
        "Ask a question about chronic kidney disease..."
    )

    # Use sidebar example if selected
    query = selected_question or user_question

    if query:

        # --------------------------------------------------------
        # USER MESSAGE
        # --------------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.chat_message("user"):

            st.markdown(query)

        # --------------------------------------------------------
        # ASSISTANT RESPONSE
        # --------------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching CKD evidence and generating answer..."
            ):

                answer, metadata = answer_query(
                    query
                )

            st.markdown(answer)

        # --------------------------------------------------------
        # SAVE METADATA
        # --------------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.session_state.metadata = metadata

        st.rerun()


# ============================================================
# RESPONSE DETAILS
# ============================================================

with details_column:

    st.subheader("Response Details")

    st.markdown(
        st.session_state.metadata
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research prototype. The bundled corpus and drug records "
    "are synthetic demo data and are not validated medical "
    "sources. This system does not diagnose, does not recommend "
    "doses, and is not a substitute for a qualified clinician."
)
