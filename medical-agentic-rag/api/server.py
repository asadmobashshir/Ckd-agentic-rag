"""FastAPI application.

    uvicorn api.server:app --reload

Endpoints: ``POST /query``, ``GET /health``, ``GET /tools``. The route handlers
own no domain logic - they call the same :class:`~agent.orchestrator.Orchestrator`
the CLI uses.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from agent.orchestrator import Orchestrator, build_orchestrator
from api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ToolInfo,
    ToolsResponse,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)

_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    """Lazily construct the shared orchestrator (models load once)."""
    global _orchestrator  # noqa: PLW0603 - single process-wide instance by design
    if _orchestrator is None:
        _orchestrator = build_orchestrator()
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.info("Starting Medical Agentic RAG API")
    get_orchestrator()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Medical Agentic RAG",
    version="0.1.0",
    description=(
        "Research prototype: an evidence-grounded medical information assistant. "
        "NOT a medical device, NOT for diagnosis or treatment decisions."
    ),
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic error; never leak internals or a stack trace."""
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error. See server logs for details."},
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest) -> QueryResponse:
    """Run the full agentic pipeline for one question."""
    try:
        result = await get_orchestrator().arun(payload.query)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query processing failed.") from exc
    return QueryResponse.from_final(result)


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Component-level health, including whether the vector index is populated."""
    components = get_orchestrator().health()
    vector_ok = bool(components.get("vector_store", {}).get("ok"))
    indexed = int(components.get("vector_store", {}).get("count", 0) or 0)
    status = "ok" if vector_ok and indexed else "degraded"
    return HealthResponse(status=status, components=components)


@app.get("/tools", response_model=ToolsResponse)
async def tools_endpoint() -> ToolsResponse:
    """List the registered tools and their argument schemas."""
    described = get_orchestrator().registry.describe_all()
    return ToolsResponse(count=len(described), tools=[ToolInfo(**item) for item in described])
