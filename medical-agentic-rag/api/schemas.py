"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.state import Citation, ConfidenceLevel, FinalResponse, RiskCategory


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1500, description="The user's question.")


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    answer: str
    confidence: ConfidenceLevel
    confidence_score: float = 0.0
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    risk_category: RiskCategory = RiskCategory.GENERAL_INFORMATION
    status: str = "ok"
    disclaimer: str = ""

    @classmethod
    def from_final(cls, response: FinalResponse) -> "QueryResponse":
        return cls(**response.model_dump())


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    components: dict[str, Any] = Field(default_factory=dict)


class ToolInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_schema: dict[str, Any] = Field(default_factory=dict)
    requires_network: bool = False
    produces_evidence: bool = True


class ToolsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    tools: list[ToolInfo] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str
    request_id: str | None = None
