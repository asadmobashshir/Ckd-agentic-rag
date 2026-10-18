"""Provider-agnostic LLM interface.

Gemini is the only fully implemented provider (see :mod:`llm.gemini_client`). This
module defines the contract another provider would have to satisfy, plus the
JSON-repair helper shared by the planner and the verifier.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field


class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMUnavailableError(LLMError):
    """No usable LLM backend (missing credentials, SDK, or network)."""


class LLMTimeoutError(LLMError):
    """The provider did not respond within the configured timeout."""


class LLMResponseError(LLMError):
    """The provider responded, but the payload was unusable."""


#: Machine-readable reasons the LLM can be unavailable. The UI shows these
#: verbatim; none of them can contain a key, because they are fixed strings.
LLM_STATUS_MISSING_KEY = "API key missing"
LLM_STATUS_SDK_MISSING = "Provider SDK not installed"
LLM_STATUS_AUTH = "Authentication failed (invalid or revoked API key)"
LLM_STATUS_MODEL = "Configured model unavailable or decommissioned"
LLM_STATUS_RATE_LIMIT = "Rate limit or quota reached"
LLM_STATUS_NETWORK = "Network error contacting the provider"
LLM_STATUS_UNKNOWN = "Provider call failed"
LLM_STATUS_OK = "OK"

#: Substrings that reliably indicate the *model id* was rejected. Deliberately
#: specific: a bare "404" also appears in unrelated transport errors, and
#: classifying those as "model decommissioned" sent earlier debugging down the
#: wrong path entirely.
_MODEL_ERROR_MARKERS = (
    "model_not_found",
    "model not found",
    "does not exist",
    "decommission",
    "is not supported",
    "unknown model",
    "invalid model",
    "no such model",
    "not found for api version",
)


def classify_llm_error(error: BaseException | str) -> str:
    """Map a provider error onto one of the fixed status strings above.

    Returning a fixed string rather than the raw exception is deliberate: it
    guarantees no request payload, header or key fragment can reach the UI.
    """
    text = str(error).lower()
    if "api_key" in text or "api key" in text:
        if "missing" in text or "not set" in text or "no api" in text:
            return LLM_STATUS_MISSING_KEY
    if (
        "package not installed" in text
        or "sdk not installed" in text
        or "no module named 'google'" in text
        or "no module named 'google.genai'" in text
        or "no module named 'groq'" in text
    ):
        return LLM_STATUS_SDK_MISSING
    if any(
        m in text
        for m in ("authentication", "unauthorized", "401", "invalid api key",
                  "invalid_api_key", "permission_denied", "api key not valid")
    ):
        return LLM_STATUS_AUTH
    if any(m in text for m in ("rate limit", "rate_limit", "429", "quota", "resource exhausted")):
        return LLM_STATUS_RATE_LIMIT
    if any(m in text for m in _MODEL_ERROR_MARKERS):
        return LLM_STATUS_MODEL
    if any(
        m in text
        for m in ("connection", "timeout", "timed out", "network", "dns",
                  "unreachable", "503", "502", "504")
    ):
        return LLM_STATUS_NETWORK
    return LLM_STATUS_UNKNOWN


def is_model_error(error: BaseException | str) -> bool:
    """Whether ``error`` specifically indicates a rejected model id."""
    return classify_llm_error(error) == LLM_STATUS_MODEL


class LLMToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = ""
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)


class BaseLLMClient(ABC):
    """Minimal chat interface used by the planner, synthesizer and verifier."""

    provider: str = "base"
    supports_tool_calling: bool = False

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this client can currently serve requests."""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request."""

    def chat_json(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Chat and parse a JSON object out of the reply.

        Raises :class:`LLMResponseError` when no JSON object can be recovered.
        """
        response = self.chat(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=True
        )
        parsed = extract_json_object(response.text)
        if parsed is None:
            raise LLMResponseError(
                f"Model did not return a JSON object (first 200 chars: {response.text[:200]!r})"
            )
        return parsed

    def list_models(self) -> list[str]:
        """Model ids this API key can actually reach.

        Returns ``[]`` when the provider cannot be queried. This exists so the
        app can *verify* a model id against the live account rather than trusting
        a name copied from documentation.
        """
        return []

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "available": self.available,
            "tool_calling": self.supports_tool_calling,
        }


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort recovery of a JSON object from a model reply.

    Handles bare JSON, fenced blocks, and prose wrapped around an object. Returns
    ``None`` when nothing parseable is found - callers must handle that rather
    than fabricating a result.
    """
    if not text:
        return None
    candidates: list[str] = [text.strip()]
    candidates.extend(match.strip() for match in _FENCE_RE.findall(text))

    depth, start = 0, None
    in_string, escape = False, False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : i + 1])
                start = None

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class UnavailableLLMClient(BaseLLMClient):
    """Stand-in used when no LLM backend can be reached.

    Every call raises :class:`LLMUnavailableError` so callers fall back to the
    deterministic, non-generative path instead of fabricating an answer.
    """

    provider = "unavailable"

    def __init__(self, reason: str = "No LLM backend configured") -> None:
        self.reason = reason

    @property
    def available(self) -> bool:
        return False

    def chat(self, messages, **kwargs) -> LLMResponse:  # noqa: ANN001, D102
        raise LLMUnavailableError(self.reason)

    def health(self) -> dict[str, Any]:
        info = super().health()
        info["reason"] = self.reason
        return info
