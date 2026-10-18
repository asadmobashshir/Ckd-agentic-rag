"""Google Gemini client with native function calling.

Previously this adapter hard-disabled tool calling and raised whenever tool
specs were passed, which forced the planner down a JSON-only path and meant tool
results were never returned to the model. It now implements the real protocol:

* declarations go out as ``types.Tool(function_declarations=[...])``
* the model replies with ``part.function_call``
* executed results go back as ``types.Part.from_function_response(...)``
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Sequence

from config.settings import Settings, get_settings
from llm.base import (
    LLM_STATUS_MISSING_KEY,
    LLM_STATUS_OK,
    LLM_STATUS_SDK_MISSING,
    BaseLLMClient,
    LLMError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    LLMToolCall,
    LLMUnavailableError,
    classify_llm_error,
    is_model_error,
)

logger = logging.getLogger(__name__)

_RETRYABLE_MARKERS = (
    "rate limit", "resource exhausted", "429", "timeout", "timed out",
    "temporarily", "503", "502", "504", "overloaded", "connection",
    "internal server error",
)

#: JSON-Schema keys Gemini's FunctionDeclaration parser rejects.
_SCHEMA_STRIP_KEYS = {
    "additionalProperties", "$schema", "title", "default", "examples",
    "definitions", "$defs", "allOf", "not", "const", "format",
}


def _sanitize_schema(node: Any) -> Any:
    """Reduce a Pydantic JSON Schema to the subset Gemini accepts.

    Pydantic emits ``title``/``additionalProperties`` and encodes optional fields
    as ``anyOf: [T, null]``; Gemini rejects both. Dropping them here keeps one
    schema definition shared by every provider.
    """
    if isinstance(node, list):
        return [_sanitize_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    # Collapse Optional[T] (anyOf [T, null]) down to T, marked nullable.
    if "anyOf" in node:
        variants = [v for v in node["anyOf"] if isinstance(v, dict)]
        non_null = [v for v in variants if v.get("type") != "null"]
        chosen = dict(non_null[0]) if non_null else {"type": "string"}
        if len(non_null) < len(variants):
            chosen["nullable"] = True
        for key in ("description",):
            if key in node and key not in chosen:
                chosen[key] = node[key]
        return _sanitize_schema(chosen)

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in _SCHEMA_STRIP_KEYS:
            continue
        cleaned[key] = _sanitize_schema(value)
    if cleaned.get("type") == "object" and "properties" not in cleaned:
        cleaned["properties"] = {}
    return cleaned


class GeminiClient(BaseLLMClient):
    """Gemini provider. Supports native function calling."""

    provider = "gemini"
    supports_tool_calling = True

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: Any | None = None
        self._init_error: str | None = None
        self.model = self._settings.gemini_model
        self._dead_models: set[str] = set()
        self._initialise()

    def _initialise(self) -> None:
        key = self._settings.gemini_api_key
        if not (key and key.strip()):
            self._init_error = f"{LLM_STATUS_MISSING_KEY}: set GEMINI_API_KEY"
            return
        try:
            from google import genai  # noqa: PLC0415
        except ImportError as exc:
            self._init_error = f"{LLM_STATUS_SDK_MISSING}: google-genai ({exc})"
            return
        try:
            self._client = genai.Client(api_key=key.strip())
        except Exception as exc:  # noqa: BLE001
            self._init_error = classify_llm_error(exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------- discovery
    def list_models(self) -> list[str]:
        """Model ids this key can reach, queried live from the provider."""
        if self._client is None:
            return []
        try:
            names: list[str] = []
            for model in self._client.models.list():
                name = str(getattr(model, "name", "") or "")
                if name.startswith("models/"):
                    name = name[len("models/"):]
                if name:
                    names.append(name)
            return sorted(set(names))
        except Exception as exc:  # noqa: BLE001
            logger.info("Gemini model listing failed: %s", classify_llm_error(exc))
            return []

    def _candidate_models(self) -> list[str]:
        ordered = [self.model] + [
            m for m in self._settings.gemini_model_fallbacks if m != self.model
        ]
        return [m for m in ordered if m not in self._dead_models]

    # --------------------------------------------------------- message prep
    @staticmethod
    def _convert_messages(messages: Sequence[dict[str, Any]]):
        """Translate provider-neutral messages into Gemini contents.

        Handles plain text, assistant turns carrying tool calls, and tool-result
        turns, so a full tool-calling conversation survives the round trip.
        """
        from google.genai import types  # noqa: PLC0415

        system_parts: list[str] = []
        contents: list[Any] = []

        for message in messages:
            role = str(message.get("role", "user")).lower()
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content
                )
            content = str(content or "")

            if role == "system":
                if content:
                    system_parts.append(content)
                continue

            if role == "tool":
                name = str(message.get("name") or message.get("tool_call_id") or "tool")
                try:
                    payload = json.loads(content) if content else {}
                except json.JSONDecodeError:
                    payload = {"result": content}
                if not isinstance(payload, dict):
                    payload = {"result": payload}
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=name, response=payload)],
                ))
                continue

            if role in {"assistant", "model"} and message.get("tool_calls"):
                parts = []
                if content:
                    parts.append(types.Part(text=content))
                for call in message["tool_calls"]:
                    fn = call.get("function", {}) if isinstance(call, dict) else {}
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    parts.append(types.Part(
                        function_call=types.FunctionCall(name=fn.get("name", ""), args=args)
                    ))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                continue

            gemini_role = "model" if role in {"assistant", "model"} else "user"
            if content:
                contents.append(types.Content(role=gemini_role, parts=[types.Part(text=content)]))

        if not contents:
            contents.append(types.Content(role="user", parts=[types.Part(text="Please respond.")]))
        return "\n\n".join(system_parts) or None, contents

    @staticmethod
    def _build_tools(tools: Sequence[dict[str, Any]]):
        """Convert OpenAI-style function specs into Gemini declarations."""
        from google.genai import types  # noqa: PLC0415

        declarations = []
        for spec in tools:
            fn = spec.get("function", spec) if isinstance(spec, dict) else {}
            name = fn.get("name")
            if not name:
                continue
            declarations.append(types.FunctionDeclaration(
                name=name,
                description=(fn.get("description") or "")[:1024],
                parameters=_sanitize_schema(fn.get("parameters") or {"type": "object", "properties": {}}),
            ))
        return [types.Tool(function_declarations=declarations)] if declarations else None

    # ------------------------------------------------------------------ chat
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
        if self._client is None:
            raise LLMUnavailableError(self._init_error or "Gemini client unavailable")
        try:
            from google.genai import types  # noqa: PLC0415
        except ImportError as exc:
            raise LLMError(f"{LLM_STATUS_SDK_MISSING}: google-genai ({exc})") from exc

        system_instruction, contents = self._convert_messages(messages)
        kwargs: dict[str, Any] = {
            "temperature": self._settings.llm_temperature if temperature is None else temperature,
            "max_output_tokens": max_tokens or self._settings.llm_max_tokens,
        }
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        gemini_tools = self._build_tools(tools) if tools else None
        if gemini_tools:
            kwargs["tools"] = gemini_tools
            # The SDK would otherwise try to invoke Python callables itself; this
            # application executes tools through its own validated registry.
            kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
        elif json_mode:
            kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**kwargs)
        last_error: Exception | None = None

        for model in self._candidate_models():
            for attempt in range(self._settings.llm_max_retries + 1):
                try:
                    response = self._client.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                    parsed = self._parse_response(response, model)
                    if model != self.model:
                        logger.warning(
                            "Gemini model %r unusable; switched to %r.", self.model, model
                        )
                        self.model = model
                    return parsed
                except LLMResponseError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if is_model_error(exc):
                        self._dead_models.add(model)
                        break  # next candidate model
                    msg = str(exc).lower()
                    if (not any(x in msg for x in _RETRYABLE_MARKERS)
                            or attempt >= self._settings.llm_max_retries):
                        return self._raise(exc)
                    delay = min(8.0, (2**attempt) * 0.5) + random.uniform(0, 0.3)
                    logger.warning("Gemini call failed (%s); retrying in %.1fs", exc, delay)
                    time.sleep(delay)

        return self._raise(last_error or RuntimeError("unknown error"))

    @staticmethod
    def _raise(error: BaseException) -> LLMResponse:  # always raises
        text = str(error)
        if "timeout" in text.lower() or "timed out" in text.lower():
            raise LLMTimeoutError(f"Gemini request timed out: {text}") from error
        raise LLMError(f"Gemini request failed: {text}") from error

    @staticmethod
    def _parse_response(response: Any, model: str) -> LLMResponse:
        """Extract text and function calls from a Gemini response."""
        tool_calls: list[LLMToolCall] = []
        raw_tool_calls: list[dict[str, Any]] = []
        text_chunks: list[str] = []

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in (getattr(content, "parts", None) or []):
                call = getattr(part, "function_call", None)
                if call is not None and getattr(call, "name", None):
                    args = dict(getattr(call, "args", None) or {})
                    call_id = f"gemini_{len(tool_calls)}_{call.name}"
                    tool_calls.append(LLMToolCall(id=call_id, name=call.name, arguments=args))
                    raw_tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(args)},
                    })
                    continue
                part_text = getattr(part, "text", None)
                if part_text:
                    text_chunks.append(str(part_text))

        text = "\n".join(text_chunks).strip()
        if not text and not tool_calls:
            # Distinguish "model chose silence" from a transport failure.
            raise LLMResponseError("Gemini returned neither text nor a function call.")

        usage: dict[str, Any] = {}
        if raw_tool_calls:
            usage["_raw_tool_calls"] = raw_tool_calls

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=str(getattr(candidates[0], "finish_reason", "")) if candidates else None,
            model=model,
            usage=usage,
        )

    # ---------------------------------------------------------------- health
    def health(self) -> dict[str, Any]:
        info = super().health()
        info["model"] = self.model
        if self._init_error:
            info["reason"] = self._init_error
        return info

    def ping(self) -> tuple[bool, str]:
        if self._client is None:
            return False, self._init_error or "unavailable"
        try:
            response = self.chat(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=8,
                temperature=0.0,
            )
            return True, response.text[:60] or "(empty reply)"
        except LLMError as exc:
            return False, classify_llm_error(exc)

    @property
    def status(self) -> str:
        return LLM_STATUS_OK if self.available else (self._init_error or "unavailable")
