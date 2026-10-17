"""Groq chat client with native (OpenAI-style) tool calling.

This client speaks the real function-calling protocol rather than asking the
model to emit JSON that the application then interprets:

* tool specifications go out as ``tools=[...]``
* the model replies with ``message.tool_calls``
* executed results go back as ``{"role": "tool", "tool_call_id": ...}`` messages

That round trip is what makes the agent loop genuinely agentic - the model
*observes* tool output and can decide to call another tool because of it.
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
    "rate limit", "rate_limit", "429", "timeout", "timed out", "temporarily",
    "503", "502", "504", "overloaded", "connection", "internal server error",
)


class GroqClient(BaseLLMClient):
    """Groq provider. Supports native tool calling."""

    provider = "groq"
    supports_tool_calling = True

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: Any | None = None
        self._init_error: str | None = None
        self.model = self._settings.groq_model
        #: Models already rejected by the API; never retried within this process.
        self._dead_models: set[str] = set()
        self._initialise()

    def _initialise(self) -> None:
        key = self._settings.groq_api_key
        if not (key and key.strip()):
            self._init_error = f"{LLM_STATUS_MISSING_KEY}: set GROQ_API_KEY"
            return
        try:
            from groq import Groq  # noqa: PLC0415
        except ImportError as exc:
            self._init_error = f"{LLM_STATUS_SDK_MISSING}: groq ({exc})"
            return
        try:
            self._client = Groq(
                api_key=key.strip(),
                timeout=self._settings.request_timeout,
                max_retries=0,  # retries handled here so backoff is observable
            )
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
            response = self._client.models.list()
            data = getattr(response, "data", None) or []
            return sorted(str(getattr(m, "id", "")) for m in data if getattr(m, "id", ""))
        except Exception as exc:  # noqa: BLE001
            logger.info("Groq model listing failed: %s", classify_llm_error(exc))
            return []

    def _candidate_models(self) -> list[str]:
        """Configured model first, then fallbacks, skipping known-dead ids."""
        ordered: list[str] = [self.model] + [
            m for m in self._settings.groq_model_fallbacks if m != self.model
        ]
        return [m for m in ordered if m not in self._dead_models]

    # ------------------------------------------------------------------ chat
    @staticmethod
    def _to_provider_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pass messages through, preserving tool_calls / tool-result messages."""
        out: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            if role == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": str(message.get("tool_call_id", "")),
                    "content": str(message.get("content", "")),
                })
                continue
            entry: dict[str, Any] = {"role": role, "content": message.get("content") or ""}
            if message.get("tool_calls"):
                entry["tool_calls"] = message["tool_calls"]
                # Groq rejects a null content field alongside tool_calls.
                entry["content"] = message.get("content") or ""
            out.append(entry)
        return out

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
            raise LLMUnavailableError(self._init_error or "Groq client unavailable")

        base_payload: dict[str, Any] = {
            "messages": self._to_provider_messages(messages),
            "temperature": (
                self._settings.llm_temperature if temperature is None else temperature
            ),
            "max_tokens": max_tokens or self._settings.llm_max_tokens,
        }
        if tools:
            base_payload["tools"] = list(tools)
            base_payload["tool_choice"] = tool_choice or "auto"
        elif json_mode:
            # response_format is incompatible with tools on several models.
            base_payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None

        # Try the configured model, then fallbacks, but only when the failure is
        # specifically "this model id is not usable".
        for model in self._candidate_models():
            payload = dict(base_payload)
            payload["model"] = model

            for attempt in range(self._settings.llm_max_retries + 1):
                try:
                    completion = self._client.chat.completions.create(**payload)
                    if model != self.model:
                        logger.warning(
                            "Groq model %r unusable; switched to %r.", self.model, model
                        )
                        self.model = model
                    return self._parse_completion(completion)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    message = str(exc).lower()
                    if json_mode and "response_format" in message:
                        payload.pop("response_format", None)
                        continue
                    if is_model_error(exc):
                        self._dead_models.add(model)
                        break  # try the next candidate model
                    if not self._is_retryable(message) or attempt >= self._settings.llm_max_retries:
                        return self._raise(exc)
                    delay = min(8.0, (2**attempt) * 0.5) + random.uniform(0, 0.3)
                    logger.warning("Groq call failed (%s); retrying in %.1fs", exc, delay)
                    time.sleep(delay)

        return self._raise(last_error or RuntimeError("unknown error"))

    @staticmethod
    def _raise(error: BaseException) -> LLMResponse:  # always raises
        text = str(error)
        if "timeout" in text.lower() or "timed out" in text.lower():
            raise LLMTimeoutError(f"Groq request timed out: {text}") from error
        raise LLMError(f"Groq request failed: {text}") from error

    @staticmethod
    def _is_retryable(message: str) -> bool:
        return any(marker in message for marker in _RETRYABLE_MARKERS)

    @staticmethod
    def _parse_completion(completion: Any) -> LLMResponse:
        try:
            choice = completion.choices[0]
            message = choice.message
        except (AttributeError, IndexError) as exc:
            raise LLMResponseError("Groq returned no choices") from exc

        tool_calls: list[LLMToolCall] = []
        raw_tool_calls: list[dict[str, Any]] = []
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            if function is None:
                continue
            name = getattr(function, "name", "") or ""
            raw_args = getattr(function, "arguments", "") or "{}"
            call_id = getattr(call, "id", "") or ""
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (json.JSONDecodeError, TypeError):
                # Keep the call so the loop can tell the model its arguments were
                # malformed, instead of silently dropping the request.
                logger.warning("Groq tool call %s had unparseable arguments", name)
                arguments = {"__malformed__": str(raw_args)[:400]}
            if not isinstance(arguments, dict):
                arguments = {"__malformed__": str(arguments)[:400]}
            tool_calls.append(LLMToolCall(id=call_id, name=name, arguments=arguments))
            raw_tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": raw_args if isinstance(raw_args, str) else json.dumps(raw_args)},
            })

        usage: dict[str, Any] = {}
        raw_usage = getattr(completion, "usage", None)
        if raw_usage is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(raw_usage, key, None)
                if value is not None:
                    usage[key] = value
        if raw_tool_calls:
            usage["_raw_tool_calls"] = raw_tool_calls

        return LLMResponse(
            text=(getattr(message, "content", None) or "").strip(),
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            model=getattr(completion, "model", None),
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
        """Live round trip. The only way to prove the key AND model work."""
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
