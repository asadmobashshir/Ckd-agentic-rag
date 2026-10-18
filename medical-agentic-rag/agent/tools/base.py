"""Base contract every tool must satisfy.

A tool declares a Pydantic argument schema; the registry validates LLM-produced
arguments against it *before* execution. Nothing in this system executes a
function named by free-form model text, and no tool ever receives unvalidated
input.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

from pydantic import BaseModel, ValidationError

from agent.state import Evidence, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutionError(RuntimeError):
    """Raised when a tool fails in a way the orchestrator should record."""


class ToolArgumentError(ValueError):
    """Raised when arguments fail schema validation."""


class BaseTool(ABC):
    """Interface shared by every registered tool."""

    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    args_model: ClassVar[Type[BaseModel]]
    requires_network: ClassVar[bool] = False
    #: Whether results from this tool may be cited as medical evidence.
    produces_evidence: ClassVar[bool] = True

    # ------------------------------------------------------------- schemas
    @classmethod
    def argument_schema(cls) -> dict[str, Any]:
        """JSON Schema for the tool arguments."""
        return cls.args_model.model_json_schema()

    @classmethod
    def result_schema(cls) -> dict[str, Any]:
        """Shape of the structured payload this tool returns."""
        return {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "ok": {"type": "boolean"},
                "evidence": {"type": "array", "items": {"type": "object"}},
                "structured": {"type": "object"},
                "error": {"type": ["string", "null"]},
            },
            "required": ["tool", "ok"],
        }

    @classmethod
    def to_function_spec(cls) -> dict[str, Any]:
        """OpenAI/Gemini-style function-calling spec for this tool."""
        schema = cls.argument_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description.strip(),
                "parameters": schema,
            },
        }

    def validate_arguments(self, arguments: dict[str, Any] | None) -> BaseModel:
        """Validate raw arguments, raising :class:`ToolArgumentError` on failure."""
        try:
            return self.args_model.model_validate(arguments or {})
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolArgumentError(f"Invalid arguments for '{self.name}': {details}") from exc

    # ----------------------------------------------------------- execution
    @abstractmethod
    def run(self, args: BaseModel) -> tuple[list[Evidence], dict[str, Any]]:
        """Execute the tool. Returns ``(evidence, structured_payload)``.

        Implementations are synchronous; the registry runs them off-thread.
        """

    async def execute(self, arguments: dict[str, Any] | None, timeout: float = 20.0) -> ToolResult:
        """Validate, run (off the event loop) and wrap the outcome."""
        started = time.perf_counter()
        result = ToolResult(tool=self.name, ok=False, arguments=dict(arguments or {}))
        try:
            validated = self.validate_arguments(arguments)
            evidence, structured = await asyncio.wait_for(
                asyncio.to_thread(self.run, validated), timeout=timeout
            )
            result.evidence = evidence
            result.structured = structured
            result.ok = True
        except ToolArgumentError as exc:
            result.error = str(exc)
            logger.warning("%s", exc)
        except asyncio.TimeoutError:
            result.error = f"Tool '{self.name}' timed out after {timeout:.0f}s"
            logger.warning("%s", result.error)
        except Exception as exc:  # noqa: BLE001 - a tool failure is data, not a crash
            result.error = f"Tool '{self.name}' failed: {exc}"
            logger.exception("Tool %s raised", self.name)
        finally:
            result.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description.strip(),
            "arguments": self.argument_schema(),
            "result_schema": self.result_schema(),
            "requires_network": self.requires_network,
            "produces_evidence": self.produces_evidence,
        }
