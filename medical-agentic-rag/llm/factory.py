"""LLM provider factory.

Two providers are implemented and both support native tool calling. The
provider is resolved from configuration; ``auto`` picks whichever API key the
deployment actually has, which is what lets a Groq-keyed deployment and a
Gemini-keyed deployment run the same code.
"""

from __future__ import annotations

import logging
from typing import Callable

from config.settings import Settings, get_settings
from llm.base import BaseLLMClient, UnavailableLLMClient, classify_llm_error
from llm.gemini_client import GeminiClient
from llm.groq_client import GroqClient

logger = logging.getLogger(__name__)

_PROVIDERS: dict[str, Callable[[Settings], BaseLLMClient]] = {
    "groq": lambda settings: GroqClient(settings),
    "gemini": lambda settings: GeminiClient(settings),
}


def build_llm_client(settings: Settings | None = None) -> BaseLLMClient:
    """Return a usable client, or an explicit unavailable stub.

    The stub is deliberate: the orchestrator must be able to detect that no
    generative backend exists and degrade to its evidence-only path rather than
    silently inventing an answer.
    """
    settings = settings or get_settings()
    provider = settings.active_provider
    factory = _PROVIDERS.get(provider)
    if factory is None:
        return UnavailableLLMClient(f"Unknown llm_provider: {provider}")

    try:
        client = factory(settings)
    except Exception as exc:  # noqa: BLE001
        reason = classify_llm_error(exc)
        logger.warning("Failed to build %s client: %s", provider, reason)
        return UnavailableLLMClient(reason)

    if client.available:
        return client

    reason = str(client.health().get("reason", "LLM backend unavailable"))

    # 'auto' with an unusable primary: try the other provider before giving up,
    # so a deployment holding only the other key still works.
    if settings.llm_provider == "auto":
        for name, other in _PROVIDERS.items():
            if name == provider:
                continue
            try:
                candidate = other(settings)
            except Exception:  # noqa: BLE001
                continue
            if candidate.available:
                logger.info("Provider %r unavailable (%s); using %r.", provider, reason, name)
                return candidate

    logger.warning("LLM backend unavailable: %s", reason)
    return UnavailableLLMClient(reason)
