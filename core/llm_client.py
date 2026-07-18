"""Ollama ve Anthropic LLM client sarmalayıcıları."""

from __future__ import annotations

import logging
import os
from typing import Any, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from core.config import settings

logger = logging.getLogger("tyda.llm")

T = TypeVar("T", bound=BaseModel)

_ANTHROPIC_MODEL = "claude-sonnet-4-6"


class OllamaClient:
    """Ollama üzerinden LLM erişimi sağlar."""

    def __init__(self) -> None:
        self.llm = ChatOllama(
            model=settings.llm.ollama_model,
            base_url=settings.llm.ollama_base_url,
        )

    def with_structured_output(self, schema: type[T] | dict[str, Any]) -> Any:
        """Structured output destekli zincir döner."""
        return self.llm.with_structured_output(schema)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Asenkron LLM çağrısı yapar."""
        return await self.llm.ainvoke(*args, **kwargs)

    async def health_check(self) -> bool:
        """Ollama bağlantısını test eder."""
        try:
            await self.llm.ainvoke("ping")
            return True
        except Exception as exc:
            logger.error("Ollama sağlık kontrolü başarısız: %s", exc)
            return False


class AnthropicClient:
    """Anthropic Claude üzerinden LLM erişimi sağlar."""

    def __init__(self) -> None:
        self.llm = ChatAnthropic(
            model=_ANTHROPIC_MODEL,
            api_key=settings.llm.anthropic_api_key,
        )

    def with_structured_output(self, schema: type[T] | dict[str, Any]) -> Any:
        """Structured output destekli zincir döner."""
        return self.llm.with_structured_output(schema)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Asenkron LLM çağrısı yapar."""
        return await self.llm.ainvoke(*args, **kwargs)

    async def health_check(self) -> bool:
        """Anthropic bağlantısını test eder."""
        try:
            await self.llm.ainvoke("ping")
            return True
        except Exception as exc:
            logger.error("Anthropic sağlık kontrolü başarısız: %s", exc)
            return False


def get_llm() -> OllamaClient | AnthropicClient:
    """USE_ANTHROPIC True ise Anthropic, aksi halde Ollama client döner."""
    use_anthropic = os.getenv("USE_ANTHROPIC", "").lower() in ("true", "1", "yes")
    if use_anthropic:
        logger.info("LLM backend: Anthropic")
        return AnthropicClient()
    logger.info("LLM backend: Ollama")
    return OllamaClient()


async def health_check() -> bool:
    """Aktif LLM backend bağlantısını test eder."""
    client = get_llm()
    return await client.health_check()
