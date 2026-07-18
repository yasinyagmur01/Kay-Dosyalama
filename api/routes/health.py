"""Sağlık kontrolü endpoint'leri."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from api.schemas import HealthResponse
from core.llm_client import health_check as llm_health_check
from vector_store.chroma_client import get_client

logger = logging.getLogger("tyda.api.health")

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """LLM ve ChromaDB bağlantı durumunu döner."""
    llm_ok = False
    vector_ok = False

    try:
        llm_ok = await llm_health_check()
    except Exception as exc:
        logger.error("LLM sağlık kontrolü başarısız: %s", exc)

    try:
        client = await get_client()
        client.heartbeat()
        vector_ok = True
    except Exception as exc:
        logger.error("ChromaDB sağlık kontrolü başarısız: %s", exc)

    status = "ok" if llm_ok and vector_ok else "degraded"
    return HealthResponse(
        status=status,
        llm_available=llm_ok,
        vector_store_available=vector_ok,
    )
