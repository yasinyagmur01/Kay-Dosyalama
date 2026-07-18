"""KayıDosyalama FastAPI uygulama girişi."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.process import router as process_router
from core.graph import get_graph

logger = logging.getLogger("tyda.api")

app = FastAPI(
    title="KayıDosyalama — Kamu Evrak Agent Sistemi",
    description="Yapay Zeka Dil Ajanları Yarışması — 1. Senaryo",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(process_router, prefix="/api/v1", tags=["process"])
app.include_router(health_router, prefix="/api/v1", tags=["health"])


@app.on_event("startup")
async def startup() -> None:
    """Graph singleton'ını ön-yükler."""
    logger.info("Pipeline graph ön-yükleniyor...")
    get_graph()
    logger.info("API hazır.")
