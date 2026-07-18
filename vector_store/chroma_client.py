"""ChromaDB client singleton ve collection yardımcıları."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.utils import embedding_functions

from core.config import settings

logger = logging.getLogger("tyda.vector_store")

_client: ClientAPI | None = None
_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


def _embedding_function() -> Any:
    """Sentence-transformers embedding fonksiyonunu döner."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=_EMBEDDING_MODEL,
    )


async def get_client() -> ClientAPI:
    """ChromaDB singleton client döner."""
    global _client
    if _client is None:
        path = settings.chroma.db_path
        logger.info("ChromaDB başlatılıyor: %s", path)
        _client = chromadb.PersistentClient(path=path)
    return _client


async def get_or_create_collection(
    name: str | None = None,
) -> chromadb.Collection:
    """mevzuat_corpus koleksiyonunu getirir veya oluşturur."""
    client = await get_client()
    collection_name = name or settings.chroma.collection_name
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
