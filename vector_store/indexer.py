"""Mevzuat metinlerini indeksleme ve semantic arama."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TypedDict

from vector_store.chroma_client import get_or_create_collection

logger = logging.getLogger("tyda.vector_store.indexer")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class SearchResult(TypedDict):
    content: str
    source: str
    relevance: float


def _tokenize(text: str) -> list[str]:
    """Basit boşluk/noktalama tabanlı tokenizasyon."""
    return re.findall(r"\S+", text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Metni token bazlı chunk'lara böler."""
    tokens = _tokenize(text)
    if not tokens:
        return []
    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(tokens), step):
        piece = tokens[start : start + chunk_size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(tokens):
            break
    return chunks


async def index_regulations(regulations_dir: str) -> int:
    """Regulations klasöründeki .txt dosyalarını ChromaDB'ye ekler."""
    directory = Path(regulations_dir)
    if not directory.exists():
        logger.error("Regulations klasörü bulunamadı: %s", directory)
        return 0

    collection = await get_or_create_collection()
    files = sorted(directory.glob("*.txt"))
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        logger.info("%s → %d chunk", file_path.name, len(chunks))
        for idx, chunk in enumerate(chunks):
            ids.append(f"{file_path.stem}_{idx}")
            documents.append(chunk)
            metadatas.append({"source": file_path.name})

    if not documents:
        logger.warning("İndekslenecek belge bulunamadı")
        return 0

    # Yeniden index için aynı id'leri upsert
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Toplam %d chunk indekslendi", len(documents))
    return len(documents)


async def search_regulations(query: str, n_results: int = 5) -> list[SearchResult]:
    """Semantic search ile ilgili mevzuat parçalarını döner."""
    collection = await get_or_create_collection()
    count = collection.count()
    if count == 0:
        return []

    k = min(n_results, count)
    raw = collection.query(query_texts=[query], n_results=k)
    results: list[SearchResult] = []

    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances, strict=False):
        relevance = 1.0 - float(dist) if dist is not None else 0.0
        source = ""
        if isinstance(meta, dict):
            source = str(meta.get("source", ""))
        results.append(
            SearchResult(
                content=str(doc),
                source=source,
                relevance=relevance,
            )
        )
    return results
