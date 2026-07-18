"""Milestone 0 altyapı doğrulama testleri."""

from __future__ import annotations

from pathlib import Path

from core.config import settings
from core.state import create_initial_state
from vector_store.chroma_client import get_client, get_or_create_collection


def test_state_creation() -> None:
    """create_initial_state çalışıyor mu?"""
    state = create_initial_state("örnek metin", "text")
    assert state["raw_input"] == "örnek metin"
    assert state["input_type"] == "text"
    assert state["validation_status"] == "complete"
    assert state["error_log"] == []
    assert state["confidence_score"] == 0.0
    assert isinstance(state["processing_time"], dict)


def test_config_loading() -> None:
    """config yükleniyor mu?"""
    assert settings.llm.ollama_base_url
    assert settings.llm.ollama_model
    assert settings.chroma.db_path
    assert settings.chroma.collection_name == "mevzuat_corpus"
    assert settings.app.log_level


async def test_chroma_connection(tmp_path: Path, monkeypatch) -> None:
    """ChromaDB bağlantısı var mı?"""
    import vector_store.chroma_client as chroma_mod

    monkeypatch.setattr(settings.chroma, "db_path", str(tmp_path / "chroma_test"))
    chroma_mod._client = None

    client = await get_client()
    assert client is not None
    collection = await get_or_create_collection("test_collection")
    assert collection.name == "test_collection"
    assert collection.count() == 0
