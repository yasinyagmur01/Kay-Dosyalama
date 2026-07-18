"""Classifier Agent birim testleri."""

from __future__ import annotations

import copy
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.classifier_agent import ClassifierAgent
from core.state import create_initial_state


def _make_llm(content: str) -> MagicMock:
    """Belirli content döndüren mock LLM."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content=content))
    return mock


DILEKCE_JSON = """{
  "document_type": "dilekce",
  "confidence_score": 0.92,
  "extracted_entities": {
    "tarih": "04/08/2026",
    "kurum": "İnsan Kaynakları ve Eğitim Müdürlüğü",
    "kisi": "Ayşe Yılmaz",
    "konu": "Yıllık izin talebi",
    "talep": "On iş günü yıllık izin"
  },
  "missing_fields": [],
  "summary": "Personel yıllık izin dilekçesi sunmuştur. İzin tarihleri belirtilmiştir. Vekalet planı hazırdır. Onay talep edilmektedir."
}"""


@pytest.mark.asyncio
async def test_classifier_dilekce_detection(
    sample_dilekce_text: str,
    mock_llm: MagicMock,
) -> None:
    """Dilekçe metninde document_type doğru tespit edilir."""
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=DILEKCE_JSON))
    agent = ClassifierAgent(llm=mock_llm)
    state = create_initial_state(sample_dilekce_text, "text")
    result = await agent.invoke(state)

    assert result["document_type"] == "dilekce"
    assert result["extracted_entities"]
    assert result["confidence_score"] > 0.5


@pytest.mark.asyncio
async def test_classifier_missing_fields(mock_llm: MagicMock) -> None:
    """Eksik bilgili metinde missing_fields dolu döner."""
    incomplete_json = """{
      "document_type": "talep",
      "confidence_score": 0.55,
      "extracted_entities": {
        "tarih": "",
        "kurum": "Müdürlük",
        "kisi": "",
        "konu": "izin",
        "talep": "izin istiyorum"
      },
      "missing_fields": ["tarih", "kisi"],
      "summary": "Kısa izin talebi. Detaylar eksik."
    }"""
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=incomplete_json))
    agent = ClassifierAgent(llm=mock_llm)
    state = create_initial_state("Müdürlüğe, izin istiyorum.", "text")
    result = await agent.invoke(state)

    assert result["missing_fields"]
    assert len(result["missing_fields"]) > 0


@pytest.mark.asyncio
async def test_classifier_merges_user_responses(mock_llm: MagicMock) -> None:
    """HITL yanıtları entity'lere yazılır ve missing_fields'tan çıkarılır."""
    incomplete_json = """{
      "document_type": "talep",
      "confidence_score": 0.55,
      "extracted_entities": {
        "tarih": "",
        "kurum": "Müdürlük",
        "kisi": "",
        "konu": "izin",
        "talep": "izin istiyorum"
      },
      "missing_fields": ["tarih", "kisi"],
      "summary": "Kısa izin talebi. Detaylar eksik."
    }"""
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=incomplete_json))
    agent = ClassifierAgent(llm=mock_llm)
    state = create_initial_state("Müdürlüğe, izin istiyorum.", "text")
    state["user_responses"] = {"tarih": "18/07/2026", "kisi": "Ali Yılmaz"}
    result = await agent.invoke(state)

    assert result["extracted_entities"]["tarih"] == "18/07/2026"
    assert result["extracted_entities"]["kisi"] == "Ali Yılmaz"
    assert "tarih" not in result["missing_fields"]
    assert "kisi" not in result["missing_fields"]


@pytest.mark.asyncio
async def test_classifier_json_parse_error_handling(mock_llm: MagicMock) -> None:
    """Geçersiz JSON'da exception fırlatılmaz, kural tabanlı fallback döner."""
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="bu geçerli json değil {{{"))
    agent = ClassifierAgent(llm=mock_llm)
    state = create_initial_state("Test evrak metni.", "text")
    result = await agent.invoke(state)

    assert result["document_type"] == "diger"
    assert "confidence_score" in result
    assert result.get("validation_status") != "error" or "error_log" not in result
    assert "extracted_entities" in result


@pytest.mark.asyncio
async def test_classifier_state_update_immutability(
    sample_dilekce_text: str,
    mock_llm: MagicMock,
) -> None:
    """Orijinal state mutasyona uğramaz (KURAL 2)."""
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=DILEKCE_JSON))
    agent = ClassifierAgent(llm=mock_llm)
    state = create_initial_state(sample_dilekce_text, "text")
    original = copy.deepcopy(dict(state))

    await agent.invoke(state)

    assert dict(state) == original
