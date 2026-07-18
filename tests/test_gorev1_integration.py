"""Görev 1 pipeline entegrasyon testleri."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.classifier_agent import ClassifierAgent
from agents.mevzuat_agent import MevzuatAgent
from agents.validator_agent import ValidatorAgent
from core import graph as graph_module
from core.graph import reset_graph, run_pipeline


DILEKCE_JSON = """{
  "document_type": "dilekce",
  "confidence_score": 0.93,
  "extracted_entities": {
    "tarih": "04/08/2026",
    "kurum": "İnsan Kaynakları ve Eğitim Müdürlüğü",
    "kisi": "Ayşe Yılmaz",
    "konu": "Yıllık izin talebi",
    "talep": "On iş günü yıllık izin"
  },
  "missing_fields": [],
  "summary": "Personel yıllık izin dilekçesi sunmuştur. Tarihler belirtilmiştir. Onay talep edilmektedir."
}"""

SIKAYET_JSON = """{
  "document_type": "sikayet",
  "confidence_score": 0.88,
  "extracted_entities": {
    "tarih": "10/07/2026",
    "kurum": "Fen İşleri Müdürlüğü",
    "kisi": "Ali Demir",
    "konu": "Yol bozukluğu",
    "talep": "Onarım yapılması"
  },
  "missing_fields": [],
  "summary": "Mahalle yolunun bozukluğu şikayet edilmiştir. Onarım talep edilmektedir."
}"""

INCOMPLETE_JSON = """{
  "document_type": "dilekce",
  "confidence_score": 0.6,
  "extracted_entities": {
    "tarih": "",
    "kurum": "Müdürlük",
    "kisi": "",
    "konu": "",
    "talep": "başvuru"
  },
  "missing_fields": ["tarih", "kisi", "konu"],
  "summary": "Kısa başvuru metni. Detaylar eksik."
}"""

MEVZUAT_JSON = """{
  "relevant_regulations": [
    {
      "title": "Resmi Yazisma Yonetmeligi",
      "article": "Madde 5",
      "relevance_score": 0.8,
      "summary": "Yazilarin dil ve usul kurallari."
    }
  ],
  "writing_rules": ["Konu satiri zorunludur."]
}"""


def _llm_with_content(content: str) -> MagicMock:
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content=content))
    return mock


def _install_mocked_graph(
    classifier_content: str,
    mevzuat_content: str = MEVZUAT_JSON,
) -> None:
    """Modül seviyesindeki agent'ları mock LLM ile değiştirir."""
    graph_module.classifier = ClassifierAgent(llm=_llm_with_content(classifier_content))
    graph_module.mevzuat = MevzuatAgent(llm=_llm_with_content(mevzuat_content))
    graph_module.validator = ValidatorAgent()
    reset_graph()


@pytest.fixture(autouse=True)
def _reset_graph_singleton() -> None:
    """Her test sonrası graph singleton'ını temizle."""
    yield
    reset_graph()


@pytest.mark.asyncio
async def test_full_gorev1_pipeline_dilekce(sample_dilekce_text: str) -> None:
    """Dilekçe ile tam Görev 1 pipeline çalışır."""
    _install_mocked_graph(DILEKCE_JSON)
    result = await run_pipeline(sample_dilekce_text, "text")

    assert result["document_type"]
    assert result["summary"]
    assert result["current_step"]
    assert result["error_log"] == []
    assert result["processing_time"]


@pytest.mark.asyncio
async def test_full_gorev1_pipeline_sikayet(sample_sikayet_text: str) -> None:
    """Şikayet metni sikayet veya dilekce olarak sınıflanır."""
    _install_mocked_graph(SIKAYET_JSON)
    result = await run_pipeline(sample_sikayet_text, "text")

    assert result["document_type"] in ("sikayet", "dilekce")


@pytest.mark.asyncio
async def test_validator_triggered() -> None:
    """Eksik bilgi validator'ı tetikler."""
    _install_mocked_graph(INCOMPLETE_JSON)
    result = await run_pipeline("Müdürlüğe başvuruyorum.", "text")

    assert result["validation_status"] == "needs_input"
    assert result["user_questions"]


@pytest.mark.asyncio
async def test_mevzuat_returns_something(sample_dilekce_text: str) -> None:
    """Mevzuat agent listeleri döner; writing_rules boş olmaz."""
    _install_mocked_graph(DILEKCE_JSON)
    result = await run_pipeline(sample_dilekce_text, "text")

    assert isinstance(result["relevant_regulations"], list)
    assert isinstance(result["writing_rules"], list)
    assert len(result["writing_rules"]) > 0


@pytest.mark.asyncio
async def test_error_does_not_crash_pipeline() -> None:
    """Boş girdi pipeline'ı çökertmez."""
    _install_mocked_graph(DILEKCE_JSON)
    result = await run_pipeline("", "text")

    assert result["validation_status"] in ("error", "complete")
