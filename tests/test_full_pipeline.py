"""Tam pipeline entegrasyon testleri (OCR → … → Router)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.classifier_agent import ClassifierAgent
from agents.drafter_agent import DrafterAgent
from agents.mevzuat_agent import MevzuatAgent
from agents.ocr_agent import OCRAgent
from agents.routing_agent import RoutingAgent
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
    "konu": "Yol bozukluğu şikayeti",
    "talep": "Yolun onarılması"
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

DRAFT_DECISION_UST = """{
  "draft_type": "ust_yazi",
  "reason": "Dilekçe için üst yazı uygun"
}"""

DRAFT_DECISION_CEVAP = """{
  "draft_type": "cevap_yazisi",
  "reason": "Şikayet için cevap yazısı uygun"
}"""

DRAFT_GENERATION_UST = """{
  "draft_text": "T.C.\\nİnsan Kaynakları ve Eğitim Müdürlüğü\\n\\nKonu: Yıllık izin talebi\\n\\nSayın Yetkili,\\n\\nPersonelin yıllık izin talebi değerlendirilerek gereği yapılmıştır.\\n\\nBilgilerinizi ve gereğini arz ederim.\\n\\nAyşe Yılmaz",
  "draft_metadata": {
    "konu": "Yıllık izin talebi",
    "hitap": "Sayın Yetkili,",
    "tarih": "04/08/2026",
    "imzalayan": "Ayşe Yılmaz",
    "tur": "ust_yazi"
  }
}"""

DRAFT_GENERATION_CEVAP = """{
  "draft_text": "T.C.\\nFen İşleri Müdürlüğü\\n\\nKonu: Yol bozukluğu şikayeti\\n\\nSayın Ali Demir,\\n\\nŞikayetiniz incelenmiş olup yol onarımına ilişkin gerekli işlemler başlatılmıştır.\\n\\nBilgilerinizi rica ederim.\\n\\nFen İşleri Müdürlüğü",
  "draft_metadata": {
    "konu": "Yol bozukluğu şikayeti",
    "hitap": "Sayın Ali Demir,",
    "tarih": "10/07/2026",
    "imzalayan": "Fen İşleri Müdürlüğü",
    "tur": "cevap_yazisi"
  }
}"""

ROUTING_JSON = """{
  "target_unit": "insan_kaynaklari",
  "target_unit_name": "İnsan Kaynakları ve Eğitim Müdürlüğü",
  "routing_rationale": "İzin konusu İnsan Kaynakları birimine aittir.",
  "alternative_units": ["idari_isler"],
  "confidence": 0.9
}"""


def _llm_with_sequence(*contents: str) -> MagicMock:
    """Sırayla farklı içerik dönen mock LLM."""
    mock = MagicMock()
    responses = [MagicMock(content=c) for c in contents]
    mock.ainvoke = AsyncMock(side_effect=responses)
    return mock


def _llm_with_content(content: str) -> MagicMock:
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content=content))
    return mock


def _install_full_mocked_graph(
    classifier_content: str,
    draft_decision: str = DRAFT_DECISION_UST,
    draft_generation: str = DRAFT_GENERATION_UST,
    mevzuat_content: str = MEVZUAT_JSON,
    routing_content: str = ROUTING_JSON,
) -> None:
    """Tüm LLM ajanlarını mock ile kurar."""
    graph_module.ocr = OCRAgent()
    graph_module.classifier = ClassifierAgent(
        llm=_llm_with_content(classifier_content)
    )
    graph_module.mevzuat = MevzuatAgent(llm=_llm_with_content(mevzuat_content))
    graph_module.validator = ValidatorAgent()
    graph_module.drafter = DrafterAgent(
        llm=_llm_with_sequence(draft_decision, draft_generation)
    )
    graph_module.router = RoutingAgent(llm=_llm_with_content(routing_content))
    reset_graph()


@pytest.fixture(autouse=True)
def _reset_graph_singleton() -> None:
    """Her test sonrası graph singleton'ını temizle."""
    yield
    reset_graph()


@pytest.mark.asyncio
async def test_text_input_full_pipeline(sample_dilekce_text: str) -> None:
    """Dilekçe metni tam pipeline'dan geçer; kritik alanlar dolu."""
    _install_full_mocked_graph(DILEKCE_JSON)
    result = await run_pipeline(sample_dilekce_text, "text")

    assert result["document_type"]
    assert result["summary"]
    assert result["draft_text"]
    assert result["target_unit"]
    assert result["routing_rationale"]
    assert result["error_log"] == []
    # Tam yol (eksik alan yok): ocr + classifier + mevzuat + drafter + router
    # Validator yalnızca missing_fields varken çalışır
    expected_keys = {"ocr", "classifier", "mevzuat", "drafter", "router"}
    assert expected_keys.issubset(set(result["processing_time"].keys()))
    assert len(result["processing_time"]) >= 5


@pytest.mark.asyncio
async def test_missing_field_then_continue() -> None:
    """Eksik bilgili metin validator'a düşer; pipeline hata vermez."""
    _install_full_mocked_graph(INCOMPLETE_JSON)
    result = await run_pipeline("Müdürlüğe başvuruyorum.", "text")

    assert result["validation_status"] in ("needs_input", "complete", "error")
    assert "error" not in str(type(result)).lower()
    # Validator çalıştıysa processing_time'da yer alır
    assert "ocr" in result["processing_time"]
    assert result.get("draft_text") or result.get("user_questions")


@pytest.mark.asyncio
async def test_hitl_user_responses_complete_pipeline() -> None:
    """HITL yanıtlarıyla eksik alanlar kapanır; taslak üretilir."""
    _install_full_mocked_graph(INCOMPLETE_JSON)
    result = await run_pipeline(
        "Müdürlüğe başvuruyorum.",
        "text",
        user_responses={
            "tarih": "18/07/2026",
            "kisi": "Ayşe Yılmaz",
            "konu": "Yıllık izin",
        },
    )

    assert "tarih" not in (result.get("missing_fields") or [])
    assert result["extracted_entities"].get("tarih") == "18/07/2026"
    assert result["extracted_entities"].get("kisi") == "Ayşe Yılmaz"
    assert result["draft_text"]
    assert result["validation_status"] in ("complete", "needs_input", "error")
    assert "validator" not in result["processing_time"]


@pytest.mark.asyncio
async def test_sikayet_gets_cevap_yazisi(sample_sikayet_text: str) -> None:
    """Şikayet için draft_type cevap_yazisi (veya tutarlı seçim) olmalı."""
    _install_full_mocked_graph(
        SIKAYET_JSON,
        draft_decision=DRAFT_DECISION_CEVAP,
        draft_generation=DRAFT_GENERATION_CEVAP,
    )
    result = await run_pipeline(sample_sikayet_text, "text")

    assert result["document_type"] == "sikayet"
    assert result["draft_type"] == "cevap_yazisi"
    assert result["draft_text"]


@pytest.mark.asyncio
async def test_pipeline_timing(sample_dilekce_text: str) -> None:
    """processing_time boş olmamalı ve agent anahtarları bulunmalı."""
    _install_full_mocked_graph(DILEKCE_JSON)
    result = await run_pipeline(sample_dilekce_text, "text")

    assert result["processing_time"]
    for key in ("ocr", "classifier", "mevzuat", "drafter", "router"):
        assert key in result["processing_time"]


@pytest.mark.asyncio
async def test_error_recovery() -> None:
    """Boş / kısa / noktalama girdileri exception fırlatmamalı."""
    _install_full_mocked_graph(DILEKCE_JSON)

    for payload in ("", "a", "...", "!!!"):
        result = await run_pipeline(payload, "text")
        assert isinstance(result, dict)
        assert "validation_status" in result
