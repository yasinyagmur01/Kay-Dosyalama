"""LangGraph pipeline için merkezi DocumentState şeması."""

from typing import Literal, TypedDict


class DocumentState(TypedDict):
    # ── INPUT ──────────────────────────────────────────
    raw_input: str              # Dosya yolu (PDF/görsel) veya ham metin
    input_type: Literal["pdf", "image", "text"]

    # ── OCR AGENT ──────────────────────────────────────
    raw_text: str               # OCR çıktısı veya direkt metin
    layout_metadata: dict       # Sayfa yapısı, başlıklar, tablolar

    # ── CLASSIFIER AGENT ───────────────────────────────
    document_type: str          # "dilekce" | "talep" | "sikayet" | "bilgi_talebi" | "resmi_yazi" | "diger"
    extracted_entities: dict    # {"tarih": ..., "kurum": ..., "kisi": ..., "konu": ..., "talep": ...}
    missing_fields: list        # Eksik zorunlu alanlar
    confidence_score: float     # 0.0 - 1.0
    summary: str                # 3-5 cümle özet (Türkçe)

    # ── MEVZUAT AGENT ──────────────────────────────────
    relevant_regulations: list  # [{"title": str, "article": str, "relevance_score": float}]
    writing_rules: list         # Uygulanacak yazışma kuralları

    # ── DRAFTER AGENT ──────────────────────────────────
    draft_type: str             # "ust_yazi" | "cevap_yazisi" | "bilgilendirme"
    draft_text: str             # Üretilen Türkçe resmi yazı taslağı
    draft_metadata: dict        # {"konu": str, "hitap": str, "tarih": str}

    # ── ROUTING AGENT ──────────────────────────────────
    target_unit: str            # Yönlendirilen birim adı
    routing_rationale: str      # Yönlendirme gerekçesi (Türkçe)
    alternative_units: list     # Alternatif birimler

    # ── VALIDATOR AGENT ────────────────────────────────
    validation_status: Literal["complete", "needs_input", "error"]
    user_questions: list        # Kullanıcıya sorulacak sorular
    user_responses: dict        # Kullanıcı yanıtları

    # ── PIPELINE META ──────────────────────────────────
    current_step: str           # Hangi agent çalışıyor
    error_log: list             # Hata geçmişi
    processing_time: dict       # Agent başına süre (saniye)


def create_initial_state(raw_input: str, input_type: str) -> DocumentState:
    """Pipeline başlangıç state'ini oluşturur."""
    # OCR henüz yokken text girdilerinde raw_input → raw_text
    initial_text = raw_input if input_type == "text" else ""
    return DocumentState(
        raw_input=raw_input,
        input_type=input_type,  # type: ignore[typeddict-item]
        raw_text=initial_text,
        layout_metadata={},
        document_type="",
        extracted_entities={},
        missing_fields=[],
        confidence_score=0.0,
        summary="",
        relevant_regulations=[],
        writing_rules=[],
        draft_type="",
        draft_text="",
        draft_metadata={},
        target_unit="",
        routing_rationale="",
        alternative_units=[],
        validation_status="complete",
        user_questions=[],
        user_responses={},
        current_step="",
        error_log=[],
        processing_time={},
    )
