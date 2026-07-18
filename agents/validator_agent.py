"""Eksik alan doğrulama ve kullanıcı sorusu üretme ajanı (kural tabanlı)."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from core.state import DocumentState

QUESTION_MAP: dict[str, str] = {
    "tarih": "Evrakın tarihi nedir? (GG/AA/YYYY formatında)",
    "kurum": "Evrakın gönderildiği kurum adı nedir?",
    "kisi": "Başvuran kişinin adı ve soyadı nedir?",
    "konu": "Evrakın konusu kısaca nedir?",
    "talep": "Talep edilen işlem veya hizmet nedir?",
    "dusuk_guven": "Evrak türü net anlaşılamadı. Evrakınızın amacı nedir?",
}


class ValidatorAgent(BaseAgent):
    """Eksik alanları Türkçe sorulara dönüştürür; LLM kullanmaz."""

    def __init__(self) -> None:
        super().__init__("validator")

    async def _run(self, state: DocumentState) -> dict:
        """missing_fields yoksa complete, varsa needs_input döner."""
        missing = state.get("missing_fields", []) or []

        if not missing:
            return {"validation_status": "complete", "user_questions": []}

        questions = self._generate_questions(list(missing))
        return {
            "validation_status": "needs_input",
            "user_questions": questions,
        }

    def _generate_questions(self, missing_fields: list) -> list[str]:
        """Eksik alanlar için deterministik Türkçe sorular üretir."""
        return [
            QUESTION_MAP.get(field, f"{field} bilgisi eksik, lütfen belirtin.")
            for field in missing_fields
        ]


def apply_user_responses(state: DocumentState, responses: dict) -> dict:
    """Kullanıcı yanıtlarını state'e uygular ve missing_fields'ı temizler."""
    updated_entities = {**state.get("extracted_entities", {}), **responses}
    return {
        "extracted_entities": updated_entities,
        "user_responses": responses,
        "missing_fields": [],
        "validation_status": "complete",
    }
