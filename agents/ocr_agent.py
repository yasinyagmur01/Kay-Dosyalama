"""PDF/görsel metin çıkarma veya metin pass-through OCR ajanı."""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from core.state import DocumentState
from tools.ocr_tools import (
    extract_text_from_image,
    extract_text_from_pdf,
    is_supported_file,
)

logger = logging.getLogger("tyda.agents.ocr")


class OCRAgent(BaseAgent):
    """Girdi tipine göre metin çıkarır; LLM kullanmaz."""

    def __init__(self) -> None:
        super().__init__("ocr")

    async def _run(self, state: DocumentState) -> dict:
        """text pass-through; pdf/image OCR; hatalarda error_log yazar."""
        input_type = state.get("input_type", "text")
        raw_input = state.get("raw_input", "") or ""

        if input_type == "text":
            return {
                "raw_text": raw_input,
                "layout_metadata": {"method": "direct_text"},
            }

        if not is_supported_file(raw_input):
            return {
                "raw_text": "",
                "layout_metadata": {"method": "unsupported"},
                "error_log": state.get("error_log", [])
                + ["ocr: Desteklenmeyen dosya formatı"],
                "validation_status": "error",
            }

        try:
            if input_type == "pdf":
                raw_text, metadata = await extract_text_from_pdf(raw_input)
            else:
                raw_text, metadata = await extract_text_from_image(raw_input)
        except Exception as exc:
            logger.error("OCR çıkarımı başarısız: %s", exc)
            return {
                "raw_text": "",
                "layout_metadata": {"method": "error"},
                "error_log": state.get("error_log", [])
                + [f"ocr: Metin çıkarma hatası: {exc}"],
                "validation_status": "error",
            }

        if not (raw_text or "").strip():
            return {
                "raw_text": "",
                "layout_metadata": metadata,
                "error_log": state.get("error_log", [])
                + ["ocr: Evraktan metin çıkarılamadı"],
                "validation_status": "error",
            }

        return {"raw_text": raw_text, "layout_metadata": metadata}
