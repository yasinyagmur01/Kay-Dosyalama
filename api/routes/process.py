"""Evrak işleme endpoint'leri."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import ProcessRequest, ProcessResponse
from core.graph import run_pipeline

logger = logging.getLogger("tyda.api.process")

router = APIRouter()

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
_PDF_EXTENSIONS = {".pdf"}


def _state_to_response(state: dict[str, Any], *, success: bool = True) -> ProcessResponse:
    """DocumentState çıktısını ProcessResponse'a çevirir."""
    return ProcessResponse(
        success=success,
        document_type=state.get("document_type", "") or "",
        confidence_score=float(state.get("confidence_score") or 0.0),
        summary=state.get("summary", "") or "",
        extracted_entities=state.get("extracted_entities") or {},
        missing_fields=state.get("missing_fields") or [],
        relevant_regulations=state.get("relevant_regulations") or [],
        writing_rules=state.get("writing_rules") or [],
        draft_type=state.get("draft_type", "") or "",
        draft_text=state.get("draft_text", "") or "",
        target_unit=state.get("target_unit", "") or "",
        routing_rationale=state.get("routing_rationale", "") or "",
        alternative_units=state.get("alternative_units") or [],
        validation_status=state.get("validation_status", "") or "",
        user_questions=state.get("user_questions") or [],
        processing_time=state.get("processing_time") or {},
        error_log=state.get("error_log") or [],
    )


def _detect_input_type(filename: str | None, content_type: str | None) -> str:
    """Dosya adından veya MIME tipinden input_type belirler."""
    suffix = Path(filename or "").suffix.lower()
    if suffix in _PDF_EXTENSIONS or (content_type and "pdf" in content_type):
        return "pdf"
    if suffix in _IMAGE_EXTENSIONS or (content_type and content_type.startswith("image/")):
        return "image"
    return "pdf"


@router.post("/process/text", response_model=ProcessResponse)
async def process_text(request: ProcessRequest) -> ProcessResponse:
    """Metin evrakını pipeline ile işler."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Evrak metni boş olamaz.")
    try:
        result = await run_pipeline(
            request.text,
            request.input_type or "text",
            request.user_responses or None,
        )
        return _state_to_response(result, success=True)
    except Exception as exc:
        logger.error("Metin işleme hatası: %s", exc)
        return ProcessResponse(
            success=False,
            validation_status="error",
            error_log=[f"İşleme hatası: {exc}"],
        )


@router.post("/process/file", response_model=ProcessResponse)
async def process_file(file: UploadFile = File(...)) -> ProcessResponse:
    """PDF veya görsel dosyayı geçici olarak kaydedip pipeline ile işler."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Dosya adı bulunamadı.")

    input_type = _detect_input_type(file.filename, file.content_type)
    suffix = Path(file.filename).suffix or (".pdf" if input_type == "pdf" else ".png")
    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Yüklenen dosya boş.")
            tmp.write(content)
            tmp_path = tmp.name

        result = await run_pipeline(tmp_path, input_type)
        return _state_to_response(result, success=True)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Dosya işleme hatası: %s", exc)
        return ProcessResponse(
            success=False,
            validation_status="error",
            error_log=[f"Dosya işleme hatası: {exc}"],
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as exc:
                logger.warning("Geçici dosya silinemedi: %s", exc)


@router.post("/process/with-responses", response_model=ProcessResponse)
async def process_with_responses(request: ProcessRequest) -> ProcessResponse:
    """HITL yanıtlarıyla pipeline'ı yeniden çalıştırır."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Evrak metni boş olamaz.")
    try:
        result = await run_pipeline(
            request.text,
            "text",
            request.user_responses or {},
        )
        return _state_to_response(result, success=True)
    except Exception as exc:
        logger.error("HITL yeniden işleme hatası: %s", exc)
        return ProcessResponse(
            success=False,
            validation_status="error",
            error_log=[f"Yeniden işleme hatası: {exc}"],
        )
