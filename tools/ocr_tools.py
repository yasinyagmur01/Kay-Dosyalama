"""PDF ve görsellerden metin çıkarma araçları."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("tyda.tools.ocr")

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}


def is_supported_file(file_path: str) -> bool:
    """Desteklenen dosya formatı mı?"""
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


async def extract_text_from_pdf(pdf_path: str) -> tuple[str, dict[str, Any]]:
    """PDF'den metin çıkarır; önce surya, ImportError'da pypdf."""
    return await asyncio.to_thread(_extract_text_from_pdf_sync, pdf_path)


def _extract_text_from_pdf_sync(pdf_path: str) -> tuple[str, dict[str, Any]]:
    """Senkron PDF metin çıkarma (thread'de çalışır)."""
    try:
        return _extract_with_surya(pdf_path, is_pdf=True)
    except ImportError:
        logger.info("surya bulunamadı, pypdf fallback kullanılıyor")
        return _extract_with_pypdf(pdf_path)
    except Exception as exc:
        logger.warning("surya PDF OCR başarısız, pypdf deneniyor: %s", exc)
        try:
            return _extract_with_pypdf(pdf_path)
        except Exception as fallback_exc:
            logger.error("pypdf extraction hatası: %s", fallback_exc)
            return "", {"pages": 0, "method": "error", "error": str(fallback_exc)}


async def extract_text_from_image(image_path: str) -> tuple[str, dict[str, Any]]:
    """Görsel dosyadan metin çıkarır."""
    return await asyncio.to_thread(_extract_text_from_image_sync, image_path)


def _extract_text_from_image_sync(image_path: str) -> tuple[str, dict[str, Any]]:
    """Senkron görsel OCR (thread'de çalışır)."""
    try:
        return _extract_with_surya(image_path, is_pdf=False)
    except ImportError:
        logger.warning("surya yüklü değil; görsel OCR yapılamıyor")
        return "", {"pages": 1, "method": "unavailable", "error": "surya yüklü değil"}
    except Exception as exc:
        logger.error("Görsel OCR hatası: %s", exc)
        return "", {"pages": 1, "method": "error", "error": str(exc)}


def _extract_with_pypdf(pdf_path: str) -> tuple[str, dict[str, Any]]:
    """pypdf ile basit PDF metin çıkarma."""
    import pypdf

    reader = pypdf.PdfReader(pdf_path)
    pages_text: list[str] = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    raw_text = "\n".join(pages_text).strip()
    return raw_text, {"pages": len(reader.pages), "method": "pypdf"}


def _extract_with_surya(file_path: str, is_pdf: bool) -> tuple[str, dict[str, Any]]:
    """surya-ocr ile metin çıkarma; ImportError yukarıya iletilir."""
    from PIL import Image

    # surya API sürümleri değişebilir; mümkün olan import yollarını dene
    try:
        from surya.ocr import run_ocr  # type: ignore[import-untyped]
        from surya.model.detection.model import load_model as load_det_model
        from surya.model.detection.model import load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
    except ImportError:
        from surya.ocr import run_ocr  # type: ignore[import-untyped]
        from surya.model.detection import segformer
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor

        load_det_model = segformer.load_model
        load_det_processor = segformer.load_processor

    images = _load_images(file_path, is_pdf=is_pdf)
    if not images:
        return "", {"pages": 0, "method": "surya"}

    det_model, det_processor = load_det_model(), load_det_processor()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()
    langs = [["tr", "en"]] * len(images)

    predictions = run_ocr(
        images,
        langs,
        det_model,
        det_processor,
        rec_model,
        rec_processor,
    )

    page_texts: list[str] = []
    for pred in predictions:
        lines = getattr(pred, "text_lines", None) or []
        page_texts.append(
            "\n".join(str(getattr(line, "text", "") or "") for line in lines)
        )

    raw_text = "\n".join(page_texts).strip()
    return raw_text, {"pages": len(images), "method": "surya"}


def _load_images(file_path: str, is_pdf: bool) -> list[Any]:
    """Dosyayı PIL Image listesine dönüştürür."""
    from PIL import Image

    path = Path(file_path)
    if not is_pdf:
        return [Image.open(path).convert("RGB")]

    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        images = []
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=2)
            images.append(bitmap.to_pil().convert("RGB"))
        return images
    except ImportError:
        # PDF sayfalarını görsele çeviremiyorsak surya yerine pypdf'e düş
        raise ImportError("PDF sayfa render için pypdfium2 gerekli") from None
