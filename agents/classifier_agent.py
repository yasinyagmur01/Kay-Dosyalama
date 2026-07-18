"""Kamu evrakı sınıflandırma ve varlık çıkarma ajanı."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.base_agent import BaseAgent
from agents.prompts.classifier_prompts import (
    DOCUMENT_TYPE_DESCRIPTIONS,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from core.llm_client import get_llm
from core.state import DocumentState, create_initial_state

logger = logging.getLogger("tyda.agents.classifier")

REQUIRED_ENTITY_KEYS = ("tarih", "kurum", "kisi", "konu", "talep")
VALID_DOCUMENT_TYPES = {
    "dilekce",
    "talep",
    "sikayet",
    "bilgi_talebi",
    "resmi_yazi",
    "diger",
}

# Öncelik: daha spesifik türler önce puanlanır
_HEURISTIC_PRIORITY = (
    "resmi_yazi",
    "bilgi_talebi",
    "dilekce",
    "sikayet",
    "talep",
    "diger",
)

_DATE_RE = re.compile(
    r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{1,2}\s*[-–]\s*\d{1,2}\s+\w+\s+\d{4})\b",
    re.IGNORECASE,
)
_SAYIN_RE = re.compile(r"Sayın\s+([^,\n]+)", re.IGNORECASE)
_KONU_RE = re.compile(r"Konu\s*:\s*(.+)", re.IGNORECASE)
_NAME_LINE_RE = re.compile(
    r"^([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)"
)


def _fallback_result() -> dict[str, Any]:
    """JSON parse hatasında güvenli varsayılan çıktı."""
    return {
        "document_type": "diger",
        "confidence_score": 0.1,
        "extracted_entities": {key: "" for key in REQUIRED_ENTITY_KEYS},
        "missing_fields": list(REQUIRED_ENTITY_KEYS),
        "summary": "",
    }


def _refine_document_type(text: str, doc_type: str) -> str:
    """LLM çıktısını kural tabanlı ince ayarla düzeltir."""
    lowered = text.casefold()
    has_4982 = "4982" in lowered or "bilgi edinme" in lowered
    if re.search(r"sayı\s*:\s*talep-", lowered) and not has_4982:
        return "talep"
    if re.search(r"sayı\s*:\s*bilgi_talebi-", lowered) or has_4982:
        return "bilgi_talebi"
    if doc_type == "bilgi_talebi" and not has_4982:
        if "bilgi ve belge talebi" in lowered or "talep ediyorum" in lowered:
            return "talep"
    return doc_type


def _heuristic_classify(text: str) -> dict[str, Any]:
    """LLM yokken anahtar kelime ve kalıplarla sınıflandırma yapar."""
    lowered = text.casefold()
    scores: dict[str, float] = {key: 0.0 for key in VALID_DOCUMENT_TYPES}

    for doc_type in _HEURISTIC_PRIORITY:
        meta = DOCUMENT_TYPE_DESCRIPTIONS.get(doc_type) or {}
        keywords = meta.get("keywords") or []
        if not isinstance(keywords, list):
            continue
        for kw in keywords:
            token = str(kw).casefold().strip()
            if token and token in lowered:
                scores[doc_type] += 1.0

    # Resmi yazı imzaları: T.C. + Sayı
    if re.search(r"\bt\.?\s*c\.?\b", lowered) and "sayı:" in lowered:
        scores["resmi_yazi"] += 3.0
    if "4982" in lowered or "bilgi edinme" in lowered:
        scores["bilgi_talebi"] += 3.0
    elif re.search(r"sayı\s*:\s*talep-", lowered) or "bilgi ve belge talebi" in lowered:
        scores["talep"] += 3.0
    if re.search(r"sayı\s*:\s*bilgi_talebi-", lowered):
        scores["bilgi_talebi"] += 3.0
    if any(k in lowered for k in ("arz ederim", "yıllık izin", "refakat izni")):
        scores["dilekce"] += 2.0
    if any(
        k in lowered
        for k in ("şikayet", "şikâyet", "bozuk", "gürültü", "ödenmemiş", "kaza")
    ):
        scores["sikayet"] += 1.5

    best_type = max(
        _HEURISTIC_PRIORITY,
        key=lambda t: (scores.get(t, 0.0), -_HEURISTIC_PRIORITY.index(t)),
    )
    best_score = scores.get(best_type, 0.0)
    if best_score <= 0:
        best_type = "diger"
        confidence = 0.2
    else:
        confidence = min(0.85, 0.45 + 0.1 * best_score)

    entities = _heuristic_entities(text)
    missing = [key for key in REQUIRED_ENTITY_KEYS if not entities.get(key)]
    summary = (text.strip().split("\n")[0] if text.strip() else "")[:160]
    return {
        "document_type": best_type,
        "confidence_score": confidence,
        "extracted_entities": entities,
        "missing_fields": missing,
        "summary": summary,
    }


def _heuristic_entities(text: str) -> dict[str, str]:
    """Metinden kural tabanlı varlık çıkarımı yapar."""
    entities = {key: "" for key in REQUIRED_ENTITY_KEYS}
    dates = _DATE_RE.findall(text)
    if dates:
        entities["tarih"] = dates[0] if isinstance(dates[0], str) else str(dates[0])

    sayin = _SAYIN_RE.search(text)
    if sayin:
        entities["kurum"] = sayin.group(1).strip().rstrip(",")
    else:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("T.C."):
                entities["kurum"] = stripped
                break
            if "Müdürlüğü" in stripped or "Başkanlığı" in stripped:
                entities["kurum"] = stripped
                break

    konu_m = _KONU_RE.search(text)
    if konu_m:
        entities["konu"] = konu_m.group(1).strip()
    else:
        for line in text.splitlines():
            low = line.casefold()
            if any(
                k in low
                for k in ("izin", "şikayet", "talep", "onarım", "erişim", "ödenek")
            ):
                entities["konu"] = line.strip()[:120]
                break

    for line in text.splitlines():
        low = line.casefold()
        if any(
            k in low
            for k in (
                "talep",
                "arz ederim",
                "rica eder",
                "istemekteyiz",
                "zorunludur",
            )
        ):
            entities["talep"] = line.strip()[:160]
            break

    # İmza / kişi: sondan isim benzeri satır
    for line in reversed(text.splitlines()):
        stripped = line.strip().rstrip(",")
        if not stripped or len(stripped) < 5:
            continue
        if stripped.upper().startswith("T.C.") or stripped.startswith("Sayın"):
            continue
        # "Ad Soyad, Unvan" veya "Ad Soyad"
        head = stripped.split(",")[0].strip()
        if _NAME_LINE_RE.match(head) or (
            " " in head and not any(ch.isdigit() for ch in head[:3])
        ):
            if any(
                k in stripped.casefold()
                for k in (
                    "müdür",
                    "uzman",
                    "demir",
                    "kaya",
                    "yıldız",
                    "çelik",
                    "şahin",
                    "yılmaz",
                    "arslan",
                    "tc:",
                )
            ) or _NAME_LINE_RE.match(head):
                entities["kisi"] = stripped[:120]
                break

    if not entities["kisi"]:
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            parts = [p for p in re.split(r"[,,]", stripped) if p.strip()]
            if parts and len(parts[0].split()) >= 2 and len(parts[0]) < 60:
                entities["kisi"] = stripped[:120]
                break

    return entities


def _merge_user_responses(
    entities: dict[str, Any],
    missing: list[str],
    user_responses: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """HITL yanıtlarını entity'lere yazar ve karşılanan eksik alanları çıkarır."""
    cleaned = {
        str(key): str(value).strip()
        for key, value in (user_responses or {}).items()
        if value is not None and str(value).strip()
    }
    if not cleaned:
        return entities, missing
    merged = {**entities, **cleaned}
    remaining = [field for field in missing if field not in cleaned]
    return merged, remaining


class ClassifierAgent(BaseAgent):
    """Gelen evrakı sınıflandırır ve varlıkları çıkarır."""

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__("classifier")
        self.llm = llm or get_llm()

    async def _run(self, state: DocumentState) -> dict:
        """Evrak metnini sınıflandırıp state alanlarını döner."""
        raw_text = state.get("raw_text") or state.get("raw_input", "")
        if not raw_text.strip():
            return {
                "document_type": "diger",
                "confidence_score": 0.0,
                "extracted_entities": {key: "" for key in REQUIRED_ENTITY_KEYS},
                "missing_fields": list(REQUIRED_ENTITY_KEYS) + ["dusuk_guven"],
                "summary": "",
                "error_log": state.get("error_log", [])
                + ["classifier: Boş evrak metni"],
                "validation_status": "error",
            }

        try:
            parsed = await self._call_llm(raw_text)
        except Exception as exc:
            logger.warning("LLM çağrısı başarısız, kural tabanlı fallback: %s", exc)
            parsed = _heuristic_classify(raw_text)

        parse_error = parsed.pop("_parse_error", None)
        if parse_error:
            logger.warning("Classifier parse hatası, kural tabanlı fallback: %s", parse_error)
            parsed = _heuristic_classify(raw_text)
            parse_error = None

        confidence = float(parsed.get("confidence_score", 0.0))
        missing = list(parsed.get("missing_fields", []))
        if confidence < 0.5 and "dusuk_guven" not in missing:
            missing.append("dusuk_guven")

        entities = dict(parsed.get("extracted_entities", {}) or {})
        entities, missing = _merge_user_responses(
            entities,
            missing,
            state.get("user_responses") or {},
        )

        doc_type = _refine_document_type(
            raw_text,
            str(parsed.get("document_type", "diger") or "diger"),
        )

        result: dict[str, Any] = {
            "document_type": doc_type,
            "confidence_score": confidence,
            "extracted_entities": entities,
            "missing_fields": missing,
            "summary": parsed.get("summary", ""),
        }
        return result

    async def _call_llm(self, text: str) -> dict:
        """Ham LLM çağrısı yapar ve parse eder."""
        from langchain_core.messages import HumanMessage, SystemMessage

        user_prompt = USER_PROMPT.format(raw_text=text)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = await self.llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join(
                part.get("text", str(part)) if isinstance(part, dict) else str(part)
                for part in content
            )
        return self._parse_response(str(content))

    def _parse_response(self, response: str) -> dict:
        """JSON parse ve alan doğrulaması; hata durumunda fallback döner."""
        try:
            payload = _extract_json_object(response)
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError("Yanıt bir JSON nesnesi değil")

            doc_type = str(data.get("document_type", "diger")).strip().lower()
            if doc_type not in VALID_DOCUMENT_TYPES:
                doc_type = "diger"

            try:
                confidence = float(data.get("confidence_score", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            entities_raw = data.get("extracted_entities") or {}
            if not isinstance(entities_raw, dict):
                entities_raw = {}
            entities = {
                key: str(entities_raw.get(key, "") or "").strip()
                for key in REQUIRED_ENTITY_KEYS
            }

            missing_raw = data.get("missing_fields") or []
            if not isinstance(missing_raw, list):
                missing_raw = []
            missing = [str(item) for item in missing_raw]
            for key in REQUIRED_ENTITY_KEYS:
                if not entities.get(key) and key not in missing:
                    missing.append(key)

            summary = str(data.get("summary", "") or "").strip()

            return {
                "document_type": doc_type,
                "confidence_score": confidence,
                "extracted_entities": entities,
                "missing_fields": missing,
                "summary": summary,
            }
        except Exception as exc:
            logger.warning("Classifier JSON parse hatası: %s", exc)
            fallback = _fallback_result()
            fallback["_parse_error"] = f"JSON parse hatası: {exc}"
            return fallback


def _extract_json_object(text: str) -> str:
    """Yanıttan JSON nesnesini ayıklar."""
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        return fence.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


async def run_classifier(text: str) -> dict:
    """Bağımsız kullanım için classifier çalıştırır."""
    agent = ClassifierAgent()
    state = create_initial_state(text, "text")
    return await agent.invoke(state)
