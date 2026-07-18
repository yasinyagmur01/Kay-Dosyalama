"""Kamu evrakı sınıflandırma ve varlık çıkarma ajanı."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.base_agent import BaseAgent
from agents.prompts.classifier_prompts import SYSTEM_PROMPT, USER_PROMPT
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


def _fallback_result() -> dict[str, Any]:
    """JSON parse hatasında güvenli varsayılan çıktı."""
    return {
        "document_type": "diger",
        "confidence_score": 0.1,
        "extracted_entities": {key: "" for key in REQUIRED_ENTITY_KEYS},
        "missing_fields": list(REQUIRED_ENTITY_KEYS),
        "summary": "",
    }


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
            logger.error("LLM çağrısı başarısız: %s", exc)
            fallback = _fallback_result()
            return {
                **fallback,
                "error_log": state.get("error_log", [])
                + [f"classifier: LLM hatası: {exc}"],
                "validation_status": "error",
            }

        parse_error = parsed.pop("_parse_error", None)
        confidence = float(parsed.get("confidence_score", 0.0))
        missing = list(parsed.get("missing_fields", []))
        if confidence < 0.5 and "dusuk_guven" not in missing:
            missing.append("dusuk_guven")

        result: dict[str, Any] = {
            "document_type": parsed.get("document_type", "diger"),
            "confidence_score": confidence,
            "extracted_entities": parsed.get("extracted_entities", {}),
            "missing_fields": missing,
            "summary": parsed.get("summary", ""),
        }
        if parse_error:
            result["error_log"] = state.get("error_log", []) + [
                f"classifier: {parse_error}"
            ]
            result["validation_status"] = "error"
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
