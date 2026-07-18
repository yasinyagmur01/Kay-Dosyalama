"""Resmî yazı taslağı üreten Drafter ajanı."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from agents.prompts.drafter_prompts import (
    DRAFT_DECISION_PROMPT,
    DRAFT_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from core.llm_client import get_llm
from core.state import DocumentState, create_initial_state

logger = logging.getLogger("tyda.agents.drafter")

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "data" / "templates"
VALID_DRAFT_TYPES = {"ust_yazi", "cevap_yazisi", "bilgilendirme"}

FALLBACK_TEMPLATE = """T.C.
{kurum}
İlgili Birim

Konu  : {konu}
Tarih : {tarih}

Sayın Yetkili,

{talep}

Bilgilerinizi ve gereğini arz ederim.

{kisi}
"""


class DrafterAgent(BaseAgent):
    """Gelen evraka uygun resmî yazı taslağı üretir."""

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__("drafter")
        self.llm = llm or get_llm()

    async def _run(self, state: DocumentState) -> dict:
        """Taslak tipini belirler, şablon yükler ve metin üretir."""
        try:
            draft_type = await self._decide_draft_type(state)
            template = self._load_template(draft_type)
            draft = await self._generate_draft(state, draft_type, template)
            return {
                "draft_type": draft_type,
                "draft_text": draft.get("draft_text", ""),
                "draft_metadata": draft.get("draft_metadata", {}),
            }
        except Exception as exc:
            logger.error("Drafter hatası: %s", exc)
            draft_type = self._rule_based_draft_type(state.get("document_type", ""))
            fallback = self._fallback_draft(state, draft_type)
            return {
                "draft_type": draft_type,
                "draft_text": fallback["draft_text"],
                "draft_metadata": fallback["draft_metadata"],
                "error_log": state.get("error_log", [])
                + [f"drafter: {exc}"],
            }

    async def _decide_draft_type(self, state: DocumentState) -> str:
        """LLM ile yazı tipini belirler; parse hatasında kural tabanlı fallback."""
        entities = state.get("extracted_entities") or {}
        document_type = state.get("document_type", "") or ""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            user_prompt = DRAFT_DECISION_PROMPT.format(
                document_type=document_type or "diger",
                konu=str(entities.get("konu", "") or ""),
                talep=str(entities.get("talep", "") or ""),
            )
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = await self.llm.ainvoke(messages)
            content = _message_content(response)
            data = json.loads(_extract_json_object(content))
            draft_type = str(data.get("draft_type", "")).strip().lower()
            if draft_type in VALID_DRAFT_TYPES:
                return draft_type
        except Exception as exc:
            logger.warning("Draft tipi LLM kararı başarısız, fallback: %s", exc)

        return self._rule_based_draft_type(document_type)

    def _rule_based_draft_type(self, document_type: str) -> str:
        """document_type'a göre varsayılan yazı tipi."""
        doc = (document_type or "").strip().lower()
        if doc in ("dilekce", "talep"):
            return "ust_yazi"
        if doc in ("sikayet", "bilgi_talebi"):
            return "cevap_yazisi"
        if doc == "resmi_yazi":
            return "bilgilendirme"
        return "ust_yazi"

    async def _generate_draft(
        self,
        state: DocumentState,
        draft_type: str,
        template: str,
    ) -> dict[str, Any]:
        """LLM ile taslak üretir; user_responses varsa entities ile birleştirir."""
        entities = {
            **(state.get("extracted_entities") or {}),
            **(state.get("user_responses") or {}),
        }
        writing_rules = state.get("writing_rules") or []
        rules_text = "\n".join(f"- {rule}" for rule in writing_rules) or "- Resmî dil kullan"

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            user_prompt = DRAFT_GENERATION_PROMPT.format(
                draft_type=draft_type,
                konu=str(entities.get("konu", "") or ""),
                talep=str(entities.get("talep", "") or ""),
                kurum=str(entities.get("kurum", "") or ""),
                kisi=str(entities.get("kisi", "") or ""),
                tarih=str(entities.get("tarih", "") or ""),
                writing_rules=rules_text,
                template_hint=template,
            )
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = await self.llm.ainvoke(messages)
            content = _message_content(response)
            data = json.loads(_extract_json_object(content))
            draft_text = str(data.get("draft_text", "") or "").strip()
            metadata_raw = data.get("draft_metadata") or {}
            if not isinstance(metadata_raw, dict):
                metadata_raw = {}
            if draft_text:
                return {
                    "draft_text": draft_text,
                    "draft_metadata": {
                        "konu": str(metadata_raw.get("konu", entities.get("konu", "")) or ""),
                        "hitap": str(metadata_raw.get("hitap", "") or ""),
                        "tarih": str(metadata_raw.get("tarih", entities.get("tarih", "")) or ""),
                        "imzalayan": str(
                            metadata_raw.get("imzalayan", entities.get("kisi", "")) or ""
                        ),
                        "tur": str(metadata_raw.get("tur", draft_type) or draft_type),
                    },
                }
        except Exception as exc:
            logger.warning("Taslak üretimi LLM hatası, şablon fallback: %s", exc)

        return self._fallback_draft(state, draft_type, entities)

    def _fallback_draft(
        self,
        state: DocumentState,
        draft_type: str,
        entities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Şablon doldurarak güvenli taslak üretir."""
        merged = entities or {
            **(state.get("extracted_entities") or {}),
            **(state.get("user_responses") or {}),
        }
        konu = str(merged.get("konu", "") or "Belirtilmemiş")
        talep = str(merged.get("talep", "") or "Gereğinin yapılmasını arz ederim.")
        kurum = str(merged.get("kurum", "") or "İlgili Kurum")
        kisi = str(merged.get("kisi", "") or "İlgili Makam")
        tarih = str(merged.get("tarih", "") or "")

        template = self._load_template(draft_type)
        try:
            draft_text = template.format(
                kurum_adi=kurum,
                birim_adi="İlgili Birim",
                sayi="...",
                konu=konu,
                tarih=tarih or "...",
                hitap="Sayın Yetkili,",
                icerik=talep,
                ilgi_yazisi="İlgili başvuru",
                imzalayan=kisi,
                unvan="",
                kurum=kurum,
                kisi=kisi,
                talep=talep,
            )
        except (KeyError, ValueError):
            draft_text = FALLBACK_TEMPLATE.format(
                kurum=kurum,
                konu=konu,
                tarih=tarih or "...",
                talep=talep,
                kisi=kisi,
            )

        return {
            "draft_text": draft_text.strip(),
            "draft_metadata": {
                "konu": konu,
                "hitap": "Sayın Yetkili,",
                "tarih": tarih,
                "imzalayan": kisi,
                "tur": draft_type,
            },
        }

    def _load_template(self, draft_type: str) -> str:
        """data/templates/{draft_type}.txt okur; yoksa inline fallback döner."""
        path = TEMPLATES_DIR / f"{draft_type}.txt"
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Şablon okunamadı (%s): %s", path, exc)
        return FALLBACK_TEMPLATE


def _message_content(response: Any) -> str:
    """LLM yanıtından metin içeriğini ayıklar."""
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = "".join(
            part.get("text", str(part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


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


async def run_drafter(text: str) -> dict:
    """Bağımsız test için classifier ardından drafter çalıştırır."""
    from agents.classifier_agent import ClassifierAgent

    classifier = ClassifierAgent()
    classified = await classifier.invoke(create_initial_state(text, "text"))
    state: DocumentState = {
        **create_initial_state(text, "text"),
        **{k: v for k, v in classified.items() if k != "processing_time"},
        "writing_rules": [
            "Yazılar Türkçe dil bilgisi kurallarına uygun olmalıdır.",
            "Hitap, konu ve imza bölümleri usulüne uygun yerleştirilmelidir.",
        ],
    }
    agent = DrafterAgent()
    return await agent.invoke(state)
