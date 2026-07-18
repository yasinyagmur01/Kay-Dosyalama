"""Mevzuat arama ve yazışma kuralı belirleme ajanı."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.base_agent import BaseAgent
from agents.prompts.mevzuat_prompts import SYSTEM_PROMPT, USER_PROMPT
from core.llm_client import get_llm
from core.state import DocumentState
from vector_store.indexer import search_regulations

logger = logging.getLogger("tyda.agents.mevzuat")

# Resmî Yazışmalarda Uygulanacak Usul ve Esaslar — temel sabit kurallar
BASE_WRITING_RULES: list[str] = [
    "Yazılar Türkçe dil bilgisi kurallarına uygun, açık ve anlaşılır olmalıdır.",
    "Hitap, konu ve imza bölümleri usulüne uygun yerleştirilmelidir.",
    "Tarih GG/AA/YYYY formatında yazılmalıdır.",
    "Gereksiz süsleme ve konuşma dili kullanılmamalıdır.",
    "İlgi yazılar varsa numarası ve tarihi belirtilmelidir.",
]


class MevzuatAgent(BaseAgent):
    """ChromaDB + LLM ile ilgili mevzuatı ve yazışma kurallarını üretir."""

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__("mevzuat")
        self.llm = llm or get_llm()

    async def _run(self, state: DocumentState) -> dict:
        """Semantic search ve LLM sıralaması ile mevzuat çıktısı döner."""
        document_type = state.get("document_type", "") or ""
        entities = state.get("extracted_entities") or {}
        konu = str(entities.get("konu", "") or "")
        query = f"{document_type} {konu}".strip() or "resmi yazisma"

        try:
            ranked = await self._search_and_rank(query, document_type, konu)
        except Exception as exc:
            logger.error("Mevzuat araması başarısız: %s", exc)
            return {
                "relevant_regulations": [],
                "writing_rules": list(BASE_WRITING_RULES),
                "error_log": state.get("error_log", [])
                + [f"mevzuat: {exc}"],
                "validation_status": "error",
            }

        writing_rules = list(BASE_WRITING_RULES)
        for rule in ranked.get("writing_rules", []):
            if rule and rule not in writing_rules:
                writing_rules.append(rule)

        return {
            "relevant_regulations": ranked.get("relevant_regulations", []),
            "writing_rules": writing_rules,
        }

    async def _search_and_rank(
        self,
        query: str,
        document_type: str,
        konu: str,
    ) -> dict[str, Any]:
        """ChromaDB search + LLM reranking."""
        try:
            hits = await search_regulations(query, n_results=5)
        except Exception as exc:
            logger.warning("ChromaDB boş veya erişilemez: %s", exc)
            hits = []

        if not hits:
            return {
                "relevant_regulations": [],
                "writing_rules": list(BASE_WRITING_RULES),
            }

        context_lines = []
        for idx, hit in enumerate(hits, start=1):
            context_lines.append(
                f"[{idx}] kaynak={hit['source']} relevance={hit['relevance']:.2f}\n"
                f"{hit['content']}"
            )
        regulations_context = "\n\n".join(context_lines)

        from langchain_core.messages import HumanMessage, SystemMessage

        user_prompt = USER_PROMPT.format(
            document_type=document_type or "diger",
            konu=konu or "belirtilmemiş",
            regulations_context=regulations_context,
        )
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
        return self._parse_llm_response(str(content), hits)

    def _parse_llm_response(
        self,
        response: str,
        hits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """LLM JSON çıktısını doğrular; hata halinde arama sonuçlarından fallback."""
        try:
            payload = _extract_json_object(response)
            data = json.loads(payload)
            regs = data.get("relevant_regulations") or []
            if not isinstance(regs, list):
                regs = []
            normalized = []
            for item in regs:
                if not isinstance(item, dict):
                    continue
                try:
                    score = float(item.get("relevance_score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                normalized.append(
                    {
                        "title": str(item.get("title", "") or ""),
                        "article": str(item.get("article", "") or ""),
                        "relevance_score": max(0.0, min(1.0, score)),
                        "summary": str(item.get("summary", "") or ""),
                    }
                )
            rules = data.get("writing_rules") or []
            if not isinstance(rules, list):
                rules = []
            return {
                "relevant_regulations": normalized,
                "writing_rules": [str(r) for r in rules if r],
            }
        except Exception as exc:
            logger.warning("Mevzuat JSON parse hatası, fallback kullanılıyor: %s", exc)
            fallback_regs = [
                {
                    "title": hit.get("source", "mevzuat"),
                    "article": hit.get("content", "")[:200],
                    "relevance_score": float(hit.get("relevance", 0.0)),
                    "summary": hit.get("content", "")[:160],
                }
                for hit in hits[:3]
            ]
            return {
                "relevant_regulations": fallback_regs,
                "writing_rules": list(BASE_WRITING_RULES),
            }


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
