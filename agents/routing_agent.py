"""Evrakı ilgili birime yönlendiren Routing ajanı."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from agents.prompts.routing_prompts import SYSTEM_PROMPT, USER_PROMPT
from core.llm_client import get_llm
from core.state import DocumentState

logger = logging.getLogger("tyda.agents.router")

UNITS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "units_config.json"

DEFAULT_ROUTING: dict[str, Any] = {
    "target_unit": "genel_evrak",
    "routing_rationale": "Otomatik yönlendirme",
    "alternative_units": [],
}


class RoutingAgent(BaseAgent):
    """Kural tabanlı + LLM fallback ile birim yönlendirmesi yapar."""

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__("router")
        self.llm = llm or get_llm()
        self.units = self._load_units()

    def _load_units(self) -> list[dict[str, Any]]:
        """data/units_config.json okur; hata halinde boş liste döner."""
        try:
            data = json.loads(UNITS_CONFIG_PATH.read_text(encoding="utf-8"))
            units = data.get("units", [])
            if isinstance(units, list):
                return [u for u in units if isinstance(u, dict)]
        except Exception as exc:
            logger.warning("Birim yapılandırması okunamadı: %s", exc)
        return []

    async def _run(self, state: DocumentState) -> dict:
        """Önce kural tabanlı, gerekirse LLM yönlendirme uygular."""
        try:
            rule_result = self._rule_based_routing(state)
            if rule_result is not None:
                return rule_result
            return await self._llm_routing(state)
        except Exception as exc:
            logger.error("Routing hatası, varsayılan birim: %s", exc)
            return dict(DEFAULT_ROUTING)

    def _rule_based_routing(self, state: DocumentState) -> dict[str, Any] | None:
        """Keyword ve document_type overlap skoru ile birim seçer."""
        if not self.units:
            return None

        entities = state.get("extracted_entities") or {}
        konu = str(entities.get("konu", "") or "").lower()
        talep = str(entities.get("talep", "") or "").lower()
        summary = str(state.get("summary", "") or "").lower()
        search_text = f"{konu} {talep} {summary}"
        document_type = str(state.get("document_type", "") or "").lower()

        best_unit: dict[str, Any] | None = None
        best_score = 0
        scored: list[tuple[int, dict[str, Any]]] = []

        for unit in self.units:
            score = 0
            keywords = unit.get("keywords") or []
            if isinstance(keywords, list):
                for kw in keywords:
                    kw_l = str(kw).lower()
                    if kw_l and kw_l in search_text:
                        score += 1
            doc_types = unit.get("document_types") or []
            if isinstance(doc_types, list) and document_type in [
                str(d).lower() for d in doc_types
            ]:
                score += 1
            scored.append((score, unit))
            if score > best_score:
                best_score = score
                best_unit = unit

        if best_unit is None or best_score < 2:
            return None

        alternatives = [
            str(u.get("id", ""))
            for s, u in sorted(scored, key=lambda x: x[0], reverse=True)
            if s > 0 and u.get("id") != best_unit.get("id")
        ][:3]

        unit_id = str(best_unit.get("id", "genel_evrak"))
        unit_name = str(best_unit.get("name", unit_id))
        return {
            "target_unit": unit_id,
            "routing_rationale": (
                f"{unit_name} birimi, konu ve evrak tipine göre "
                f"kural tabanlı eşleşmeyle seçilmiştir (skor={best_score})."
            ),
            "alternative_units": [a for a in alternatives if a],
        }

    async def _llm_routing(self, state: DocumentState) -> dict[str, Any]:
        """Birim listesini LLM'e vererek yönlendirme yapar."""
        entities = state.get("extracted_entities") or {}
        units_lines = []
        for unit in self.units:
            units_lines.append(
                f"- id={unit.get('id', '')} | ad={unit.get('name', '')} | "
                f"tipler={','.join(unit.get('document_types') or [])} | "
                f"anahtar={','.join(unit.get('keywords') or [])}"
            )
        units_list = "\n".join(units_lines) or "- id=genel_evrak | ad=Genel Evrak"

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            user_prompt = USER_PROMPT.format(
                document_type=state.get("document_type", "") or "diger",
                konu=str(entities.get("konu", "") or ""),
                talep=str(entities.get("talep", "") or ""),
                units_list=units_list,
            )
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = await self.llm.ainvoke(messages)
            content = _message_content(response)
            data = json.loads(_extract_json_object(content))
            target = str(data.get("target_unit", "") or "").strip()
            rationale = str(data.get("routing_rationale", "") or "").strip()
            alts = data.get("alternative_units") or []
            if not isinstance(alts, list):
                alts = []
            if not target:
                return dict(DEFAULT_ROUTING)
            return {
                "target_unit": target,
                "routing_rationale": rationale or "Otomatik yönlendirme",
                "alternative_units": [str(a) for a in alts if a],
            }
        except Exception as exc:
            logger.warning("LLM yönlendirme başarısız: %s", exc)
            return dict(DEFAULT_ROUTING)


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
