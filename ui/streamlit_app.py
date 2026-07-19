"""KayıDosyalama Streamlit arayüzü — core.graph.run_pipeline üzerinden."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.classifier_agent import ClassifierAgent
from agents.drafter_agent import DrafterAgent
from agents.mevzuat_agent import MevzuatAgent
from agents.ocr_agent import OCRAgent
from agents.routing_agent import RoutingAgent
from agents.validator_agent import ValidatorAgent
from core import graph as graph_module
from core.graph import reset_graph, run_pipeline
from core.llm_client import get_llm
from data.demo_samples import SAMPLE_DOCS

logger = logging.getLogger("tyda.ui.streamlit")

st.set_page_config(page_title="KayıDosyalama", layout="wide")

RENKLER = {
    "dilekce": "🔵",
    "sikayet": "🔴",
    "talep": "🟠",
    "bilgi_talebi": "🟣",
    "resmi_yazi": "🟢",
    "diger": "⚪",
}
LABELS = {
    "tarih": "Tarih",
    "kurum": "Kurum",
    "kisi": "Kişi",
    "konu": "Konu",
    "talep": "Talep",
}
MODEL_OPTIONS = {
    "Qwen (Local)": "qwen",
    "Claude API": "claude",
}


def _init_session() -> None:
    """Session state anahtarlarını başlatır."""
    defaults: dict[str, Any] = {
        "result": None,
        "raw_input": "",
        "input_type": "text",
        "user_responses": {},
        "model": "qwen",
        "hitl_done": False,
        "_tmp_pdf": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _cleanup_tmp(path: str | None) -> None:
    """Geçici PDF dosyasını siler."""
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            logger.warning("Geçici dosya silinemedi: %s", path)


def _apply_model(choice_label: str) -> None:
    """Model seçimini env + agent singleton'larına uygular."""
    model_key = MODEL_OPTIONS[choice_label]
    st.session_state["model"] = model_key

    # UI sözleşmesi
    os.environ["LLM_PROVIDER"] = "anthropic" if model_key == "claude" else "ollama"
    # Mevcut llm_client USE_ANTHROPIC okur
    os.environ["USE_ANTHROPIC"] = "true" if model_key == "claude" else "false"

    if st.session_state.get("_applied_model") == model_key:
        return

    llm = get_llm()
    graph_module.ocr = OCRAgent()
    graph_module.classifier = ClassifierAgent(llm=llm)
    graph_module.mevzuat = MevzuatAgent(llm=llm)
    graph_module.validator = ValidatorAgent()
    graph_module.drafter = DrafterAgent(llm=llm)
    graph_module.router = RoutingAgent(llm=llm)
    reset_graph()
    st.session_state["_applied_model"] = model_key


_pipeline_loop: asyncio.AbstractEventLoop | None = None
_pipeline_loop_lock = threading.Lock()


def _get_pipeline_loop() -> asyncio.AbstractEventLoop:
    """Streamlit/uvloop dışında kalıcı asyncio loop tutar."""
    global _pipeline_loop
    with _pipeline_loop_lock:
        if _pipeline_loop is not None and _pipeline_loop.is_running():
            return _pipeline_loop

        loop = asyncio.new_event_loop()

        def _run_forever() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_run_forever, daemon=True, name="pipeline-loop")
        thread.start()
        _pipeline_loop = loop
        return loop


def _run_pipeline(
    raw_input: str,
    input_type: str,
    user_responses: dict[str, str] | None = None,
) -> dict[str, Any]:
    """run_pipeline'ı Streamlit thread'inden güvenli şekilde çalıştırır."""
    loop = _get_pipeline_loop()
    future = asyncio.run_coroutine_threadsafe(
        run_pipeline(raw_input, input_type, user_responses),
        loop,
    )
    return dict(future.result())


def _entities_dataframe(entities: dict[str, Any]) -> pd.DataFrame:
    """Çıkarılan varlıkları tabloya çevirir."""
    rows = []
    for key, label in LABELS.items():
        value = entities.get(key, "")
        rows.append({"Alan": label, "Değer": value if value else "—"})
    for key, value in (entities or {}).items():
        if key not in LABELS:
            rows.append({"Alan": str(key), "Değer": value if value else "—"})
    return pd.DataFrame(rows)


def _render_sidebar() -> None:
    """Model seçimi ve processing_time metrikleri."""
    with st.sidebar:
        labels = list(MODEL_OPTIONS.keys())
        current = st.session_state.get("model", "qwen")
        index = 0 if current == "qwen" else 1
        choice = st.selectbox("Model", labels, index=index)
        if st.session_state.get("_applied_model") != MODEL_OPTIONS[choice]:
            _apply_model(choice)

        result = st.session_state.get("result")
        if result and result.get("processing_time"):
            st.subheader("Çalışma süresi")
            times = result["processing_time"]
            total = sum(float(v) for v in times.values())
            st.metric("Toplam", f"{total:.2f}s")
            for agent, seconds in times.items():
                st.metric(str(agent), f"{float(seconds):.2f}s")


def _render_error_banner(result: dict[str, Any]) -> None:
    """error_log varsa uyarı expander'ı gösterir."""
    error_log = result.get("error_log") or []
    if not error_log:
        return
    with st.expander("⚠️ Uyarılar"):
        for err in error_log:
            st.error(str(err))


def _render_tab_classification(result: dict[str, Any]) -> None:
    """TAB 1 — Sınıflandırma."""
    doc_type = str(result.get("document_type") or "diger")
    emoji = RENKLER.get(doc_type, "⚪")
    st.markdown(
        f"<p style='font-size:2rem;font-weight:700;margin:0;'>"
        f"{emoji} {doc_type}</p>",
        unsafe_allow_html=True,
    )

    confidence = float(result.get("confidence_score") or 0.0)
    col_bar, col_pct = st.columns([4, 1])
    with col_bar:
        st.progress(min(max(confidence, 0.0), 1.0))
    with col_pct:
        st.markdown(f"**%{confidence * 100:.0f}**")

    summary = result.get("summary") or ""
    if summary:
        st.info(summary)
    else:
        st.info("Özet üretilmedi.")

    st.markdown("**Çıkarılan varlıklar**")
    st.dataframe(
        _entities_dataframe(result.get("extracted_entities") or {}),
        use_container_width=True,
        hide_index=True,
    )


def _render_tab_hitl(result: dict[str, Any]) -> None:
    """TAB 2 — Eksik bilgi / HITL."""
    if result.get("validation_status") != "needs_input":
        st.success("Tüm bilgiler tam ✓")
        return

    st.warning("Bazı bilgiler eksik, lütfen doldurun:")
    questions = result.get("user_questions") or []
    missing = result.get("missing_fields") or []

    with st.form("hitl_form"):
        responses: dict[str, str] = {}
        for field, question in zip(missing, questions):
            responses[str(field)] = st.text_input(
                str(question),
                key=f"hitl_{field}",
            )
        send = st.form_submit_button("✅ Gönder ve Devam Et")

    if send:
        cleaned = {
            key: str(value).strip()
            for key, value in responses.items()
            if value is not None and str(value).strip()
        }
        if len(cleaned) < len(missing):
            st.error("Lütfen tüm eksik alanları doldurun.")
            return

        st.session_state["user_responses"] = cleaned
        st.session_state["hitl_done"] = True
        _apply_model(
            "Claude API" if st.session_state["model"] == "claude" else "Qwen (Local)"
        )
        with st.spinner("Yanıtlarla pipeline devam ediyor..."):
            new_result = _run_pipeline(
                st.session_state["raw_input"],
                st.session_state["input_type"],
                user_responses=st.session_state["user_responses"],
            )
        st.session_state["result"] = new_result
        _cleanup_tmp(st.session_state.get("_tmp_pdf"))
        st.session_state["_tmp_pdf"] = None
        st.rerun()


def _render_tab_mevzuat(result: dict[str, Any]) -> None:
    """TAB 3 — Mevzuat."""
    regulations = result.get("relevant_regulations") or []
    if not regulations:
        st.info("Mevzuat bulunamadı")
        return

    for reg in regulations:
        if not isinstance(reg, dict):
            st.write(str(reg))
            continue
        title = reg.get("title") or "Başlıksız"
        score = float(reg.get("relevance_score") or 0.0)
        with st.expander(f"{title} — %{score * 100:.0f}"):
            st.write(reg.get("article", "") or "")


def _render_tab_draft(result: dict[str, Any]) -> None:
    """TAB 4 — Taslak."""
    st.caption(f"Taslak türü: {result.get('draft_type') or '—'}")
    edited = st.text_area(
        "Taslak (düzenleyebilirsiniz)",
        value=result.get("draft_text") or "",
        height=400,
    )
    if st.button("📋 Kopyala"):
        st.code(edited, language=None)


def _render_tab_routing(result: dict[str, Any]) -> None:
    """TAB 5 — Yönlendirme."""
    target = result.get("target_unit") or "—"
    st.success(f"📌 {target}")
    st.write(result.get("routing_rationale") or "")
    alts = result.get("alternative_units") or []
    if alts:
        st.caption("Alternatif birimler: " + " · ".join(str(a) for a in alts))


# ── Sayfa ──────────────────────────────────────────────────────────
_init_session()
_render_sidebar()

st.title("KayıDosyalama — Kamu Evrak İşleme")

col_in, col_out = st.columns([5, 7])

with col_in:
    ornek_etiketler = ["(Manuel yaz / yapıştır)"] + list(SAMPLE_DOCS.keys())
    secim = st.selectbox("Deneme evrakı (10 örnek)", ornek_etiketler)
    if secim != "(Manuel yaz / yapıştır)":
        if st.button("Örneği metin kutusuna yükle", use_container_width=True):
            st.session_state["draft_input"] = SAMPLE_DOCS[secim]
            st.rerun()

    with st.form("input_form"):
        metin = st.text_area(
            "Evrak metni",
            height=280,
            placeholder="Dilekçe, şikayet, talep...",
            value=st.session_state.get("draft_input", ""),
        )
        dosya = st.file_uploader(
            "veya PDF yükle",
            type=["pdf"],
            accept_multiple_files=False,
        )
        submitted = st.form_submit_button("🔍 Analiz Et")

    if submitted:
        _cleanup_tmp(st.session_state.get("_tmp_pdf"))
        st.session_state["_tmp_pdf"] = None
        st.session_state["user_responses"] = {}
        st.session_state["hitl_done"] = False
        _apply_model(
            "Claude API" if st.session_state["model"] == "claude" else "Qwen (Local)"
        )

        tmp_path: str | None = None
        try:
            if dosya is not None:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    f.write(dosya.getbuffer())
                    tmp_path = f.name
                st.session_state["raw_input"] = tmp_path
                st.session_state["input_type"] = "pdf"
                st.session_state["_tmp_pdf"] = tmp_path
                with st.spinner("Pipeline çalışıyor..."):
                    result = _run_pipeline(tmp_path, "pdf")
            else:
                if not (metin or "").strip():
                    st.error("Lütfen evrak metni girin veya PDF yükleyin.")
                    st.stop()
                st.session_state["raw_input"] = metin
                st.session_state["input_type"] = "text"
                with st.spinner("Pipeline çalışıyor..."):
                    result = _run_pipeline(metin, "text")

            st.session_state["result"] = result

            # HITL için PDF yolunu koru; needs_input değilse hemen sil
            if (
                tmp_path
                and result.get("validation_status") != "needs_input"
            ):
                _cleanup_tmp(tmp_path)
                st.session_state["_tmp_pdf"] = None

            error_log = result.get("error_log") or []
            if error_log:
                for err in error_log:
                    st.error(str(err))
        except Exception as exc:
            logger.exception("Pipeline hatası")
            st.error(f"Pipeline hatası: {exc}")
            if tmp_path:
                _cleanup_tmp(tmp_path)
                st.session_state["_tmp_pdf"] = None

with col_out:
    result = st.session_state.get("result")
    if not result:
        st.info("Sol taraftan evrak girin")
    else:
        _render_error_banner(result)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📄 Sınıflandırma",
                "❓ Eksik Bilgi",
                "📚 Mevzuat",
                "✍️ Taslak",
                "🏢 Yönlendirme",
            ]
        )
        with tab1:
            _render_tab_classification(result)
        with tab2:
            _render_tab_hitl(result)
        with tab3:
            _render_tab_mevzuat(result)
        with tab4:
            _render_tab_draft(result)
        with tab5:
            _render_tab_routing(result)
