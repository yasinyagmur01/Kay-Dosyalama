# KayıDosyalama — Kamu Evrak Akıllı Agent Sistemi

## Proje Hakkında

- **Yarışma:** TEKNOFEST 2026 Yapay Zeka Dil Ajanları Yarışması (1. Senaryo)
- **Problem:** Kamu evrak işleme süreçlerinin manuel, yavaş ve hatalı yapısı
- **Çözüm:** 6 ajanlı LangGraph pipeline — sınıflandırma + taslak + yönlendirme

Sistem, dilekçe / talep / şikayet / bilgi talebi / resmî yazı gibi kamu evraklarını otomatik sınıflandırır, ilgili mevzuatı bulur, resmî yazı taslağı üretir ve doğru birime yönlendirir.

## Mimari

```
                    ┌─────────┐
                    │   OCR   │
                    └────┬────┘
                         │
              (text hazır / hata)
                         │
                    ┌────▼────────┐
                    │ Classifier  │
                    └────┬────────┘
                         │
                    ┌────▼────────┐
                    │  Mevzuat    │
                    └────┬────────┘
                         │
              eksik alan var mı?
                    /         \
                   evet        hayır
                    │           │
             ┌──────▼──────┐    │
             │  Validator  │    │
             │   (HITL)    │    │
             └──────┬──────┘    │
                    │           │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  Drafter  │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  Router   │
                    └─────┬─────┘
                          │
                         END
```

LangGraph `StateGraph` üzerinden ajanlar yalnızca `DocumentState` okur/yazar; birbirini doğrudan çağırmaz.

## Agent Açıklamaları

- **OCR Agent:** PDF/görsel girdiyi Surya-OCR ile metne çevirir; text girdide pass-through yapar.
- **Classifier Agent:** Evrak türünü belirler, varlıkları çıkarır, özet ve güven skoru üretir.
- **Mevzuat Agent:** ChromaDB ile ilgili yönetmelik maddelerini bulur ve yazışma kurallarını çıkarır.
- **Validator Agent:** Eksik zorunlu alanlarda HITL soruları üretir (`needs_input`).
- **Drafter Agent:** Resmî yazışma usullerine uygun Türkçe taslak metin üretir.
- **Routing Agent:** Kural tabanlı + LLM fallback ile hedef birim ve gerekçe belirler.

## Teknoloji Yığını

| Katman | Araç | Lisans | Not |
|--------|------|--------|-----|
| Orkestrasyon | LangGraph 0.2+ | MIT | Merkezi koordinasyon |
| LLM (local) | Qwen2.5-7B-Instruct via Ollama | Apache 2.0 | Türkçe ana model |
| LLM (dev/fallback) | Claude API (claude-sonnet-4-6) | Commercial | Geliştirme hızı için |
| OCR | Surya-OCR | GPL-3 | PDF/görsel → metin |
| Vector DB | ChromaDB | Apache 2.0 | Mevzuat RAG |
| Embedding | multilingual-e5-large | MIT | Türkçe semantic search |
| API | FastAPI | MIT | Backend |
| Demo UI | Streamlit | Apache 2.0 | Jüri sunumu |
| Tracing | LangSmith | Commercial | Debug + görsel pipeline |
| Package mgr | uv | MIT | Hız için |

## Kurulum

### Gereksinimler

- Python 3.11+
- Ollama
- uv

### Adımlar

```bash
git clone https://github.com/kayidosyalama/kayidosyalama.git
cd kayidosyalama
uv sync
cp .env.example .env
# .env dosyasına ANTHROPIC_API_KEY ekle (opsiyonel)
ollama pull qwen2.5:7b
python scripts/generate_synthetic_data.py
python scripts/build_vector_store.py
```

OCR (Surya) gerektiğinde:

```bash
uv sync --extra ocr
```

## Çalıştırma

```bash
# API
uv run uvicorn api.main:app --reload --port 8000

# Demo UI
uv run streamlit run ui/streamlit_app.py --server.port 8501

# Demo senaryoları (terminal)
uv run python scripts/demo_runner.py
```

## Test

```bash
uv run pytest tests/ -v
```

## Değerlendirme

```bash
uv run python scripts/evaluate.py
```

Sonuçlar `data/evaluation_results.json` dosyasına yazılır.

## Lisans

Apache 2.0
