# Jüri Demo Rehberi — KayıDosyalama

TEKNOFEST 2026 · Yapay Zeka Dil Ajanları · 1. Senaryo

## 1. 60 saniyelik pitch

**Problem:** Kamu kurumlarında dilekçe / şikayet / bilgi talebi gibi evraklar elle okunuyor, yanlış birime gidiyor, taslak yazımı zaman alıyor.

**Çözüm:** 6 ajanlı LangGraph pipeline — OCR → sınıflandırma → mevzuat (RAG) → HITL doğrulama → resmî taslak → birim yönlendirme.

**Kanıt:** Yerel LLM (Qwen) veya kural tabanlı fallback; ChromaDB mevzuat araması; Streamlit jüri arayüzü + FastAPI.

## 2. Sunum öncesi checklist (5 dk)

```bash
cd kayidosyalama
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_MODELS="$HOME/.local/share/ollama/models"
uv sync
# Ollama (yerel kurulum)
ollama serve &
# ilk seferde: ollama pull qwen2.5:7b
uv run python scripts/build_vector_store.py   # chroma_db yoksa
uv run uvicorn api.main:app --port 8000 &
uv run streamlit run ui/streamlit_app.py --server.port 8501
```

Doğrulama:

- `http://127.0.0.1:8000/api/v1/health` → `vector_store_available: true`
- `http://127.0.0.1:8501` → Streamlit açılır
- LLM yoksa sistem yine çalışır (heuristic fallback); health `degraded` olabilir — bunu jüriye “offline dayanıklılık” olarak anlatın.

## 3. Canlı demo akışı (~8 dk)

### Senaryo A — Tam evrak (hızlı mutluluk yolu)

Streamlit’e `scripts/demo_runner.py` içindeki **senaryo1** (yıllık izin dilekçesi) metnini yapıştır → İşle.

Gösterilecekler:

1. Evrak türü: `dilekce`
2. Varlıklar (tarih, kişi, kurum, konu)
3. İlgili mevzuat parçaları (Chroma)
4. Taslak üst yazı
5. Hedef birim: `insan_kaynaklari`

### Senaryo B — HITL (eksik alan)

Kısa / eksik bir metin kullanın, örn.:

> Yol çukuru şikayetim var. Hasan Demir.

Beklenen: `needs_input` + sorular → formdan tarih/konu doldur → devam → taslak + birim.

### Senaryo C — Bilgi talebi / şikayet

`demo_runner` senaryo2 (şikayet) veya senaryo3 (bilgi talebi) ile tür + yönlendirme farkını gösterin.

## 4. Terminal yedek demo

UI bozulursa:

```bash
uv run python scripts/demo_runner.py
uv run python scripts/evaluate.py
```

## 5. Jüriye söylenecek teknik noktalar

| Soru | Cevap |
|------|--------|
| Agent’lar birbirini mi çağırıyor? | Hayır — sadece `core/graph.py` orkestre eder; state immutable return. |
| LLM offline olursa? | Classifier / drafter / router heuristic fallback; pipeline çökmez. |
| Mevzuat nereden? | `data/regulations/` → ChromaDB + multilingual-e5 embedding. |
| İnsan nerede? | Validator HITL: `user_questions` → UI yanıtları → entity merge. |
| Lisans | Apache-2.0; OCR opsiyonel (Surya GPL-3). |

## 6. Riskler ve hazır cevaplar

- **Ollama / GPU yok:** Fallback ile demo yapılır; kalite LLM ile artar.
- **İlk embedding yüklemesi yavaş:** İlk istekte model indirilir; önceden `build_vector_store.py` çalıştırın.
- **OCR:** Metin demo yeter; PDF için `uv sync --extra ocr`.

## 7. Değerlendirme kanıtı

```bash
uv run pytest tests/ -v
uv run python scripts/evaluate.py
```

Sonuç: `data/evaluation_results.json` (gitignore’da; jüri öncesi üretip ekranda gösterin).

Hedef metrikler (Qwen2.5:7B, düzeltme sonrası):

| Metrik | Hedef |
|--------|--------|
| Sınıflandırma | ≥ %90 (talep vs bilgi_talebi ayrımı kritik) |
| Taslak | %100 geçerli draft |
| Yönlendirme | %100 birim + gerekçe |

## 8. Sunum günü tek komut hatırlatma

```bash
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_MODELS="$HOME/.local/share/ollama/models"
ollama serve &
cd ~/kayidosyalama
uv run python scripts/build_vector_store.py
uv run uvicorn api.main:app --port 8000 &
uv run streamlit run ui/streamlit_app.py --server.port 8501
```

Tarayıcı: http://127.0.0.1:8501 · Health: http://127.0.0.1:8000/api/v1/health
