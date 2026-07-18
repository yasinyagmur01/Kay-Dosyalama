# KayıDosyalama — TEKNOFEST 2026 (TYDA)

Kamu evrak işleme sistemi: sınıflandırma, mevzuat arama, taslak üretimi ve yönlendirme.

## Kurulum

```bash
uv sync
cp .env.example .env
```

OCR (Surya) gerektiğinde:

```bash
uv sync --extra ocr
```

## Milestone 0 komutları

```bash
uv sync
cp .env.example .env
python scripts/generate_synthetic_data.py
python scripts/build_vector_store.py
pytest tests/test_infrastructure.py -v
```

## Milestone 1 komutları

```bash
pytest tests/test_classifier.py tests/test_gorev1_integration.py -v
```

Görev 1 pipeline (OCR/Drafter hariç):

```python
from core.graph import run_pipeline
result = await run_pipeline("Türkçe evrak metni", "text")
```

## Ortam değişkenleri

`.env.example` dosyasındaki anahtarları kullanın. Sentetik veri üretimi için `ANTHROPIC_API_KEY` ve isteğe bağlı `USE_ANTHROPIC=true` gerekir; anahtar yoksa script şablon metinlerle devam eder.
