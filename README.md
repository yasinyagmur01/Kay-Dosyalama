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

## Ortam değişkenleri

`.env.example` dosyasındaki anahtarları kullanın. Sentetik veri üretimi için `ANTHROPIC_API_KEY` ve isteğe bağlı `USE_ANTHROPIC=true` gerekir; anahtar yoksa script şablon metinlerle devam eder.
