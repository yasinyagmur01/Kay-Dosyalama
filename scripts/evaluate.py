"""Sınıflandırma, taslak ve yönlendirme değerlendirme scripti."""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graph import run_pipeline

BEKLENEN_TIPLER: dict[str, str] = {
    "dilekce": "dilekce",
    "talep": "talep",
    "sikayet": "sikayet",
}

SYNTHETIC_ROOT = ROOT / "data" / "synthetic_docs"
RESULTS_PATH = ROOT / "data" / "evaluation_results.json"


def _collect_files(klasorler: list[str] | None = None) -> list[tuple[str, Path]]:
    """Sentetik doküman dosyalarını (beklenen_tip, path) listesi olarak döner."""
    hedefler = klasorler or list(BEKLENEN_TIPLER.keys())
    dosyalar: list[tuple[str, Path]] = []
    for klasor in hedefler:
        klasor_path = SYNTHETIC_ROOT / klasor
        if not klasor_path.is_dir():
            continue
        for path in sorted(klasor_path.glob("*.txt")):
            dosyalar.append((klasor, path))
    return dosyalar


def _collect_all_files() -> list[Path]:
    """Tüm sentetik dokümanları döner."""
    if not SYNTHETIC_ROOT.is_dir():
        return []
    return sorted(SYNTHETIC_ROOT.rglob("*.txt"))


async def evaluate_classification() -> dict[str, Any]:
    """Sınıflandırma doğruluğunu ölçer."""
    dosyalar = _collect_files()
    if not dosyalar:
        print("Uyarı: data/synthetic_docs içinde sınıflandırma dosyası bulunamadı.")
        return {"accuracy": 0.0, "dogru": 0, "toplam": 0, "detay": []}

    detay: list[dict[str, Any]] = []
    dogru = 0
    for beklenen, path in dosyalar:
        try:
            metin = path.read_text(encoding="utf-8")
            sonuc = await run_pipeline(metin, "text")
            tahmin = str(sonuc.get("document_type", "") or "")
            is_dogru = tahmin == BEKLENEN_TIPLER[beklenen]
            if is_dogru:
                dogru += 1
            detay.append(
                {
                    "dosya": str(path.relative_to(ROOT)),
                    "beklenen": beklenen,
                    "tahmin": tahmin,
                    "dogru": is_dogru,
                }
            )
        except Exception as exc:  # noqa: BLE001
            detay.append(
                {
                    "dosya": str(path.relative_to(ROOT)),
                    "beklenen": beklenen,
                    "tahmin": "",
                    "dogru": False,
                    "hata": f"Sınıflandırma değerlendirme hatası: {exc}",
                }
            )

    toplam = len(dosyalar)
    accuracy = (dogru / toplam) if toplam else 0.0
    return {
        "accuracy": accuracy,
        "dogru": dogru,
        "toplam": toplam,
        "detay": detay,
    }


async def evaluate_drafting() -> dict[str, Any]:
    """Taslak üretiminin başarı oranını ölçer."""
    tumu = _collect_all_files()
    if not tumu:
        print("Uyarı: data/synthetic_docs boş; taslak değerlendirmesi atlandı.")
        return {"basari_orani": 0.0, "bos_draft": 0, "toplam": 0}

    ornekler = random.sample(tumu, k=min(10, len(tumu)))
    basarili = 0
    bos_draft = 0
    for path in ornekler:
        try:
            metin = path.read_text(encoding="utf-8")
            sonuc = await run_pipeline(metin, "text")
            draft_text = str(sonuc.get("draft_text", "") or "")
            draft_metadata = sonuc.get("draft_metadata") or {}
            if not draft_text.strip():
                bos_draft += 1
                continue
            if len(draft_text) > 50 and bool(draft_metadata):
                basarili += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Taslak değerlendirme hatası ({path.name}): {exc}")
            bos_draft += 1

    toplam = len(ornekler)
    return {
        "basari_orani": (basarili / toplam) if toplam else 0.0,
        "bos_draft": bos_draft,
        "toplam": toplam,
    }


async def evaluate_routing() -> dict[str, Any]:
    """Yönlendirme çıktılarının geçerliliğini ölçer."""
    tumu = _collect_all_files()
    if not tumu:
        print("Uyarı: data/synthetic_docs boş; yönlendirme değerlendirmesi atlandı.")
        return {"basari_orani": 0.0, "toplam": 0}

    ornekler = random.sample(tumu, k=min(10, len(tumu)))
    basarili = 0
    for path in ornekler:
        try:
            metin = path.read_text(encoding="utf-8")
            sonuc = await run_pipeline(metin, "text")
            target_unit = str(sonuc.get("target_unit", "") or "")
            rationale = str(sonuc.get("routing_rationale", "") or "")
            if target_unit.strip() and len(rationale.strip()) > 10:
                basarili += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Yönlendirme değerlendirme hatası ({path.name}): {exc}")

    toplam = len(ornekler)
    return {
        "basari_orani": (basarili / toplam) if toplam else 0.0,
        "toplam": toplam,
    }


async def evaluate_all() -> None:
    """Üç değerlendirmeyi çalıştırır, yazdırır ve JSON kaydeder."""
    try:
        sinif = await evaluate_classification()
        taslak = await evaluate_drafting()
        yonlendirme = await evaluate_routing()

        sonuclar = {
            "siniflandirma": sinif,
            "taslak": taslak,
            "yonlendirme": yonlendirme,
        }

        sinif_pct = sinif["accuracy"] * 100
        taslak_pct = taslak["basari_orani"] * 100
        yon_pct = yonlendirme["basari_orani"] * 100

        print("=== Değerlendirme Sonuçları ===")
        print(
            f"Sınıflandırma: {sinif['dogru']}/{sinif['toplam']} "
            f"({sinif_pct:.1f}%)"
        )
        print(
            f"Taslak: {taslak['toplam'] - taslak['bos_draft']}/"
            f"{taslak['toplam']} geçerli draft "
            f"({taslak_pct:.1f}%), boş={taslak['bos_draft']}"
        )
        print(
            f"Yönlendirme: başarı {yon_pct:.1f}% "
            f"(toplam={yonlendirme['toplam']})"
        )
        print(
            f"Sınıflandırma: {sinif_pct:.0f}% | "
            f"Taslak: {taslak_pct:.0f}% | "
            f"Yönlendirme: {yon_pct:.0f}%"
        )

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(
            json.dumps(sonuclar, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Sonuçlar kaydedildi: {RESULTS_PATH}")
    except Exception as exc:  # noqa: BLE001
        print(f"Değerlendirme sırasında hata oluştu: {exc}")


if __name__ == "__main__":
    asyncio.run(evaluate_all())
