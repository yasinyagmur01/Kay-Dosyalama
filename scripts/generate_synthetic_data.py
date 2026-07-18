"""LLM ile sentetik kamu evrakı üretir."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage

from core.llm_client import AnthropicClient

DOC_TYPES: dict[str, str] = {
    "dilekce": "vatandaş dilekçesi",
    "talep": "kurum içi veya vatandaş talep yazısı",
    "sikayet": "şikayet dilekçesi",
    "bilgi_talebi": "bilgi edinme talebi",
    "resmi_yazi": "kurumlar arası resmî yazı",
}

DOCS_PER_TYPE = 10
OUTPUT_ROOT = ROOT / "data" / "synthetic_docs"

GENERATION_PROMPT = """Sen Türk kamu yazışmaları konusunda uzman bir asistansın.
Aşağıdaki tipte gerçekçi bir Türkçe evrak metni yaz.

Evrak tipi: {doc_type_label} ({doc_type})
Sıra no: {index}

Kurallar:
- Tamamen Türkçe yaz
- 100-300 kelime arasında olsun
- Gerçekçi kişi/kurum/konu kullan (uydurma ama inandırıcı)
- Sadece evrak metnini döndür, açıklama ekleme
"""


async def generate_one(client: AnthropicClient, doc_type: str, index: int) -> str:
    """Tek bir sentetik evrak metni üretir."""
    prompt = GENERATION_PROMPT.format(
        doc_type=doc_type,
        doc_type_label=DOC_TYPES[doc_type],
        index=index,
    )
    response = await client.ainvoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


def _fallback_text(doc_type: str, index: int) -> str:
    """API yoksa kullanılacak şablon metin."""
    topics = {
        "dilekce": "yıllık izin talebi",
        "talep": "bilgi ve belge talebi",
        "sikayet": "hizmet aksaması şikayeti",
        "bilgi_talebi": "proje durumu hakkında bilgi talebi",
        "resmi_yazi": "koordinasyon ve görüş talebi",
    }
    topic = topics[doc_type]
    return (
        f"T.C. ÖRNEK BELEDİYESİ\n"
        f"İlgili Birim\n\n"
        f"Sayı : {doc_type.upper()}-{index:03d}\n"
        f"Konu : {topic.title()}\n"
        f"Tarih : 18/07/2026\n\n"
        f"Sayın Yetkili,\n\n"
        f"İşbu yazımız, {topic} kapsamında hazırlanmış örnek bir {DOC_TYPES[doc_type]} "
        f"metnidir. Başvuru numarası {index} olup işlemlerin ilgili mevzuat çerçevesinde "
        f"yürütülmesi hususunda gereğini arz/rica ederim. Başvuruya konu olay/talep; "
        f"vatandaş veya birim ihtiyacına dayalı olup kayıt altına alınmış örnek veridir. "
        f"Gerekli incelemenin yapılarak tarafıma/birimimize bilgi verilmesi önem arz "
        f"etmektedir. Bu metin, sistem testleri için üretilmiş sentetik içeriktir ve "
        f"gerçek bir kişiye ait değildir. Konuya ilişkin ek belgeler talep halinde "
        f"sunulacaktır. Sonuç olarak işlemin en kısa sürede sonuçlandırılmasını "
        f"saygılarımla rica ederim.\n\n"
        f"Ad Soyad\nVatandaş / Yetkili"
    )


async def generate_all() -> int:
    """Tüm evrak tipleri için sentetik dosyalar üretir."""
    use_api = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("USE_ANTHROPIC"))
    client: AnthropicClient | None = None
    if use_api:
        try:
            client = AnthropicClient()
            print("Anthropic API ile üretim başlıyor...")
        except Exception as exc:
            print(f"Anthropic client başlatılamadı, şablon moda geçiliyor: {exc}")
            client = None
    else:
        print("ANTHROPIC_API_KEY bulunamadı; şablon metinlerle üretim yapılıyor...")

    total = 0
    for doc_type in DOC_TYPES:
        out_dir = OUTPUT_ROOT / doc_type
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{doc_type}] {DOCS_PER_TYPE} dosya üretiliyor...")
        for i in range(1, DOCS_PER_TYPE + 1):
            try:
                if client is not None:
                    text = await generate_one(client, doc_type, i)
                else:
                    text = _fallback_text(doc_type, i)
            except Exception as exc:
                print(f"  ! {doc_type}/{i:02d} API hatası, şablon kullanılıyor: {exc}")
                text = _fallback_text(doc_type, i)

            path = out_dir / f"{doc_type}_{i:02d}.txt"
            path.write_text(text + "\n", encoding="utf-8")
            total += 1
            print(f"  ✓ {path.relative_to(ROOT)} ({len(text.split())} kelime)")
    return total


def main() -> None:
    """Script giriş noktası."""
    print("=== Sentetik Evrak Üretimi ===")
    count = asyncio.run(generate_all())
    print(f"\nTamamlandı: {count} dosya üretildi → {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
