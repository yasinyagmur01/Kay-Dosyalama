"""Mevzuat corpus'unu ChromaDB'ye indeksleyen script."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_store.chroma_client import get_client, get_or_create_collection
from vector_store.indexer import index_regulations, search_regulations

REGULATIONS_DIR = ROOT / "data" / "regulations"
TEST_QUERY = "resmi yazı formatı başlık"


async def main_async() -> None:
    """Indexleme ve test sorgusunu çalıştırır."""
    print("=== Vector Store Kurulumu ===")
    print("1) ChromaDB başlatılıyor...")
    await get_client()
    await get_or_create_collection()
    print("   ChromaDB hazır.")

    print(f"2) Mevzuat indeksleniyor: {REGULATIONS_DIR}")
    count = await index_regulations(str(REGULATIONS_DIR))
    print(f"   {count} chunk eklendi.")

    print(f"3) Test sorgusu: \"{TEST_QUERY}\"")
    results = await search_regulations(TEST_QUERY, n_results=5)
    if not results:
        print("   Sonuç bulunamadı.")
        return

    print(f"   {len(results)} sonuç:")
    for i, item in enumerate(results, start=1):
        preview = item["content"][:160].replace("\n", " ")
        print(
            f"   [{i}] kaynak={item['source']} "
            f"relevance={item['relevance']:.3f}\n"
            f"       {preview}..."
        )


def main() -> None:
    """Script giriş noktası."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
