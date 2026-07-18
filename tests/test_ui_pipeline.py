import asyncio
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.graph import run_pipeline

METINLER = [
    ("dilekce", """Sayın İnsan Kaynakları Müdürlüğü,
Ankara Vergi Dairesi Başkanlığı'nda gelir uzmanı olarak görev yapmaktayım. 
15/07/2026 - 25/07/2026 tarihleri arasında 8 iş günü yıllık izin 
kullanmak istiyorum.
Gereğini saygılarımla arz ederim.
Mehmet Kaya, Gelir Uzmanı, Sicil No: 12345"""),

    ("bilgi_talebi", """Sayın Belediye Başkanlığı Bilgi Edinme Birimi,
Atatürk Bulvarı genişletme projesiyle ilgili başlangıç-bitiş tarihleri,
maliyet ve ihale bilgilerini 4982 sayılı Kanun kapsamında talep ediyorum.
Ayşe Demir, Çankaya Mah. Gül Sok. No:5, Ankara"""),

    ("sikayet", """Sayın Fen İşleri Müdürlüğü,
Keçiören Bağlarbası Mahallesi Kiraz Sokak'ta 3 aydır bozuk asfalt
nedeniyle trafik kazaları yaşanmaktadır. Onarım yapılmasını talep ediyoruz.
Ali Yıldız, 18/07/2026"""),

    ("talep_eksik", """Sayın Bilgi İşlem Müdürlüğü,
Strateji Geliştirme Birimi adına gösterge panosu erişimi talep ediyoruz.
Gereğini rica ederiz."""),

    ("resmi_yazi", """T.C. ANKARA VALİLİĞİ
İl Afet ve Acil Durum Müdürlüğü
Sayı: 2026/1847 Konu: Tatbikat Programı
28-29 Temmuz 2026 deprem tatbikatına tüm kurumların katılımı zorunludur.
İrtibat bilgilerini 22/07/2026'ya kadar iletin.
İl Afet Müdürü"""),

    ("sikayet2", """Sayın Çevre Müdürlüğü,
Tekstil fabrikası gece 23:00-06:00 arası gürültü ve koku yapıyor.
Denetim yapılmasını talep ediyoruz.
Fatma Çelik, Sincan/Ankara, 17/07/2026"""),

    ("dilekce2", """Sayın Sağlık Müdürlüğü,
Eşim 10/07/2026 ameliyat oldu, refakat izni kullanmak istiyorum.
Belgeler ektedir.
Kadir Şahin, Ankara, 18/07/2026"""),

    ("resmi_yazi2", """T.C. MALİYE BAKANLIĞI
Bütçe ve Mali Kontrol Genel Müdürlüğü
Sayı: 2026/4521 Konu: Ek Ödenek Talebi
2026 bütçesinde öngörülmeyen harcamalar için 500.000 TL
ek ödenek talebinde bulunulması zorunludur.
Genel Müdür"""),

    ("talep2", """Sayın Tapu Müdürlüğü,
Çankaya Kızılay Mah. 142 ada 7 parseldeki ipotek şerhinin
kaldırılmasını talep ediyorum. Ödeme dekontu ektedir.
Zeynep Yılmaz, TC: 12345678901"""),

    ("sikayet3", """Sayın Milli Eğitim Müdürlüğü,
Öğretmenlerin Mart 2026'dan bu yana ek ders ücretleri ödenmemiştir.
İvedilikle çözüm talep ediyoruz.
Okul Müdürü Ahmet Arslan, Etimesgut/Ankara"""),
]

async def test_all():
    sonuclar = []
    for beklenen_tur, metin in METINLER:
        try:
            result = await run_pipeline(metin, "text")
            sonuclar.append({
                "beklenen": beklenen_tur,
                "gelen_tur": result.get("document_type"),
                "guven": f"%{result.get('confidence_score', 0)*100:.0f}",
                "validation": result.get("validation_status"),
                "birim": result.get("target_unit"),
                "draft_var": bool(result.get("draft_text")),
                "hata": result.get("error_log") or [],
            })
        except Exception as e:
            sonuclar.append({
                "beklenen": beklenen_tur,
                "gelen_tur": "EXCEPTION",
                "guven": "—",
                "validation": "—",
                "birim": "—",
                "draft_var": False,
                "hata": [str(e)],
            })
    
    print("\n" + "="*70)
    print(f"{'Beklenen':<15} {'Gelen':<15} {'Güven':<8} {'Birim':<20} {'Draft':<6} {'Hata'}")
    print("="*70)
    for s in sonuclar:
        hata_str = s['hata'][0][:30] if s['hata'] else "—"
        print(f"{s['beklenen']:<15} {s['gelen_tur']:<15} {s['guven']:<8} "
              f"{str(s['birim']):<20} {str(s['draft_var']):<6} {hata_str}")
    
    hatali = [s for s in sonuclar if s['hata'] or s['gelen_tur'] == 'EXCEPTION']
    print(f"\nToplam: {len(sonuclar)} | Hatalı: {len(hatali)}")
    return sonuclar

if __name__ == "__main__":
    asyncio.run(test_all())
