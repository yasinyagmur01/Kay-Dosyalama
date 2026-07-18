"""Üç sabit demo senaryosunu terminalde çalıştırır."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graph import run_pipeline

DEMO_SCENARIOS: dict[str, str] = {
    "senaryo1": (
        "T.C. ANKARA BÜYÜKŞEHİR BELEDİYESİ\n"
        "Personel ve Eğitim Dairesi Başkanlığı\n\n"
        "Sayı : 2026/PERS-1847\n"
        "Konu : Yıllık izin talebi\n"
        "Tarih : 18/07/2026\n\n"
        "Sayın Yetkili,\n\n"
        "Ben Ayşe Yılmaz, Personel ve Eğitim Dairesi Başkanlığı bünyesinde "
        "memur olarak görev yapmaktayım. 657 sayılı Devlet Memurları Kanunu "
        "uyarınca hak ettiğim yıllık iznimi kullanmak istiyorum. İzin "
        "tarihlerim 04/08/2026 ile 22/08/2026 arasındadır. Toplam on beş "
        "iş günü talep etmekteyim. Görev yerim Çankaya hizmet binasıdır "
        "ve iletişim numarım 0532 111 22 33'tür. İzin süresince acil "
        "durumlarda ulaşılabilir olacağım. Yerime bakacak personel olarak "
        "Mehmet Kaya görevlendirilmiştir. Gerekli belgeler ekte "
        "sunulmuştur. İşbu dilekçem ile yıllık izin talebimin uygun "
        "görülerek onaylanmasını saygılarımla arz ederim.\n\n"
        "Ad Soyad : Ayşe Yılmaz\n"
        "T.C. Kimlik No : 12345678901\n"
        "Adres : Çankaya / Ankara\n"
        "İmza"
    ),
    "senaryo2": (
        "T.C. İSTANBUL BÜYÜKŞEHİR BELEDİYESİ\n"
        "Başvuru Sahibi : Hasan Demir\n"
        "Adres : Kadıköy / İstanbul\n"
        "Telefon : 0533 444 55 66\n\n"
        "Konu : Yol bozulması ve çukur şikayeti\n\n"
        "Sayın Yetkili,\n\n"
        "Mahallemizde bulunan ana cadde üzerindeki asfalt yüzey ciddi "
        "şekilde bozulmuş olup çok sayıda derin çukur oluşmuştur. Özellikle "
        "yağmur sonrası biriken su nedeniyle yaya ve araç trafiği tehlike "
        "altına girmektedir. Bölgede yaşayan vatandaşlar olarak defalarca "
        "sözlü bildirimde bulunulmasına rağmen kalıcı bir onarım "
        "yapılmamıştır. Çocukların okul yolu da aynı güzergâh üzerindedir. "
        "Acilen yol onarımının yapılması, geçici uyarı tabelalarının "
        "konulması ve aydınlatmanın kontrol edilmesi gerekmektedir. "
        "Şikayetimin incelenerek ilgili birime iletilmesini ve tarafıma "
        "bilgi verilmesini talep ederim. Olayın meydana geldiği kesin "
        "tarih şu an hatırlanamadığı için belgede belirtilmemiştir; "
        "ancak sorun haftalardır devam etmektedir. Gereğinin yapılmasını "
        "saygılarımla rica ederim.\n\n"
        "Hasan Demir"
    ),
    "senaryo3": (
        "T.C. İZMİR VALİLİĞİ\n"
        "Bilgi Edinme Birimi\n\n"
        "Sayı : 2026/BE-0921\n"
        "Konu : Bilgi erişim talebi\n"
        "Tarih : 15/07/2026\n\n"
        "Sayın Yetkili,\n\n"
        "4982 sayılı Bilgi Edinme Hakkı Kanunu kapsamında, İzmir ili "
        "Bornova ilçesinde 2025 yılı içinde tamamlanan yeşil alan ve "
        "park düzenleme projelerine ilişkin kamuoyu bilgilendirmesi "
        "talep etmekteyim. Özellikle proje adları, uygulama bütçeleri, "
        "yüklenici firma bilgileri ve tamamlanma tarihlerini içeren "
        "özet listenin tarafıma elektronik ortamda iletilmesini "
        "istiyorum. Başvuru sahibinin adı Elif Kara'dır; T.C. kimlik "
        "numarası 10987654321, açık adresi Bornova / İzmir, e-posta "
        "adresi elif.kara@ornek.gov.tr ve telefon numarası "
        "0532 777 88 99'dur. Talep edilen bilgiler kişisel veri "
        "içermemekte olup kamuya açık niteliktedir. Başvurumun yasal "
        "süresi içinde sonuçlandırılmasını ve cevap yazısının "
        "yukarıdaki e-posta adresine gönderilmesini saygılarımla "
        "arz ederim.\n\n"
        "Elif Kara\n"
        "İmza"
    ),
}


async def run_demo_scenario(isim: str, metin: str) -> None:
    """Tek bir demo senaryosunu çalıştırıp sonucu yazdırır."""
    print(f"\n{'=' * 60}")
    print(f"SENARYO: {isim}")
    print(f"{'=' * 60}")
    try:
        sonuc = await run_pipeline(metin, "text")
        print(f"Evrak Türü    : {sonuc['document_type']}")
        print(f"Güven Skoru   : {sonuc['confidence_score']:.0%}")
        ozet = sonuc.get("summary") or ""
        print(f"Özet          : {ozet[:100]}...")
        print(f"Yazı Türü     : {sonuc['draft_type']}")
        print(f"Hedef Birim   : {sonuc['target_unit']}")
        sureler = sonuc.get("processing_time") or {}
        print(f"Süre          : {sum(sureler.values()):.2f}s")
        if sonuc.get("validation_status") == "needs_input":
            print(f"HITL Soruları : {sonuc['user_questions']}")
        if sonuc.get("error_log"):
            print(f"Hatalar       : {sonuc['error_log']}")
    except Exception as exc:  # noqa: BLE001
        print(f"Senaryo çalıştırılırken hata oluştu: {exc}")


async def main() -> None:
    """Tüm demo senaryolarını sırayla çalıştırır."""
    try:
        print("KayıDosyalama — Demo Senaryoları")
        print("TEKNOFEST 2026 | Yapay Zeka Dil Ajanları Yarışması\n")
        for isim, metin in DEMO_SCENARIOS.items():
            await run_demo_scenario(isim, metin)
        print(f"\n{'=' * 60}")
        print("Tüm demo senaryoları tamamlandı.")
    except Exception as exc:  # noqa: BLE001
        print(f"Demo çalıştırılırken hata oluştu: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
