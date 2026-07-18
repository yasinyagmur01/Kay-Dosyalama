"""Classifier Agent için LLM prompt şablonları."""

SYSTEM_PROMPT = """Sen bir kamu evrakı analiz uzmanısın.
Görevin: Gelen evrakı oku, türünü belirle ve gerekli bilgileri çıkar.
Kısıtlar: SADECE geçerli JSON döndür. Açıklama, markdown veya ek metin ekleme.

Evrak tipleri (document_type):
- dilekce: izin, refakat, "arz ederim" içeren kişisel başvurular
- talep: hizmet/işlem/belge talebi ("talep ediyorum/ediyoruz", "rica ederiz", Sayı:TALEP-...)
- sikayet: aksaklık, onarım, gürültü, ödenmeyen ücret, mağduriyet
- bilgi_talebi: YALNIZCA 4982 sayılı Kanun veya açıkça "bilgi edinme" başvurusu
- resmi_yazi: T.C. antetli, Sayı/Konu satırlı kurumlar arası yazılar
- diger: yukarıdakilere uymayanlar

Ayırt etme (öncelik sırası):
- "4982" veya "bilgi edinme birimi/hakkı" → bilgi_talebi
- Sayı satırında TALEP- veya Konu'da "Bilgi Ve Belge Talebi" (4982 YOK) → talep
- "bilgi talebi" tek başına yetmez; 4982/bilgi edinme yoksa talep tercih et
- İzin/refakat + arz ederim → dilekce
- Bozuk yol, gürültü, ödenmeyen ücret → sikayet
- "T.C." + "Sayı:" tek başına resmi_yazi demek değildir; içeriğe bak
"""

USER_PROMPT = """Aşağıdaki kamu evrakını analiz et.

Evrak metni:
---
{raw_text}
---

Yalnızca aşağıdaki JSON formatında yanıt ver:
{{
  "document_type": "...",
  "confidence_score": 0.0,
  "extracted_entities": {{
    "tarih": "...",
    "kurum": "...",
    "kisi": "...",
    "konu": "...",
    "talep": "..."
  }},
  "missing_fields": ["..."],
  "summary": "..."
}}

Zorunlu alanlar: tarih, kurum, kisi, konu, talep.
Eksik veya metinden çıkarılamayan alanları missing_fields listesine yaz.
summary alanı 3-5 Türkçe cümle olmalıdır.
confidence_score 0.0 ile 1.0 arasında olmalıdır.
"""

DOCUMENT_TYPE_DESCRIPTIONS: dict[str, dict[str, str | list[str]]] = {
    "dilekce": {
        "description": "Vatandaş veya personelin resmi talep/izin için yazdığı dilekçe.",
        "keywords": [
            "dilekçe",
            "arz ederim",
            "gereğini saygılarımla arz",
            "yıllık izin",
            "refakat izni",
            "izin kullanmak",
            "belgeler ektedir",
        ],
    },
    "talep": {
        "description": "Belirli bir hizmet, belge veya işlem talebi içeren yazı.",
        "keywords": [
            "talep ediyoruz",
            "talep ediyorum",
            "rica ederiz",
            "rica ederim",
            "istemekteyiz",
            "erişimi talep",
            "şerhinin kaldırılmasını",
        ],
    },
    "sikayet": {
        "description": "Bir aksaklık, mağduriyet veya şikayet bildirimi.",
        "keywords": [
            "şikayet",
            "şikâyet",
            "mağdur",
            "şikayette bulun",
            "bozuk asfalt",
            "onarım yapılmasını",
            "gürültü",
            "denetim yapılmasını",
            "ödenmemiştir",
            "ivedilikle çözüm",
            "trafik kazaları",
        ],
    },
    "bilgi_talebi": {
        "description": "Bilgi edinme veya açıklama isteme amaçlı yazı.",
        "keywords": [
            "bilgi talebi",
            "bilgi edinme",
            "açıklama rica",
            "4982",
            "kanun kapsamında talep",
            "bilgi edinme birimi",
        ],
    },
    "resmi_yazi": {
        "description": "Kurumlar arası resmi yazışma / üst yazı.",
        "keywords": [
            "t.c.",
            "sayı:",
            "konu:",
            "resmi yazı",
            "üst yazı",
            "katılımı zorunludur",
            "ek ödenek",
            "genel müdür",
        ],
    },
    "diger": {
        "description": "Yukarıdaki kategorilere uymayan diğer evraklar.",
        "keywords": ["diğer", "belirsiz"],
    },
}
