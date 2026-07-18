"""Classifier Agent için LLM prompt şablonları."""

SYSTEM_PROMPT = """Sen bir kamu evrakı analiz uzmanısın.
Görevin: Gelen evrakı oku, türünü belirle ve gerekli bilgileri çıkar.
Kısıtlar: SADECE geçerli JSON döndür. Açıklama, markdown veya ek metin ekleme.

Evrak tipleri (document_type):
- dilekce
- talep
- sikayet
- bilgi_talebi
- resmi_yazi
- diger
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
        "keywords": ["dilekçe", "arz ederim", "gereğini arz", "izin", "başvuru"],
    },
    "talep": {
        "description": "Belirli bir hizmet, belge veya işlem talebi içeren yazı.",
        "keywords": ["talep", "rica ederim", "talep ederiz", "istemekteyiz"],
    },
    "sikayet": {
        "description": "Bir aksaklık, mağduriyet veya şikayet bildirimi.",
        "keywords": ["şikayet", "şikâyet", "mağdur", "şikayette bulun"],
    },
    "bilgi_talebi": {
        "description": "Bilgi edinme veya açıklama isteme amaçlı yazı.",
        "keywords": ["bilgi talebi", "bilgi edinme", "açıklama rica", "4982"],
    },
    "resmi_yazi": {
        "description": "Kurumlar arası resmi yazışma / üst yazı.",
        "keywords": ["resmi yazı", "üst yazı", "ilgi", "gereği", "makam"],
    },
    "diger": {
        "description": "Yukarıdaki kategorilere uymayan diğer evraklar.",
        "keywords": ["diğer", "belirsiz"],
    },
}
