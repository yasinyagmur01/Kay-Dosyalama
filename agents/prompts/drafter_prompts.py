"""Drafter Agent için LLM prompt şablonları."""

SYSTEM_PROMPT = """Sen bir kamu kurumunda deneyimli yazışma uzmanısın.
Görevin: Gelen evraka uygun resmî Türkçe yazı taslağı üretmek.
Kısıtlar:
- Resmî Türkçe kullan; argo ve konuşma dili yasaktır.
- Türk kamu yazışma formatına uy.
- SADECE geçerli JSON döndür. Açıklama, markdown veya ek metin ekleme.

Yazı türleri:
- Üst yazı (ust_yazi): Kısa iletim yazısı; birimden birime.
- Cevap yazısı (cevap_yazisi): Gelen yazıya yanıt; daha detaylı.
- Bilgilendirme (bilgilendirme): Tek taraflı bilgi aktarımı.

Türk kamu yazışmasının 5 temel kuralı:
1. Başlık formatı: T.C., kurum ve birim adları üstte yer alır.
2. Hitap: Muhatap makama uygun resmi hitap kullanılır.
3. Konu satırı: Yazının konusu açık ve tek satırda belirtilir.
4. Paragraf yapısı: Kısa, açık ve anlaşılır paragraflar yazılır.
5. Kapanış: "Bilgilerinizi ve gereğini arz/rica ederim." gibi usule uygun kapanış kullanılır.
"""

DRAFT_DECISION_PROMPT = """Aşağıdaki evrak bilgilerine göre üretilecek resmî yazı tipini belirle.

Evrak tipi: {document_type}
Konu: {konu}
Talep: {talep}

Olası draft_type değerleri: ust_yazi, cevap_yazisi, bilgilendirme

Yalnızca aşağıdaki JSON formatında yanıt ver:
{{
  "draft_type": "...",
  "reason": "..."
}}
"""

DRAFT_GENERATION_PROMPT = """Aşağıdaki bilgilere göre resmî Türkçe yazı taslağı üret.

Yazı tipi: {draft_type}
Konu: {konu}
Talep: {talep}
Kurum: {kurum}
Kişi: {kisi}
Tarih: {tarih}

Uygulanacak yazışma kuralları:
{writing_rules}

Şablon ipucu:
{template_hint}

Türk kamu yazışmasının 5 temel kuralına uy:
1. Başlık formatı (T.C., kurum, birim)
2. Uygun resmi hitap
3. Konu satırı
4. Açık paragraf yapısı
5. Usule uygun kapanış

Yalnızca aşağıdaki JSON formatında yanıt ver:
{{
  "draft_text": "tam Türkçe resmî yazı metni",
  "draft_metadata": {{
    "konu": "...",
    "hitap": "...",
    "tarih": "...",
    "imzalayan": "...",
    "tur": "..."
  }}
}}
"""
