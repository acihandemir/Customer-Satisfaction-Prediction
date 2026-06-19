"""Ortak metin temizleme — baseline ve API'nin AYNI fonksiyonu kullanması şart.

Önemli tasarım kararı: olumsuzluk (negation) sinyalini KORUyoruz.
"değil", "yok", "ama" gibi kelimeler Türkçe duygu için kritik; bu yüzden
stopword temizliği YAPMIYORUZ. char + word n-gram modeli bağlamı yakalar.
"""
import re

_URL_RE = re.compile(r"http\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
# Harf, rakam, boşluk ve birkaç anlamlı noktalama (! ? ...) dışını at.
# Ünlem/soru işareti duygu sinyali taşır, bu yüzden tutuyoruz.
_KEEP_RE = re.compile(r"[^0-9a-zçğıöşü!?\s]", re.IGNORECASE)
_PUNCT_REPEAT_RE = re.compile(r"([!?])\1+")


def light_clean(text: str) -> str:
    """Hafif temizlik: küçük harf, URL at, gürültü işaretlerini at.
    Olumsuzluk kelimeleri ve ünlem/soru işaretleri KORUNUR."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _KEEP_RE.sub(" ", text)
    # "!!!" -> "!" (sinyali tut ama patlamayı sönümle)
    text = _PUNCT_REPEAT_RE.sub(r"\1", text)
    text = _WS_RE.sub(" ", text).strip()
    return text
