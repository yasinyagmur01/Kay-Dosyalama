"""TYDA pipeline özel exception sınıfları."""


class TYDABaseException(Exception):
    """Tüm TYDA exception'larının taban sınıfı."""


class OCRException(TYDABaseException):
    """OCR işlemi sırasında oluşan hatalar."""


class ClassificationException(TYDABaseException):
    """Evrak sınıflandırma sırasında oluşan hatalar."""


class VectorStoreException(TYDABaseException):
    """Vektör veritabanı işlemleri sırasında oluşan hatalar."""


class DraftingException(TYDABaseException):
    """Resmi yazı taslaklama sırasında oluşan hatalar."""
