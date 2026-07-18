"""Pytest ortak fixture'ları."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.state import DocumentState, create_initial_state

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_dilekce_text() -> str:
    """Test için örnek dilekçe metni."""
    path = FIXTURES_DIR / "sample_dilekce.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (
        "Sayın Müdürlük,\n"
        "Ankara ilinde ikamet eden vatandaş olarak yıllık izin talebimi "
        "sunarım. Ailevi nedenlerle 10 iş günü izin kullanmak istiyorum. "
        "Gereğini arz ederim.\nAyşe Yılmaz"
    )


@pytest.fixture
def sample_talep_text() -> str:
    """Test için örnek talep metni."""
    path = FIXTURES_DIR / "sample_talep.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (
        "Sayın Bilgi İşlem Müdürlüğü,\n"
        "Strateji birimi adına gösterge panosuna erişim talep ederiz. "
        "Bilgi ve gereğini rica ederim.\nMehmet Kaya"
    )


@pytest.fixture
def sample_sikayet_text() -> str:
    """Test için örnek şikayet metni."""
    path = FIXTURES_DIR / "sample_sikayet.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (
        "Sayın Fen İşleri Müdürlüğü,\n"
        "Mahallemizdeki yol bozukluğu nedeniyle şikayette bulunuyoruz. "
        "Onarım yapılmasını arz ederim.\nAli Demir"
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    """LLM çağrısı yapmadan test için mock."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content="mock yanıt"))
    mock.with_structured_output = MagicMock(return_value=mock)
    mock.health_check = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def initial_state_dilekce(sample_dilekce_text: str) -> DocumentState:
    """Dilekçe metni ile başlangıç state'i."""
    return create_initial_state(sample_dilekce_text, "text")
