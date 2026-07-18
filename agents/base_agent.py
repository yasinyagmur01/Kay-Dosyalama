"""Tüm ajanlar için soyut taban sınıf."""

from abc import ABC, abstractmethod
import logging
import time

from core.state import DocumentState


class BaseAgent(ABC):
    """Pipeline ajanları için ortak invoke ve hata yönetimi."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(f"tyda.agents.{name}")

    async def invoke(self, state: DocumentState) -> dict:
        """Pipeline'dan çağrılan ana metod."""
        start = time.time()
        self.logger.info(f"{self.name} başlatıldı")
        try:
            result = await self._run(state)
            elapsed = time.time() - start
            result["processing_time"] = {
                **state.get("processing_time", {}),
                self.name: elapsed,
            }
            result["current_step"] = self.name
            self.logger.info(f"{self.name} tamamlandı ({elapsed:.2f}s)")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} hatası: {e}")
            return {
                "error_log": state.get("error_log", []) + [f"{self.name}: {str(e)}"],
                "validation_status": "error",
                "current_step": self.name,
            }

    @abstractmethod
    async def _run(self, state: DocumentState) -> dict:
        """Alt sınıflar bu metodu implement eder."""
        ...
