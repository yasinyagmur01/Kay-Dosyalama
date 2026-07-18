"""Uygulama yapılandırma sınıfları (.env üzerinden)."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """LLM bağlantı ayarları."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    anthropic_api_key: str = ""


class ChromaConfig(BaseSettings):
    """ChromaDB ayarları."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    db_path: str = Field(default="./chroma_db", validation_alias="CHROMA_DB_PATH")
    collection_name: str = "mevzuat_corpus"


class AppConfig(BaseSettings):
    """Uygulama genel ayarları."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    log_level: str = "INFO"
    langsmith_project: str = Field(
        default="tyda-agent-system",
        validation_alias="LANGCHAIN_PROJECT",
    )
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = True


class Config:
    """Tüm yapılandırmaları birleştiren kök ayar nesnesi."""

    def __init__(self) -> None:
        self.llm: LLMConfig = LLMConfig()
        self.chroma: ChromaConfig = ChromaConfig()
        self.app: AppConfig = AppConfig()


settings = Config()
