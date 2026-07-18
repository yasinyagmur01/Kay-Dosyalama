"""FastAPI istek/yanıt şemaları."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    """Evrak işleme isteği."""

    text: str | None = None
    input_type: str = "text"
    user_responses: dict = Field(default_factory=dict)


class ProcessResponse(BaseModel):
    """Pipeline çıktı alanları."""

    success: bool
    document_type: str = ""
    confidence_score: float = 0.0
    summary: str = ""
    extracted_entities: dict = Field(default_factory=dict)
    missing_fields: list = Field(default_factory=list)
    relevant_regulations: list = Field(default_factory=list)
    writing_rules: list = Field(default_factory=list)
    draft_type: str = ""
    draft_text: str = ""
    target_unit: str = ""
    routing_rationale: str = ""
    alternative_units: list = Field(default_factory=list)
    validation_status: str = ""
    user_questions: list = Field(default_factory=list)
    processing_time: dict = Field(default_factory=dict)
    error_log: list = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Sağlık kontrolü yanıtı."""

    status: str
    llm_available: bool
    vector_store_available: bool
