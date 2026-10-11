"""Public request/response types for the single-turn inference API."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    prompt: str = Field(min_length=1, max_length=20000)
    max_new_tokens: int | None = Field(default=None, ge=1)
    do_sample: bool | None = None
    temperature: float | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    repetition_penalty: float | None = Field(default=None, gt=0)

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace text")
        return value


class ChatResponse(BaseModel):
    request_id: str
    kind: Literal["base", "sft", "dpo"]
    answer: str
    input_tokens: int
    output_tokens: int
    generation_seconds: float
    finish_reason: Literal["stop", "length"]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message: str = Field(min_length=1, max_length=20000)

    @field_validator("message")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("message must contain non-whitespace text")
        return value


class AnalyzeResponse(BaseModel):
    request_id: str
    intent: Literal["cancel_order", "change_order", "track_order", "get_refund", "track_refund"]
    order_id: str | None
    generation_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ready"]
    kind: Literal["base", "sft", "dpo"]
    model: str
    max_input_tokens: int
    max_new_tokens: int
