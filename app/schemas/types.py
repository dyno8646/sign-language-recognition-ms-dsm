from __future__ import annotations

from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    session_id: str
    status: str = "ok"


class FramePayload(BaseModel):
    session_id: str
    image_b64: str = Field(..., description="Base64 JPEG data URL or raw base64")
    timestamp_ms: int | None = None


class PredictionResponse(BaseModel):
    text: str
    confidence: float
    raw_tokens: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    updated: bool
    engine: str
