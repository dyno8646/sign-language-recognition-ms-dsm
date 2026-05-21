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
    updated: bool
    engine: str
