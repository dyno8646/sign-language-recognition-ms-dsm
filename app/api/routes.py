from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.schemas.types import FramePayload, PredictionResponse, SessionResponse
from app.services.inference_service import InferenceService
from app.services.session_manager import SessionManager
from app.utils.image_io import decode_b64_image

router = APIRouter()

templates = Jinja2Templates(directory="frontend/templates")
session_manager = SessionManager()
inference_service = InferenceService()


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@router.get("/health")
def health() -> dict:
    status = inference_service.engine_status()
    return {"status": "ok", **status}


@router.post("/api/session/start", response_model=SessionResponse)
def start_session() -> SessionResponse:
    session_id = session_manager.create_session()
    return SessionResponse(session_id=session_id)


@router.post("/api/session/stop", response_model=SessionResponse)
def stop_session(payload: SessionResponse) -> SessionResponse:
    session_manager.close_session(payload.session_id)
    return SessionResponse(session_id=payload.session_id)


@router.post("/api/frame", response_model=PredictionResponse)
def ingest_frame(payload: FramePayload) -> PredictionResponse:
    state = session_manager.get(payload.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    frame = decode_b64_image(payload.image_b64)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    text, confidence, raw_tokens, latency_ms, updated = inference_service.push_and_predict(state, frame)
    return PredictionResponse(
        text=text,
        confidence=confidence,
        raw_tokens=raw_tokens,
        latency_ms=latency_ms,
        updated=updated,
        engine=inference_service.engine_name,
    )
