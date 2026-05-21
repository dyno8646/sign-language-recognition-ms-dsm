from __future__ import annotations

import logging

import numpy as np

from app.core.config import settings
from app.services.engines.base import BaseInferenceEngine
from app.services.engines.mock_engine import MockInferenceEngine
from app.services.engines.slrt_online_engine import SLRTOnlineEngine
from app.services.postprocess import dedupe_tokens, smooth_prediction
from app.services.session_manager import SessionState

logger = logging.getLogger(__name__)


def build_engine() -> BaseInferenceEngine:
    if settings.inference.engine == "mock":
        engine = MockInferenceEngine()
        engine.warmup()
        return engine
    engine = SLRTOnlineEngine()
    try:
        engine.warmup()
    except Exception:
        logger.exception("SLRT warmup failed; engine stays unavailable")
    return engine


class InferenceService:
    def __init__(self, engine: BaseInferenceEngine | None = None) -> None:
        self.startup_error: str | None = None
        if engine is not None:
            self.engine = engine
            return
        self.engine = build_engine()
        if not self.engine.ready:
            self.startup_error = self.engine.status_message

    @property
    def engine_name(self) -> str:
        return self.engine.name

    def engine_status(self) -> dict:
        device = getattr(self.engine, "device", None)
        device_name = str(device) if device is not None else "unknown"
        return {
            "engine": self.engine.name,
            "ready": self.engine.ready,
            "status_message": self.engine.status_message,
            "startup_error": self.startup_error,
            "device": device_name,
        }

    def push_and_predict(
        self, state: SessionState, frame: np.ndarray
    ) -> tuple[str, float, list[str], int, bool]:
        state.frames.append(frame)
        state.frame_counter += 1

        infer_every = settings.inference.inference_every_n_frames
        if not self.engine.ready:
            infer_every = max(infer_every, 12)

        should_infer = (
            len(state.frames) >= settings.inference.window_size
            and state.frame_counter % infer_every == 0
        )
        if not should_infer:
            return state.last_text, state.last_confidence, state.last_raw_tokens, state.last_latency_ms, False

        pred = self.engine.predict_sequence(list(state.frames))
        tokens = dedupe_tokens(pred.raw_tokens)
        candidate_text = " ".join(tokens) if tokens else pred.text
        text, updated = smooth_prediction(
            candidate_text=candidate_text,
            confidence=pred.confidence,
            state=state.postprocess,
            min_confidence=settings.inference.min_confidence,
            cooldown_seconds=settings.inference.cooldown_seconds,
        )
        if updated:
            state.last_text = text
            state.last_confidence = pred.confidence
            state.last_raw_tokens = tokens
            state.last_latency_ms = pred.latency_ms
        return (
            state.last_text,
            state.last_confidence,
            state.last_raw_tokens,
            state.last_latency_ms,
            updated,
        )
