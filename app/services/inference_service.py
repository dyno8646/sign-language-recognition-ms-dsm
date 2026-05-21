from __future__ import annotations

import logging

import numpy as np

from app.core.config import settings
from app.services.engines.base import BaseInferenceEngine
from app.services.engines.mock_engine import MockInferenceEngine
from app.services.engines.slrt_online_engine import SLRTOnlineEngine
from app.services.postprocess import smooth_prediction
from app.services.session_manager import SessionState

logger = logging.getLogger(__name__)


def build_engine() -> BaseInferenceEngine:
    if settings.inference.engine == "slrt_online":
        engine = SLRTOnlineEngine()
        try:
            engine.warmup()
            return engine
        except Exception as exc:
            logger.exception("SLRT warmup failed, using mock engine: %s", exc)
    engine = MockInferenceEngine()
    engine.warmup()
    return engine


class InferenceService:
    def __init__(self, engine: BaseInferenceEngine | None = None) -> None:
        self.engine = engine or build_engine()

    @property
    def engine_name(self) -> str:
        return self.engine.name

    def push_and_predict(self, state: SessionState, frame: np.ndarray) -> tuple[str, float, bool]:
        state.frames.append(frame)
        state.frame_counter += 1

        should_infer = (
            len(state.frames) >= settings.inference.window_size
            and state.frame_counter % settings.inference.inference_every_n_frames == 0
        )
        if not should_infer:
            return state.last_text, state.last_confidence, False

        pred = self.engine.predict_sequence(list(state.frames))
        text, updated = smooth_prediction(
            candidate_text=pred.text,
            confidence=pred.confidence,
            state=state.postprocess,
            min_confidence=settings.inference.min_confidence,
            cooldown_seconds=settings.inference.cooldown_seconds,
        )
        if updated:
            state.last_text = text
            state.last_confidence = pred.confidence
        return state.last_text, state.last_confidence, updated
