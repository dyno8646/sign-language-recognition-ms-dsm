from __future__ import annotations

from itertools import cycle
from typing import Sequence

import numpy as np

from app.services.engines.base import BaseInferenceEngine, EnginePrediction


class MockInferenceEngine(BaseInferenceEngine):
    name = "mock"

    def __init__(self) -> None:
        self._phrases = cycle(
            [
                ("hello nice to meet you", 0.62),
                ("thank you", 0.74),
                ("how are you", 0.66),
                ("good morning", 0.71),
            ]
        )

    def warmup(self) -> None:
        return

    def predict_sequence(self, frames: Sequence[np.ndarray]) -> EnginePrediction:
        if len(frames) < 4:
            return EnginePrediction(text="", confidence=0.0, raw_tokens=[], latency_ms=0)
        text, conf = next(self._phrases)
        return EnginePrediction(text=text, confidence=conf, raw_tokens=text.split(), latency_ms=5)
