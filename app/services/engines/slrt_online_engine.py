from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from app.core.config import settings
from app.services.engines.base import BaseInferenceEngine, EnginePrediction
from app.services.slrt_runtime.runtime import SLRTRuntime

logger = logging.getLogger(__name__)


class SLRTOnlineEngine(BaseInferenceEngine):
    name = "slrt_online"

    def __init__(self) -> None:
        self.project_root = settings.root_dir
        self.repo_path = Path(settings.slrt.repo_path)
        self.checkpoint_path = Path(settings.slrt.checkpoint_path)
        self.vocab_path = Path(settings.slrt.vocab_path)
        self.device = self._resolve_device()
        self._ready = False
        self._status_message = "not initialized"
        self.runtime: SLRTRuntime | None = None

    def _resolve_device(self) -> torch.device:
        if settings.slrt.device == "cpu":
            return torch.device("cpu")
        if settings.slrt.device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def warmup(self) -> None:
        try:
            self.runtime = SLRTRuntime(
                project_root=self.project_root,
                repo_path=self.project_root / self.repo_path,
                checkpoint_path=self.project_root / self.checkpoint_path,
                vocab_path=self.project_root / self.vocab_path,
                device=self.device,
                config_override_path=(
                    (self.project_root / settings.slrt.config_path) if settings.slrt.config_path else None
                ),
            )
            self.runtime.warmup()
            self._ready = True
            self._status_message = self.runtime.status_message
        except Exception as exc:
            self._ready = False
            self._status_message = f"warmup failed: {exc}"
            logger.exception("SLRT warmup failed")
            raise

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def status_message(self) -> str:
        return self._status_message

    def predict_sequence(self, frames: Sequence[np.ndarray]) -> EnginePrediction:
        if not self.ready or self.runtime is None:
            return EnginePrediction(text="", confidence=0.0, raw_tokens=[], latency_ms=0)
        if len(frames) < settings.inference.window_size:
            return EnginePrediction(text="", confidence=0.0, raw_tokens=[], latency_ms=0)
        pred = self.runtime.predict(frames[-settings.inference.window_size :])
        return EnginePrediction(
            text=pred.text,
            confidence=pred.confidence,
            raw_tokens=pred.raw_tokens,
            latency_ms=pred.latency_ms,
        )
