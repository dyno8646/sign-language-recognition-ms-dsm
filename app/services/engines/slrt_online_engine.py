from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from app.core.config import settings
from app.services.engines.base import BaseInferenceEngine, EnginePrediction


class SLRTOnlineEngine(BaseInferenceEngine):
    """
    Lightweight inference adapter point for SLRT Online checkpoints.
    This class intentionally keeps a strict failure path so the app can
    safely fallback to mock mode when model assets are unavailable.
    """

    name = "slrt_online"

    def __init__(self) -> None:
        self.repo_path = Path(settings.slrt.repo_path)
        self.checkpoint_path = Path(settings.slrt.checkpoint_path)
        self.vocab_path = Path(settings.slrt.vocab_path)
        self.device = self._resolve_device()
        self.vocab: list[str] = []
        self.model = None
        self.ready = False

    def _resolve_device(self) -> torch.device:
        if settings.slrt.device == "cpu":
            return torch.device("cpu")
        if settings.slrt.device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def warmup(self) -> None:
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Missing vocab file: {self.vocab_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {self.checkpoint_path}")
        with self.vocab_path.open("r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        # MVP note: actual SLRT module init requires dataset/config coupling.
        # We only verify checkpoint readability here to keep startup stable.
        _ = torch.load(self.checkpoint_path, map_location=self.device)
        self.ready = True

    def _frames_to_tensor(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        resized = [cv2.resize(f, (224, 224)) for f in frames]
        arr = np.stack(resized, axis=0)  # T,H,W,C
        arr = arr[:, :, :, ::-1]  # BGR -> RGB
        arr = arr.astype(np.float32) / 255.0
        arr = np.transpose(arr, (0, 3, 1, 2))  # T,C,H,W
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)  # B,T,C,H,W

    def predict_sequence(self, frames: Sequence[np.ndarray]) -> EnginePrediction:
        if not self.ready or len(frames) < settings.inference.window_size:
            return EnginePrediction(text="", confidence=0.0)

        # Placeholder decoding logic for MVP integration stage.
        # Once a project-specific SLRT checkpoint wrapper is added, replace
        # this block with true model forward + CTC decoding.
        _ = self._frames_to_tensor(frames[-settings.inference.window_size :])
        if not self.vocab:
            return EnginePrediction(text="", confidence=0.0)
        candidate = self.vocab[min(1, len(self.vocab) - 1)]
        return EnginePrediction(text=str(candidate), confidence=0.35)
