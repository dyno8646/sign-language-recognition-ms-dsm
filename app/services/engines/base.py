from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class EnginePrediction:
    text: str
    confidence: float


class BaseInferenceEngine(ABC):
    name: str = "base"

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_sequence(self, frames: Sequence[np.ndarray]) -> EnginePrediction:
        raise NotImplementedError
