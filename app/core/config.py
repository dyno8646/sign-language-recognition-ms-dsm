from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InferenceSettings:
    engine: str = os.getenv("SLR_ENGINE", "mock")
    target_fps: int = int(os.getenv("SLR_TARGET_FPS", "4"))
    frame_width: int = int(os.getenv("SLR_FRAME_WIDTH", "320"))
    frame_height: int = int(os.getenv("SLR_FRAME_HEIGHT", "240"))
    window_size: int = int(os.getenv("SLR_WINDOW_SIZE", "16"))
    inference_every_n_frames: int = int(os.getenv("SLR_INFER_EVERY_N", "8"))
    min_confidence: float = float(os.getenv("SLR_MIN_CONFIDENCE", "0.25"))
    cooldown_seconds: float = float(os.getenv("SLR_COOLDOWN_SECONDS", "0.8"))


@dataclass
class SLRTSettings:
    repo_path: str = os.getenv("SLRT_REPO_PATH", "third_party_SLRT")
    checkpoint_path: str = os.getenv("SLRT_CHECKPOINT_PATH", "checkpoints/slrt/best.ckpt")
    vocab_path: str = os.getenv("SLRT_VOCAB_PATH", "checkpoints/slrt/vocab.json")
    device: str = os.getenv("SLRT_DEVICE", "auto")


@dataclass
class AppSettings:
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "8000"))
    debug: bool = os.getenv("APP_DEBUG", "true").lower() == "true"
    root_dir: Path = Path(__file__).resolve().parents[2]
    inference: InferenceSettings = InferenceSettings()
    slrt: SLRTSettings = SLRTSettings()


settings = AppSettings()
