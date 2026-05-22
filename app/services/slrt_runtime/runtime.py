from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from app.services.slrt_runtime.decoder import DecodedPrediction, ctc_greedy_decode
from app.services.slrt_runtime.loader import RuntimeAssets, discover_assets, load_yaml
from app.services.slrt_runtime.preprocess import (
    calc_keypoint_count,
    frames_to_video_tensor,
    generate_stub_keypoints,
    sample_to_window,
)

logger = logging.getLogger(__name__)


@dataclass
class RuntimePrediction:
    text: str
    confidence: float
    raw_tokens: list[str]
    latency_ms: int


class SLRTRuntime:
    def __init__(
        self,
        project_root: Path,
        repo_path: Path,
        checkpoint_path: Path,
        vocab_path: Path,
        device: torch.device,
        config_override_path: Path | None = None,
    ) -> None:
        self.project_root = project_root
        self.repo_path = repo_path
        self.device = device
        self.assets: RuntimeAssets = discover_assets(
            project_root=project_root,
            configured_checkpoint=checkpoint_path,
            configured_vocab=vocab_path,
            configured_config=config_override_path,
        )
        self.vocab: list[str] = []
        self.model = None
        self.blank_id = 0
        self.win_size = 16
        self.keypoint_count = 63
        self.raw_size = (224, 224)
        self.loaded = False
        self._status_message = "not initialized"

    @property
    def status_message(self) -> str:
        return self._status_message

    def _bootstrap_imports(self) -> None:
        cslr_dir = self.repo_path / "Online" / "CSLR"
        if not cslr_dir.exists():
            raise FileNotFoundError(f"Missing SLRT Online/CSLR directory: {cslr_dir}")
        cslr_dir_str = str(cslr_dir.resolve())
        if cslr_dir_str not in sys.path:
            sys.path.insert(0, cslr_dir_str)
        import utils.misc as slrt_misc  # type: ignore

        slrt_misc.logger = logging.getLogger("slrt_cslr")

    def _infer_class_count_from_state(self, state: dict) -> int:
        candidate_keys = [
            "recognition_network.visual_head.gloss_output_layer.weight",
            "recognition_network.visual_head_keypoint.gloss_output_layer.weight",
            "recognition_network.visual_head_fuse.gloss_output_layer.weight",
        ]
        for key in candidate_keys:
            if key in state and hasattr(state[key], "shape"):
                return int(state[key].shape[0])
        for key, value in state.items():
            if key.endswith("gloss_output_layer.weight") and hasattr(value, "shape"):
                return int(value.shape[0])
        raise RuntimeError("Could not infer class count from checkpoint state dict")

    def _load_vocab(self, class_count: int) -> list[str]:
        path = self.assets.vocab_path
        if path is None:
            vocab = ["<blank>"] + [f"token_{i}" for i in range(1, class_count)]
            self.blank_id = 0
            return vocab
        with path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        if "<blank>" in vocab:
            self.blank_id = vocab.index("<blank>")
        else:
            vocab = ["<blank>"] + vocab
            self.blank_id = 0
        if len(vocab) != class_count:
            logger.warning("Vocab size %s does not match checkpoint classes %s; adjusting.", len(vocab), class_count)
            if len(vocab) < class_count:
                vocab.extend([f"token_{i}" for i in range(len(vocab), class_count)])
            else:
                vocab = vocab[:class_count]
        return vocab

    def _prepare_cfg(self) -> dict:
        cfg = load_yaml(self.assets.config_path)
        cfg["device"] = self.device
        cfg["task"] = "ISLR"
        cfg.setdefault("data", {})
        cfg["data"]["transform_cfg"] = cfg["data"].get("transform_cfg", {})
        cfg["data"]["transform_cfg"]["color_jitter"] = False
        cfg["data"]["transform_cfg"]["center_crop"] = True
        cfg["data"]["transform_cfg"]["randomcrop_threshold"] = 1
        cfg["data"]["num_output_frames"] = -1
        cfg["data"]["win_size"] = cfg["data"].get("win_size", 16)
        self.win_size = int(cfg["data"]["win_size"])

        use_keypoints = cfg["data"].get("use_keypoints", ["pose", "mouth_half", "hand"])
        self.keypoint_count = calc_keypoint_count(use_keypoints)
        heatmap_cfg = cfg["model"]["RecognitionNetwork"].get("heatmap_cfg", {})
        raw = heatmap_cfg.get("raw_size", [224, 224])
        self.raw_size = (int(raw[1]), int(raw[0]))  # w,h

        rec_cfg = cfg["model"]["RecognitionNetwork"]
        if "s3d" in rec_cfg:
            rec_cfg["s3d"]["pretrained_ckpt"] = "scratch"
        if "keypoint_s3d" in rec_cfg:
            rec_cfg["keypoint_s3d"]["pretrained_ckpt"] = "scratch"
            rec_cfg["keypoint_s3d"]["in_channel"] = self.keypoint_count
        return cfg

    def warmup(self) -> None:
        self._bootstrap_imports()
        raw_ckpt = torch.load(self.assets.checkpoint_path, map_location=self.device)
        state = raw_ckpt.get("model_state", raw_ckpt.get("state_dict", raw_ckpt))
        class_count = self._infer_class_count_from_state(state)
        self.vocab = self._load_vocab(class_count=class_count)
        cfg = self._prepare_cfg()
        from modelling.model import build_model  # type: ignore
        from utils.misc import neq_load_customized  # type: ignore

        self.model = build_model(cfg, cls_num=class_count, word_emb_tab=None)
        neq_load_customized(self.model, state, verbose=False)
        self.model.eval()
        self.loaded = True
        self._status_message = (
            f"loaded config={self.assets.config_path.name} "
            f"checkpoint={self.assets.checkpoint_path.name} "
            f"vocab={self.assets.vocab_path.name} "
            f"device={self.device.type}"
        )
        logger.info("SLRT runtime warmup complete: %s", self._status_message)

    def _select_logits(self, forward_output: dict) -> torch.Tensor:
        for name in [
            "ensemble_last_gloss_logits",
            "fuse_gloss_logits",
            "rgb_gloss_logits",
            "gloss_logits",
        ]:
            value = forward_output.get(name)
            if isinstance(value, torch.Tensor):
                return value
        raise RuntimeError(f"No decodable gloss logits in output keys: {list(forward_output.keys())}")

    def predict(self, frames: Sequence[np.ndarray]) -> RuntimePrediction:
        if not self.loaded or self.model is None:
            raise RuntimeError("SLRT runtime not loaded")
        start = time.perf_counter()
        sampled = sample_to_window(frames, self.win_size)
        video = frames_to_video_tensor(sampled, img_size=224).to(self.device)  # B,T,H,W,C
        kps = generate_stub_keypoints(
            win_size=self.win_size,
            keypoint_count=self.keypoint_count,
            raw_w=self.raw_size[0],
            raw_h=self.raw_size[1],
        ).to(self.device)
        labels = torch.zeros((1,), dtype=torch.long, device=self.device)

        with torch.no_grad():
            forward_output = self.model(
                is_train=False,
                labels=labels,
                sgn_videos=[video],
                sgn_keypoints=[kps],
                epoch=0,
            )
        logits = self._select_logits(forward_output)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("video_shape=%s keypoints_shape=%s logits_shape=%s", tuple(video.shape), tuple(kps.shape), tuple(logits.shape))
        decoded: DecodedPrediction = ctc_greedy_decode(logits=logits, vocab=self.vocab, blank_id=self.blank_id)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("decoded_tokens=%s confidence=%.4f", decoded.tokens, decoded.confidence)
        text = " ".join(decoded.tokens)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return RuntimePrediction(
            text=text,
            confidence=decoded.confidence,
            raw_tokens=decoded.tokens,
            latency_ms=latency_ms,
        )
