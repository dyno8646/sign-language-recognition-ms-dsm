from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
import torch


def calc_keypoint_count(use_keypoints: list[str]) -> int:
    hrnet_parts = {
        "pose": list(range(11)),
        "hand": list(range(91, 133)),
        "mouth": list(range(71, 91)),
        "face_others": list(range(23, 71)),
    }
    for k in ["mouth", "face_others", "hand"]:
        hrnet_parts[f"{k}_half"] = hrnet_parts[k][::2]
        hrnet_parts[f"{k}_1_3"] = hrnet_parts[k][::3]
    total = 0
    for key in sorted(use_keypoints):
        total += len(hrnet_parts[key])
    return total


def sample_to_window(frames: Sequence[np.ndarray], win_size: int) -> list[np.ndarray]:
    if len(frames) == win_size:
        return list(frames)
    if len(frames) < win_size:
        pad_count = win_size - len(frames)
        return list(frames) + [frames[-1]] * pad_count
    idx = np.linspace(0, len(frames) - 1, win_size).astype(np.int32)
    return [frames[i] for i in idx]


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    crop = min(h, w)
    y0 = max(0, (h - crop) // 2)
    x0 = max(0, (w - crop) // 2)
    cropped = img[y0 : y0 + crop, x0 : x0 + crop]
    if crop != size:
        return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    return cropped


def frames_to_video_tensor(frames: Sequence[np.ndarray], img_size: int = 224) -> torch.Tensor:
    stacked = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = _center_crop(rgb, img_size)
        stacked.append(rgb.astype(np.float32) / 255.0)
    arr = np.stack(stacked, axis=0)  # T,H,W,C
    return torch.from_numpy(arr).unsqueeze(0)  # B,T,H,W,C


def generate_stub_keypoints(
    win_size: int,
    keypoint_count: int,
    raw_w: int,
    raw_h: int,
) -> torch.Tensor:
    # Generates deterministic center-biased points so heatmap pipeline is valid.
    kps = np.zeros((1, win_size, keypoint_count, 3), dtype=np.float32)
    center_x = raw_w * 0.5
    center_y = raw_h * 0.5
    radius = min(raw_w, raw_h) * 0.15
    for t in range(win_size):
        for i in range(keypoint_count):
            angle = (i / max(1, keypoint_count)) * 2.0 * math.pi
            kps[0, t, i, 0] = center_x + radius * math.cos(angle)
            kps[0, t, i, 1] = center_y + radius * math.sin(angle)
            kps[0, t, i, 2] = 0.8
    return torch.from_numpy(kps)
