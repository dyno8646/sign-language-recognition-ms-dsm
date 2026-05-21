from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np


def decode_b64_image(image_b64: str) -> Optional[np.ndarray]:
    payload = image_b64
    if "," in image_b64:
        payload = image_b64.split(",", 1)[1]
    try:
        binary = base64.b64decode(payload)
        arr = np.frombuffer(binary, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None
