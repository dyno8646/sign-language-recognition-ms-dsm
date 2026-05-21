from __future__ import annotations

import base64
import json
from io import BytesIO

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.services.engines.slrt_online_engine import SLRTOnlineEngine


def _encode_frame(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode synthetic frame")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def main() -> None:
    print("== Verify SLR Pipeline ==")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, "SLR", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

    engine = SLRTOnlineEngine()
    try:
        engine.warmup()
        print("[OK] Engine warmup:", engine.status_message)
    except Exception as exc:
        print("[WARN] Engine warmup failed:", exc)
        print("[INFO] Continue checks for graceful API behavior.")

    if engine.ready:
        frames = [frame.copy() for _ in range(16)]
        pred = engine.predict_sequence(frames)
        print("[OK] Inference text:", pred.text)
        print("[OK] Inference confidence:", pred.confidence)
        print("[OK] Tokens:", pred.raw_tokens)
        print("[OK] Latency ms:", pred.latency_ms)

    client = TestClient(app)
    health = client.get("/health")
    print("[OK] /health:", health.status_code, json.dumps(health.json()))

    start = client.post("/api/session/start", json={})
    assert start.status_code == 200, start.text
    session_id = start.json()["session_id"]
    print("[OK] Session started:", session_id)

    payload = {"session_id": session_id, "image_b64": _encode_frame(frame), "timestamp_ms": 0}
    resp = client.post("/api/frame", json=payload)
    print("[OK] /api/frame:", resp.status_code, resp.json())

    stop = client.post("/api/session/stop", json={"session_id": session_id})
    print("[OK] Session stop:", stop.status_code, stop.json())


if __name__ == "__main__":
    main()
