"""
Phase 0 — Real-time inference
==============================
Loads trained BiLSTM, runs webcam, predicts sign word per 30-frame window.

Usage:
    python src/inference/realtime_inference.py
    python src/inference/realtime_inference.py --config configs/config.yaml

Controls:  Q = quit
"""
import argparse
import collections
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mediapipe.holistic_tracker import HolisticTracker
from src.models.bilstm.model import SignBiLSTM


def load_artifacts(config: dict):
    ckpt_dir = Path(config["checkpoints"]["dir"])
    enc_path = ckpt_dir / config["checkpoints"]["label_encoder"]
    mdl_path = ckpt_dir / config["checkpoints"]["best_model"]

    if not enc_path.exists() or not mdl_path.exists():
        raise FileNotFoundError(
            f"Checkpoints not found in {ckpt_dir}. "
            "Run training/train_bilstm.py first."
        )

    with open(enc_path, "rb") as f:
        le = pickle.load(f)

    _dev   = config["inference"]["device"]
    device = "cuda" if torch.cuda.is_available() else "cpu" if _dev == "auto" else _dev

    cfg = config["model"]
    model = SignBiLSTM(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_classes=len(le.classes_),
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(mdl_path, map_location=device))
    model.eval()

    return model, le, device


def overlay(frame, text, pos, scale=0.9, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def run(config: dict) -> None:
    model, le, device = load_artifacts(config)
    seq_len   = config["data"]["sequence_length"]
    threshold = config["inference"]["confidence_threshold"]
    print(f"Model loaded | {len(le.classes_)} classes | device={device}")
    print("Press Q to quit\n")

    tracker  = HolisticTracker()
    cap      = cv2.VideoCapture(0)
    sequence = []
    sentence = collections.deque(maxlen=5)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            results   = tracker.process_frame(frame)
            keypoints = tracker.extract_keypoints(results)
            sequence.append(keypoints)

            # predict on full window
            if len(sequence) == seq_len:
                x = torch.tensor([sequence], dtype=torch.float32).to(device)
                with torch.no_grad():
                    probs         = torch.softmax(model(x), dim=1)[0]
                    conf, idx     = probs.max(0)

                if conf.item() >= threshold:
                    word = le.inverse_transform([idx.item()])[0]
                    if not sentence or sentence[-1] != word:
                        sentence.append(word)

                sequence = []

            frame = tracker.draw_landmarks(frame, results)

            # ── HUD ───────────────────────────────────────────────────────────
            cv2.rectangle(frame, (0, 0), (640, 55), (0, 0, 0), -1)
            text = " ".join(sentence) if sentence else "Waiting for signs..."
            overlay(frame, text, (10, 38))

            # buffer bar
            filled = min(len(sequence), seq_len)
            bar_w  = int(filled / seq_len * 200)
            cv2.rectangle(frame, (10, 62), (210, 74), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, 62), (10 + bar_w, 74), (0, 220, 100), -1)

            cv2.imshow("Sign Language — Phase 0", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)


if __name__ == "__main__":
    main()
