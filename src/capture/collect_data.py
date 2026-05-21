"""
Phase 0 — Dataset collection
=============================
Records webcam keypoint sequences for 20 ASL words.

Saves to:  datasets/custom/keypoints/<word>/<seq_index>/<frame_index>.npy
Controls:  Q = quit   SPACE = skip current sequence

Usage:
    python src/capture/collect_data.py
    python src/capture/collect_data.py --config configs/config.yaml
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mediapipe.holistic_tracker import HolisticTracker


# ── defaults (overridden by config file) ──────────────────────────────────────
DEFAULT_WORDS = [
    "hello", "thanks", "yes", "no", "please", "sorry", "help",
    "more", "stop", "good", "bad", "eat", "drink", "home", "love",
    "name", "what", "where", "who", "how",
]
DEFAULT_SEQUENCES = 30
DEFAULT_SEQ_LEN   = 30
DEFAULT_DATA_PATH = Path("datasets/custom/keypoints")
COUNTDOWN_SECS    = 2


def load_config(path: str | None) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def overlay(frame, text, pos, scale=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def make_dirs(data_path: Path, words: list, sequences: int) -> None:
    for word in words:
        for seq in range(sequences):
            (data_path / word / str(seq)).mkdir(parents=True, exist_ok=True)


def collect(config: dict) -> None:
    words      = config.get("data", {}).get("words",           DEFAULT_WORDS)
    sequences  = config.get("data", {}).get("sequences",       DEFAULT_SEQUENCES)
    seq_len    = config.get("data", {}).get("sequence_length", DEFAULT_SEQ_LEN)
    data_path  = Path(config.get("data", {}).get("path",       DEFAULT_DATA_PATH))

    make_dirs(data_path, words, sequences)

    tracker = HolisticTracker()
    cap     = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (device 0).")

    print(f"\nCollecting {len(words)} words × {sequences} seqs × {seq_len} frames")
    print("Controls: Q = quit   SPACE = skip sequence\n")

    try:
        for word in words:
            for seq in range(sequences):

                # ── countdown ─────────────────────────────────────────────────
                deadline = time.time() + COUNTDOWN_SECS
                while time.time() < deadline:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    results = tracker.process_frame(frame)
                    frame   = tracker.draw_landmarks(frame, results)
                    overlay(frame, f"GET READY: {word.upper()}",   (10, 40),  color=(0, 255, 255))
                    overlay(frame, f"Seq {seq+1}/{sequences}",     (10, 75),  scale=0.6)
                    overlay(frame, f"In {deadline-time.time():.1f}s", (10, 105), scale=0.6, color=(180, 180, 180))
                    cv2.imshow("Collection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("\nQuit by user.")
                        return
                    if key == ord(" "):
                        break

                # ── record frames ──────────────────────────────────────────────
                for frame_num in range(seq_len):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    results   = tracker.process_frame(frame)
                    keypoints = tracker.extract_keypoints(results)
                    frame     = tracker.draw_landmarks(frame, results)

                    np.save(str(data_path / word / str(seq) / f"{frame_num}.npy"), keypoints)

                    overlay(frame, f"RECORDING: {word.upper()}", (10, 40), color=(0, 0, 255))
                    overlay(frame, f"Seq {seq+1}/{sequences}  Frame {frame_num+1}/{seq_len}",
                            (10, 75), scale=0.6)
                    cv2.imshow("Collection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\nQuit by user.")
                        return

                print(f"  [{word}]  seq {seq+1:02d}/{sequences} saved")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()

    print(f"\nDone. Data saved to: {data_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Collect keypoint sequences from webcam.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    collect(load_config(args.config))


if __name__ == "__main__":
    main()
