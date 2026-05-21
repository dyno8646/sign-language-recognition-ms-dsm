"""
Collect sign language training keypoints from a webcam.

Captures 30 sequences × 30 frames per word for 20 ASL glosses and saves
flattened Holistic keypoints to ``datasets/custom/keypoints/``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on path when run as: python webcam/webcam_capture.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mediapipe.holistic_tracker import HolisticTracker

WORDS: list[str] = [
    "hello",
    "thanks",
    "yes",
    "no",
    "please",
    "sorry",
    "help",
    "more",
    "stop",
    "good",
    "bad",
    "eat",
    "drink",
    "home",
    "love",
    "name",
    "what",
    "where",
    "who",
    "how",
]

SEQUENCES: int = 30
SEQUENCE_LENGTH: int = 30
DATA_PATH: Path = _PROJECT_ROOT / "datasets" / "custom" / "keypoints"

WINDOW_NAME = "Sign Language Data Collection"
COUNTDOWN_SECONDS = 2
QUIT_KEY = ord("q")


def _ensure_directories() -> None:
    """Create the base data path and per-word folders."""
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    for word in WORDS:
        (DATA_PATH / word).mkdir(parents=True, exist_ok=True)


def _draw_status(frame: np.ndarray, text: str, subtext: str = "") -> np.ndarray:
    """Draw status lines on the top-left of the frame."""
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 90), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if subtext:
        cv2.putText(
            output,
            subtext,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        "Press Q to quit",
        (10, output.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return output


def _wait_countdown(
    cap: cv2.VideoCapture,
    tracker: HolisticTracker,
    word: str,
    sequence_num: int,
) -> bool:
    """
    Show a get-ready countdown for ``COUNTDOWN_SECONDS`` seconds.

    Returns:
        False if the user pressed Q to quit, True otherwise.
    """
    deadline = time.time() + COUNTDOWN_SECONDS
    label = f"Get ready: {word.upper()} (seq {sequence_num}/{SEQUENCES})"

    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            print("Error: could not read from webcam during countdown.")
            return False

        results = tracker.process_frame(frame)
        display = tracker.draw_landmarks(frame, results)
        remaining = max(0.0, deadline - time.time())
        display = _draw_status(
            display,
            label,
            f"Starting in {remaining:.1f}s",
        )
        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == QUIT_KEY:
            return False

    return True


def _capture_sequence(
    cap: cv2.VideoCapture,
    tracker: HolisticTracker,
    word: str,
    sequence_num: int,
) -> bool:
    """
    Capture ``SEQUENCE_LENGTH`` frames and save keypoints per frame.

    Returns:
        False if the user quit early, True on success.
    """
    seq_dir = DATA_PATH / word / str(sequence_num)
    seq_dir.mkdir(parents=True, exist_ok=True)

    for frame_num in range(SEQUENCE_LENGTH):
        ok, frame = cap.read()
        if not ok:
            print(f"Error: could not read frame {frame_num} for '{word}' seq {sequence_num}.")
            return False

        results = tracker.process_frame(frame)
        keypoints = tracker.extract_keypoints(results)

        out_path = seq_dir / f"{frame_num}.npy"
        np.save(out_path, keypoints)

        display = tracker.draw_landmarks(frame, results)
        display = _draw_status(
            display,
            f"Recording: {word.upper()} (seq {sequence_num}/{SEQUENCES})",
            f"Frame {frame_num + 1}/{SEQUENCE_LENGTH}",
        )
        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == QUIT_KEY:
            return False

    return True


def collect_data() -> None:
    """Run the full multi-word, multi-sequence keypoint collection loop."""
    _ensure_directories()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check that a camera is connected.")

    print("=" * 60)
    print("Sign Language Keypoint Collection")
    print(f"  Words:           {len(WORDS)}")
    print(f"  Sequences/word:  {SEQUENCES}")
    print(f"  Frames/sequence: {SEQUENCE_LENGTH}")
    print(f"  Save path:       {DATA_PATH}")
    print("=" * 60)

    try:
        with HolisticTracker() as tracker:
            for word_idx, word in enumerate(WORDS, start=1):
                print(f"\n[{word_idx}/{len(WORDS)}] Word: {word}")

                for sequence_num in range(1, SEQUENCES + 1):
                    print(
                        f"  Sequence {sequence_num}/{SEQUENCES} — "
                        f"countdown {COUNTDOWN_SECONDS}s..."
                    )

                    if not _wait_countdown(cap, tracker, word, sequence_num):
                        print("Collection stopped by user (Q pressed).")
                        return

                    if not _capture_sequence(cap, tracker, word, sequence_num):
                        print("Collection stopped by user (Q pressed).")
                        return

                    print(f"  Sequence {sequence_num}/{SEQUENCES} saved.")

                print(f"Finished all sequences for '{word}'.")

        print("\nCollection complete.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Entry point for webcam keypoint collection."""
    collect_data()


if __name__ == "__main__":
    main()
