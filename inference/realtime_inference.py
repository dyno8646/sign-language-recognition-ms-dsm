"""
Real-time sign language recognition from webcam using a trained BiLSTM.

Accumulates 30 frames of Holistic keypoints, runs inference, and overlays
the predicted gloss on the live feed.
"""

from __future__ import annotations

import pickle
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mediapipe.holistic_tracker import HolisticTracker
from models.bilstm.model import SignBiLSTM

SEQUENCE_LENGTH = 30
INPUT_SIZE = 258
MODEL_PATH = _PROJECT_ROOT / "checkpoints" / "bilstm_best.pt"
ENCODER_PATH = _PROJECT_ROOT / "checkpoints" / "label_encoder.pkl"
WINDOW_NAME = "Sign Language Recognition"
QUIT_KEY = ord("q")
CONFIDENCE_THRESHOLD = 0.5


def load_model_and_encoder(device: torch.device) -> tuple[SignBiLSTM, object]:
    """Load trained BiLSTM weights and label encoder from checkpoints."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run: python training/train_bilstm.py"
        )
    if not ENCODER_PATH.is_file():
        raise FileNotFoundError(
            f"Label encoder not found at {ENCODER_PATH}. "
            "Run: python training/train_bilstm.py"
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = SignBiLSTM(
        input_size=checkpoint.get("input_size", INPUT_SIZE),
        hidden_size=checkpoint.get("hidden_size", 128),
        num_layers=checkpoint.get("num_layers", 2),
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint.get("dropout", 0.5),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder


def predict_sequence(
    model: SignBiLSTM,
    label_encoder: object,
    sequence: np.ndarray,
    device: torch.device,
) -> tuple[str, float]:
    """
    Predict sign label from a (seq_len, 258) keypoint sequence.

    Returns:
        Predicted word and confidence score.
    """
    tensor = torch.from_numpy(sequence).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

    word = str(label_encoder.inverse_transform([idx.item()])[0])
    return word, conf.item()


def main() -> None:
    """Run live webcam inference loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, label_encoder = load_model_and_encoder(device)
    keypoint_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Live inference started. Press Q to quit.")
    predicted_word = "..."
    confidence = 0.0

    try:
        with HolisticTracker() as tracker:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                results = tracker.process_frame(frame)
                keypoints = tracker.extract_keypoints(results)
                keypoint_buffer.append(keypoints)

                display = tracker.draw_landmarks(frame, results)

                if len(keypoint_buffer) == SEQUENCE_LENGTH:
                    sequence = np.stack(list(keypoint_buffer), axis=0)
                    predicted_word, confidence = predict_sequence(
                        model, label_encoder, sequence, device
                    )

                color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                label = (
                    f"{predicted_word.upper()} ({confidence:.0%})"
                    if len(keypoint_buffer) == SEQUENCE_LENGTH
                    else f"Buffering {len(keypoint_buffer)}/{SEQUENCE_LENGTH}..."
                )
                cv2.rectangle(display, (0, 0), (display.shape[1], 50), (0, 0, 0), -1)
                cv2.putText(
                    display,
                    label,
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, display)

                if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
