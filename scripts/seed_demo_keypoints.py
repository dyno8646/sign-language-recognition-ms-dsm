"""Generate minimal synthetic keypoint data for pipeline smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from webcam.webcam_capture import SEQUENCE_LENGTH, WORDS

DATA_PATH = _PROJECT_ROOT / "datasets" / "custom" / "keypoints"
SEQUENCES_PER_WORD = 5  # minimum for 80/20 stratified split across 20 classes
FRAMES_PER_SEQUENCE = SEQUENCE_LENGTH
FEATURES = 258


def main() -> None:
    """Write 2 synthetic sequences per word (for stratified train/val split)."""
    rng = np.random.default_rng(42)
    total = 0

    for word in WORDS:
        for seq in range(1, SEQUENCES_PER_WORD + 1):
            seq_dir = DATA_PATH / word / str(seq)
            seq_dir.mkdir(parents=True, exist_ok=True)
            base = rng.standard_normal(FRAMES_PER_SEQUENCE * FEATURES).astype(np.float32)
            base = base.reshape(FRAMES_PER_SEQUENCE, FEATURES) + (hash(word) % 100) * 0.01

            for frame in range(FRAMES_PER_SEQUENCE):
                np.save(seq_dir / f"{frame}.npy", base[frame])
                total += 1

    print(f"Seeded {len(WORDS)} words x {SEQUENCES_PER_WORD} sequences x {FRAMES_PER_SEQUENCE} frames")
    print(f"Saved {total} .npy files under {DATA_PATH}")


if __name__ == "__main__":
    main()
