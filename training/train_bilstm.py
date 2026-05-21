"""
Train a BiLSTM classifier on collected Holistic keypoint sequences.

Loads data from ``datasets/custom/keypoints/``, trains with an 80/20 stratified
split, and saves the best checkpoint plus a fitted label encoder.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.bilstm.model import SignBiLSTM

DATA_PATH = _PROJECT_ROOT / "datasets" / "custom" / "keypoints"
CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "bilstm_best.pt"
ENCODER_PATH = CHECKPOINT_DIR / "label_encoder.pkl"

SEQUENCE_LENGTH = 30
INPUT_SIZE = 258
NUM_CLASSES = 20
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.5

EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
VAL_SPLIT = 0.2
RANDOM_SEED = 42


def load_sequence(sequence_dir: Path, sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Load all frame ``.npy`` files in a sequence folder and pad/truncate to fixed length.

    Args:
        sequence_dir: Directory containing ``0.npy``, ``1.npy``, ...
        sequence_length: Target number of frames.

    Returns:
        Array of shape ``(sequence_length, features)``.
    """
    frame_files = sorted(sequence_dir.glob("*.npy"), key=lambda p: int(p.stem))
    if not frame_files:
        return np.zeros((sequence_length, INPUT_SIZE), dtype=np.float32)

    frames = [np.load(path).astype(np.float32).reshape(-1) for path in frame_files]
    stacked = np.stack(frames, axis=0)

    if stacked.shape[1] != INPUT_SIZE:
        raise ValueError(
            f"Expected {INPUT_SIZE} features per frame in {sequence_dir}, "
            f"got shape {stacked.shape}"
        )

    if stacked.shape[0] >= sequence_length:
        return stacked[:sequence_length]

    pad = np.zeros((sequence_length - stacked.shape[0], INPUT_SIZE), dtype=np.float32)
    return np.vstack([stacked, pad])


def discover_samples(data_path: Path) -> tuple[list[Path], list[str]]:
    """
    Discover all sequence directories under ``data_path/<word>/<seq>/``.

    Returns:
        Tuple of sequence directory paths and string labels (word names).
    """
    sequence_dirs: list[Path] = []
    labels: list[str] = []

    if not data_path.is_dir():
        raise FileNotFoundError(
            f"Data path not found: {data_path}. "
            "Run webcam/webcam_capture.py first to collect keypoints."
        )

    for word_dir in sorted(data_path.iterdir()):
        if not word_dir.is_dir():
            continue
        word = word_dir.name
        for seq_dir in sorted(word_dir.iterdir(), key=lambda p: int(p.name)):
            if not seq_dir.is_dir():
                continue
            if list(seq_dir.glob("*.npy")):
                sequence_dirs.append(seq_dir)
                labels.append(word)

    if not sequence_dirs:
        raise RuntimeError(f"No .npy sequences found under {data_path}")

    return sequence_dirs, labels


class SignSequenceDataset(Dataset):
    """
    PyTorch dataset of sign keypoint sequences stored as per-frame ``.npy`` files.

    Each item is one sequence folder (30 frames padded/truncated) with an integer label.
    """

    def __init__(
        self,
        sequence_dirs: list[Path],
        labels: list[int],
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> None:
        """
        Args:
            sequence_dirs: Paths to ``.../<word>/<sequence_id>/`` folders.
            labels: Encoded class index per sequence.
            sequence_length: Fixed temporal length.
        """
        self.sequence_dirs = sequence_dirs
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.sequence_dirs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        seq_dir = self.sequence_dirs[index]
        sequence = load_sequence(seq_dir, self.sequence_length)
        x = torch.from_numpy(sequence).float()
        y = int(self.labels[index])
        return x, y


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_epoch(
    model: SignBiLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    count = 0

    for batch_x, batch_y in tqdm(loader, desc="Train", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        count += batch_x.size(0)

    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(
    model: SignBiLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate validation loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0

    for batch_x, batch_y in tqdm(loader, desc="Val", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        correct += (logits.argmax(dim=1) == batch_y).sum().item()
        count += batch_x.size(0)

    mean_loss = total_loss / max(count, 1)
    acc = correct / max(count, 1)
    return mean_loss, acc


def main() -> None:
    """Load data, train BiLSTM, and save best checkpoint."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    sequence_dirs, word_labels = discover_samples(DATA_PATH)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(word_labels)

    if len(label_encoder.classes_) != NUM_CLASSES:
        print(
            f"Warning: found {len(label_encoder.classes_)} classes, "
            f"expected {NUM_CLASSES}. Training with discovered labels."
        )

    (
        train_dirs,
        val_dirs,
        train_labels,
        val_labels,
    ) = train_test_split(
        sequence_dirs,
        encoded_labels,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=encoded_labels,
    )

    train_dataset = SignSequenceDataset(train_dirs, train_labels.tolist())
    val_dataset = SignSequenceDataset(val_dirs, val_labels.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = SignBiLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=len(label_encoder.classes_),
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    print("=" * 60)
    print("BiLSTM Training")
    print(f"  Device:      {device}")
    print(f"  Samples:     {len(train_dataset)} train / {len(val_dataset)} val")
    print(f"  Classes:     {list(label_encoder.classes_)}")
    print(f"  Checkpoint:  {MODEL_PATH}")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                    "input_size": INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "num_classes": len(label_encoder.classes_),
                    "dropout": DROPOUT,
                    "sequence_length": SEQUENCE_LENGTH,
                },
                MODEL_PATH,
            )
            with open(ENCODER_PATH, "wb") as f:
                pickle.dump(label_encoder, f)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining finished. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
