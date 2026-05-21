"""
Phase 0 — Train BiLSTM on collected keypoint sequences.

Usage:
    python training/train_bilstm.py
    python training/train_bilstm.py --config configs/config.yaml
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.bilstm.model import SignBiLSTM


# ── Dataset ───────────────────────────────────────────────────────────────────
class SignSequenceDataset(Dataset):
    """Loads saved .npy keypoint sequences."""

    def __init__(self, sequences: list, labels: list, seq_len: int = 30) -> None:
        self.sequences = sequences
        self.labels    = labels
        self.seq_len   = seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        T, F = seq.shape

        # pad or truncate to fixed length
        if T < self.seq_len:
            pad = np.zeros((self.seq_len - T, F), dtype=np.float32)
            seq = np.vstack([seq, pad])
        else:
            seq = seq[: self.seq_len]

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ── Data loading ──────────────────────────────────────────────────────────────
def load_dataset(data_path: Path) -> tuple[list, list]:
    sequences, labels = [], []
    words = sorted(p.name for p in data_path.iterdir() if p.is_dir())
    if not words:
        return sequences, labels
    print(f"Words found ({len(words)}): {words}")
    for word in words:
        for seq_dir in sorted((data_path / word).iterdir(), key=lambda p: int(p.name)):
            frames = sorted(seq_dir.glob("*.npy"), key=lambda p: int(p.stem))
            if not frames:
                continue
            seq = np.stack([np.load(str(f)) for f in frames])  # (T, 258)
            sequences.append(seq)
            labels.append(word)
    return sequences, labels


# ── Training loop ─────────────────────────────────────────────────────────────
def train(config: dict) -> None:
    data_path  = Path(config["data"]["path"])
    ckpt_dir   = Path(config["checkpoints"]["dir"])
    seq_len    = config["data"]["sequence_length"]
    epochs     = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr         = config["training"]["learning_rate"]
    val_split  = config["training"]["val_split"]

    _dev = config["training"]["device"]
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
        if _dev == "auto"
        else _dev
    )

    ckpt_dir.mkdir(exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    sequences, raw_labels = load_dataset(data_path)
    if not sequences:
        print(f"No data found at {data_path}. Run collect_data.py first.")
        return

    le     = LabelEncoder()
    labels = le.fit_transform(raw_labels)

    X_tr, X_val, y_tr, y_val = train_test_split(
        sequences, labels, test_size=val_split, stratify=labels, random_state=42
    )

    tr_dl  = DataLoader(SignSequenceDataset(X_tr,  y_tr,  seq_len), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(SignSequenceDataset(X_val, y_val, seq_len), batch_size=batch_size)

    # ── model ─────────────────────────────────────────────────────────────────
    model_cfg = config["model"]
    model = SignBiLSTM(
        input_size=model_cfg["input_size"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_classes=len(le.classes_),
        dropout=model_cfg["dropout"],
    ).to(device)

    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    best_acc = 0.0
    print(f"\nTraining on {device} | {len(X_tr)} train  {len(X_val)} val | {len(le.classes_)} classes")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(tr_dl, desc=f"Epoch {epoch:03d}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        # validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total   += yb.size(0)

        avg_loss = total_loss / len(tr_dl)
        val_acc  = correct / total
        scheduler.step(avg_loss)

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / config["checkpoints"]["best_model"])
            with open(ckpt_dir / config["checkpoints"]["label_encoder"], "wb") as f:
                pickle.dump(le, f)
            marker = "  ✓ saved"

        print(f"Epoch {epoch:03d}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}{marker}")

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"Checkpoint: {ckpt_dir / config['checkpoints']['best_model']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args   = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
