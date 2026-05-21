"""Bidirectional LSTM classifier for sign language keypoint sequences."""

from __future__ import annotations

import torch
import torch.nn as nn


class SignBiLSTM(nn.Module):
    """
    Bidirectional LSTM classifier for fixed-length Holistic keypoint sequences.

    Expects per-frame feature vectors (default 258-d) and outputs per-sequence
    class logits.
    """

    def __init__(
        self,
        input_size: int = 258,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 20,
        dropout: float = 0.5,
    ) -> None:
        """
        Initialize the BiLSTM classifier.

        Args:
            input_size: Number of features per frame (Holistic keypoints).
            hidden_size: LSTM hidden dimension per direction.
            num_layers: Number of stacked BiLSTM layers.
            num_classes: Number of sign word classes.
            dropout: Dropout probability after LayerNorm and between LSTM layers.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, features)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        # lstm_out: (batch, seq_len, hidden_size * 2)
        lstm_out, _ = self.lstm(x)
        # Use final timestep representation for sequence classification
        last_step = lstm_out[:, -1, :]
        normalized = self.layer_norm(last_step)
        dropped = self.dropout(normalized)
        logits = self.fc(dropped)
        return logits
