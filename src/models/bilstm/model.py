"""
SignBiLSTM
==========
Bidirectional LSTM for isolated sign language word classification.

Architecture:
  Input  (batch, seq_len, 258)
    → BiLSTM ×2 layers  → (batch, seq_len, 256)
    → last timestep     → (batch, 256)
    → LayerNorm
    → Dropout
    → Linear            → (batch, num_classes)
"""
import torch
import torch.nn as nn


class SignBiLSTM(nn.Module):
    """Two-layer bidirectional LSTM for sign word classification."""

    def __init__(
        self,
        input_size:  int   = 258,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        num_classes: int   = 20,
        dropout:     float = 0.5,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.drop       = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, num_classes)
        """
        out, _  = self.lstm(x)          # (batch, seq_len, hidden*2)
        out     = self.layer_norm(out[:, -1])  # last timestep
        out     = self.drop(out)
        return self.fc(out)
