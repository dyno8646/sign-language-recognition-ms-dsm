from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DecodedPrediction:
    tokens: list[str]
    confidence: float


def ctc_greedy_decode(logits: torch.Tensor, vocab: list[str], blank_id: int) -> DecodedPrediction:
    # logits [B, T, V]
    probs = torch.softmax(logits, dim=-1)
    max_prob, indices = torch.max(probs, dim=-1)
    ids = indices[0].tolist()
    confs = max_prob[0].tolist()

    dedup_ids: list[int] = []
    dedup_confs: list[float] = []
    for i, idx in enumerate(ids):
        if not dedup_ids or idx != dedup_ids[-1]:
            dedup_ids.append(idx)
            dedup_confs.append(confs[i])

    keep_tokens: list[str] = []
    keep_confs: list[float] = []
    for idx, conf in zip(dedup_ids, dedup_confs):
        if idx == blank_id or idx < 0 or idx >= len(vocab):
            continue
        keep_tokens.append(str(vocab[idx]))
        keep_confs.append(float(conf))

    confidence = float(sum(keep_confs) / len(keep_confs)) if keep_confs else 0.0
    return DecodedPrediction(tokens=keep_tokens, confidence=confidence)
