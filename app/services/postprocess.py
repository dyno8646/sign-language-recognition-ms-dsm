from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class PostprocessState:
    last_text: str = ""
    last_emit_at: float = 0.0


def normalize_phrase(text: str) -> str:
    text = " ".join(text.strip().split())
    return text.lower()


def dedupe_tokens(tokens: list[str]) -> list[str]:
    cleaned: list[str] = []
    for t in tokens:
        token = t.strip().upper()
        if not token:
            continue
        if cleaned and cleaned[-1] == token:
            continue
        cleaned.append(token)
    return cleaned


def smooth_prediction(
    candidate_text: str,
    confidence: float,
    state: PostprocessState,
    min_confidence: float,
    cooldown_seconds: float,
) -> tuple[str, bool]:
    now = time.time()
    norm = normalize_phrase(candidate_text)
    if not norm or confidence < min_confidence:
        return state.last_text, False
    if norm == state.last_text and (now - state.last_emit_at) < cooldown_seconds:
        return state.last_text, False
    state.last_text = norm
    state.last_emit_at = now
    return state.last_text, True
