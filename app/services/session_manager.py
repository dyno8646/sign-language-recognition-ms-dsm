from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4

import numpy as np

from app.core.config import settings
from app.services.postprocess import PostprocessState


@dataclass
class SessionState:
    frames: deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=settings.inference.window_size * 2)
    )
    frame_counter: int = 0
    last_text: str = ""
    last_confidence: float = 0.0
    postprocess: PostprocessState = field(default_factory=PostprocessState)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = Lock()

    def create_session(self) -> str:
        session_id = uuid4().hex
        with self._lock:
            self._sessions[session_id] = SessionState()
        return session_id

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)
