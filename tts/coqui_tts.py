"""Text-to-speech for predicted glosses (Edge TTS / pyttsx3; Coqui when available)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_BACKEND: Optional[str] = None
_pyttsx3_engine: Optional[object] = None


def _detect_backend() -> str:
    """Pick the best available TTS backend."""
    global _BACKEND
    if _BACKEND:
        return _BACKEND

    try:
        import TTS  # noqa: F401

        _BACKEND = "coqui"
    except ImportError:
        try:
            import edge_tts  # noqa: F401

            _BACKEND = "edge"
        except ImportError:
            import pyttsx3  # noqa: F401

            _BACKEND = "pyttsx3"

    return _BACKEND


def _speak_edge(text: str) -> None:
    import edge_tts

    async def _run() -> None:
        out = _PROJECT_ROOT / "checkpoints" / "_tts_preview.mp3"
        out.parent.mkdir(parents=True, exist_ok=True)
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        await communicate.save(str(out))
        _play_audio(out)

    asyncio.run(_run())


def _speak_pyttsx3(text: str) -> None:
    global _pyttsx3_engine
    import pyttsx3

    if _pyttsx3_engine is None:
        _pyttsx3_engine = pyttsx3.init()
    _pyttsx3_engine.say(text)
    _pyttsx3_engine.runAndWait()


def _speak_coqui(text: str) -> None:
    from TTS.api import TTS

    out = _PROJECT_ROOT / "checkpoints" / "_tts_preview.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
    tts.tts_to_file(text=text, file_path=str(out))
    _play_audio(out)


def _play_audio(path: Path) -> None:
    try:
        import winsound

        if path.suffix.lower() == ".wav":
            winsound.PlaySound(str(path), winsound.SND_FILENAME)
        else:
            import os

            os.startfile(str(path))  # noqa: S606 — Windows default player for mp3
    except Exception:
        pass


def speak_text(text: str) -> None:
    """Speak text using the best available TTS engine."""
    text = text.strip()
    if not text:
        return

    backend = _detect_backend()
    if backend == "coqui":
        _speak_coqui(text)
    elif backend == "edge":
        _speak_edge(text)
    else:
        _speak_pyttsx3(text)


def speak_word(word: str) -> None:
    """Speak a sign gloss or word label aloud."""
    speak_text(word.replace("_", " "))
