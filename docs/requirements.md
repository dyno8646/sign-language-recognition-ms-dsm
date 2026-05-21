# Requirement Elicitation

## Functional Requirements

- Access webcam from browser.
- Show live video preview and recognition status.
- Capture frames continuously at controlled FPS.
- Send sampled frames to backend for buffered inference.
- Run sliding-window recognition repeatedly (continuous mode).
- Display recognized text/gloss in real time.
- Display confidence score and last update timestamp.
- Start/stop recognition controls.
- Manual and auto speech playback of recognized text.
- Session-based buffering to isolate multiple users/tabs.

## Non-Functional Requirements

- Low perceived latency (incremental updates every few seconds).
- Local-first execution (no cloud requirement).
- Stable fallback behavior under missing GPU/dependency failures.
- Responsive lightweight UI (HTML/CSS/Vanilla JS).
- Modular backend services and pluggable inference engines.
- Demo-friendly observability (health/status endpoints, clear error states).

## Technical Requirements

- Backend: Python + FastAPI.
- Frontend: HTML/CSS/Vanilla JavaScript.
- Inference runtime: PyTorch-compatible adapter.
- Video transport: JPEG frame upload over HTTP.
- Postprocessing: duplicate suppression + confidence filtering + phrase smoothing.
- Speech: Browser `SpeechSynthesis` API.

## Constraints

- Inference only; no training/fine-tuning.
- Must be buildable quickly (hours, not days).
- Must support laptop webcam.
- Must degrade gracefully to CPU.
- Must avoid over-engineering.
- Must keep architecture clear and extensible for later checkpoint upgrades.
