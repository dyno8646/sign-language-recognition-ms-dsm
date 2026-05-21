# System Architecture

## End-to-End Flow

Laptop Webcam  
-> Frontend (`HTML/CSS/JS`)  
-> Frame Streaming API (`FastAPI`)  
-> Session Frame Buffer (sliding window)  
-> Inference Service (`SLRT` adapter)  
-> Postprocessing (dedupe + smoothing + confidence gate)  
-> Readable text output  
-> Browser speech synthesis

## Frontend Architecture

- `index.html`: webcam view, controls, transcript panel.
- `app.js`:
  - acquires webcam stream (`getUserMedia`)
  - samples frames to an offscreen canvas
  - posts frame payloads to backend at target FPS
  - updates transcript/confidence in UI
  - triggers speech synthesis
- `styles.css`: responsive demo layout.

## Backend Architecture

- `app/main.py`: app factory, static/templates mounting.
- `app/api/routes.py`:
  - `GET /` UI page
  - `GET /health`
  - `POST /api/session/start`
  - `POST /api/session/stop`
  - `POST /api/frame` (streamed frame ingestion + incremental prediction)
- `SessionManager`: per-session buffer + state.
- `InferenceService`: orchestrates frame gating and engine calls.

## Model Wrapper Architecture

- `BaseInferenceEngine`: unified contract `predict_sequence(frames)`.
- `MockInferenceEngine`: deterministic fallback for demo continuity.
- `SLRTOnlineEngine`: integration point for SLRT-style checkpoint inference.

## Preprocessing Pipeline

- Decode JPEG/base64 frame -> NumPy BGR image.
- Optional resize normalization.
- Append frame to fixed-size deque window.
- Run inference every `N` frames (not every frame).

## Inference Scheduling

- Sliding window buffer (`window_size`).
- Trigger interval (`inference_every_n_frames`) to reduce load.
- Cooldown to prevent repetitive noisy outputs.

## Buffering Strategy

- Per-session deque of recent frames.
- Window truncates oldest frames automatically.
- Inference always runs on the latest contiguous clip.

## Speech Pipeline

- Backend returns sanitized text + confidence.
- Frontend invokes `SpeechSynthesisUtterance`.
- Auto-speak toggle and manual "Speak" button are supported.
