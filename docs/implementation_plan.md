# Phased Implementation Plan

## Phase 0 - Environment Setup

- Clone reference repo (`FangyunWei/SLRT`) for validation and adapters.
- Install Python dependencies.
- Verify FastAPI boot and browser page load.
- Validate checkpoint and vocab paths (if using `slrt_online`).

## Phase 1 - Frontend

- Build webcam preview and connection health.
- Add start/stop controls.
- Add transcript + confidence panels.
- Add speak button and optional auto-speak.

## Phase 2 - Backend

- Build FastAPI routes and session lifecycle.
- Implement frame upload endpoint.
- Implement per-session queue/sliding window.

## Phase 3 - Model Integration

- Add inference engine abstraction.
- Implement `predict_sequence(frames)`.
- Integrate SLRT adapter path (checkpoint + vocab + device).
- Add robust fallback to mock mode.

## Phase 4 - Postprocessing

- Deduplicate repeated token bursts.
- Confidence thresholding.
- Phrase smoothing and cooldown policy.

## Phase 5 - Speech

- Integrate browser TTS.
- Add auto-play switch.
- Prevent repetitive speech loops for identical outputs.

## Performance Strategy

- Low-FPS capture (`~3-5 FPS`).
- Sliding-window inference (`window_size ~ 16`).
- Trigger inference every `N` frames.
- Reduce frame resolution for CPU fallback.

## Debugging and Fallback Strategy

- Missing GPU: switch to CPU automatically.
- CUDA/runtime errors: continue in mock engine mode.
- Webcam denial: show UI guidance and retry controls.
- Slow inference: increase interval, reduce resolution/FPS.
- Model load failure: return explicit diagnostics in health output.
