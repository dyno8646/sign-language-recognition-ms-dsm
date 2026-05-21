# Real-Time Sign Language Recognition MVP

Inference-first, demo-ready Sign Language Recognition (SLR) web app using:

- `FangyunWei/SLRT` `Online` as the primary design reference for sliding-window online inference.
- `FangyunWei/SLRT` `NLA-SLR` as a secondary reference for model/checkpoint conventions and post-classification semantics.

This project is intentionally scoped for rapid MVP delivery:

- no training
- no dataset creation
- local execution
- webcam -> text -> speech loop

## Project Structure

```text
app/
  api/                 # HTTP routes
  core/                # Settings and app bootstrap config
  schemas/             # Pydantic contracts
  services/
    engines/           # Inference engine adapters (mock + SLRT)
    inference_service.py
    postprocess.py
    session_manager.py
  utils/               # Image/frame helpers
frontend/
  static/              # JS/CSS
  templates/           # HTML
configs/
  app_config.yaml      # Runtime knobs (fps/window/inference interval)
docs/
  repository_validation.md
  requirements.md
  architecture.md
  implementation_plan.md
scripts/
  run_dev.ps1
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open:

- [http://localhost:8000](http://localhost:8000)

## Engine Modes

Set `SLR_ENGINE` environment variable:

- `mock` (default): stable demo pipeline without heavyweight model dependencies.
- `slrt_online`: attempts SLRT-style inference flow (checkpoint + vocab required).

## Notes

- Browser `SpeechSynthesis` is used for TTS.
- Inference executes periodically on a sliding frame window.
- CPU fallback is automatic when CUDA is unavailable.
