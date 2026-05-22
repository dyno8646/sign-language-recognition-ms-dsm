# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Pure Python sign language recognition pipeline (Phase 0). Core working modules: `models/bilstm/model.py`, `training/train_bilstm.py`, `mediapipe/holistic_tracker.py`, `webcam/webcam_capture.py`. Most other modules (api, frontend, tts, llm, inference) are stubs.

### Known issues

- **MediaPipe namespace conflict**: The local `mediapipe/` directory shadows the installed Google `mediapipe` package. The `holistic_tracker.py` has a workaround (`_import_google_mediapipe()`) that attempts to load Google's mediapipe from site-packages, but it fails because Google's mediapipe `__init__.py` does `from mediapipe.python import *` which resolves to the local package. This requires a webcam anyway and cannot be tested in headless environments.
- **Coqui TTS (`TTS>=0.22.0`)**: Not installable on Python 3.12 (requires Python < 3.12). The `tts/` module is a stub and not needed for Phase 0.

### Running services

- **BiLSTM training**: `python training/train_bilstm.py` (requires keypoint data in `datasets/custom/keypoints/`)
- **FastAPI server**: `uvicorn api.fastapi_server:app --host 0.0.0.0 --port 8000` (currently a stub module)
- **Streamlit UI**: `streamlit run frontend/streamlit_app/app.py` (currently a stub)

### Testing

No test framework or test files are configured. Use `python3 -m py_compile <file>` for syntax validation. The BiLSTM model can be tested with synthetic data:

```python
import torch, sys; sys.path.insert(0, '.')
from models.bilstm.model import SignBiLSTM
model = SignBiLSTM()
x = torch.randn(1, 30, 258)
logits = model(x)  # shape: (1, 20)
```

### Dependencies

Install with `pip install -r requirements.txt`. Note that `TTS>=0.22.0` will fail on Python 3.12; all other dependencies install successfully. Use `mediapipe==0.10.14` (the version with `solutions` API that works with Python 3.12).

### Environment notes

- Python 3.12 is the runtime.
- No GPU/CUDA available in cloud VMs; PyTorch falls back to CPU automatically.
- Webcam-dependent features (`webcam/`, `mediapipe/holistic_tracker.py`) require a physical camera and cannot run in headless cloud environments.
