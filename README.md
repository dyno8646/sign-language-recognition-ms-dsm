# 🤟 Sign Language Recognition — Real-Time AI Pipeline

> **MS DSM Capstone Project** · Webcam → MediaPipe → BiLSTM → Gloss → LLM → Speech

Real-time sign language recognition system that captures hand/body gestures via webcam,
recognizes continuous sign sequences, converts them to natural language, and speaks the
output live.

---

## 🎯 Project aim

````
Person signs in webcam
        ↓
AI understands continuous signs (MediaPipe Holistic + Temporal Model)
        ↓
AI converts to gloss  →  YOU GO WHERE
        ↓
LLM converts gloss → natural language  →  "Where are you going?"
        ↓
Text shown live  +  Speech generated live (Coqui TTS)
````

---

## 🔁 Full system pipeline

| Stage | Component | Tech |
|---|---|---|
| 1 | Webcam capture | OpenCV |
| 2 | Landmark extraction | MediaPipe Holistic — 543 3D keypoints |
| 3 | Preprocessing | Normalization · sequence buffer · temporal alignment |
| 4 | Sequence model | BiLSTM (Phase 0) → Temporal Transformer (Phase 2) |
| 5 | Decoder | CTC / Attention decoder → gloss output |
| 6 | LLM refinement | Gloss → fluent natural language |
| 7 | TTS | Coqui TTS → live audio |
| 8 | UI | Streamlit + FastAPI WebSockets |

---

## 🗓 Delivery phases

### ⚡ Phase 0 — Working demo (weeks 1–3) ← current focus
- Webcam capture + MediaPipe Holistic extraction
- Collect 20-word dataset (WLASL subset or self-recorded)
- Train BiLSTM classifier on keypoint sequences
- Wire Coqui TTS for predicted word
- Streamlit UI with live landmark overlay

### 🟢 Phase 1 — Continuous recognition (weeks 4–9)
- Replace classifier with CTC decoder for sentence-level sequences
- Sliding window inference + confidence filter
- Expand vocabulary: WLASL-100 / MSASL
- Dataset agent: auto-download and normalize datasets
- FastAPI + WebSocket streaming backend

### 🔵 Phase 2 — Transformer + LLM (weeks 10–17)
- Port to Temporal Transformer (reference: SLRT, CorrNet)
- Attention decoder for gloss prediction
- LLM grammar refiner: gloss → fluent text via Claude API
- Experiment agent: hyperparameter sweeps + W&B logging
- PHOENIX-2014T dataset for sentence-level training

### 🟣 Phase 3 — Production (ongoing)
- ONNX export + TensorRT optimization
- Docker Compose (API + model + TTS)
- Deployment agent: containerization + smoke tests
- Multi-language support: ISL, ASL, DGS

---

## 📁 Project structure

````
sign-language-recognition-ms-dsm/
│
├── webcam/                    # Frame capture + realtime stream
│   ├── webcam_capture.py
│   ├── frame_buffer.py
│   └── realtime_stream.py
│
├── mediapipe/                 # Holistic tracker + landmark extraction
│   ├── holistic_tracker.py
│   ├── landmark_extractor.py
│   └── pose_visualizer.py
│
├── preprocessing/             # Sequence builder, normalization, augmentation
│   ├── sequence_builder.py
│   ├── normalization.py
│   ├── augmentation.py
│   └── temporal_alignment.py
│
├── models/                    # BiLSTM (phase 0) → Transformer (phase 2)
│   ├── bilstm/               ← start here
│   ├── transformer/
│   ├── slrt/                 ← ref: FangyunWei/SLRT
│   └── corrnet/              ← ref: hulianyuyy/CorrNet (CVPR 2023)
│
├── inference/
│   ├── realtime_inference.py
│   ├── sliding_window.py
│   ├── gloss_decoder.py
│   └── confidence_filter.py
│
├── llm/                       # Gloss → natural language
│   ├── gloss_to_text.py
│   ├── grammar_refiner.py
│   └── contextual_translation.py
│
├── tts/                       # Coqui TTS integration
│   ├── coqui_tts.py
│   └── live_audio.py
│
├── api/                       # FastAPI + WebSockets
│   ├── fastapi_server.py
│   ├── websocket_api.py
│   └── streaming_routes.py
│
├── frontend/
│   └── streamlit_app/
│       └── app.py
│
├── agents/                    # Agentic automation
│   ├── dataset_agent/        # Download + normalize datasets
│   ├── labeling_agent/       # Auto-label via MediaPipe + OCR
│   ├── experiment_agent/     # Hyperparameter tuning + eval
│   └── deployment_agent/    # Docker + ONNX + FastAPI deploy
│
├── datasets/                  # WLASL · MSASL · PHOENIX · ISL
├── training/                  # Train scripts per model
├── configs/                   # YAML configs per experiment
├── checkpoints/               # Saved model weights
├── notebooks/                 # EDA + experimentation
├── docker/                    # Compose + Dockerfiles
├── requirements.txt
└── README.md
````

---

## 🤖 Agents

| Agent | Automates |
|---|---|
| **Dataset agent** | Download WLASL/MSASL/PHOENIX, normalize annotations, format conversion |
| **Labeling agent** | Run MediaPipe on raw video, OCR subtitle tracks, auto-label gloss sequences |
| **Experiment agent** | Hyperparameter tuning, training runs, WER eval, Slack + Linear notifications |
| **Deployment agent** | ONNX export, TensorRT opt, Docker build, FastAPI smoke tests, GitHub PR |

---

## 🛠 Tech stack

| Area | Tool |
|---|---|
| CV capture | OpenCV |
| Tracking | MediaPipe Holistic |
| ML framework | PyTorch + CUDA |
| Sequence model | BiLSTM → Temporal Transformer |
| Decoder | CTC / Attention |
| LLM refiner | Claude API |
| TTS | Coqui TTS |
| Backend | FastAPI + WebSockets |
| UI | Streamlit |
| Optimization | ONNX / TensorRT |
| Containers | Docker |
| Dev | Cursor AI |
| Agentic coding | Claude |

---

## 📚 Reference repos

- [FangyunWei/SLRT](https://github.com/FangyunWei/SLRT) — Temporal transformer for online SLR (Phase 2 model)
- [hulianyuyy/CorrNet](https://github.com/hulianyuyy/CorrNet) — CVPR 2023, CTC decoding reference
- [coqui-ai/TTS](https://github.com/coqui-ai/TTS) — TTS integration

---

## ⚙️ Setup

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## 🚀 Quick start (Phase 0)

```bash
# Step 1 — Capture landmark sequences for 20-word dataset
python webcam/webcam_capture.py

# Step 2 — Train BiLSTM classifier
python training/train_bilstm.py

# Step 3 — Run live recognition + speech
python inference/realtime_inference.py
```

---

## 🔗 Project tools

| Tool | Purpose |
|---|---|
| GitHub | Source control + CI |
| Linear | Issue tracking per phase |
| Slack | Agent notifications + build alerts |
| Cursor | AI-assisted development |

---

## License

MIT
