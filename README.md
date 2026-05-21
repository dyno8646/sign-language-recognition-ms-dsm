# 🤟 Sign Language Recognition

> Real-time ASL recognition · Webcam → MediaPipe → BiLSTM → Speech  
> **MS DSM Capstone Project**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

This project converts live sign language gestures into spoken natural language. A webcam captures the signer, MediaPipe Holistic extracts 3D skeletal keypoints, a deep learning model predicts glosses, an LLM refines them into fluent text, and Coqui TTS speaks the result.

Webcam → MediaPipe Holistic → Sequence Model → Gloss → LLM → Text + Speech

---

## Pipeline

| # | Stage | Tech | Status |
|---|-------|------|--------|
| 1 | Webcam capture | OpenCV | ✅ Phase 0 |
| 2 | Landmark extraction | MediaPipe Holistic — 258-dim keypoints | ✅ Phase 0 |
| 3 | Preprocessing | Normalization · sliding window | ✅ Phase 0 |
| 4 | Sequence model | BiLSTM (Phase 0) → Temporal Transformer (Phase 2) | ✅ Phase 0 |
| 5 | Decoder | CTC / Attention → gloss output | 🔜 Phase 1 |
| 6 | LLM refinement | Gloss → fluent text via Claude API | 🔜 Phase 2 |
| 7 | TTS | Coqui TTS → live audio | 🔜 Phase 1 |
| 8 | Streaming UI | Streamlit + FastAPI WebSockets | 🔜 Phase 1 |

---

## Phases

| Phase | Goal | Timeline |
|-------|------|----------|
| **0 ← current** | Webcam → MediaPipe → BiLSTM → 20-word vocab → live display | Weeks 1–3 |
| **1** | CTC decoder · continuous sentences · TTS · FastAPI WebSockets | Weeks 4–9 |
| **2** | Temporal Transformer · LLM refiner · PHOENIX dataset | Weeks 10–17 |
| **3** | ONNX · TensorRT · Docker · multi-language (ISL / ASL / DGS) | Ongoing |

---

## Project Structure

```
sign-language-recognition-ms-dsm/
│
├── src/
│   ├── capture/
│   │   └── collect_data.py       # Webcam data collection
│   ├── mediapipe/
│   │   └── holistic_tracker.py   # 258-dim keypoint extraction
│   ├── models/
│   │   ├── bilstm/
│   │   │   └── model.py          # SignBiLSTM definition
│   │   └── transformer/          # Phase 2
│   ├── inference/
│   │   └── realtime_inference.py # Live prediction loop
│   ├── preprocessing/            # Normalization, augmentation
│   ├── llm/                      # Gloss → natural language (Phase 2)
│   ├── tts/                      # Coqui TTS (Phase 1)
│   ├── api/                      # FastAPI + WebSockets (Phase 1)
│   └── agents/                   # Dataset · Labeling · Experiment · Deploy
│
├── app/                          # Streamlit frontend
├── training/
│   └── train_bilstm.py           # Phase 0 training script
├── configs/
│   └── config.yaml               # Single config for all phases
├── datasets/
│   ├── custom/keypoints/         # Your recorded data
│   ├── WLASL/                    # Phase 1+
│   ├── MSASL/
│   └── PHOENIX/
├── checkpoints/                  # Saved model weights (git-ignored)
├── notebooks/                    # EDA and experimentation
├── tests/                        # Unit tests
├── docker/                       # Dockerfiles + compose
├── docs/                         # Architecture diagrams
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 — Setup

```bash
git clone https://github.com/dyno8646/sign-language-recognition-ms-dsm.git
cd sign-language-recognition-ms-dsm

python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

pip install -r requirements.txt
```

### 2 — Collect your dataset (~15 min)

```bash
python src/capture/collect_data.py
```

Webcam opens. Sign each word shown on screen (2-sec countdown then 30 frames). Repeat 30 times per word across 20 words. Press `Q` to stop, `SPACE` to skip a sequence.

### 3 — Train

```bash
python training/train_bilstm.py
```

Prints loss + val accuracy each epoch. Saves best checkpoint to `checkpoints/bilstm_best.pt`.

### 4 — Run live recognition

```bash
python src/inference/realtime_inference.py
```

---

## Tech Stack

| Area | Tool |
|------|------|
| CV capture | OpenCV |
| Tracking | MediaPipe Holistic |
| ML framework | PyTorch + CUDA |
| Sequence model | BiLSTM → Temporal Transformer |
| LLM refiner | Claude API |
| TTS | Coqui TTS |
| Backend | FastAPI + WebSockets |
| UI | Streamlit |
| Optimization | ONNX / TensorRT |
| Dev | Cursor + Claude |

---

## Reference Repos

- [FangyunWei/SLRT](https://github.com/FangyunWei/SLRT) — Temporal transformer for online SLR
- [hulianyuyy/CorrNet](https://github.com/hulianyuyy/CorrNet) — CVPR 2023, CTC decoding
- [coqui-ai/TTS](https://github.com/coqui-ai/TTS) — TTS toolkit

---

## License

MIT
