# Sign Language Recognition

**MS DSM Project** — Real-time sign language recognition using computer vision and deep learning.

## Overview

This project detects hand gestures from a webcam feed and classifies them into sign language labels. It is built for the MS Data Science & Machine Learning (DSM) capstone.

## Features

- Webcam-based hand landmark extraction (MediaPipe)
- Gesture classification with a trained ML/DL model
- Live prediction overlay
- Dataset collection and model training pipeline

## Project Structure

```
├── data/              # Raw and processed datasets
├── models/            # Saved model weights
├── notebooks/         # EDA and experimentation
├── src/
│   ├── capture.py     # Collect training samples from webcam
│   ├── train.py       # Train the classifier
│   └── predict.py     # Real-time inference
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

```bash
# Collect gesture samples
python src/capture.py

# Train the model
python src/train.py

# Run live recognition
python src/predict.py
```

## Tech Stack

- Python 3.10+
- OpenCV
- MediaPipe
- scikit-learn / TensorFlow (configurable)

## Team

MS DSM Project — Sign Language Recognition

## License

MIT
