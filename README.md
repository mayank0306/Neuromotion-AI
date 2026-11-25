# NeuroMotion: AI-Assisted Rehabilitation Movement Assessment

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-blue?logo=typescript)](https://www.typescriptlang.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-0096D9?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose-00A389)](https://google.github.io/mediapipe/)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-2496ED?logo=docker)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-85%2B%20coverage-brightgreen)](tests/)

NeuroMotion is an AI-powered clinical and digital healthcare platform for **AI-Assisted Rehabilitation Movement Assessment**.

---

## Core System Architecture & Plan

### 1. Flagship Module: Rehab Movement Quality Assessment
- **Clinician-Equivalent Quality Scoring:** Predicts a continuous movement quality score (0–100) via a multi-task MLP regression head trained on the KIMORE clinical dataset.
- **Form Correctness:** Binary classification head trained on UI-PRMD (correct vs. incorrect execution).
- **Explainable Clinical Diagnostics:** Pinpoints the exact joint and movement phase causing low scores based on deviations from training-set class means.
- **Supporting Module:** Yoga pose classification (Kaggle Yoga 5-Class) is a supporting secondary module, not the headline.

### 2. Single Shared Pipeline & Feature Engineering
- **End-to-End Pipeline:** Uploaded video $\to$ OpenCV $\to$ MediaPipe BlazePose (pretrained, not retrained) $\to$ shared landmark processing $\to$ shared feature engineering (joint angles, ROM, bilateral symmetry, temporal velocities/accelerations).
- **Zero Duplication:** Training pipelines and runtime inference reuse the exact same `FeatureEngineering` layer (`backend/app/analysis/feature_engineering.py`).

### 3. Exactly Two Trained PyTorch Models (Simple MLPs)
- No LSTM, GRU, ST-GCN, or YOLO.
- **Model 1 (Flagship Rehab):** `RehabQualityMLP` — Multi-task MLP with:
  1. UI-PRMD classification head (correct/incorrect form)
  2. KIMORE regression head (continuous clinician quality score 0–100)
- **Model 2 (Supporting Yoga):** `YogaClassifierMLP` — 5-class MLP for Kaggle Yoga 5-Class (Warrior II, Downward Dog, Tree, Goddess, Plank).

### 4. Deterministic Geometric Modules
- **Posture, Gait, Balance, and Flexibility** are implemented using deterministic geometric algorithms on 3D landmarks.
- No machine learning models are trained for these modules.

### 5. Measured vs. Estimated Metric Tagging
Every output metric carries an explicit tag in response schemas:
- `"measured"`: Direct geometric computation (e.g., joint angle, active ROM, bilateral symmetry).
- `"estimated"`: Derived proxy or model inference (e.g., predicted quality score, balance stability index).

### 6. Configurable Weighted Composite Scoring
- **Formula:**
  $$\text{Overall Score} = (\text{Angle Accuracy} \times 0.40) + (\text{Symmetry} \times 0.20) + (\text{Stability} \times 0.25) + (\text{ROM} \times 0.15)$$
- Configurable via `settings.SCORING_WEIGHTS`.
- Corrections and explanations are generated from joint-level deviations against persisted training-set class means (`reference_stats.json`).

### 7. V1 Input Mode: Recorded Video Upload Only
- V1 processes recorded video uploads via `POST /api/v1/analysis/video`.
- Real-time/webcam scaffolding is kept behind the `ENABLE_REALTIME_ANALYSIS` feature flag (default `False`).

### 8. Subject-Based Dataset Splitting (Never by Frame or Rep)
- **UI-PRMD:** 5 train / 2 validation / 3 test rotating subject folds (`GroupKFold`).
- **KIMORE:** ~70 / 15 / 15 split stratified by diagnosis group (healthy, stroke, parkinsons, back_pain).
- **Kaggle Yoga 5-Class:** Official train/test split with an 80/20 train/validation carve-out from the training set.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/username/neuromotion.git
cd neuromotion

# 2. Run test suite
make test
# Or directly with pytest
PYTHONPATH=backend pytest

# 3. Train AI models
make ai-train
# Or run unified training entry point
PYTHONPATH=backend python3 ai-models/train.py --model all --epochs 5

# 4. Start backend server
cd backend && uvicorn app.main:app --reload
```

Then visit:
- Backend API Docs: http://localhost:8000/docs
- Video Analysis Endpoint: `POST /api/v1/analysis/video`

---

## Project Structure

```
neuromotion/
├── backend/
│   ├── app/
│   │   ├── analysis/             # Shared FeatureEngineering & PyTorch models
│   │   ├── ai-models/            # Model configs, pipelines, checkpoints
│   │   ├── api/v1/endpoints/     # REST API (video upload, analysis)
│   │   ├── core/                 # Config (scoring weights, feature flags)
│   │   ├── data/                 # Exercises & persisted reference stats
│   │   ├── schemas/              # Pydantic schemas (measured/estimated tags)
│   │   └── services/ai_service.py # Core AIService orchestrator
│   └── tests/                    # Unit and integration test suite
├── ai-models/                    # Unified training entry point (train.py)
├── docs/                         # Architecture docs & gap analysis
└── Makefile                      # Build and test shortcuts
```
