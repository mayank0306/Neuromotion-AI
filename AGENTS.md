# NeuroMotion Development Guide

## Project Overview

**NeuroMotion** is an AI-powered clinical healthcare platform for **AI-Assisted Rehabilitation Movement Assessment**.

### Flagship vs. Supporting Modules
1. **Flagship Module: Rehab Movement Quality Assessment**
   - Predicts a continuous quality score (0–100) via a multi-task MLP regression head trained on the KIMORE clinical dataset.
   - Evaluates form correctness (correct vs. incorrect) via a UI-PRMD classification head.
   - Generates explainable clinical diagnostics identifying the exact joint and movement phase causing low scores based on deviations from training-set class means.
2. **Supporting Module: Yoga Pose Classification**
   - Kaggle Yoga 5-Class classifier (Warrior II, Downward Dog, Tree, Goddess, Plank). Secondary supporting module, not the headline.
3. **Deterministic Geometric Modules: Posture, Gait, Balance, Flexibility**
   - Strictly rule-based geometric algorithms on landmarks. No ML training for these modules.

## Architecture Summary

```
Uploaded Video (.mp4/.mov)
       │
       ▼
OpenCV Frame Extraction
       │
       ▼
MediaPipe BlazePose (Pretrained, not retrained)
       │
       ▼
Shared Landmark Processing
       │
       ▼
Shared Feature Engineering (Joint angles, ROM, bilateral symmetry, temporal features)
  ├── Flagship Rehab: RehabQualityMLP (UI-PRMD classification + KIMORE regression)
  ├── Supporting Yoga: YogaClassifierMLP (Kaggle Yoga 5-Class)
  └── Rule Modules: Posture, Gait, Balance, Flexibility (Deterministic geometry)
       │
       ▼
Weighted Composite Scoring & Joint Deviation Explanations
```

### Key Decisions
- **Flagship Focus:** Digital rehabilitation movement quality assessment and explainable diagnostics.
- **Single Shared Feature Layer:** `FeatureEngineering` in `backend/app/analysis/feature_engineering.py` is reused identically everywhere; never duplicate between training and inference.
- **Exactly Two Trained PyTorch Models:** Simple MLPs only (no LSTM, GRU, ST-GCN, or YOLO).
- **Metric Tagging:** Every output metric carries an explicit tag: `"measured"` (direct geometry) or `"estimated"` (derived proxy).
- **Composite Scoring:** Weighted formula with configurable constants:
  $$\text{Score} = (0.40 \times \text{angle\_accuracy}) + (0.20 \times \text{symmetry}) + (0.25 \times \text{stability}) + (0.15 \times \text{ROM})$$
- **Persisted Class Means:** Corrections are deviations from training-set class means, computed during training and persisted for inference in `reference_stats.json`.
- **V1 Input Mode:** Recorded video upload only (`POST /api/v1/analysis/video`). Real-time scaffolding is gated behind `ENABLE_REALTIME_ANALYSIS = False`.
- **Subject-Based Splits:** Splits are strictly by subject (never by frame or rep):
  - UI-PRMD: 5 train / 2 validation / 3 test rotating folds (`GroupKFold`).
  - KIMORE: ~70 / 15 / 15 split stratified by diagnosis group.
  - Yoga: Official train/test split with an 80/20 train/val carve-out.

## Directory Structure

```
neuromotion/
├── backend/
│   ├── app/
│   │   ├── analysis/             # FeatureEngineering & PyTorch models (models.py)
│   │   ├── ai-models/            # Model configs, pipelines, checkpoints
│   │   ├── api/v1/endpoints/     # REST API (video upload, analysis)
│   │   ├── core/config.py        # Settings (SCORING_WEIGHTS, ENABLE_REALTIME_ANALYSIS)
│   │   ├── data/                 # Exercises & persisted reference stats
│   │   ├── schemas/movement.py   # Schemas with measured/estimated tags
│   │   └── services/ai_service.py # Core AIService
│   └── tests/                    # Unit and integration test suite
├── ai-models/                    # Unified training entry point (train.py)
├── docs/                         # Architecture documentation & gap analysis
└── Makefile                      # Development commands
```

## Common Commands

| Command | Description |
|---------|-------------|
| `make test` | Run pytest test suite |
| `make ai-train` | Train both PyTorch models and persist artifacts |
| `make lint` | Run code linters |
| `make typecheck` | Run type checking |
| `make dev` | Start development servers |

## Testing Strategy
- Unit tests for feature engineering, models, and subject-level splits.
- Integration tests for video upload and analysis endpoints.
- Coverage target: 85% minimum for unit tests.
