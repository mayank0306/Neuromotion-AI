# NeuroMotion Gap Analysis: AI-Assisted Rehabilitation Movement Assessment

This document analyzes the current state of the NeuroMotion codebase against the new architectural and product plan.

## Overview & System Repositioning

- **Target Identity:** NeuroMotion is an **AI-Assisted Rehabilitation Movement Assessment** platform.
  - **Flagship Module:** Rehab Movement Quality Assessment. It predicts a continuous quality score (via a KIMORE-trained regression head) and explains which joint/phase caused a low score based on deviations from training-set class means.
  - **Supporting Module:** Yoga pose classification (Kaggle Yoga 5-Class) is a secondary supporting module.
  - **Deterministic Modules:** Posture, Gait, Balance, and Flexibility are strictly rule-based geometric algorithms on landmarks. No ML models are trained for these.

---

## Gap Matrix

| Component | Current State | Target State | Gap / Action Required |
| :--- | :--- | :--- | :--- |
| **Headline Identity & Docs** | `README.md`, `CLAUDE.md`, and `AGENTS.md` feature Yoga as the flagship product. | Flagship is Rehabilitation Movement Quality Assessment; Yoga is supporting. | Update `README.md`, `CLAUDE.md`, `AGENTS.md`, and documentation to center on digital rehabilitation. |
| **Shared Feature Engineering** | `backend/app/analysis/feature_engineering.py` has basic methods, but lacks zero-division safety and does not produce a unified feature vector. ML training pipelines use raw random noise. | Shared feature engineering layer (joint angles, ROM, symmetry, temporal velocity/acceleration) must be reused identically between training and inference. | Enhance `FeatureEngineering` with zero-safe math, full kinematic vector extraction, and make training/inference share the exact same feature layer. |
| **PyTorch Models (Exactly Two)** | Prototype scripts with simulated data exist in `backend/app/ai-models/ml-pipelines/` but are detached from `AIService`. `AIService` uses hardcoded mock scores. | Exactly two simple MLP models: 1) Yoga 5-Class MLP, 2) Rehab Multi-task MLP (UI-PRMD classification head + KIMORE regression head). Loaded in `AIService`. | Implement clean, shared MLP architectures; generate and persist calibrated model checkpoints and training-set class reference means for inference. |
| **Explainable Scoring & Corrections** | Hardcoded score breakdown in `AIService`; no systematic comparison against persisted class reference means. | Continuous quality score + explainability identifying which joint/phase deviated from persisted training-set class mean. Scoring = $(0.40 \times \text{angle}) + (0.20 \times \text{symmetry}) + (0.25 \times \text{stability}) + (0.15 \times \text{ROM})$. | Persist training-set class means and standard deviations; implement joint/phase deviation engine in `AIService`. |
| **Deterministic Rule Modules** | Posture, gait, balance, flexibility implemented with basic heuristics; some missing imports (`MetricWithTag` NameError). | Strictly deterministic geometric algorithms on 3D landmarks; no ML training. Each output metric carries `"measured"` or `"estimated"`. | Harden geometric computations in `AIService` using `FeatureEngineering`. |
| **Response Schema Tags** | `MetricWithTag` has `is_measured: bool`, but lacks explicit string tag field `"measured"` / `"estimated"`. | Every output metric carries a tag: `"measured"` (direct geometry) or `"estimated"` (derived proxy). | Add `tag: Literal["measured", "estimated"]` to `MetricWithTag` and update all analysis responses. |
| **V1 Input & Real-time Flag** | `analysis.py` only handles single base64 images; video handling is marked TODO. No feature flag for real-time. | V1 input is recorded video upload only. Real-time/webcam scaffolding is kept behind a feature flag (`ENABLE_REALTIME_ANALYSIS = False`). | Add `ENABLE_REALTIME_ANALYSIS` setting; implement video processing via OpenCV (`VideoProcessor`) extracting frames and landmark sequences for analysis. |
| **Data Splitting Strategy** | Training pipelines in `ml-pipelines/` used naive random splits without subject grouping. | Subject-level splits only: UI-PRMD 5/2/3 rotating subject folds (`GroupKFold`); KIMORE ~70/15/15 stratified by diagnosis group; Yoga 80/20 train/val from official train split. | Implement subject-level split algorithms in dataset loaders and training pipelines. |

---

## Action Plan

1. **Step 1:** Document this Gap Analysis in `docs/gap-analysis.md`. (Done)
2. **Step 2:** Enhance the shared `FeatureEngineering` layer with kinematic extraction, temporal features, and zero-safe geometry; add unit tests.
3. **Step 3:** Update Pydantic schemas in `backend/app/schemas/movement.py` to include `"measured"` / `"estimated"` tags on all metrics.
4. **Step 4:** Construct the two PyTorch MLP models, subject-based dataset splitting logic (GroupKFold and stratified), and reference means generation & persistence.
5. **Step 5:** Enhance `AIService` to load the PyTorch models, run the video-to-landmarks pipeline, compute the weighted composite score, and provide joint/phase explanations based on persisted class means.
6. **Step 6:** Implement the recorded video upload endpoint in `backend/app/api/v1/endpoints/analysis.py` and gate real-time scaffolding behind `ENABLE_REALTIME_ANALYSIS`.
7. **Step 7:** Run full unit and integration tests, verify coverage, and update `CLAUDE.md`, `AGENTS.md`, and `README.md`.
