# NeuroMotion AI models

This directory is reserved for reproducible model experiments. The production API currently uses MediaPipe BlazePose through `backend/app/ai/pose_estimator.py`.

Training code must record the dataset version, evaluation cohort, model version, and MLflow run before a model can be promoted to production. It must not use protected health information without the appropriate consent and controls.
