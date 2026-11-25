"""
AI Models unified training entry point for NeuroMotion.
Supports:
1. Flagship: Rehab Movement Quality Assessment (UI-PRMD classification + KIMORE regression)
2. Supporting: Yoga 5-Class classifier (Kaggle Yoga 5-Class)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure backend is on PYTHONPATH
repo_root = Path(__file__).resolve().parent.parent
backend_dir = repo_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ai_models.ml_pipelines.rehab_assessment_train import train_rehab_assessment_model
from app.ai_models.ml_pipelines.yoga_classifier_train import train_yoga_classifier_model


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroMotion Model Training Pipeline")
    parser.add_argument(
        "--model",
        choices=["rehab", "yoga", "all"],
        default="all",
        help="Which model to train: 'rehab' (flagship), 'yoga' (supporting), or 'all'",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--config", type=Path, default=None, help="Optional config path")
    args = parser.parse_args()

    print(f"Starting training run for model: {args.model}")

    if args.model in ("rehab", "all"):
        print("Training Rehab Movement Quality Assessment Model (UI-PRMD + KIMORE)...")
        rehab_res = train_rehab_assessment_model(epochs=args.epochs)
        print("Rehab model training complete:", rehab_res)

    if args.model in ("yoga", "all"):
        print("Training Yoga Classifier Model (Kaggle Yoga 5-Class)...")
        yoga_res = train_yoga_classifier_model(epochs=args.epochs)
        print("Yoga classifier training complete:", yoga_res)

    print("All configured models trained and artifacts persisted successfully.")


if __name__ == "__main__":
    main()
