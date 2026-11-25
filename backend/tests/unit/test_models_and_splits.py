"""
Unit tests for PyTorch models and subject-based splitting logic.
Verifies:
- Splitting by subject only (never frame or rep)
- UI-PRMD rotating subject folds (5/2/3)
- KIMORE stratified subject splits (~70/15/15)
- Yoga official train/test with 80/20 train/val carve-out
- MLP architectures and forward pass shapes
"""

import pytest
import torch
import numpy as np
from app.analysis.models import (
    YogaClassifierMLP,
    RehabQualityMLP,
    split_ui_prmd_subjects,
    split_kimore_stratified,
    split_yoga_dataset,
    DEFAULT_REFERENCE_STATS,
)


def test_ui_prmd_splits_by_subject():
    """Verify UI-PRMD rotates across 5/2/3 subject folds without overlap."""
    subjects = list(range(1, 11))
    for fold in range(5):
        splits = split_ui_prmd_subjects(subjects, fold=fold)
        train_s = set(splits["train"])
        val_s = set(splits["validation"])
        test_s = set(splits["test"])

        assert len(splits["train"]) == 5
        assert len(splits["validation"]) == 2
        assert len(splits["test"]) == 3

        # Disjoint sets (never leaking subjects across splits)
        assert train_s.isdisjoint(val_s)
        assert train_s.isdisjoint(test_s)
        assert val_s.isdisjoint(test_s)
        assert train_s | val_s | test_s == set(subjects)


def test_kimore_stratified_by_diagnosis():
    """Verify KIMORE subject split ~70/15/15 stratified by diagnosis group."""
    diagnoses = ["healthy", "stroke", "parkinsons", "back_pain"]
    records = []
    for i in range(1, 79):
        records.append({
            "subject_id": f"sub_{i}",
            "diagnosis": diagnoses[i % len(diagnoses)]
        })

    splits = split_kimore_stratified(records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_set = set(splits["train"])
    val_set = set(splits["validation"])
    test_set = set(splits["test"])

    # Disjoint
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set) + len(val_set) + len(test_set) == 78

    # Check ratios approximately 70 / 15 / 15
    assert 50 <= len(train_set) <= 58
    assert 10 <= len(val_set) <= 15
    assert 10 <= len(test_set) <= 15


def test_yoga_carve_out_split():
    """Verify Kaggle Yoga uses 80/20 train/val carve-out from official train split."""
    official_train = [f"train_{i}.jpg" for i in range(100)]
    official_test = [f"test_{i}.jpg" for i in range(20)]

    splits = split_yoga_dataset(official_train, official_test, val_ratio=0.20)
    assert len(splits["train"]) == 80
    assert len(splits["validation"]) == 20
    assert len(splits["test"]) == 20
    assert set(splits["train"]).isdisjoint(set(splits["validation"]))


def test_rehab_mlp_forward():
    """Verify RehabQualityMLP outputs dual heads: binary classification + continuous regression."""
    model = RehabQualityMLP(input_features=32)
    model.eval()

    x = torch.randn(4, 32)
    with torch.no_grad():
        logits, quality_score = model(x)

    assert logits.shape == (4, 2)
    assert quality_score.shape == (4,)
    # Score should be continuous 0-100
    assert (quality_score >= 0.0).all() and (quality_score <= 100.0).all()


def test_yoga_mlp_forward():
    """Verify YogaClassifierMLP outputs 5 class logits."""
    model = YogaClassifierMLP(input_features=32, num_classes=5)
    model.eval()

    x = torch.randn(3, 32)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (3, 5)


def test_reference_stats_present():
    """Verify reference class means are defined for rehabilitation exercises."""
    assert "shoulder-abduction" in DEFAULT_REFERENCE_STATS
    assert "sit-to-stand" in DEFAULT_REFERENCE_STATS
    assert "warrior_ii" in DEFAULT_REFERENCE_STATS
