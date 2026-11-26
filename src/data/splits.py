"""
Subject-independent dataset splitting helpers for REHAB24-6.

Supports Leave-One-Person-Out (LOPO) and grouped K-fold splits by person_id,
ensuring no person ever appears in both train and test.
Person 10 (only 8 reps) is always kept in the training set and never placed in test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


@dataclass(frozen=True)
class SubjectFold:
    """Single subject-independent train/test fold."""

    fold_index: int
    train_indices: List[int]
    test_indices: List[int]
    train_persons: List[int]
    test_persons: List[int]


def get_person_splits(
    samples_or_person_ids: Sequence[Any],
    n_folds: Optional[int] = None,
    always_train_persons: Iterable[int] = (10,),
) -> List[SubjectFold]:
    """
    Create subject-independent train/test folds grouped by person_id.

    Args:
        samples_or_person_ids: Sequence of sample dicts (each having 'person_id')
            or a sequence of integer person_ids.
        n_folds: Number of grouped folds. If None, uses Leave-One-Person-Out (LOPO)
            over all eligible test persons.
        always_train_persons: Person IDs forced into the training set in every fold
            (default: (10,), since Person 10 has only 8 reps).

    Returns:
        List of SubjectFold objects with disjoint train/test person_ids and indices.
    """
    person_ids: List[int] = [
        int(item["person_id"]) if isinstance(item, dict) else int(item)
        for item in samples_or_person_ids
    ]

    pinned_train: Set[int] = {int(p) for p in always_train_persons}
    all_persons = sorted(set(person_ids))
    eligible_test_persons = [p for p in all_persons if p not in pinned_train]

    if not eligible_test_persons:
        raise ValueError("No eligible persons available for test folds after excluding always_train_persons.")

    if n_folds is None or n_folds >= len(eligible_test_persons):
        # Leave-One-Person-Out
        person_groups: List[List[int]] = [[p] for p in eligible_test_persons]
    else:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2 or None for LOPO, got {n_folds}")
        person_groups = [[] for _ in range(n_folds)]
        for idx, p in enumerate(eligible_test_persons):
            person_groups[idx % n_folds].append(p)

    folds: List[SubjectFold] = []
    for fold_idx, test_group in enumerate(person_groups):
        test_set = set(test_group)
        train_set = set(all_persons) - test_set

        train_indices = [i for i, pid in enumerate(person_ids) if pid in train_set]
        test_indices = [i for i, pid in enumerate(person_ids) if pid in test_set]

        folds.append(
            SubjectFold(
                fold_index=fold_idx,
                train_indices=train_indices,
                test_indices=test_indices,
                train_persons=sorted(train_set),
                test_persons=sorted(test_set),
            )
        )

    return folds
