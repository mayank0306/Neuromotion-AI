"""
Sanity check script for the REHAB24-6 data pipeline.

Prints:
- Total reps loaded, dropped reps, and drop reasons
- Counts of correct/incorrect per exercise_id
- Reps per person_id
- Subject-independent split verification (confirming Person 10 is always in train)
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from src.data.rehab24_dataset import Rehab24Dataset, load_segmentation_rows
from src.data.splits import get_person_splits


def run_sanity_check(
    segmentation_csv: Path,
    landmarks_dir: Path,
    target_frames: int = 60,
    one_based_frames: bool = True,
) -> None:
    """Run and print REHAB24-6 dataset and split sanity statistics."""
    dataset = Rehab24Dataset(
        segmentation_csv=segmentation_csv,
        landmarks_dir=landmarks_dir,
        target_frames=target_frames,
        one_based_frames=one_based_frames,
    )
    stats = dataset.stats

    print("=== REHAB24-6 Dataset Sanity Report ===")
    print(f"Total CSV rows:                {stats.total_csv_rows}")
    print(f"Total reps loaded:             {stats.loaded_reps}")
    print(f"Total reps dropped:            {stats.total_dropped}")
    print(f"  - mocap_erroneous == 1:      {stats.dropped_mocap_erroneous}")
    print(f"  - missing .npz landmarks:    {stats.dropped_missing_npz}")
    print(f"  - empty slice:               {stats.dropped_empty_slice}")
    print(f"  - undetected > 20%:          {stats.dropped_undetected_gt_20pct}")

    if len(dataset) > 0:
        samples_for_stats = dataset.samples
        source_label = "Loaded Samples"
    else:
        # Fallback to valid non-erroneous CSV rows when .npz files haven't been extracted yet
        raw_rows = load_segmentation_rows(segmentation_csv)
        samples_for_stats = [
            {
                "exercise_id": int(r["exercise_id"]),
                "person_id": int(r["person_id"]),
                "correctness": int(r["correctness"]),
            }
            for r in raw_rows
            if int(r["mocap_erroneous"]) == 0
        ]
        source_label = "Eligible CSV Rows (no .npz files found yet; mocap_erroneous == 0)"

    print(f"\n--- Breakdown ({source_label}) ---")
    ex_counts: Dict[int, Counter] = defaultdict(Counter)
    person_counts: Counter = Counter()

    for sample in samples_for_stats:
        ex_counts[sample["exercise_id"]][sample["correctness"]] += 1
        person_counts[sample["person_id"]] += 1

    print("\nCorrect / Incorrect per exercise_id:")
    for ex_id in sorted(ex_counts):
        c1 = ex_counts[ex_id][1]
        c0 = ex_counts[ex_id][0]
        print(f"  exercise_id {ex_id}: correct={c1}, incorrect={c0}, total={c1 + c0}")

    print("\nReps per person_id:")
    for pid in sorted(person_counts):
        print(f"  person_id {pid:2d}: {person_counts[pid]} reps")

    folds = get_person_splits(samples_for_stats, n_folds=None, always_train_persons=(10,))
    print(f"\nSubject-Independent LOPO Splits: {len(folds)} folds")
    for fold in folds:
        assert 10 in fold.train_persons, "Person 10 must always be in train!"
        assert 10 not in fold.test_persons, "Person 10 must never be in test!"
        assert set(fold.train_persons).isdisjoint(set(fold.test_persons))
    print(
        f"  Verified: 0 person overlap across all {len(folds)} folds; "
        "Person 10 is always in train."
    )


def main() -> None:
    """CLI entry point for sanity check."""
    parser = argparse.ArgumentParser(description="REHAB24-6 data pipeline sanity check.")
    parser.add_argument(
        "--segmentation-csv",
        type=Path,
        default=Path("datasets/rehab24_6/segmentation/Segmentation.csv"),
        help="Path to Segmentation.csv.",
    )
    parser.add_argument(
        "--landmarks-dir",
        type=Path,
        default=Path("datasets/rehab24_6/landmarks"),
        help="Directory containing extracted .npz landmark files.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=60,
        help="Fixed frame count per repetition.",
    )
    parser.add_argument(
        "--zero-based",
        action="store_true",
        help="Treat CSV frame indices as 0-based instead of 1-based.",
    )
    args = parser.parse_args()

    run_sanity_check(
        segmentation_csv=args.segmentation_csv,
        landmarks_dir=args.landmarks_dir,
        target_frames=args.target_frames,
        one_based_frames=not args.zero_based,
    )


if __name__ == "__main__":
    main()
