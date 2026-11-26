"""
Extract MediaPipe BlazePose (33 landmarks) from REHAB24-6 Camera17 videos.

Saves one compressed .npz file per video in the landmarks output directory:
- landmarks: float32 array of shape (num_frames, 33, 4) -> (x, y, z, visibility)
- detected: bool array of shape (num_frames,)
- fps: float32 scalar

Resumable: skips videos that already have a corresponding .npz file.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")


class BlazePoseExtractor:
    """Wrapper around MediaPipe BlazePose supporting both solutions and Tasks APIs."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Initialize BlazePose extractor.

        Args:
            model_path: Optional path to a MediaPipe PoseLandmarker .task bundle
                (required when using MediaPipe 1.2+ Tasks API).
            min_detection_confidence: Minimum pose detection confidence threshold.
            min_tracking_confidence: Minimum pose tracking confidence threshold.
        """
        import mediapipe as mp

        self._mode: str
        self._pose = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._mode = "solutions"
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        elif hasattr(mp, "tasks") and model_path is not None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            if not model_path.exists():
                raise FileNotFoundError(f"MediaPipe task model not found: {model_path}")

            self._mode = "tasks"
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._pose = vision.PoseLandmarker.create_from_options(options)
            self._mp = mp
        else:
            raise RuntimeError(
                "MediaPipe Pose requires either `mediapipe.solutions.pose` (mediapipe==0.10.x) "
                "or a PoseLandmarker `.task` bundle passed via `--model-path` for mediapipe>=1.2."
            )

    def process_frame(
        self, frame_bgr: np.ndarray, timestamp_ms: int
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Run BlazePose on a single BGR frame.

        Args:
            frame_bgr: OpenCV BGR frame.
            timestamp_ms: Frame timestamp in milliseconds (used by Tasks VIDEO mode).

        Returns:
            Tuple of (landmarks[33, 4], world_landmarks[33, 3], detected_bool).
            If no detection, world_landmarks is filled with NaN.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks_arr = np.zeros((33, 4), dtype=np.float32)
        world_arr = np.full((33, 3), np.nan, dtype=np.float32)

        if self._mode == "solutions":
            results = self._pose.process(frame_rgb)
            if results and results.pose_landmarks:
                for idx, lm in enumerate(results.pose_landmarks.landmark[:33]):
                    landmarks_arr[idx] = (lm.x, lm.y, lm.z, lm.visibility)
                if getattr(results, "pose_world_landmarks", None):
                    for idx, wlm in enumerate(results.pose_world_landmarks.landmark[:33]):
                        world_arr[idx] = (wlm.x, wlm.y, wlm.z)
                return landmarks_arr, world_arr, True
            return landmarks_arr, world_arr, False

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        results = self._pose.detect_for_video(mp_image, timestamp_ms)
        if results and results.pose_landmarks:
            pose_lms = results.pose_landmarks[0]
            for idx, lm in enumerate(pose_lms[:33]):
                vis = getattr(lm, "visibility", 1.0)
                landmarks_arr[idx] = (lm.x, lm.y, lm.z, float(vis if vis is not None else 1.0))
            if getattr(results, "pose_world_landmarks", None):
                world_lms = results.pose_world_landmarks[0]
                for idx, wlm in enumerate(world_lms[:33]):
                    world_arr[idx] = (wlm.x, wlm.y, wlm.z)
            return landmarks_arr, world_arr, True
        return landmarks_arr, world_arr, False

    def close(self) -> None:
        """Release underlying MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None


def load_video_frame_bounds(segmentation_csv: Path) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """
    Read Segmentation.csv and compute (min_first_frame, max_last_frame) per video_id.

    Args:
        segmentation_csv: Path to semicolon-separated Segmentation.csv.

    Returns:
        Tuple of ({video_id: (min_first_frame, max_last_frame)}, global_min_first_frame).
    """
    bounds: Dict[str, Tuple[int, int]] = {}
    global_min_first: Optional[int] = None

    with segmentation_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            vid = row["video_id"].strip()
            first_f = int(row["first_frame"])
            last_f = int(row["last_frame"])
            if global_min_first is None or first_f < global_min_first:
                global_min_first = first_f
            if vid not in bounds:
                bounds[vid] = (first_f, last_f)
            else:
                prev_min, prev_max = bounds[vid]
                bounds[vid] = (min(prev_min, first_f), max(prev_max, last_f))

    return bounds, (global_min_first if global_min_first is not None else 0)


def match_videos_by_prefix(
    videos_dir: Path, video_ids: List[str]
) -> List[Tuple[str, Path]]:
    """
    Match Camera17 video files in videos_dir to video_ids by filename prefix.

    Args:
        videos_dir: Directory containing video files.
        video_ids: Ordered list of video_ids from Segmentation.csv.

    Returns:
        List of (video_id, video_path) tuples sorted by video_id.
    """
    if not videos_dir.exists():
        return []

    candidate_files = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTENSIONS
    )

    matched: List[Tuple[str, Path]] = []
    for vid in sorted(set(video_ids)):
        matches = [p for p in candidate_files if p.name.startswith(vid)]
        if matches:
            # Prefer exact stem match or first sorted match
            matched.append((vid, matches[0]))
    return matched


def extract_video_landmarks(
    video_path: Path,
    extractor: BlazePoseExtractor,
    max_last_frame: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Extract BlazePose landmarks and world_landmarks for every frame of a video.

    Args:
        video_path: Path to the video file.
        extractor: Initialized BlazePoseExtractor instance.
        max_last_frame: Optional max last_frame from CSV for bounds checking.

    Returns:
        Tuple of (landmarks (N, 33, 4), world_landmarks (N, 33, 3), detected (N,), fps, num_frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if abs(fps - 30.0) > 0.5:
        warnings.warn(
            f"Video '{video_path.name}' has FPS={fps:.2f} (expected 30.0 FPS).",
            UserWarning,
            stacklevel=2,
        )

    if max_last_frame is not None and reported_frames > 0 and max_last_frame > reported_frames:
        warnings.warn(
            f"Video '{video_path.name}' has max last_frame={max_last_frame} > "
            f"reported frame_count={reported_frames}.",
            UserWarning,
            stacklevel=2,
        )

    all_landmarks: List[np.ndarray] = []
    all_world_landmarks: List[np.ndarray] = []
    all_detected: List[bool] = []
    frame_idx = 0
    effective_fps = fps if fps > 0 else 30.0

    with tqdm(
        total=reported_frames if reported_frames > 0 else None,
        desc=f"  {video_path.name}",
        unit="frame",
        leave=False,
    ) as frame_bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_ms = int((frame_idx / effective_fps) * 1000.0)
            lms, world_lms, det = extractor.process_frame(frame, timestamp_ms)
            all_landmarks.append(lms)
            all_world_landmarks.append(world_lms)
            all_detected.append(det)
            frame_idx += 1
            frame_bar.update(1)

    cap.release()

    if all_landmarks:
        landmarks_np = np.stack(all_landmarks, axis=0).astype(np.float32)
        world_landmarks_np = np.stack(all_world_landmarks, axis=0).astype(np.float32)
        detected_np = np.array(all_detected, dtype=bool)
    else:
        landmarks_np = np.zeros((0, 33, 4), dtype=np.float32)
        world_landmarks_np = np.full((0, 33, 3), np.nan, dtype=np.float32)
        detected_np = np.zeros((0,), dtype=bool)

    return landmarks_np, world_landmarks_np, detected_np, fps, frame_idx


def run_extraction(
    videos_dir: Path,
    segmentation_csv: Path,
    output_dir: Path,
    limit: Optional[int] = None,
    model_path: Optional[Path] = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Dict[str, int]:
    """
    Run landmark extraction across matched Camera17 videos.

    Args:
        videos_dir: Directory containing Camera17 videos.
        segmentation_csv: Path to Segmentation.csv.
        output_dir: Directory where .npz files will be saved.
        limit: Optional limit on number of videos to process.
        model_path: Optional MediaPipe .task bundle path.
        min_detection_confidence: BlazePose detection confidence.
        min_tracking_confidence: BlazePose tracking confidence.

    Returns:
        Summary counts dict ('matched', 'processed', 'skipped').
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_bounds, global_min_first = load_video_frame_bounds(segmentation_csv)
    matched = match_videos_by_prefix(videos_dir, list(frame_bounds.keys()))

    print(f"CSV min(first_frame) = {global_min_first} (across {len(frame_bounds)} video_ids)")
    print(f"Matched {len(matched)} video file(s) in {videos_dir}")

    if matched:
        print("First 5 matched (video_id -> filename):")
        for vid, vpath in matched[:5]:
            print(f"  {vid} -> {vpath.name}")
    else:
        print("No video files found to process.")
        return {"matched": 0, "processed": 0, "skipped": 0}

    to_process = matched[:limit] if limit is not None else matched
    extractor: Optional[BlazePoseExtractor] = None
    processed_count = 0
    skipped_count = 0
    all_last_le_num_frames = True
    any_last_eq_num_frames = False

    try:
        for vid, vpath in tqdm(to_process, desc="Extracting landmarks", unit="video"):
            out_path = output_dir / f"{vid}.npz"
            if out_path.exists():
                skipped_count += 1
                continue

            if extractor is None:
                extractor = BlazePoseExtractor(
                    model_path=model_path,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )

            _, max_last = frame_bounds[vid]
            lms, world_lms, det, fps, num_frames = extract_video_landmarks(
                vpath, extractor, max_last_frame=max_last
            )

            if max_last > num_frames:
                all_last_le_num_frames = False
                print(
                    f"[FRAME CHECK] {vid}: max last_frame={max_last} > actual num_frames={num_frames}"
                )
            elif max_last == num_frames:
                any_last_eq_num_frames = True

            np.savez_compressed(
                out_path,
                landmarks=lms,
                world_landmarks=world_lms,
                detected=det,
                fps=np.float32(fps),
            )
            det_rate = float(det.mean() * 100.0) if len(det) > 0 else 0.0
            print(
                f"Saved {out_path.name}: frames={num_frames}, fps={fps:.2f}, "
                f"max_last_frame={max_last}, detected={det_rate:.1f}%"
            )
            processed_count += 1
    finally:
        if extractor is not None:
            extractor.close()

    if processed_count > 0:
        if global_min_first >= 1 and all_last_le_num_frames:
            indexing_note = (
                "1-based confirmed (min first_frame >= 1 and last_frame <= num_frames"
                + ("; some videos hit last_frame == num_frames" if any_last_eq_num_frames else "")
                + ")"
            )
        else:
            indexing_note = f"Check needed (min first_frame={global_min_first}, all last_frame<=num_frames={all_last_le_num_frames})"
        print(f"Frame indexing check: {indexing_note}")

    return {"matched": len(matched), "processed": processed_count, "skipped": skipped_count}


def main() -> None:
    """CLI entry point for REHAB24-6 landmark extraction."""
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe BlazePose landmarks from REHAB24-6 Camera17 videos."
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("datasets/rehab24_6/videos"),
        help="Directory containing Camera17 videos.",
    )
    parser.add_argument(
        "--segmentation-csv",
        type=Path,
        default=Path("datasets/rehab24_6/segmentation/Segmentation.csv"),
        help="Path to Segmentation.csv (semicolon-separated).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/rehab24_6/landmarks"),
        help="Directory to write .npz landmark files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N matched videos.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to MediaPipe PoseLandmarker .task file (if using mediapipe>=1.2).",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for BlazePose.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence for BlazePose.",
    )
    args = parser.parse_args()

    summary = run_extraction(
        videos_dir=args.videos_dir,
        segmentation_csv=args.segmentation_csv,
        output_dir=args.output_dir,
        limit=args.limit,
        model_path=args.model_path,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    print(f"Done: {summary}")


if __name__ == "__main__":
    main()
