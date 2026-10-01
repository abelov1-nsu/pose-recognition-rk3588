"""
extract.py

Extract normalized human poses from the raw video dataset.

Input:
    dataset/raw/
        tpose/
        bored/
        smoking/
        combat/
        other/
        67/

Output:
    dataset/keypoints/
        tpose/
        bored/
        smoking/
        combat/
        other/
        67/

Each output .npy file contains:

    (number_of_frames, 17, 3)

where each keypoint contains:

    x
    y
    confidence

The poses are normalized relative to the shoulder center and
shoulder distance.

Temporal sequences are NOT created here.
train.py handles that later.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from temporal_features import normalize_pose


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POSE_MODEL = "yolov8n-pose.pt"

RAW_DIR = Path("dataset/raw")
OUTPUT_DIR = Path("dataset/keypoints")

CLASSES = [
    "tpose",
    "bored",
    "smoking",
    "combat",
    "other",
    "67"
]

CONFIDENCE = 0.5
IMG_SIZE = 640

# Process every Nth frame.
# 1 = every frame
# 2 = every second frame
# 3 = every third frame
FRAME_STEP = 1


# ---------------------------------------------------------------------------
# Video extraction
# ---------------------------------------------------------------------------

def extract_video(
    model: YOLO,
    video_path: Path,
) -> np.ndarray | None:
    """
    Extract normalized poses from one video.

    Returns
    -------
    np.ndarray | None

        Shape:

            (frames, 17, 3)

        or None if no usable poses were found.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"WARNING: Could not open {video_path}")
        return None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    poses = []

    frame_index = 0

    with tqdm(
        total=total_frames,
        desc=video_path.name,
        unit="frame",
        leave=False,
    ) as progress:

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Update progress even when skipping the frame.
            progress.update(1)

            if frame_index % FRAME_STEP != 0:
                frame_index += 1
                continue

            frame_index += 1

            # ----------------------------------------------------------
            # YOLO pose inference
            # ----------------------------------------------------------

            results = model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONFIDENCE,
                verbose=False,
            )

            if not results:
                continue

            result = results[0]

            if result.keypoints is None:
                continue

            keypoints = result.keypoints.data

            if keypoints is None:
                continue

            if len(keypoints) == 0:
                continue

            # ----------------------------------------------------------
            # Dataset recordings contain one intended person.
            #
            # If multiple people are detected, use the person with
            # the largest bounding box.
            # ----------------------------------------------------------

            person_index = 0

            if result.boxes is not None and len(result.boxes) > 1:

                boxes = result.boxes.xyxy.cpu().numpy()

                areas = (
                    (boxes[:, 2] - boxes[:, 0])
                    *
                    (boxes[:, 3] - boxes[:, 1])
                )

                person_index = int(
                    np.argmax(areas)
                )

            pose = (
                keypoints[person_index]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # ----------------------------------------------------------
            # Validate pose
            # ----------------------------------------------------------

            if pose.shape != (17, 3):
                continue

            # Shoulders are required for normalization.
            shoulder_confidence = (
                float(pose[5, 2])
                +
                float(pose[6, 2])
            ) / 2.0

            if shoulder_confidence < CONFIDENCE:
                continue

            # ----------------------------------------------------------
            # Normalize
            # ----------------------------------------------------------

            pose = normalize_pose(pose)

            poses.append(pose)

    cap.release()

    if not poses:
        return None

    return np.stack(
        poses,
        axis=0,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("POSE DATASET EXTRACTION")
    print("=" * 60)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Model:       {POSE_MODEL}")
    print(f"Input:       {RAW_DIR}")
    print(f"Output:      {OUTPUT_DIR}")
    print(f"Frame step:  {FRAME_STEP}")
    print()

    # --------------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------------

    print("Loading YOLO pose model...")

    model = YOLO(POSE_MODEL)

    print("Model loaded.")
    print()

    total_videos = 0
    successful = 0
    failed = 0

    # --------------------------------------------------------------
    # Process each class
    # --------------------------------------------------------------

    for class_name in CLASSES:

        input_dir = RAW_DIR / class_name
        output_dir = OUTPUT_DIR / class_name

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not input_dir.exists():
            print(
                f"WARNING: Missing class directory: "
                f"{input_dir}"
            )
            continue

        video_files = sorted(
            [
                path
                for path in input_dir.iterdir()
                if path.suffix.lower()
                in {
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".mkv",
                    ".webm",
                }
            ]
        )

        print()
        print(
            f"Class: {class_name} "
            f"({len(video_files)} videos)"
        )

        for video_path in video_files:

            total_videos += 1

            output_path = (
                output_dir
                / f"{video_path.stem}.npy"
            )

            # Do not redo work unnecessarily.
            if output_path.exists():

                print(
                    f"SKIP: {video_path.name} "
                    f"(already extracted)"
                )

                successful += 1
                continue

            poses = extract_video(
                model,
                video_path,
            )

            if poses is None:

                print(
                    f"WARNING: No usable poses in "
                    f"{video_path.name}"
                )

                failed += 1
                continue

            np.save(
                output_path,
                poses,
            )

            print(
                f"OK: {video_path.name} "
                f"→ {poses.shape}"
            )

            successful += 1

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Videos found:     {total_videos}")
    print(f"Successfully done: {successful}")
    print(f"Failed:           {failed}")

    print()
    print("Output:")
    print(OUTPUT_DIR)

    print()
    print(
        "Each .npy file contains normalized poses with shape:"
    )

    print(
        "    (frames, 17, 3)"
    )


if __name__ == "__main__":
    main()