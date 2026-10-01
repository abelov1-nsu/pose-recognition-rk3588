"""
recognize.py

Temporal pose-sequence recognition using:

    YOLOv8n-pose
        ↓
    ByteTrack
        ↓
    normalized pose history
        ↓
    temporal_features.py
        ↓
    small GRU classifier
        ↓
    TPOSE / BORED / SMOKING / COMBAT / 67 / OTHER / ...

The classifier is trained on sequences of poses rather than
individual frames.

The same script supports:

    - webcam
    - video files
    - screen capture

Example:

    python recognize.py

    python recognize.py --source video.mp4

    python recognize.py --source screen

    python recognize.py --source video.mp4 --video-mode realtime

    python recognize.py --source video.mp4 --no-display
"""

from __future__ import annotations

import argparse
import time

from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from ultralytics import YOLO

from temporal_features import (
    normalize_pose,
    build_temporal_features,
    SEQUENCE_LENGTH,
    SAMPLE_RATE,
)


# Optional dependency: only required for --source screen
try:
    import mss
except ImportError:
    mss = None


# ============================================================
# CONFIGURATION
# ============================================================

POSE_MODEL = "yolov8n-pose.pt"

CLASSIFIER = "pose_sequence_model.pt"

CONFIDENCE = 0.5
IMG_SIZE = 640

# The temporal model looks at approximately this much history.
WINDOW_SECONDS = 1.0

# Maximum amount of pose history kept for each person.
# This gives us some safety margin around the 1-second window.
MAX_HISTORY_SECONDS = 2.0

# Small amount of output smoothing.
# This is NOT the main temporal recognition mechanism anymore.
OUTPUT_SMOOTHING = 3

# Remove a tracked person's state after this much time
# without seeing them.
MAX_MISSING_SECONDS = 2.0


# ============================================================
# DISPLAY
# ============================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX

LABEL_SCALE = 0.65
LABEL_THICKNESS = 2

INFO_SCALE = 0.45
INFO_THICKNESS = 1

SKELETON_THICKNESS = 2


# ============================================================
# TEMPORAL MODEL
# ============================================================

class PoseSequenceGRU(nn.Module):
    """
    Small GRU for temporal pose classification.

    Input:
        (batch, sequence_length, input_size)

    Default:
        sequence_length = 15
        input_size = 85

    Output:
        class logits
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        # Last timestep's hidden representation
        features = output[:, -1, :]

        return self.classifier(features)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

def load_classifier(path):
    """
    Load the temporal GRU checkpoint.

    Expected checkpoint format:

        {
            "model_state_dict": ...,
            "classes": [...],
            "input_size": 85,
            "hidden_size": 64,
            "num_layers": 1,
            ...
        }
    """

    print("Loading temporal classifier...")

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    classes = checkpoint["classes"]

    input_size = checkpoint.get(
        "input_size",
        85,
    )

    hidden_size = checkpoint.get(
        "hidden_size",
        64,
    )

    num_layers = checkpoint.get(
        "num_layers",
        1,
    )

    model = PoseSequenceGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=len(classes),
        num_layers=num_layers,
    )

    # Support the normal format we will use in train.py.
    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint["model_state_dict"]
        )

    # Also support a raw state_dict if we ever save one.
    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "Classes:",
        ", ".join(classes)
    )

    print(
        f"Input size: {input_size}"
    )

    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )

    return model, classes


# ============================================================
# PERSON STATE
# ============================================================

class PersonState:
    """
    State belonging to one tracked person.

    Stores:

        pose history
        prediction history
        last seen time

    The GRU operates on the pose history.
    """

    def __init__(
        self,
        max_history_seconds,
        sample_fps,
    ):

        max_frames = max(
            30,
            int(
                max_history_seconds
                * sample_fps
            ),
        )

        self.poses = deque(
            maxlen=max_frames
        )

        self.timestamps = deque(
            maxlen=max_frames
        )

        self.predictions = deque(
            maxlen=OUTPUT_SMOOTHING
        )

        self.confidences = deque(
            maxlen=OUTPUT_SMOOTHING
        )

        self.stable_label = "..."
        self.stable_confidence = 0.0

        self.last_seen = 0.0

    def add_pose(
        self,
        pose,
        timestamp,
    ):

        self.poses.append(
            pose
        )

        self.timestamps.append(
            timestamp
        )

        self.last_seen = timestamp

    def ready(
        self,
        timestamp,
    ):
        """
        Whether enough temporal history exists
        to make a prediction.
        """

        if len(self.poses) < 2:
            return False

        elapsed = (
            timestamp
            - self.timestamps[0]
        )

        return (
            elapsed >= WINDOW_SECONDS
        )

    def get_recent_poses(
        self,
        timestamp,
    ):
        """
        Return approximately the last WINDOW_SECONDS
        of pose history.
        """

        cutoff = (
            timestamp
            - WINDOW_SECONDS
        )

        selected = [
            pose
            for pose, pose_time
            in zip(
                self.poses,
                self.timestamps,
            )
            if pose_time >= cutoff
        ]

        return selected

    def update_prediction(
        self,
        prediction,
        confidence,
        classes,
    ):
        """
        Apply a very small amount of output smoothing.

        The GRU itself performs the real temporal reasoning.
        """

        self.predictions.append(
            prediction
        )

        self.confidences.append(
            confidence
        )

        if not self.predictions:
            return

        # Majority vote among the last few GRU predictions.
        counts = np.bincount(
            list(self.predictions),
            minlength=len(classes),
        )

        candidate = int(
            np.argmax(counts)
        )

        candidate_confidences = [
            conf
            for pred, conf in zip(
                self.predictions,
                self.confidences,
            )
            if pred == candidate
        ]

        average_confidence = float(
            np.mean(
                candidate_confidences
            )
        )

        self.stable_label = (
            classes[candidate]
        )

        self.stable_confidence = (
            average_confidence
        )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "YOLO pose + temporal GRU "
            "movement recognition"
        )
    )

    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Webcam index, video file path, "
            "or 'screen' for monitor capture"
        ),
    )

    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help=(
            "Monitor number for screen capture "
            "(1 = first monitor, 2 = second monitor)"
        ),
    )

    parser.add_argument(
        "--video-mode",
        choices=[
            "realtime",
            "per-frame",
        ],
        default="per-frame",
        help=(
            "Video processing mode: "
            "'realtime' simulates live playback; "
            "'per-frame' processes every frame"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output video path",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without displaying video",
    )

    return parser.parse_args()


# ============================================================
# SCREEN CAPTURE
# ============================================================

def get_screen_capture(
    monitor_number,
):
    """
    Create an MSS screen capture.

    MSS numbering:

        0 = combined virtual desktop
        1 = first physical monitor
        2 = second physical monitor
        ...
    """

    if mss is None:

        raise RuntimeError(
            "The 'mss' package is required "
            "for screen capture.\n"
            "Install it with:\n"
            "    pip install mss"
        )

    sct = mss.mss()

    monitors = sct.monitors

    if (
        monitor_number < 1
        or monitor_number >= len(monitors)
    ):

        available = (
            len(monitors) - 1
        )

        sct.close()

        raise ValueError(
            f"Invalid monitor number: "
            f"{monitor_number}\n"
            f"Available monitors: "
            f"1-{available}"
        )

    return (
        sct,
        monitors[monitor_number],
    )


def capture_screen_frame(
    sct,
    monitor,
):
    """
    Capture one monitor frame.

    MSS returns BGRA.
    OpenCV expects BGR.
    """

    screenshot = sct.grab(
        monitor
    )

    frame = np.asarray(
        screenshot
    )

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGRA2BGR,
    )


# ============================================================
# CLASSIFY ONE PERSON
# ============================================================

@torch.no_grad()
def classify_person(
    state,
    timestamp,
    model,
    classes,
):
    """
    Build the latest temporal sequence and classify it.

    Returns:

        (label, confidence)

    or:

        (None, None)

    if the history is not ready yet.
    """

    if not state.ready(timestamp):
        return None, None

    recent_poses = (
        state.get_recent_poses(
            timestamp
        )
    )

    if len(recent_poses) < 2:
        return None, None

    poses = np.stack(
        recent_poses,
        axis=0,
    )

    # --------------------------------------------------------
    # Convert raw normalized poses into the exact feature
    # representation used during training.
    # --------------------------------------------------------

    features = build_temporal_features(
        poses,
        target_length=SEQUENCE_LENGTH,
        sample_rate=SAMPLE_RATE,
    )

    # (15, 85) → (1, 15, 85)
    tensor = torch.from_numpy(
        features
    ).unsqueeze(0)

    logits = model(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    prediction = int(
        torch.argmax(
            probabilities
        ).item()
    )

    confidence = float(
        probabilities[
            prediction
        ].item()
    )

    label = classes[
        prediction
    ]

    return label, confidence


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def analyze_frame(
    frame,
    pose_model,
    classifier,
    classes,
    person_states,
    frame_timestamp,
):
    """
    Process one frame.

    YOLO handles detection + tracking.
    Each tracked person gets an independent temporal
    pose sequence.
    """

    results = pose_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        verbose=False,
    )

    result = results[0]

    # --------------------------------------------------------
    # Draw YOLO skeleton
    # --------------------------------------------------------

    annotated = result.plot(
        kpt_line=True,
        kpt_radius=3,
        line_width=SKELETON_THICKNESS,
    )

    # --------------------------------------------------------
    # No people
    # --------------------------------------------------------

    if (
        result.keypoints is None
        or result.boxes is None
        or len(result.keypoints) == 0
    ):
        return annotated

    people = (
        result.keypoints.data
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Tracking IDs
    # --------------------------------------------------------

    if result.boxes.is_track:

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

    else:

        track_ids = list(
            range(len(people))
        )

    # --------------------------------------------------------
    # Process each person
    # --------------------------------------------------------

    for index, (
        person,
        track_id,
    ) in enumerate(
        zip(
            people,
            track_ids,
        )
    ):

        # ----------------------------------------------------
        # Normalize pose
        # ----------------------------------------------------

        normalized = normalize_pose(
            person
        )

        # Restore (17, 3) because the temporal feature
        # pipeline expects the individual keypoints.
        normalized = normalized.reshape(
            17,
            3,
        )

        # ----------------------------------------------------
        # Create state if necessary
        # ----------------------------------------------------

        if track_id not in person_states:

            person_states[
                track_id
            ] = PersonState(
                max_history_seconds=(
                    MAX_HISTORY_SECONDS
                ),
                sample_fps=SAMPLE_RATE,
            )

        state = person_states[
            track_id
        ]

        state.add_pose(
            normalized,
            frame_timestamp,
        )

        # ----------------------------------------------------
        # Temporal classification
        # ----------------------------------------------------

        raw_label, confidence = (
            classify_person(
                state,
                frame_timestamp,
                classifier,
                classes,
            )
        )

        if raw_label is not None:

            prediction_index = (
                classes.index(
                    raw_label
                )
            )

            state.update_prediction(
                prediction_index,
                confidence,
                classes,
            )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        box = result.boxes[
            index
        ]

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        if state.stable_label == "...":

            label = (
                f"ID {track_id}: "
                "WARMING UP"
            )

            info = (
                f"{len(state.poses)} "
                "poses"
            )

        else:

            label = (
                f"ID {track_id}: "
                f"{state.stable_label.upper()}"
            )

            info = (
                f"confidence: "
                f"{state.stable_confidence:.2f}"
            )

        # ----------------------------------------------------
        # Text dimensions
        # ----------------------------------------------------

        (
            label_size,
            _,
        ) = cv2.getTextSize(
            label,
            FONT,
            LABEL_SCALE,
            LABEL_THICKNESS,
        )

        (
            info_size,
            _,
        ) = cv2.getTextSize(
            info,
            FONT,
            INFO_SCALE,
            INFO_THICKNESS,
        )

        box_width = (
            max(
                label_size[0],
                info_size[0],
            )
            + 12
        )

        box_height = (
            label_size[1]
            + info_size[1]
            + 16
        )

        label_x = max(
            int(x1),
            0,
        )

        label_y = max(
            int(y1) - 8,
            box_height + 4,
        )

        # ----------------------------------------------------
        # Background
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (
                label_x,
                label_y - box_height,
            ),
            (
                label_x + box_width,
                label_y,
            ),
            (0, 0, 0),
            -1,
        )

        # ----------------------------------------------------
        # Stable classification
        # ----------------------------------------------------

        cv2.putText(
            annotated,
            label,
            (
                label_x + 6,
                label_y - info_size[1] - 6,
            ),
            FONT,
            LABEL_SCALE,
            (0, 255, 0),
            LABEL_THICKNESS,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        cv2.putText(
            annotated,
            info,
            (
                label_x + 6,
                label_y - 5,
            ),
            FONT,
            INFO_SCALE,
            (220, 220, 220),
            INFO_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated


# ============================================================
# REMOVE OLD TRACKS
# ============================================================

def cleanup_person_states(
    person_states,
    current_time,
):
    """
    Remove people who have disappeared for too long.

    ByteTrack IDs are only useful while the person is tracked.
    """

    expired = []

    for track_id, state in (
        person_states.items()
    ):

        if (
            current_time
            - state.last_seen
            > MAX_MISSING_SECONDS
        ):

            expired.append(
                track_id
            )

    for track_id in expired:

        del person_states[
            track_id
        ]


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Determine source
    # --------------------------------------------------------

    is_screen = (
        args.source.lower()
        == "screen"
    )

    if is_screen:

        source = None
        is_video = False

    elif args.source.isdigit():

        source = int(
            args.source
        )
        is_video = False

    else:

        source = args.source
        is_video = True

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print(
        "Loading YOLO pose model..."
    )

    pose_model = YOLO(
        POSE_MODEL
    )

    # --------------------------------------------------------
    # Load temporal GRU
    # --------------------------------------------------------

    classifier, classes = (
        load_classifier(
            CLASSIFIER
        )
    )

    # --------------------------------------------------------
    # Open source
    # --------------------------------------------------------

    cap = None
    sct = None
    monitor = None

    if is_screen:

        try:

            sct, monitor = (
                get_screen_capture(
                    args.monitor
                )
            )

        except (
            RuntimeError,
            ValueError,
        ) as error:

            print(
                f"ERROR: {error}"
            )

            return

        width = monitor[
            "width"
        ]

        height = monitor[
            "height"
        ]

        fps = 30.0
        total_frames = 0

        print()
        print(
            f"Screen monitor: "
            f"{args.monitor}"
        )

        print(
            f"Resolution: "
            f"{width} × {height}"
        )

    else:

        cap = cv2.VideoCapture(
            source
        )

        if not cap.isOpened():

            print(
                "ERROR: Could not open "
                f"source: {source}"
            )

            return

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30.0

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        print()
        print(
            f"Resolution: "
            f"{width} × {height}"
        )

        print(
            f"FPS: {fps:.2f}"
        )

        if is_video:

            print(
                f"Frames: "
                f"{total_frames}"
            )

    # --------------------------------------------------------
    # Output writer
    # --------------------------------------------------------

    writer = None

    if args.output is not None:

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        writer = cv2.VideoWriter(
            args.output,
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():

            print(
                "ERROR: Could not open "
                "output video."
            )

            if cap is not None:
                cap.release()

            if sct is not None:
                sct.close()

            return

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    person_states = {}

    frame_number = 0

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    progress = None

    if is_video:

        progress = tqdm(
            total=total_frames,
            desc="Processing",
            unit="frame",
            dynamic_ncols=True,
        )

    # --------------------------------------------------------
    # Real-time timing
    # --------------------------------------------------------

    realtime_start = None

    if (
        is_video
        and args.video_mode
        == "realtime"
    ):

        realtime_start = (
            time.perf_counter()
        )

    # --------------------------------------------------------
    # Main processing loop
    # --------------------------------------------------------

    while True:

        # ----------------------------------------------------
        # Read frame
        # ----------------------------------------------------

        if is_screen:

            frame = (
                capture_screen_frame(
                    sct,
                    monitor,
                )
            )

            ok = True

            frame_timestamp = (
                time.perf_counter()
            )

        else:

            ok, frame = (
                cap.read()
            )

            if not ok:
                break

            # Use source-video time rather than processing
            # time. This is important when processing a video
            # faster or slower than real time.
            frame_timestamp = (
                frame_number
                / fps
            )

        frame_number += 1

        # ----------------------------------------------------
        # Realtime video mode
        # ----------------------------------------------------

        skip_analysis = False

        if (
            is_video
            and args.video_mode
            == "realtime"
        ):

            expected_time = (
                (frame_number - 1)
                / fps
            )

            elapsed = (
                time.perf_counter()
                - realtime_start
            )

            if elapsed > expected_time:

                skip_analysis = True

        # ----------------------------------------------------
        # Analyze
        # ----------------------------------------------------

        if skip_analysis:

            annotated = frame

        else:

            annotated = analyze_frame(
                frame,
                pose_model,
                classifier,
                classes,
                person_states,
                frame_timestamp,
            )

        # ----------------------------------------------------
        # Remove stale people
        # ----------------------------------------------------

        cleanup_person_states(
            person_states,
            frame_timestamp,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        if writer is not None:

            writer.write(
                annotated
            )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        if not args.no_display:

            cv2.imshow(
                "Pose Recognition",
                annotated,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if progress is not None:

            progress.update(1)

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    if cap is not None:
        cap.release()

    if sct is not None:
        sct.close()

    if writer is not None:
        writer.release()

    if progress is not None:
        progress.close()

    cv2.destroyAllWindows()

    print()
    print("Finished.")

    if args.output is not None:

        print(
            f"Saved: {args.output}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()