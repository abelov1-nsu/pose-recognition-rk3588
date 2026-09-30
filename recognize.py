import argparse
import cv2
import numpy as np
import joblib
import time

from collections import defaultdict, deque
from tqdm import tqdm
from ultralytics import YOLO

# Optional dependency: only required for --source screen
try:
    import mss
except ImportError:
    mss = None


# ============================================================
# CONFIGURATION
# ============================================================

POSE_MODEL = "yolov8n-pose.pt"
CLASSIFIER = "pose_classifier.joblib"

CONFIDENCE = 0.5
IMG_SIZE = 640

# Temporal smoothing
# Actual window size is calculated from FPS:
# 30 FPS -> 30 frames
# 6 FPS  -> 6 frames
TEMPORAL_WINDOW_SECONDS = 1.0

MIN_VOTES = 5
MIN_AVG_CONFIDENCE = 0.60

# Remove tracking state after this many unseen frames
MAX_MISSING_FRAMES = 30

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


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
# POSE NORMALIZATION
# ============================================================

def normalize_pose(keypoints):
    """
    Normalize COCO keypoints relative to the shoulder center
    and shoulder width.

    Input:
        (17, 3) -> x, y, confidence

    Output:
        (51,) flattened feature vector
    """

    xy = keypoints[:, :2]
    conf = keypoints[:, 2:3]

    left = xy[LEFT_SHOULDER]
    right = xy[RIGHT_SHOULDER]

    center = (left + right) / 2
    scale = np.linalg.norm(left - right)

    if scale < 1e-6:
        scale = 1.0

    xy = (xy - center) / scale

    normalized = np.concatenate(
        [xy, conf],
        axis=1
    )

    return normalized.flatten()


# ============================================================
# TEMPORAL STATE
# ============================================================

class PersonState:
    """
    Stores recent predictions for one tracked person and
    produces a temporally stabilized classification.

    The window size represents approximately one real-world
    second of source video.
    """

    def __init__(self, window_size):

        self.predictions = deque(
            maxlen=window_size
        )

        self.confidences = deque(
            maxlen=window_size
        )

        self.stable_label = "OTHER"
        self.last_seen = 0

    def update(
        self,
        prediction,
        confidence,
        classes
    ):

        self.predictions.append(
            prediction
        )

        self.confidences.append(
            confidence
        )

        if len(self.predictions) < MIN_VOTES:
            return self.stable_label

        predictions = list(
            self.predictions
        )

        confidences = list(
            self.confidences
        )

        counts = np.bincount(
            predictions,
            minlength=len(classes)
        )

        candidate = int(
            np.argmax(counts)
        )

        votes = counts[candidate]

        candidate_confidences = [
            conf
            for pred, conf in zip(
                predictions,
                confidences
            )
            if pred == candidate
        ]

        avg_confidence = (
            float(
                np.mean(
                    candidate_confidences
                )
            )
            if candidate_confidences
            else 0.0
        )

        if (
            votes >= MIN_VOTES
            and avg_confidence >= MIN_AVG_CONFIDENCE
        ):
            self.stable_label = classes[
                candidate
            ]

        return self.stable_label


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "YOLO pose + temporal gesture recognition"
        )
    )

    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Webcam index, video file path, "
            "or 'screen' for monitor capture"
        )
    )

    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help=(
            "Monitor number for screen capture "
            "(1 = first monitor, 2 = second monitor)"
        )
    )

    parser.add_argument(
        "--video-mode",
        choices=[
            "realtime",
            "per-frame"
        ],
        default="per-frame",
        help=(
            "How video files are analyzed: "
            "'realtime' simulates live playback and may "
            "skip frames; 'per-frame' analyzes every frame"
        )
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output video path"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without displaying the video"
    )

    return parser.parse_args()


# ============================================================
# SCREEN CAPTURE
# ============================================================

def get_screen_capture(monitor_number):
    """
    Create an MSS screen capture for the requested monitor.

    MSS uses:
        0 = virtual desktop / all monitors
        1 = first monitor
        2 = second monitor
        ...

    Returns:
        (sct, monitor)
    """

    if mss is None:
        raise RuntimeError(
            "The 'mss' package is required for screen capture.\n"
            "Install it with:\n"
            "    pip install mss"
        )

    sct = mss.mss()

    monitors = sct.monitors

    if (
        monitor_number < 1
        or monitor_number >= len(monitors)
    ):
        sct.close()

        available = len(monitors) - 1

        raise ValueError(
            f"Invalid monitor number: {monitor_number}\n"
            f"Available monitors: 1-{available}"
        )

    monitor = monitors[
        monitor_number
    ]

    return sct, monitor


def capture_screen_frame(
    sct,
    monitor
):
    """
    Capture one monitor frame using MSS.

    MSS returns BGRA.
    OpenCV expects BGR.
    """

    screenshot = sct.grab(
        monitor
    )

    frame = np.asarray(
        screenshot
    )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGRA2BGR
    )

    return frame


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def analyze_frame(
    frame,
    pose_model,
    classifier,
    classes,
    person_states,
    last_seen,
    frame_number,
    window_size
):
    """
    Run the complete YOLO + classifier + temporal
    recognition pipeline on one frame.

    Returns:
        annotated_frame
    """

    results = pose_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # Draw YOLO skeletons and detections
    # --------------------------------------------------------

    annotated = result.plot(
        kpt_line=True,
        kpt_radius=3,
        line_width=SKELETON_THICKNESS
    )

    # --------------------------------------------------------
    # Process detected people
    # --------------------------------------------------------

    if (
        result.keypoints is not None
        and result.boxes is not None
        and len(result.keypoints) > 0
    ):

        people = (
            result.keypoints.data
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Get tracking IDs
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Each person independently
        # ----------------------------------------------------

        for index, (
            person,
            track_id
        ) in enumerate(
            zip(
                people,
                track_ids
            )
        ):

            features = normalize_pose(
                person
            )

            # ------------------------------------------------
            # Classifier probabilities
            # ------------------------------------------------

            probabilities = (
                classifier.predict_proba(
                    [features]
                )[0]
            )

            prediction = int(
                np.argmax(
                    probabilities
                )
            )

            confidence = float(
                probabilities[
                    prediction
                ]
            )

            raw_label = classes[
                prediction
            ]

            # ------------------------------------------------
            # Temporal smoothing
            # ------------------------------------------------

            if track_id not in person_states:

                person_states[
                    track_id
                ] = PersonState(
                    window_size
                )

            state = person_states[
                track_id
            ]

            state.last_seen = (
                frame_number
            )

            last_seen[
                track_id
            ] = frame_number

            stable_label = state.update(
                prediction,
                confidence,
                classes
            )

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            box = result.boxes[
                index
            ]

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label = (
                f"ID {track_id}: "
                f"{stable_label.upper()}"
            )

            raw_info = (
                f"raw: {raw_label} "
                f"{confidence:.2f}"
            )

            # ------------------------------------------------
            # Label background
            # ------------------------------------------------

            (label_w, label_h), _ = (
                cv2.getTextSize(
                    label,
                    FONT,
                    LABEL_SCALE,
                    LABEL_THICKNESS
                )
            )

            (info_w, info_h), _ = (
                cv2.getTextSize(
                    raw_info,
                    FONT,
                    INFO_SCALE,
                    INFO_THICKNESS
                )
            )

            box_width = max(
                label_w,
                info_w
            ) + 12

            box_height = (
                label_h
                + info_h
                + 16
            )

            label_x = max(
                x1,
                0
            )

            label_y = max(
                y1 - 8,
                box_height + 4
            )

            # ------------------------------------------------
            # Background
            # ------------------------------------------------

            cv2.rectangle(
                annotated,
                (
                    label_x,
                    label_y - box_height
                ),
                (
                    label_x + box_width,
                    label_y
                ),
                (0, 0, 0),
                -1
            )

            # ------------------------------------------------
            # Stable classification
            # ------------------------------------------------

            cv2.putText(
                annotated,
                label,
                (
                    label_x + 6,
                    label_y - info_h - 6
                ),
                FONT,
                LABEL_SCALE,
                (0, 255, 0),
                LABEL_THICKNESS,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # Raw classifier result
            # ------------------------------------------------

            cv2.putText(
                annotated,
                raw_info,
                (
                    label_x + 6,
                    label_y - 5
                ),
                FONT,
                INFO_SCALE,
                (220, 220, 220),
                INFO_THICKNESS,
                cv2.LINE_AA
            )

    # --------------------------------------------------------
    # Remove old tracking states
    # --------------------------------------------------------

    expired_ids = [
        track_id
        for track_id, last_frame
        in last_seen.items()
        if (
            frame_number - last_frame
            > MAX_MISSING_FRAMES
        )
    ]

    for track_id in expired_ids:

        person_states.pop(
            track_id,
            None
        )

        last_seen.pop(
            track_id,
            None
        )

    return annotated


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Determine source type
    # --------------------------------------------------------

    is_screen = (
        args.source.lower()
        == "screen"
    )

    if is_screen:

        source = None
        is_camera = False
        is_video = False

    elif args.source.isdigit():

        source = int(
            args.source
        )

        is_camera = True
        is_video = False

    else:

        source = args.source

        is_camera = False
        is_video = True

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print(
        "Loading YOLO pose model..."
    )

    pose_model = YOLO(
        POSE_MODEL
    )

    print(
        "Loading classifier..."
    )

    data = joblib.load(
        CLASSIFIER
    )

    classifier = data["model"]
    classes = data["classes"]

    print(
        "Classes:",
        ", ".join(classes)
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
            ValueError
        ) as e:

            print(
                f"ERROR: {e}"
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
            "Screen capture:"
        )

        print(
            f"  Monitor    : "
            f"{args.monitor}"
        )

        print(
            f"  Resolution : "
            f"{width} x {height}"
        )

        print(
            "  FPS        : "
            "real-time"
        )

        print()

    else:

        cap = cv2.VideoCapture(
            source
        )

        if not cap.isOpened():

            print(
                "ERROR: Could not "
                f"open source: {source}"
            )

            return

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30.0

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

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

        print()
        print(
            "Video information:"
        )

        print(
            f"  Resolution : "
            f"{width} x {height}"
        )

        print(
            f"  FPS        : "
            f"{fps:.2f}"
        )

        if is_video:

            print(
                f"  Frames     : "
                f"{total_frames}"
            )

    # --------------------------------------------------------
    # Calculate one-second temporal window
    # --------------------------------------------------------

    window_size = max(
        1,
        round(
            fps
            * TEMPORAL_WINDOW_SECONDS
        )
    )

    # Don't allow the vote requirement to exceed
    # the available temporal window.
    effective_min_votes = min(
        MIN_VOTES,
        window_size
    )

    print()
    print(
        "Temporal analysis:"
    )

    print(
        f"  Window     : "
        f"{window_size} frames "
        f"(~{window_size / fps:.2f} sec)"
    )

    print(
        f"  Min votes  : "
        f"{effective_min_votes}"
    )

    # --------------------------------------------------------
    # Output writer
    # --------------------------------------------------------

    writer = None

    if args.output is not None:

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            args.output,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():

            print(
                "ERROR: Could not "
                "open output video."
            )

            if cap is not None:
                cap.release()

            if sct is not None:
                sct.close()

            return

        print(
            f"  Output     : "
            f"{args.output}"
        )

    print()

    # --------------------------------------------------------
    # Per-person tracking state
    # --------------------------------------------------------

    person_states = {}
    last_seen = {}

    frame_number = 0

    # --------------------------------------------------------
    # Progress bar
    # --------------------------------------------------------

    progress = None

    if is_video:

        progress = tqdm(
            total=total_frames,
            desc="Processing",
            unit="frame",
            dynamic_ncols=True
        )

    # --------------------------------------------------------
    # Real-time video timing
    # --------------------------------------------------------

    realtime_start = None

    if (
        is_video
        and args.video_mode
        == "realtime"
    ):

        realtime_start = time.perf_counter()

    # --------------------------------------------------------
    # Processing loop
    # --------------------------------------------------------

    while True:

        # ----------------------------------------------------
        # Get frame
        # ----------------------------------------------------

        if is_screen:

            frame = (
                capture_screen_frame(
                    sct,
                    monitor
                )
            )

            ok = True

        else:

            ok, frame = (
                cap.read()
            )

            if not ok:
                break

        frame_number += 1

        # ====================================================
        # REAL-TIME VIDEO MODE
        # ====================================================
        #
        # The source video is treated as though it were
        # actually playing at its original FPS.
        #
        # If analysis cannot keep up, this frame is skipped
        # instead of waiting for the analysis to finish.
        #
        # The raw frame is still written to the output so
        # that the output video remains the original duration
        # and FPS.
        # ====================================================

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

            # We are already behind the source video's
            # real-time schedule.
            if elapsed > expected_time:

                skip_analysis = True

        if skip_analysis:

            annotated = frame

        else:

            # ------------------------------------------------
            # Full analysis
            # ------------------------------------------------

            annotated = analyze_frame(
                frame,
                pose_model,
                classifier,
                classes,
                person_states,
                last_seen,
                frame_number,
                window_size
            )

        # ----------------------------------------------------
        # Write output
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
                annotated
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
    print(
        "Finished."
    )

    if args.output is not None:

        print(
            f"Saved: "
            f"{args.output}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()