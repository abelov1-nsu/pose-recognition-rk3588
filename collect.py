"""
collect.py

Lightweight GUI-style pose dataset recorder.

Features:
    - Automatic class discovery from dataset/raw/
    - Create new classes without editing code
    - Actual camera resolution/FPS detection
    - Measured capture FPS
    - Clean source videos (no overlays burned into recordings)
    - Preview-only overlays
    - Recording countdown
    - Recording frame counter
    - Configurable recording duration
    - Discard the last recording
    - Keyboard-driven interface
    - Camera warm-up
    - Per-class recording counts

Controls:

    Number keys 1-9
        Select class

    N
        Create new class

    SPACE
        Start recording

    D
        Discard the most recently recorded video

    Q
        Quit
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("dataset/raw")

CAMERA_INDEX = 0

# These are requested values.
# The actual values returned by the camera are used afterward.
REQUESTED_WIDTH = 1280
REQUESTED_HEIGHT = 720
REQUESTED_FPS = 30

RECORD_SECONDS = 6

# Give the camera a moment to stabilize before showing the GUI.
WARMUP_SECONDS = 2.0

# Default classes.
# Any additional folders inside dataset/raw/ are discovered
# automatically.
DEFAULT_CLASSES = [
    "tpose",
    "bored",
    "smoking",
    "combat",
    "other",
    "67",
]

WINDOW_NAME = "Pose Dataset Recorder"


# ============================================================
# CLASS MANAGEMENT
# ============================================================

def get_classes() -> list[str]:
    """
    Discover classes from folders in dataset/raw/.

    Default classes are included even if their folders do not
    exist yet.
    """

    BASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    classes = set(DEFAULT_CLASSES)

    for path in BASE_DIR.iterdir():

        if path.is_dir():
            classes.add(path.name)

    return sorted(classes)


def is_valid_class_name(name: str) -> bool:
    """
    Check whether a class name is safe as a folder name.
    """

    name = name.strip()

    if not name:
        return False

    if name in (".", ".."):
        return False

    if "/" in name or "\\" in name:
        return False

    if any(
        character in name
        for character in '<>:"|?*'
    ):
        return False

    return True


def create_new_class() -> str | None:
    """
    Create a new class from terminal input.

    This is intentionally kept outside the OpenCV window because
    entering arbitrary text is much easier through the terminal.
    """

    print()
    print("======================================")
    print("           CREATE NEW CLASS")
    print("======================================")
    print()
    print("Examples:")
    print("  walking")
    print("  waving")
    print("  sitting")
    print("  falling")
    print()

    while True:

        name = input(
            "Class name (Enter to cancel): "
        ).strip()

        if not name:
            return None

        if not is_valid_class_name(name):

            print()
            print("Invalid class name.")
            print(
                "Use a simple folder-safe name."
            )
            print()

            continue

        class_dir = BASE_DIR / name

        if class_dir.exists():

            if class_dir.is_dir():

                print()
                print(
                    f"Class '{name}' already exists."
                )

                return name

            print()
            print(
                f"A file already exists at:"
                f" {class_dir}"
            )

            continue

        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print(
            f"Created class: {name}"
        )

        return name


# ============================================================
# VIDEO FILE MANAGEMENT
# ============================================================

def get_recordings(class_name: str) -> list[Path]:
    """
    Return all recorded videos for a class.
    """

    class_dir = BASE_DIR / class_name

    if not class_dir.exists():
        return []

    extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    )


def get_next_filename(class_name: str) -> Path:
    """
    Generate the next unique filename.
    """

    class_dir = BASE_DIR / class_name

    class_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = get_recordings(
        class_name
    )

    used_numbers = []

    prefix = f"{class_name}_"

    for path in existing:

        stem = path.stem

        if not stem.startswith(prefix):
            continue

        number_text = stem[len(prefix):]

        if number_text.isdigit():
            used_numbers.append(
                int(number_text)
            )

    index = (
        max(used_numbers) + 1
        if used_numbers
        else 1
    )

    return (
        class_dir
        / f"{class_name}_{index:03d}.mp4"
    )


def delete_last_recording(
    class_name: str,
) -> bool:
    """
    Delete the most recent recording for a class.
    """

    recordings = get_recordings(
        class_name
    )

    if not recordings:

        return False

    latest = max(
        recordings,
        key=lambda path: path.stat().st_mtime,
    )

    try:

        latest.unlink()

        print()
        print(
            f"Deleted: {latest}"
        )

        return True

    except OSError as error:

        print()
        print(
            f"Could not delete {latest}:"
        )

        print(error)

        return False


# ============================================================
# CAMERA
# ============================================================

def get_camera_info(camera):
    """
    Read the camera's actual reported properties.
    """

    width = int(
        round(
            camera.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )
    )

    height = int(
        round(
            camera.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )
    )

    fps = camera.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 0.0

    return width, height, fps


def measure_camera_fps(
    camera,
    duration: float = 1.0,
) -> float:
    """
    Measure how quickly frames are actually arriving.
    """

    count = 0

    start = time.perf_counter()

    while (
        time.perf_counter() - start
        < duration
    ):

        ok, _ = camera.read()

        if not ok:
            break

        count += 1

    elapsed = (
        time.perf_counter() - start
    )

    if elapsed <= 0:
        return 0.0

    return count / elapsed


# ============================================================
# VIDEO WRITER
# ============================================================

def create_writer(
    path: Path,
    width: int,
    height: int,
    fps: float,
):
    """
    Create an MP4 writer using the camera's actual dimensions
    and reported FPS.
    """

    # VideoWriter requires a sensible positive FPS.
    writer_fps = fps if fps > 1 else REQUESTED_FPS

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        writer_fps,
        (width, height),
    )

    return writer


# ============================================================
# DRAWING
# ============================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.7,
    thickness=2,
):
    """
    Draw readable text with a black background shadow.
    """

    x, y = position

    cv2.putText(
        frame,
        text,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_interface(
    frame,
    classes,
    selected_index,
    camera_fps,
    measured_fps,
    recording,
    recording_elapsed=0.0,
    recording_frames=0,
    status="READY",
):
    """
    Draw the complete preview interface.
    """

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Top panel
    # --------------------------------------------------------

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, 150),
        (20, 20, 20),
        -1,
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
    )

    selected_class = (
        classes[selected_index]
        if classes
        else "none"
    )

    draw_text(
        frame,
        "POSE DATASET RECORDER",
        (20, 35),
        scale=0.9,
        thickness=2,
    )

    draw_text(
        frame,
        f"CLASS: {selected_class}",
        (20, 72),
        scale=0.75,
        thickness=2,
    )

    draw_text(
        frame,
        (
            f"Camera: "
            f"{width}x{height}  "
            f"{camera_fps:.1f} FPS"
        ),
        (20, 108),
        scale=0.55,
        thickness=1,
    )

    draw_text(
        frame,
        f"Measured: {measured_fps:.1f} FPS",
        (20, 136),
        scale=0.55,
        thickness=1,
    )

    # --------------------------------------------------------
    # Right-side status
    # --------------------------------------------------------

    if recording:

        remaining = max(
            0.0,
            RECORD_SECONDS
            - recording_elapsed,
        )

        draw_text(
            frame,
            "RECORDING",
            (
                width - 250,
                40,
            ),
            scale=0.8,
            thickness=2,
        )

        draw_text(
            frame,
            f"{remaining:.1f}s",
            (
                width - 150,
                82,
            ),
            scale=0.9,
            thickness=2,
        )

        draw_text(
            frame,
            f"Frames: {recording_frames}",
            (
                width - 250,
                120,
            ),
            scale=0.55,
            thickness=1,
        )

    else:

        draw_text(
            frame,
            status,
            (
                width - 220,
                45,
            ),
            scale=0.7,
            thickness=2,
        )

    # --------------------------------------------------------
    # Bottom panel
    # --------------------------------------------------------

    bottom_height = 95

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            0,
            height - bottom_height,
        ),
        (
            width,
            height,
        ),
        (20, 20, 20),
        -1,
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
    )

    controls = (
        "1-9 Select Class    "
        "SPACE Record    "
        "N New Class    "
        "D Delete Last    "
        "Q Quit"
    )

    draw_text(
        frame,
        controls,
        (
            20,
            height - 55,
        ),
        scale=0.55,
        thickness=1,
    )

    draw_text(
        frame,
        "Videos: "
        + str(
            len(
                get_recordings(
                    selected_class
                )
            )
        ),
        (
            20,
            height - 25,
        ),
        scale=0.55,
        thickness=1,
    )


# ============================================================
# RECORDING
# ============================================================

def record_video(
    camera,
    class_name,
    width,
    height,
    fps,
):
    """
    Record one clean video.

    The preview receives overlays, but the saved frame does not.
    """

    output_path = get_next_filename(
        class_name
    )

    writer = create_writer(
        output_path,
        width,
        height,
        fps,
    )

    if not writer.isOpened():

        print()
        print(
            "ERROR: Could not create video:"
        )
        print(output_path)

        return None

    # --------------------------------------------------------
    # Countdown
    # --------------------------------------------------------

    countdown_start = time.perf_counter()

    while True:

        elapsed = (
            time.perf_counter()
            - countdown_start
        )

        remaining = (
            3.0 - elapsed
        )

        if remaining <= 0:
            break

        ok, frame = camera.read()

        if not ok:
            writer.release()
            return None

        preview = frame.copy()

        draw_text(
            preview,
            f"GET READY: {remaining:.1f}",
            (
                40,
                height // 2,
            ),
            scale=1.5,
            thickness=3,
        )

        draw_text(
            preview,
            f"CLASS: {class_name}",
            (
                40,
                height // 2 + 60,
            ),
            scale=0.9,
            thickness=2,
        )

        cv2.imshow(
            WINDOW_NAME,
            preview,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            writer.release()

            return False

    # --------------------------------------------------------
    # Actual recording
    # --------------------------------------------------------

    start_time = time.perf_counter()

    frame_count = 0

    while True:

        ok, frame = camera.read()

        if not ok:

            print(
                "Camera read failed."
            )

            break

        elapsed = (
            time.perf_counter()
            - start_time
        )

        if elapsed >= RECORD_SECONDS:
            break

        # ----------------------------------------------------
        # SAVE CLEAN FRAME
        # ----------------------------------------------------

        writer.write(frame)

        frame_count += 1

        # ----------------------------------------------------
        # DISPLAY PREVIEW COPY
        # ----------------------------------------------------

        preview = frame.copy()

        remaining = max(
            0.0,
            RECORD_SECONDS - elapsed,
        )

        draw_text(
            preview,
            f"RECORDING: {class_name}",
            (20, 45),
            scale=0.8,
            thickness=2,
        )

        draw_text(
            preview,
            f"TIME: {remaining:.1f}s",
            (20, 85),
            scale=0.7,
            thickness=2,
        )

        draw_text(
            preview,
            f"FRAMES: {frame_count}",
            (20, 120),
            scale=0.6,
            thickness=1,
        )

        cv2.imshow(
            WINDOW_NAME,
            preview,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            writer.release()

            return False

        if key == ord("d"):

            writer.release()

            try:
                output_path.unlink()
            except OSError:
                pass

            print()
            print(
                "Recording discarded."
            )

            return True

    writer.release()

    print()
    print("======================================")
    print("Recording complete")
    print("======================================")
    print(
        f"Class:    {class_name}"
    )
    print(
        f"File:     {output_path}"
    )
    print(
        f"Frames:   {frame_count}"
    )
    print(
        f"Duration: {elapsed:.2f}s"
    )

    if elapsed > 0:

        actual_fps = (
            frame_count / elapsed
        )

        print(
            f"Actual FPS: {actual_fps:.2f}"
        )

    print()

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    BASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        print(
            "Could not open webcam."
        )

        return

    # --------------------------------------------------------
    # Request preferred settings
    # --------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        REQUESTED_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        REQUESTED_HEIGHT,
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        REQUESTED_FPS,
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print()
    print(
        "Warming up camera..."
    )

    warmup_start = time.perf_counter()

    while (
        time.perf_counter()
        - warmup_start
        < WARMUP_SECONDS
    ):

        ok, _ = camera.read()

        if not ok:
            break

    # --------------------------------------------------------
    # Read actual camera properties
    # --------------------------------------------------------

    width, height, camera_fps = (
        get_camera_info(camera)
    )

    print()
    print("======================================")
    print("       CAMERA INFORMATION")
    print("======================================")
    print()
    print(
        f"Resolution: "
        f"{width} × {height}"
    )
    print(
        f"Reported FPS: "
        f"{camera_fps:.2f}"
    )

    # --------------------------------------------------------
    # Measure actual incoming FPS
    # --------------------------------------------------------

    print(
        "Measuring actual capture FPS..."
    )

    measured_fps = measure_camera_fps(
        camera,
        duration=1.0,
    )

    print(
        f"Measured FPS: "
        f"{measured_fps:.2f}"
    )

    # --------------------------------------------------------
    # Main GUI
    # --------------------------------------------------------

    classes = get_classes()

    if not classes:

        print(
            "No classes available."
        )

        camera.release()

        return

    selected_index = 0

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        width,
        height,
    )

    print()
    print("======================================")
    print("       DATASET RECORDER READY")
    print("======================================")
    print()
    print(
        "1-9  Select class"
    )
    print(
        "SPACE  Record"
    )
    print(
        "N  Create new class"
    )
    print(
        "D  Delete last recording"
    )
    print(
        "Q  Quit"
    )
    print()

    try:

        while True:

            # ------------------------------------------------
            # Refresh classes
            # ------------------------------------------------

            classes = get_classes()

            if selected_index >= len(classes):
                selected_index = (
                    len(classes) - 1
                )

            selected_class = (
                classes[selected_index]
            )

            # ------------------------------------------------
            # Capture preview frame
            # ------------------------------------------------

            ok, frame = camera.read()

            if not ok:

                print(
                    "Failed to read camera."
                )

                break

            # ------------------------------------------------
            # Draw interface
            # ------------------------------------------------

            preview = frame.copy()

            draw_interface(
                preview,
                classes,
                selected_index,
                camera_fps,
                measured_fps,
                recording=False,
                status="READY",
            )

            cv2.imshow(
                WINDOW_NAME,
                preview,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ------------------------------------------------
            # Quit
            # ------------------------------------------------

            if key == ord("q"):
                break

            # ------------------------------------------------
            # New class
            # ------------------------------------------------

            if key == ord("n"):

                cv2.destroyWindow(
                    WINDOW_NAME
                )

                new_class = (
                    create_new_class()
                )

                cv2.namedWindow(
                    WINDOW_NAME,
                    cv2.WINDOW_NORMAL,
                )

                cv2.resizeWindow(
                    WINDOW_NAME,
                    width,
                    height,
                )

                if new_class is not None:

                    classes = get_classes()

                    if new_class in classes:

                        selected_index = (
                            classes.index(
                                new_class
                            )
                        )

                continue

            # ------------------------------------------------
            # Select class
            # ------------------------------------------------

            if ord("1") <= key <= ord("9"):

                index = (
                    key - ord("1")
                )

                if index < len(classes):

                    selected_index = index

                continue

            # ------------------------------------------------
            # Delete last recording
            # ------------------------------------------------

            if key == ord("d"):

                delete_last_recording(
                    selected_class
                )

                continue

            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            if key == 32:  # SPACE

                result = record_video(
                    camera,
                    selected_class,
                    width,
                    height,
                    camera_fps,
                )

                if result is False:
                    break

    finally:

        camera.release()

        cv2.destroyAllWindows()

    print()
    print("Done.")


if __name__ == "__main__":
    main()