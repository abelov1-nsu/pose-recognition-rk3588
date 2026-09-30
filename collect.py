import cv2
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30

RECORD_SECONDS = 6

CLASSES = [
    "tpose",
    "bored",
    "smoking",
    "combat",
    "other",
]

BASE_DIR = "dataset/raw"


# ============================================================
# HELPERS
# ============================================================

def get_next_filename(class_name):
    class_dir = os.path.join(BASE_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    existing = [
        f for f in os.listdir(class_dir)
        if f.endswith(".mp4")
    ]

    index = len(existing) + 1

    while True:
        filename = f"{class_name}_{index:03d}.mp4"
        path = os.path.join(class_dir, filename)

        if not os.path.exists(path):
            return path

        index += 1


def record_video(camera, class_name):
    output_path = get_next_filename(class_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        FPS,
        (VIDEO_WIDTH, VIDEO_HEIGHT)
    )

    print()
    print(f"Recording: {class_name}")
    print(f"Output: {output_path}")
    print("Starting in 3 seconds...")

    time.sleep(3)

    start_time = time.time()

    while True:
        ok, frame = camera.read()

        if not ok:
            print("Failed to read camera.")
            break

        elapsed = time.time() - start_time

        if elapsed >= RECORD_SECONDS:
            break

        # ----------------------------------------------------
        # Display recording information
        # ----------------------------------------------------

        remaining = RECORD_SECONDS - elapsed

        cv2.putText(
            frame,
            f"CLASS: {class_name.upper()}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"TIME: {remaining:.1f}s",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        writer.write(frame)

        cv2.imshow("Gesture Dataset Recorder", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            writer.release()
            return False

    writer.release()

    print(f"Saved: {output_path}")

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(BASE_DIR, exist_ok=True)

    camera = cv2.VideoCapture(0)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    print()
    print("======================================")
    print("       POSE DATASET COLLECTOR")
    print("======================================")
    print()
    print("Classes:")

    for i, name in enumerate(CLASSES, 1):
        print(f"  {i}. {name}")

    print()
    print("Press Q during recording to quit.")
    print()

    try:

        while True:

            print()
            print("--------------------------------------")
            print("Choose a class:")
            print()

            for i, name in enumerate(CLASSES, 1):
                print(f"{i}. {name}")

            print("q. quit")
            print()

            choice = input("> ").strip().lower()

            if choice == "q":
                break

            if not choice.isdigit():
                print("Invalid choice.")
                continue

            index = int(choice) - 1

            if index < 0 or index >= len(CLASSES):
                print("Invalid choice.")
                continue

            class_name = CLASSES[index]

            record_video(camera, class_name)

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()