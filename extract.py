import os
import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "yolov8n-pose.pt"

RAW_DIR = "dataset/raw"
OUTPUT_DIR = "dataset/keypoints"

CONFIDENCE = 0.5


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


def normalize_pose(keypoints):
    """
    Convert 17 keypoints into a position-independent representation.
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

    return np.concatenate(
        [xy, conf],
        axis=1
    )


def process_video(model, path):

    cap = cv2.VideoCapture(path)

    frames = []

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        result = model(
            frame,
            imgsz=640,
            conf=CONFIDENCE,
            verbose=False
        )[0]


        if result.keypoints is None:
            continue


        people = result.keypoints.data.cpu().numpy()


        if len(people) == 0:
            continue


        # First person only for now
        pose = people[0]

        normalized = normalize_pose(pose)

        frames.append(normalized)


    cap.release()


    if len(frames) == 0:
        return None


    return np.array(
        frames,
        dtype=np.float32
    )


def main():

    model = YOLO(MODEL_PATH)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    for class_name in os.listdir(RAW_DIR):

        input_dir = os.path.join(
            RAW_DIR,
            class_name
        )

        if not os.path.isdir(input_dir):
            continue


        output_dir = os.path.join(
            OUTPUT_DIR,
            class_name
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )


        for file in os.listdir(input_dir):

            if not file.endswith(".mp4"):
                continue


            input_path = os.path.join(
                input_dir,
                file
            )

            output_path = os.path.join(
                output_dir,
                file.replace(".mp4", ".npy")
            )


            if os.path.exists(output_path):
                print("Skipping", file)
                continue


            print(
                "Processing",
                input_path
            )


            data = process_video(
                model,
                input_path
            )


            if data is None:

                print(
                    "No pose found:",
                    file
                )

                continue


            np.save(
                output_path,
                data
            )


            print(
                "Saved",
                data.shape
            )


if __name__ == "__main__":
    main()