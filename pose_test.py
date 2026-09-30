import cv2
from ultralytics import YOLO


MODEL = "yolov8n-pose.pt"


def main():
    print("Loading YOLOv8n-pose...")
    model = YOLO(MODEL)

    print("Opening camera...")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)

    print("Camera ready.")
    print("Press Q to quit.")

    while True:
        ok, frame = camera.read()

        if not ok:
            print("Failed to read frame.")
            break

        results = model(
            frame,
            imgsz=640,
            conf=0.5,
            verbose=False,
        )

        result = results[0]

        # YOLO's annotated frame:
        annotated = result.plot()

        # Print information about detected people.
        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()

            for person_index, person in enumerate(keypoints):

                print(
                    f"\rPeople detected: {len(keypoints)}",
                    end="",
                    flush=True,
                )

                # 17 keypoints:
                #
                # 0  nose
                # 1  left eye
                # 2  right eye
                # 3  left ear
                # 4  right ear
                # 5  left shoulder
                # 6  right shoulder
                # 7  left elbow
                # 8  right elbow
                # 9  left wrist
                # 10 right wrist
                # 11 left hip
                # 12 right hip
                # 13 left knee
                # 14 right knee
                # 15 left ankle
                # 16 right ankle

                left_wrist = person[9]
                right_wrist = person[10]

                print(
                    f" | "
                    f"L wrist: "
                    f"({left_wrist[0]:.0f}, "
                    f"{left_wrist[1]:.0f}) "
                    f"conf={left_wrist[2]:.2f} | "
                    f"R wrist: "
                    f"({right_wrist[0]:.0f}, "
                    f"{right_wrist[1]:.0f}) "
                    f"conf={right_wrist[2]:.2f}",
                    end="",
                    flush=True,
                )

                # Only process the first person for now.
                break

        cv2.imshow("YOLOv8 Pose Test", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("\nDone.")


if __name__ == "__main__":
    main()