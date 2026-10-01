from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rknnlite.api import RKNNLite


IMG_SIZE = 640
OBJECT_THRESH = 0.5
NMS_THRESH = 0.4


@dataclass
class PersonDetection:
    box: np.ndarray          # [x1, y1, x2, y2]
    keypoints: np.ndarray    # (17, 3)
    confidence: float


def letterbox_resize(image, size=(640, 640), bg_color=56):
    target_width, target_height = size

    image_height, image_width = image.shape[:2]

    scale = min(
        target_width / image_width,
        target_height / image_height,
    )

    new_width = int(image_width * scale)
    new_height = int(image_height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.ones(
        (target_height, target_width, 3),
        dtype=np.uint8,
    ) * bg_color

    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2

    canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width,
    ] = resized

    return canvas, scale, offset_x, offset_y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    intersection = w * h

    area1 = max(0.0, box1[2] - box1[0]) * \
            max(0.0, box1[3] - box1[1])

    area2 = max(0.0, box2[2] - box2[0]) * \
            max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def nms(detections):
    detections = sorted(
        detections,
        key=lambda d: d.confidence,
        reverse=True,
    )

    kept = []

    for detection in detections:
        keep = True

        for existing in kept:
            if iou(detection.box, existing.box) > NMS_THRESH:
                keep = False
                break

        if keep:
            kept.append(detection)

    return kept


def decode_head(
    output,
    keypoints,
    index,
    model_w,
    model_h,
    stride,
    scale_x,
    scale_y,
):
    """
    Decode one YOLOv8-pose detection head.

    This follows the output layout used by the Rockchip
    RKNN Model Zoo YOLOv8-pose implementation.
    """

    xywh = output[:, :64, :]
    conf = sigmoid(output[:, 64:, :])

    detections = []

    for h in range(model_h):
        for w in range(model_w):

            confidence = float(conf[0, 0, h * model_w + w])

            if confidence < OBJECT_THRESH:
                continue

            raw_box = xywh[0, :, h * model_w + w]

            raw_box = raw_box.reshape(1, 4, 16, 1)

            distribution = softmax(raw_box, axis=2)

            bins = np.arange(16).reshape(1, 1, 16, 1)

            distances = np.sum(
                distribution * bins,
                axis=2,
            ).reshape(-1)

            x1 = (w + 0.5) - distances[0]
            y1 = (h + 0.5) - distances[1]
            x2 = (w + 0.5) + distances[2]
            y2 = (h + 0.5) + distances[3]

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1

            cx *= stride
            cy *= stride
            bw *= stride
            bh *= stride

            box = np.array(
                [
                    (cx - bw / 2) * scale_x,
                    (cy - bh / 2) * scale_y,
                    (cx + bw / 2) * scale_x,
                    (cy + bh / 2) * scale_y,
                ],
                dtype=np.float32,
            )

            kp = keypoints[
                ...,
                h * model_w + w + index,
            ]

            kp = kp.reshape(-1, 3).copy()

            # Keypoints are still in model-image coordinates.
            kp[:, 0] *= scale_x
            kp[:, 1] *= scale_y

            detections.append(
                PersonDetection(
                    box=box,
                    keypoints=kp,
                    confidence=confidence,
                )
            )

    return detections


class RKNNPose:
    """
    Lightweight YOLOv8-pose RKNN backend.

    Returns PersonDetection objects compatible with the
    rest of the recognition pipeline.
    """

    def __init__(
        self,
        model_path="models/yolov8n-pose.rknn",
    ):
        self.model = RKNNLite()

        ret = self.model.load_rknn(model_path)

        if ret != 0:
            raise RuntimeError(
                f"Failed to load RKNN model: {model_path}"
            )

        ret = self.model.init_runtime()

        if ret != 0:
            raise RuntimeError(
                "Failed to initialize RKNN runtime."
            )

    def infer(self, frame):
        original_h, original_w = frame.shape[:2]

        image, scale, offset_x, offset_y = letterbox_resize(
            frame,
            (IMG_SIZE, IMG_SIZE),
        )

        # RKNN YOLOv8 model expects RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # RKNN pose model expects a single image.
        outputs = self.model.inference(
            inputs=[image]
        )

        if len(outputs) < 4:
            raise RuntimeError(
                f"Unexpected RKNN output count: {len(outputs)}"
            )

        keypoints = outputs[3]

        detections = []

        # YOLOv8 uses 80x80, 40x40 and 20x20 heads.
        for output in outputs[:3]:

            if output.shape[2] == 80:
                stride = 8
                index = 0

            elif output.shape[2] == 40:
                stride = 16
                index = 20 * 4 * 20 * 4

            elif output.shape[2] == 20:
                stride = 32
                index = (
                    20 * 4 * 20 * 4
                    + 20 * 2 * 20 * 2
                )

            else:
                continue

            feature = output.reshape(1, 65, -1)

            head_detections = decode_head(
                feature,
                keypoints,
                index,
                output.shape[3],
                output.shape[2],
                stride,
                1.0,
                1.0,
            )

            detections.extend(head_detections)

        detections = nms(detections)

        # Convert model-space coordinates back to
        # original frame coordinates.
        for detection in detections:

            detection.box[[0, 2]] = (
                detection.box[[0, 2]] - offset_x
            ) / scale

            detection.box[[1, 3]] = (
                detection.box[[1, 3]] - offset_y
            ) / scale

            detection.keypoints[:, 0] = (
                detection.keypoints[:, 0] - offset_x
            ) / scale

            detection.keypoints[:, 1] = (
                detection.keypoints[:, 1] - offset_y
            ) / scale

            detection.box[[0, 2]] = np.clip(
                detection.box[[0, 2]],
                0,
                original_w - 1,
            )

            detection.box[[1, 3]] = np.clip(
                detection.box[[1, 3]],
                0,
                original_h - 1,
            )

            detection.keypoints[:, 0] = np.clip(
                detection.keypoints[:, 0],
                0,
                original_w - 1,
            )

            detection.keypoints[:, 1] = np.clip(
                detection.keypoints[:, 1],
                0,
                original_h - 1,
            )

        return detections

    def release(self):
        self.model.release()