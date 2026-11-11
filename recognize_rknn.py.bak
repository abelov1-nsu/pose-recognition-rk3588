"""
recognize_rknn.py

RK3588 deployment of the temporal pose-sequence recognizer.

Pipeline:

    YOLOv8n-pose (RKNN, NPU)
        ↓
    CPU post-processing (DFL decode, NMS, keypoint scaling)
        ↓
    ByteTrack (via ultralytics, CPU)
        ↓
    normalize_pose → temporal_features
        ↓
    GRU (RKNN, NPU)
        ↓
    TPOSE / BORED / SMOKING / COMBAT / 67 / OTHER / ...

Both models run on the RK3588 NPU via rknn-toolkit-lite2.
ByteTrack runs on CPU inside ultralytics.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque

import cv2
import numpy as np

from rknnlite.api import RKNNLite
from output_to_i2c import display_results

from temporal_features import (
    normalize_pose,
    build_temporal_features,
    SEQUENCE_LENGTH,
    SAMPLE_RATE,
)

try:
    import mss
except ImportError:
    mss = None

# ultralytics is used ONLY for ByteTrack + NMS on CPU.
# It does not load a PyTorch YOLO model in this script.
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace


# ============================================================
# CONFIG
# ============================================================

POSE_MODEL_RKNN = "yolov8n-pose.rknn"
GRU_MODEL_RKNN  = "pose_sequence_gru.rknn"
CLASSES_JSON    = "classes.json"

IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45

WINDOW_SECONDS = 1.0
MAX_HISTORY_SECONDS = 3.0
OUTPUT_SMOOTHING = 3
MAX_MISSING_SECONDS = 3.0

NUM_KEYPOINTS = 17
KPT_DIM = 3

# YOLOv8 pose anchors-per-location (reg_max)
REG_MAX = 16
STRIDES = [8, 16, 32]

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.65
LABEL_THICKNESS = 2
INFO_SCALE = 0.45
INFO_THICKNESS = 1
SKELETON_THICKNESS = 2


# ============================================================
# RKNN HELPERS
# ============================================================

def load_rknn(path: str, core_mask=RKNNLite.NPU_CORE_0) -> RKNNLite:
    rknn = RKNNLite()
    ret = rknn.load_rknn(path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed for {path}: {ret}")

    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed for {path}: {ret}")

    return rknn


# ============================================================
# YOLOv8-POSE CPU POST-PROCESSING
# ============================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _dfl_decode(box_logits: np.ndarray) -> np.ndarray:
    """
    box_logits: (..., 4 * REG_MAX)  →  (..., 4)
    Apply softmax over the REG_MAX bins and take the expectation.
    """
    shape = box_logits.shape
    box_logits = box_logits.reshape(*shape[:-1], 4, REG_MAX)
    probs = _softmax(box_logits, axis=-1)
    bins = np.arange(REG_MAX, dtype=np.float32)
    dist = (probs * bins).sum(axis=-1)          # (..., 4)
    return dist


def _make_anchors(h, w, stride, device="cpu"):
    """Return anchor centers in pixel coords, shape (h*w, 2)."""
    sx = np.arange(w, dtype=np.float32) * stride + stride / 2
    sy = np.arange(h, dtype=np.float32) * stride + stride / 2
    grid_x, grid_y = np.meshgrid(sx, sy)
    anchors = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return anchors


def decode_yolov8_pose(outputs, img_w, img_h):
    """
    Decode the 4 RKNN outputs into boxes, scores, and keypoints.

    Expected outputs (from airockchip yolov8-pose fork, INT8):
        cat_14: (1, 65, 80, 80)   stride 8
        cat_15: (1, 65, 40, 40)   stride 16
        cat_16: (1, 65, 20, 20)   stride 32
        cat_25: (1, 17, 3, 8400)  keypoints for all anchors

    Returns:
        boxes      (N, 4)  xyxy in original image coords
        scores     (N,)
        keypoints  (N, 17, 3)  x, y, conf in original image coords
    """
    all_boxes  = []
    all_scores = []

    for out, stride in zip(outputs[:3], STRIDES):
        # out shape: (1, 65, H, W)
        _, c, h, w = out.shape
        assert c == 4 * REG_MAX + 1, f"unexpected channel count: {c}"

        out = out[0].transpose(1, 2, 0).reshape(-1, c)   # (H*W, 65)
        box_logits = out[:, :4 * REG_MAX]                # (HW, 64)
        cls_logits = out[:, 4 * REG_MAX:5 * REG_MAX if False else 4 * REG_MAX + 1]

        # Actually: layout is [64 box | 1 class]
        cls_logits = out[:, 4 * REG_MAX:]                # (HW, 1)
        score = _sigmoid(cls_logits[:, 0])

        dist = _dfl_decode(box_logits)                   # (HW, 4) in stride units
        anchors = _make_anchors(h, w, stride)            # (HW, 2) pixel coords

        x1 = anchors[:, 0] - dist[:, 0] * stride
        y1 = anchors[:, 1] - dist[:, 1] * stride
        x2 = anchors[:, 0] + dist[:, 2] * stride
        y2 = anchors[:, 1] + dist[:, 3] * stride
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        all_boxes.append(boxes.astype(np.float32))
        all_scores.append(score.astype(np.float32))

    boxes  = np.concatenate(all_boxes, axis=0)   # (8400, 4)
    scores = np.concatenate(all_scores, axis=0)  # (8400,)

    # -------- keypoints --------
    # cat_25: (1, 17, 3, 8400)  →  (8400, 17, 3)
    kpts = outputs[3][0].transpose(2, 0, 1).astype(np.float32)

    # -------- confidence filter --------
    keep = scores > CONFIDENCE_THRESHOLD
    boxes  = boxes[keep]
    scores = scores[keep]
    kpts   = kpts[keep]

    if len(boxes) == 0:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0, 17, 3), np.float32))

    # -------- NMS (class-agnostic since 1 class) --------
    keep = nms_xyxy(boxes, scores, NMS_IOU_THRESHOLD)
    boxes  = boxes[keep]
    scores = scores[keep]
    kpts   = kpts[keep]

    # -------- scale from model input to original image --------
    scale_x = img_w / IMG_SIZE
    scale_y = img_h / IMG_SIZE

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # keypoints: if the model was trained with kpt_shape=(17,3) the
    # fork outputs (x, y, conf) with x,y in input-image pixels and
    # conf already sigmoided. If your export differs, adjust here.
    kpts[..., 0] *= scale_x
    kpts[..., 1] *= scale_y

    return boxes, scores, kpts


def nms_xyxy(boxes, scores, iou_thres):
    """Fast NumPy NMS. boxes are xyxy."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union

        remaining = np.where(iou <= iou_thres)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.int64)


# ============================================================
# BYTETRACK WRAPPER
# ============================================================

def build_bytetrack():
    """
    Create a ByteTrack instance with the settings ultralytics uses
    by default. Requires `ultralytics` to be importable.
    """
    cfg = IterableSimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    return BYTETracker(cfg)


# ============================================================
# PERSON STATE
# ============================================================

class PersonState:
    def __init__(self, max_history_seconds, sample_fps):
        max_frames = max(30, int(max_history_seconds * sample_fps))
        self.poses = deque(maxlen=max_frames)
        self.timestamps = deque(maxlen=max_frames)
        self.predictions = deque(maxlen=OUTPUT_SMOOTHING)
        self.confidences = deque(maxlen=OUTPUT_SMOOTHING)
        self.stable_label = "..."
        self.stable_confidence = 0.0
        self.last_seen = 0.0

    def add_pose(self, pose, timestamp):
        self.poses.append(pose)
        self.timestamps.append(timestamp)
        self.last_seen = timestamp

    def ready(self, timestamp):
        if len(self.poses) < 2:
            return False
        return (timestamp - self.timestamps[0]) >= WINDOW_SECONDS

    def get_recent_poses(self, timestamp):
        cutoff = timestamp - WINDOW_SECONDS
        return [p for p, t in zip(self.poses, self.timestamps) if t >= cutoff]

    def update_prediction(self, prediction, confidence, classes):
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        if not self.predictions:
            return
        counts = np.bincount(list(self.predictions), minlength=len(classes))
        candidate = int(np.argmax(counts))
        confs = [c for p, c in zip(self.predictions, self.confidences) if p == candidate]
        self.stable_label = classes[candidate]
        self.stable_confidence = float(np.mean(confs))


# ============================================================
# GRU INFERENCE (RKNN)
# ============================================================

def classify_person(state, timestamp, gru_rknn, classes):
    if not state.ready(timestamp):
        return None, None

    recent = state.get_recent_poses(timestamp)
    if len(recent) < 2:
        return None, None

    poses = np.stack(recent, axis=0)   # (T, 17, 3)

    features = build_temporal_features(
        poses,
        target_length=SEQUENCE_LENGTH,
        sample_rate=SAMPLE_RATE,
    )                                   # (15, 85)

    inp = features[None, ...].astype(np.float32)  # (1, 15, 85)

    logits = gru_rknn.inference(inputs=[inp])[0]  # (1, C)

    # softmax
    x = logits[0] - logits[0].max()
    e = np.exp(x)
    probs = e / e.sum()

    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])


# ============================================================
# ARGS / SCREEN CAPTURE
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="RK3588 pose-sequence recognition")
    p.add_argument("--source", default="0")
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--video-mode", choices=["realtime", "per-frame"], default="per-frame")
    p.add_argument("--output", default=None)
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def get_screen_capture(monitor_number):
    if mss is None:
        raise RuntimeError("Install 'mss' for screen capture: pip install mss")
    sct = mss.mss()
    monitors = sct.monitors
    if monitor_number < 1 or monitor_number >= len(monitors):
        sct.close()
        raise ValueError(f"Invalid monitor {monitor_number}; available 1..{len(monitors)-1}")
    return sct, monitors[monitor_number]


def capture_screen_frame(sct, monitor):
    shot = sct.grab(monitor)
    frame = np.asarray(shot)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


# ============================================================
# FRAME ANALYSIS
# ============================================================

def analyze_frame(frame, pose_rknn, gru_rknn, classes, tracker, person_states,
                  frame_timestamp, source_fps):
    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # YOLO via NPU
    # --------------------------------------------------------
    # Resize to the model input, letterbox to preserve aspect ratio.
    # The airockchip export takes BGR, normalized to [0,1] by the
    # rknn.config() mean/std settings during conversion. If you set
    # those to [[0,0,0]] / [[1,1,1]], the model expects raw 0..255.
    # Adjust here to match your conversion config.
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[None, ...].astype(np.uint8)

    outputs = pose_rknn.inference(inputs=[img])

    boxes, scores, kpts = decode_yolov8_pose(outputs, w, h)

    # --------------------------------------------------------
    # ByteTrack
    # --------------------------------------------------------
    dets = np.concatenate([boxes, scores[:, None]], axis=1) if len(boxes) else np.zeros((0, 5))
    online_targets = tracker.update(dets, frame, (h, w))

    # online_targets is a list of STrack; each has .tlbr and .track_id
    annotated = frame.copy()
    for t in online_targets:
        x1, y1, x2, y2 = t.tlbr.astype(int)
        track_id = int(t.track_id)

        # Match this track to the closest detection by IoU to get its keypoints
        person_kpts = match_keypoints(boxes, kpts, (x1, y1, x2, y2))
        if person_kpts is None:
            continue

        normalized = normalize_pose(person_kpts).reshape(17, 3)

        if track_id not in person_states:
            person_states[track_id] = PersonState(MAX_HISTORY_SECONDS, source_fps)
        state = person_states[track_id]
        state.add_pose(normalized, frame_timestamp)

        label_raw, conf = classify_person(state, frame_timestamp, gru_rknn, classes)
        if label_raw is not None:
            state.update_prediction(classes.index(label_raw), conf, classes)

        draw_person(annotated, (x1, y1, x2, y2), track_id, state)

    return annotated


def match_keypoints(boxes, kpts, track_box, iou_thresh=0.3):
    if len(boxes) == 0:
        return None
    tx1, ty1, tx2, ty2 = track_box
    xx1 = np.maximum(tx1, boxes[:, 0])
    yy1 = np.maximum(ty1, boxes[:, 1])
    xx2 = np.minimum(tx2, boxes[:, 2])
    yy2 = np.minimum(ty2, boxes[:, 3])
    w = (xx2 - xx1).clip(min=0)
    h = (yy2 - yy1).clip(min=0)
    inter = w * h
    area_a = (tx2 - tx1) * (ty2 - ty1)
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = inter / (area_a + area_b - inter + 1e-9)
    idx = int(np.argmax(iou))
    if iou[idx] < iou_thresh:
        return None
    return kpts[idx]


def draw_person(img, box, track_id, state):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if state.stable_label == "...":
        label = f"ID {track_id}: WARMING UP"
        info = f"{len(state.poses)} poses"
    else:
        label = f"ID {track_id}: {state.stable_label.upper()}"
        info = f"confidence: {state.stable_confidence:.2f}"

    (lw, lh), _ = cv2.getTextSize(label, FONT, LABEL_SCALE, LABEL_THICKNESS)
    (iw, ih), _ = cv2.getTextSize(info, FONT, INFO_SCALE, INFO_THICKNESS)
    bw = max(lw, iw) + 12
    bh = lh + ih + 16

    lx = max(x1, 0)
    ly = max(y1 - 8, bh + 4)

    cv2.rectangle(img, (lx, ly - bh), (lx + bw, ly), (0, 0, 0), -1)
    cv2.putText(img, label, (lx + 6, ly - ih - 6), FONT, LABEL_SCALE,
                (0, 255, 0), LABEL_THICKNESS, cv2.LINE_AA)
    cv2.putText(img, info, (lx + 6, ly - 5), FONT, INFO_SCALE,
                (220, 220, 220), INFO_THICKNESS, cv2.LINE_AA)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    print("Loading RKNN models...")
    pose_rknn = load_rknn(POSE_MODEL_RKNN, RKNNLite.NPU_CORE_0)
    gru_rknn  = load_rknn(GRU_MODEL_RKNN,  RKNNLite.NPU_CORE_1)

    with open(CLASSES_JSON) as f:
        classes = json.load(f)
    print("Classes:", classes)

    tracker = build_bytetrack()

    is_screen = args.source.lower() == "screen"
    if is_screen:
        sct, monitor = get_screen_capture(args.monitor)
        cap = None
        width, height = monitor["width"], monitor["height"]
        fps = 30.0
        total_frames = 0
        is_video = False
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")
        sct = None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        is_video = not isinstance(src, int)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    person_states = {}
    frame_number = 0
    realtime_start = time.perf_counter() if args.video_mode == "realtime" else None

    while True:
        if is_screen:
            frame = capture_screen_frame(sct, monitor)
            frame_timestamp = time.perf_counter()
        else:
            ok, frame = cap.read()
            if not ok:
                break
            frame_timestamp = frame_number / fps

        frame_number += 1

        skip = False
        if is_video and args.video_mode == "realtime":
            expected = (frame_number - 1) / fps
            if (time.perf_counter() - realtime_start) > expected:
                skip = True

        annotated = frame if skip else analyze_frame(
            frame, pose_rknn, gru_rknn, classes, tracker,
            person_states, frame_timestamp, fps,
        )

        oled_lines = []
        for track_id, state in person_states.items():
            if state.stable_label == "...":
                oled_lines.append(f"ID {track_id}: WARMUP")
            else:
                oled_lines.append(
                    f"ID {track_id}: {state.stable_label.upper()} "
                    f"{state.stable_confidence:.2f}"
                )
        display_results(oled_lines)

        # cleanup stale states
        expired = [tid for tid, s in person_states.items()
                if frame_timestamp - s.last_seen > MAX_MISSING_SECONDS]
        for tid in expired:
            del person_states[tid]

        if writer:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Pose Recognition (RKNN)", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    display_results([])
    if cap: cap.release()
    if sct: sct.close()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print("Finished.")


if __name__ == "__main__":
    main()