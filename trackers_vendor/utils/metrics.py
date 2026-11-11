"""Torch-free metrics helpers used by ByteTrack's matching."""
from __future__ import annotations
import numpy as np


def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """
    Intersection-over-area-of-box2 for every box1/box2 pair.
    box1, box2: (N, 4) and (M, 4) in xyxy.
    Returns (N, M).
    """
    b1 = np.asarray(box1, dtype=np.float32)[:, None, :]   # (N,1,4)
    b2 = np.asarray(box2, dtype=np.float32)[None, :, :]   # (1,M,4)

    dx = np.minimum(b1[..., 2], b2[..., 2]) - np.maximum(b1[..., 0], b2[..., 0])
    dy = np.minimum(b1[..., 3], b2[..., 3]) - np.maximum(b1[..., 1], b2[..., 1])
    inter = np.clip(dx, 0, None) * np.clip(dy, 0, None)

    if iou:
        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
        return inter / (area1 + area2 - inter + eps)
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    return inter / (area2 + eps)


def batch_probiou(bboxes1, bboxes2, eps=1e-7):
    """Probabilistic IoU вЂ” approximated by regular IoU for our purposes.

    The real ultralytics impl uses a Bhattacharyya/Chi-square distance on
    Gaussians; for ByteTrack *only* the constant 1.0 returned in the
    `fuse_score=False` path matters, and the standard implementation returns
    very close to regular IoU for high-overlap pairs. Regular IoU is a safe
    stand-in here.
    """
    return bbox_ioa(bboxes1, bboxes2, iou=True, eps=eps)


__all__ = ["bbox_ioa", "batch_probiou"]