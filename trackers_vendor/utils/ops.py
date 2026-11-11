"""NumPy-only reimplementations of the ultralytics.utils.ops helpers used by ByteTrack."""
from __future__ import annotations

import numpy as np


def xyxy2xywh(x):
    y = np.array(x, dtype=np.float32, copy=True)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2.0
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2.0
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def xywh2xyxy(x):
    y = np.array(x, dtype=np.float32, copy=True)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0
    y[..., 2] = x[..., 0] + x[..., 2] / 2.0
    y[..., 3] = x[..., 1] + x[..., 3] / 2.0
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    if clip:
        x = np.clip(x, 0, max(w, h) - eps)
    y = xyxy2xywh(x)
    y[..., 0] /= w
    y[..., 1] /= h
    y[..., 2] /= w
    y[..., 3] /= h
    return y


def xywh2ltwh(x):
    """(cx, cy, w, h) -> (left, top, w, h)."""
    y = np.array(x, dtype=np.float32, copy=True)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0
    return y


def ltwh2xywh(x):
    """(left, top, w, h) -> (cx, cy, w, h)."""
    y = np.array(x, dtype=np.float32, copy=True)
    y[..., 0] = x[..., 0] + x[..., 2] / 2.0
    y[..., 1] = x[..., 1] + x[..., 3] / 2.0
    return y


def xyxy2ltwh(x):
    """(x1, y1, x2, y2) -> (left, top, w, h)."""
    y = np.array(x, dtype=np.float32, copy=True)
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def ltwh2xyxy(x):
    """(left, top, w, h) -> (x1, y1, x2, y2)."""
    y = np.array(x, dtype=np.float32, copy=True)
    y[..., 2] = x[..., 0] + x[..., 2]
    y[..., 3] = x[..., 1] + x[..., 3]
    return y


def ltwh2xywhn(x, w=640, h=640, eps=0.0):
    y = ltwh2xywh(x)
    y[..., 0] /= w
    y[..., 1] /= h
    y[..., 2] /= w
    y[..., 3] /= h
    return y

def linear_sum_assignment(cost_matrix, maximize=False):
    """Thin wrapper around scipy's LSA вЂ” matching.py imports it from here."""
    from scipy.optimize import linear_sum_assignment as _lsa
    return _lsa(cost_matrix, maximize=maximize)

__all__ = [
    "xyxy2xywh", "xywh2xyxy", "xyxy2xywhn",
    "xywh2ltwh", "ltwh2xywh", "xyxy2ltwh", "ltwh2xyxy", "ltwh2xywhn",
]