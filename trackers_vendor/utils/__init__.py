"""Vendored subset of ultralytics.trackers.utils for ByteTrack. Torch-free."""
from __future__ import annotations

import logging
from types import SimpleNamespace as _SimpleNamespace

LOGGER = logging.getLogger("trackers_vendor")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(_h)
    LOGGER.setLevel(logging.INFO)


class IterableSimpleNamespace(_SimpleNamespace):
    def __iter__(self):
        return iter(vars(self).items())

    def __str__(self):
        return ", ".join(f"{k}={v}" for k, v in vars(self).items())


from . import ops  # noqa: E402
from . import metrics  # noqa: E402
from . import checks  # noqa: E402
from . import matching  # noqa: E402
from .ops import (  # noqa: E402,F401
    xyxy2xywh, xywh2xyxy, xyxy2xywhn,
    xywh2ltwh, ltwh2xywh, xyxy2ltwh, ltwh2xyxy, ltwh2xywhn,
    linear_sum_assignment,
)

__all__ = [
    "LOGGER", "IterableSimpleNamespace",
    "ops", "matching", "metrics", "checks",
    "xyxy2xywh", "xywh2xyxy", "xyxy2xywhn",
    "xywh2ltwh", "ltwh2xywh", "xyxy2ltwh", "ltwh2xyxy", "ltwh2xywhn",
    "linear_sum_assignment",
]
