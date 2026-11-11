#!/usr/bin/env python3
"""Finish vendoring: add local utils pkg + fix ..utils relative imports."""
from __future__ import annotations
import re, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
VENDOR = ROOT / "trackers_vendor"
if not VENDOR.is_dir():
    sys.exit("run from repo root; trackers_vendor/ not found")

# --- 1. create trackers_vendor/utils/ ---
utils = VENDOR / "utils"
utils.mkdir(exist_ok=True)
(utils / "__init__.py").write_text(
    '"""Local torch-free stand-in for ultralytics.utils (ByteTrack needs)."""\n'
    "from __future__ import annotations\n"
    "import logging\n"
    "from types import SimpleNamespace as _SimpleNamespace\n"
    "\n"
    'LOGGER = logging.getLogger("trackers_vendor")\n'
    "if not LOGGER.handlers:\n"
    "    _h = logging.StreamHandler()\n"
    "    _h.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))\n"
    "    LOGGER.addHandler(_h)\n"
    "    LOGGER.setLevel(logging.INFO)\n"
    "\n"
    "class IterableSimpleNamespace(_SimpleNamespace):\n"
    "    def __iter__(self):\n"
    "        return iter(vars(self).items())\n"
    "    def __str__(self):\n"
    "        return ', '.join(f'{k}={v}' for k, v in vars(self).items())\n"
    "\n"
    "from . import ops  # noqa: E402\n"
    "\n"
    "__all__ = ['LOGGER', 'IterableSimpleNamespace', 'ops']\n"
)
(utils / "ops.py").write_text(
    '"""NumPy-only reimplementations of ultralytics.utils.ops helpers."""\n'
    "from __future__ import annotations\n"
    "import numpy as np\n"
    "\n"
    "def xyxy2xywh(x):\n"
    "    y = np.array(x, dtype=np.float32, copy=True)\n"
    "    y[..., 0] = (x[..., 0] + x[..., 2]) / 2.0\n"
    "    y[..., 1] = (x[..., 1] + x[..., 3]) / 2.0\n"
    "    y[..., 2] = x[..., 2] - x[..., 0]\n"
    "    y[..., 3] = x[..., 3] - x[..., 1]\n"
    "    return y\n"
    "\n"
    "def xywh2xyxy(x):\n"
    "    y = np.array(x, dtype=np.float32, copy=True)\n"
    "    y[..., 0] = x[..., 0] - x[..., 2] / 2.0\n"
    "    y[..., 1] = x[..., 1] - x[..., 3] / 2.0\n"
    "    y[..., 2] = x[..., 0] + x[..., 2] / 2.0\n"
    "    y[..., 3] = x[..., 1] + x[..., 3] / 2.0\n"
    "    return y\n"
    "\n"
    "def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):\n"
    "    if clip:\n"
    "        x = np.clip(x, 0, max(w, h) - eps)\n"
    "    y = xyxy2xywh(x)\n"
    "    y[..., 0] /= w\n"
    "    y[..., 1] /= h\n"
    "    y[..., 2] /= w\n"
    "    y[..., 3] /= h\n"
    "    return y\n"
    "\n"
    "__all__ = ['xyxy2xywh', 'xywh2xyxy', 'xyxy2xywhn']\n"
)
print(f"[fix] wrote {utils}/__init__.py and ops.py")

# --- 2. rewrite "from ..X" -> "from .X" everywhere under trackers_vendor ---
changed = 0
for py in VENDOR.rglob("*.py"):
    text = py.read_text()
    orig = text
    text = re.sub(r"(?m)^(\s*)from\s+\.\.([\w.]*)\s+import\b", r"\1from .\2 import", text)
    if text != orig:
        py.write_text(text)
        changed += 1
print(f"[fix] rewrote '..x' -> '.x' in {changed} file(s)")

# --- 3. self-test with ultralytics + torch blocked ---
test = (
    "import sys;"
    "sys.modules['ultralytics']=None;"
    "sys.modules['torch']=None;"
    "from ultralytics_compat import BYTETracker, IterableSimpleNamespace;"
    "from types import SimpleNamespace;"
    "cfg = IterableSimpleNamespace(tracker_type='bytetrack',"
    " track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6,"
    " track_buffer=30, match_thresh=0.8, fuse_score=True);"
    "t = BYTETracker(cfg);"
    "print('SELFTEST OK', type(t).__module__, type(t).__name__)"
)
r = subprocess.run([sys.executable, "-c", test], cwd=ROOT, capture_output=True, text=True)
print(r.stdout, end="")
if r.returncode:
    print(r.stderr, file=sys.stderr, end="")
    sys.exit("[fix] self-test FAILED — see traceback above")
print("[fix] done. Next: pip uninstall -y ultralytics torch torchvision")