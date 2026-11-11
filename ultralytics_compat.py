"""
Local stand-in for the two ultralytics symbols that recognize_rknn.py
uses. Backed by the vendored trackers_vendor/ tree, which is
torch-free.
"""

from __future__ import annotations

from types import SimpleNamespace as IterableSimpleNamespace

from trackers_vendor.byte_tracker import BYTETracker

__all__ = ["BYTETracker", "IterableSimpleNamespace"]
