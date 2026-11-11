import importlib, sys
sys.modules["ultralytics"] = None
sys.modules["torch"] = None
sys.modules["torchvision"] = None

mods = [
    "trackers_vendor",
    "trackers_vendor.utils",
    "trackers_vendor.utils.ops",
    "trackers_vendor.utils.metrics",
    "trackers_vendor.utils.checks",
    "trackers_vendor.utils.matching",
    "trackers_vendor.utils.kalman_filter",
    "trackers_vendor.utils.stracks",
    "trackers_vendor.basetrack",
    "trackers_vendor.byte_tracker",
]
ok = fail = 0
for m in mods:
    try:
        importlib.import_module(m)
        print(f"OK    {m}")
        ok += 1
    except Exception as e:
        print(f"FAIL  {m}: {type(e).__name__}: {e}")
        fail += 1
print(f"\n{ok} ok, {fail} failed")