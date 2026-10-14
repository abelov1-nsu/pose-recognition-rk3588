import os
import numpy as np
from pathlib import Path
from temporal_features import build_temporal_features, SEQUENCE_LENGTH, SAMPLE_RATE

# Adjust to wherever your raw keypoint recordings live
DATA_DIR = Path("/mnt/c/study/projects/EdgeDeviceGestureRecognition/dataset/keypoints")
OUT_DIR  = Path("/mnt/c/study/projects/EdgeDeviceGestureRecognition/calib")
OUT_DIR.mkdir(exist_ok=True)

all_files = sorted(DATA_DIR.rglob("*.npy"))
print(f"Found {len(all_files)} recordings")

count = 0
for path in all_files:
    poses = np.load(path)
    if poses.ndim != 3 or poses.shape[1:] != (17, 3):
        continue

    # Build (15, 85) features, add batch dim → (1, 15, 85)
    feats = build_temporal_features(poses, SEQUENCE_LENGTH, SAMPLE_RATE)
    feats = feats[None, ...].astype(np.float32)

    np.save(OUT_DIR / f"{count:04d}.npy", feats)
    count += 1
    if count >= 100:      # 100 samples is plenty
        break

print(f"Wrote {count} calibration samples to {OUT_DIR}")

# dataset.txt for RKNN
with open("/mnt/c/study/projects/EdgeDeviceGestureRecognition/dataset.txt", "w") as f:
    for name in sorted(os.listdir(OUT_DIR)):
        f.write(str((OUT_DIR / name).resolve()) + "\n")

print("Wrote dataset.txt")