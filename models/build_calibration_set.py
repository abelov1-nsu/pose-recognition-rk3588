# build_calibration_set.py  — run this where you have your training data
import os
import numpy as np
from temporal_features import build_temporal_features, SEQUENCE_LENGTH, SAMPLE_RATE
from recognize import normalize_pose  # or however you get a (T,17,3) array

OUT_DIR = "calib"
os.makedirs(OUT_DIR, exist_ok=True)

# `pose_sequences` is a list of (T, 17, 3) arrays captured from real videos,
# already passed through normalize_pose. T can vary; build_temporal_features
# resamples to SEQUENCE_LENGTH anyway.
for i, poses in enumerate(pose_sequences[:100]):
    feats = build_temporal_features(
        poses,
        target_length=SEQUENCE_LENGTH,
        sample_rate=SAMPLE_RATE,
    )                                   # -> (15, 85)
    feats = feats[None, ...]            # -> (1, 15, 85)
    np.save(os.path.join(OUT_DIR, f"{i:04d}.npy"), feats)

with open("dataset.txt", "w") as f:
    for name in sorted(os.listdir(OUT_DIR)):
        f.write(os.path.abspath(os.path.join(OUT_DIR, name)) + "\n")