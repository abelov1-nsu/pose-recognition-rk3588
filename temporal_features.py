"""
temporal_features.py

Shared temporal feature construction for pose-sequence recognition.

Input:
    A sequence of normalized COCO poses.

Each pose:
    17 keypoints × 3 values
    = x, y, confidence
    = 51 values

We add:
    x, y
    confidence
    velocity_x
    velocity_y

Total:
    17 × 5 = 85 features per timestep.

The temporal model therefore receives:

    (sequence_length, 85)

Example:
    15 timesteps × 85 features
"""

from __future__ import annotations

import numpy as np


# COCO keypoint layout
NUM_KEYPOINTS = 17

# x, y, confidence
POSE_FEATURES = 3

# x, y, confidence, velocity_x, velocity_y
TEMPORAL_FEATURES = 5

FEATURES_PER_FRAME = NUM_KEYPOINTS * TEMPORAL_FEATURES

# One second sampled at 15 Hz.
SEQUENCE_LENGTH = 15
SAMPLE_RATE = 15


# ---------------------------------------------------------------------------
# Pose normalization
# ---------------------------------------------------------------------------

def normalize_pose(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize a single pose relative to the shoulder center.

    Parameters
    ----------
    keypoints:
        Shape (17, 3):
        x, y, confidence

    Returns
    -------
    np.ndarray
        Shape (17, 3)
    """

    keypoints = np.asarray(keypoints, dtype=np.float32)

    if keypoints.shape != (NUM_KEYPOINTS, 3):
        raise ValueError(
            f"Expected pose shape (17, 3), got {keypoints.shape}"
        )

    xy = keypoints[:, :2]
    confidence = keypoints[:, 2:3]

    left_shoulder = xy[5]
    right_shoulder = xy[6]

    center = (left_shoulder + right_shoulder) / 2.0

    scale = np.linalg.norm(left_shoulder - right_shoulder)

    if scale < 1e-6:
        scale = 1.0

    normalized_xy = (xy - center) / scale

    return np.concatenate(
        [normalized_xy, confidence],
        axis=1,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

def pose_sequence_to_features(
    poses: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Convert a sequence of normalized poses into temporal features.

    Parameters
    ----------
    poses:
        Shape (T, 17, 3)

    sample_rate:
        Samples per second. Used for velocity calculation.

    Returns
    -------
    np.ndarray
        Shape (T, 85)

    Features for each keypoint:

        x
        y
        confidence
        velocity_x
        velocity_y
    """

    poses = np.asarray(poses, dtype=np.float32)

    if poses.ndim != 3:
        raise ValueError(
            f"Expected poses with 3 dimensions, got {poses.shape}"
        )

    if poses.shape[1:] != (NUM_KEYPOINTS, 3):
        raise ValueError(
            f"Expected shape (T, 17, 3), got {poses.shape}"
        )

    if len(poses) < 1:
        raise ValueError("Pose sequence is empty.")

    xy = poses[:, :, :2]
    confidence = poses[:, :, 2:3]

    # ------------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------------

    # Difference between consecutive poses.
    velocity = np.zeros_like(xy)

    if len(poses) > 1:
        velocity[1:] = (
            xy[1:] - xy[:-1]
        ) * float(sample_rate)

    # First frame has no previous frame.
    # Copy the second-frame velocity if possible so that the first
    # timestep does not contain an artificial zero.
    if len(poses) > 1:
        velocity[0] = velocity[1]

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------

    features = np.concatenate(
        [
            xy,
            confidence,
            velocity,
        ],
        axis=2,
    )

    # (T, 17, 5) → (T, 85)
    features = features.reshape(
        len(poses),
        FEATURES_PER_FRAME,
    )

    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal resampling
# ---------------------------------------------------------------------------

def resample_sequence(
    poses: np.ndarray,
    target_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Resample a pose sequence to a fixed number of timesteps.

    This allows videos with different FPS values to be used by the
    same temporal model.

    Parameters
    ----------
    poses:
        Shape (T, 17, 3)

    target_length:
        Number of timesteps in the resulting sequence.

    Returns
    -------
    np.ndarray
        Shape (target_length, 17, 3)
    """

    poses = np.asarray(poses, dtype=np.float32)

    if poses.ndim != 3 or poses.shape[1:] != (17, 3):
        raise ValueError(
            f"Expected shape (T, 17, 3), got {poses.shape}"
        )

    if len(poses) == 0:
        raise ValueError("Cannot resample an empty sequence.")

    if target_length <= 0:
        raise ValueError("target_length must be positive.")

    # Already correct size.
    if len(poses) == target_length:
        return poses.copy()

    old_positions = np.linspace(
        0.0,
        1.0,
        len(poses),
    )

    new_positions = np.linspace(
        0.0,
        1.0,
        target_length,
    )

    result = np.empty(
        (target_length, 17, 3),
        dtype=np.float32,
    )

    for keypoint in range(17):
        for feature in range(3):
            result[:, keypoint, feature] = np.interp(
                new_positions,
                old_positions,
                poses[:, keypoint, feature],
            )

    return result


# ---------------------------------------------------------------------------
# Complete conversion
# ---------------------------------------------------------------------------

def build_temporal_features(
    poses: np.ndarray,
    target_length: int = SEQUENCE_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Complete preprocessing pipeline.

    Input:
        arbitrary-length normalized pose sequence

    Output:
        fixed-size temporal feature sequence

        (15, 85) by default
    """

    poses = resample_sequence(
        poses,
        target_length=target_length,
    )

    return pose_sequence_to_features(
        poses,
        sample_rate=sample_rate,
    )