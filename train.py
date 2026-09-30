import os
import numpy as np
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset/keypoints"
OUTPUT_MODEL = "pose_classifier.joblib"

CLASSES = [
    "tpose",
    "bored",
    "smoking",
    "combat",
    "other"
]

# Take every Nth frame from each recording.
FRAME_STEP = 5

# Training augmentation
USE_MIRROR_AUGMENTATION = True
USE_STRETCH_AUGMENTATION = True

# Number of mildly stretched versions generated per frame.
NUM_STRETCH_VARIANTS = 1

# Maximum amount of horizontal / vertical deformation.
#
# Example:
#   0.05 means the dimension can be scaled between
#   0.95 and 1.05.
STRETCH_AMOUNT = 0.05

# Random seed for reproducibility.
RANDOM_SEED = 42


# ============================================================
# COCO KEYPOINT MIRRORING
# ============================================================

# COCO 17-keypoint indices:
#
# 0  nose
# 1  left eye
# 2  right eye
# 3  left ear
# 4  right ear
# 5  left shoulder
# 6  right shoulder
# 7  left elbow
# 8  right elbow
# 9  left wrist
# 10 right wrist
# 11 left hip
# 12 right hip
# 13 left knee
# 14 right knee
# 15 left ankle
# 16 right ankle

LEFT_RIGHT_PAIRS = [
    (1, 2),    # eyes
    (3, 4),    # ears
    (5, 6),    # shoulders
    (7, 8),    # elbows
    (9, 10),   # wrists
    (11, 12),  # hips
    (13, 14),  # knees
    (15, 16),  # ankles
]


# ============================================================
# AUGMENTATION
# ============================================================

def mirror_pose(frame):
    """
    Create a left-right mirrored version of a normalized pose.

    Input:
        (17, 3)

    Output:
        (17, 3)

    The X coordinate is negated and all left/right keypoints
    are swapped.
    """

    mirrored = frame.copy()

    # Mirror horizontally.
    mirrored[:, 0] *= -1

    # Swap left/right anatomical keypoints.
    for left, right in LEFT_RIGHT_PAIRS:
        mirrored[[left, right]] = mirrored[[right, left]]

    return mirrored


def stretch_pose(frame, rng):
    """
    Apply a small independent horizontal/vertical stretch.

    This simulates mild differences in camera geometry,
    person proportions, and pose estimation.

    The transformation is performed around the normalized
    shoulder center (approximately x=0, y=0).

    Confidence values are unchanged.
    """

    augmented = frame.copy()

    x_scale = rng.uniform(
        1.0 - STRETCH_AMOUNT,
        1.0 + STRETCH_AMOUNT
    )

    y_scale = rng.uniform(
        1.0 - STRETCH_AMOUNT,
        1.0 + STRETCH_AMOUNT
    )

    augmented[:, 0] *= x_scale
    augmented[:, 1] *= y_scale

    return augmented


# ============================================================
# LOAD RECORDINGS
# ============================================================

def get_recordings():

    recordings = []

    for label, cls in enumerate(CLASSES):

        folder = os.path.join(
            DATASET_DIR,
            cls
        )

        if not os.path.isdir(folder):
            print("WARNING: Missing folder:", folder)
            continue

        files = sorted(
            f for f in os.listdir(folder)
            if f.endswith(".npy")
        )

        for file in files:

            path = os.path.join(
                folder,
                file
            )

            recordings.append(
                {
                    "path": path,
                    "label": label,
                    "class": cls,
                    "file": file
                }
            )

    return recordings


# ============================================================
# LOAD FRAMES FROM RECORDINGS
# ============================================================

def load_frames(recordings):

    X = []
    y = []

    for recording in recordings:

        data = np.load(
            recording["path"]
        )

        # Take every 5th frame.
        samples = data[::FRAME_STEP]

        for frame in samples:

            X.append(
                frame.flatten()
            )

            y.append(
                recording["label"]
            )

    if not X:
        return (
            np.empty((0, 51), dtype=np.float32),
            np.empty((0,), dtype=np.int64)
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64)
    )


# ============================================================
# AUGMENT TRAINING DATA
# ============================================================

def augment_recordings(recordings, rng):

    X = []
    y = []

    for recording in recordings:

        data = np.load(
            recording["path"]
        )

        samples = data[::FRAME_STEP]

        label = recording["label"]

        for frame in samples:

            # ------------------------------------------------
            # Original
            # ------------------------------------------------

            X.append(
                frame.flatten()
            )

            y.append(label)

            # ------------------------------------------------
            # Mirrored
            # ------------------------------------------------

            if USE_MIRROR_AUGMENTATION:

                mirrored = mirror_pose(frame)

                X.append(
                    mirrored.flatten()
                )

                y.append(label)

            # ------------------------------------------------
            # Mild stretching / compression
            # ------------------------------------------------

            if USE_STRETCH_AUGMENTATION:

                for _ in range(NUM_STRETCH_VARIANTS):

                    stretched = stretch_pose(
                        frame,
                        rng
                    )

                    X.append(
                        stretched.flatten()
                    )

                    y.append(label)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Find recordings
    # --------------------------------------------------------

    recordings = get_recordings()

    if not recordings:
        print("ERROR: No .npy recordings found.")
        return

    print(
        "Recordings:",
        len(recordings)
    )

    # --------------------------------------------------------
    # Print recording counts
    # --------------------------------------------------------

    print()
    print("Recordings per class:")

    for cls in CLASSES:

        count = sum(
            r["class"] == cls
            for r in recordings
        )

        print(
            f"  {cls:8s}: {count}"
        )

    # --------------------------------------------------------
    # Split by recording, NOT by frame.
    #
    # This prevents frames from the same recording appearing
    # in both training and testing sets.
    # --------------------------------------------------------

    labels = np.array([
        r["label"]
        for r in recordings
    ])

    train_recordings, test_recordings = train_test_split(
        recordings,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    print()
    print(
        "Training recordings:",
        len(train_recordings)
    )

    print(
        "Testing recordings:",
        len(test_recordings)
    )

    # --------------------------------------------------------
    # Build training dataset
    #
    # Augmentation happens ONLY here.
    # --------------------------------------------------------

    print()
    print("Preparing training data...")

    X_train, y_train = augment_recordings(
        train_recordings,
        rng
    )

    # --------------------------------------------------------
    # Build untouched test dataset
    # --------------------------------------------------------

    print("Preparing test data...")

    X_test, y_test = load_frames(
        test_recordings
    )

    print()
    print(
        "Training samples:",
        X_train.shape
    )

    print(
        "Test samples:",
        X_test.shape
    )

    # --------------------------------------------------------
    # Report augmentation
    # --------------------------------------------------------

    augmentation_multiplier = 1

    if USE_MIRROR_AUGMENTATION:
        augmentation_multiplier += 1

    if USE_STRETCH_AUGMENTATION:
        augmentation_multiplier += NUM_STRETCH_VARIANTS

    print()
    print(
        "Approx. training augmentation multiplier:",
        augmentation_multiplier
    )

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------

    model = Pipeline([

        (
            "scale",
            StandardScaler()
        ),

        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(
                    64,
                    32
                ),
                max_iter=500,
                random_state=RANDOM_SEED
            )
        )
    ])

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("Training...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    print()
    print("Classification report:")
    print()

    print(
        classification_report(
            y_test,
            predictions,
            target_names=CLASSES
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    joblib.dump(
        {
            "model": model,
            "classes": CLASSES
        },
        OUTPUT_MODEL
    )

    print(
        "Saved:",
        OUTPUT_MODEL
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
