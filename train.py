"""
train.py

Train a temporal GRU classifier on pose sequences.

Pipeline:

    dataset/keypoints/<class>/<recording>.npy
                    ↓
             temporal windows
                    ↓
             15 × 85 features
                    ↓
                  GRU
                    ↓
              class prediction

Important:
    Recordings are split into train/validation BEFORE windows are created.
    This prevents windows from the same recording appearing in both sets.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from temporal_features import (
    FEATURES_PER_FRAME,
    SAMPLE_RATE,
    SEQUENCE_LENGTH,
    build_temporal_features,
)


# ============================================================
# Configuration
# ============================================================

KEYPOINT_DIR = Path("dataset/keypoints")
MODEL_PATH = Path("pose_sequence_model.pt")

VAL_SPLIT = 0.20

BATCH_SIZE = 64
EPOCHS = 40

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_SIZE = 64
NUM_LAYERS = 1

WINDOW_SECONDS = 1.0
WINDOW_FRAMES = 30

WINDOW_STRIDE = 8

RANDOM_SEED = 42

NUM_WORKERS = 0


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Model
# ============================================================

class PoseSequenceGRU(nn.Module):
    """
    Small GRU for temporal pose classification.

    Input:
        (batch, sequence_length, 85)

    Output:
        (batch, num_classes)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        output, _ = self.gru(x)

        # Last timestep
        last_output = output[:, -1, :]

        return self.classifier(last_output)


# ============================================================
# Dataset
# ============================================================

class TemporalPoseDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        features, label = self.samples[index]

        return (
            torch.from_numpy(features).float(),
            torch.tensor(label, dtype=torch.long),
        )


# ============================================================
# Discover classes
# ============================================================

def discover_classes():
    if not KEYPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Keypoint directory does not exist:\n{KEYPOINT_DIR}"
        )

    classes = sorted(
        path.name
        for path in KEYPOINT_DIR.iterdir()
        if path.is_dir()
    )

    if not classes:
        raise RuntimeError(
            f"No class directories found in {KEYPOINT_DIR}"
        )

    return classes


# ============================================================
# Discover recordings
# ============================================================

def discover_recordings(classes):

    recordings = []

    for class_index, class_name in enumerate(classes):

        class_dir = KEYPOINT_DIR / class_name

        files = sorted(class_dir.glob("*.npy"))

        if not files:
            print(
                f"Warning: no .npy files found for class '{class_name}'"
            )

        for path in files:

            recordings.append(
                {
                    "path": path,
                    "label": class_index,
                    "class_name": class_name,
                }
            )

    if not recordings:
        raise RuntimeError(
            "No keypoint recordings were found."
        )

    return recordings


# ============================================================
# File-level train/validation split
# ============================================================

def split_recordings(recordings, val_split=VAL_SPLIT):

    by_class = {}

    for recording in recordings:

        label = recording["label"]

        by_class.setdefault(label, []).append(recording)

    train_recordings = []
    val_recordings = []

    rng = random.Random(RANDOM_SEED)

    for label, items in by_class.items():

        items = items.copy()

        rng.shuffle(items)

        if len(items) == 1:
            print(
                f"Warning: class '{items[0]['class_name']}' "
                f"has only one recording."
            )

            train_recordings.extend(items)
            continue

        val_count = max(
            1,
            round(len(items) * val_split),
        )

        val_items = items[:val_count]
        train_items = items[val_count:]

        train_recordings.extend(train_items)
        val_recordings.extend(val_items)

    rng.shuffle(train_recordings)
    rng.shuffle(val_recordings)

    return train_recordings, val_recordings


# ============================================================
# Create temporal windows
# ============================================================

def recording_to_windows(recording):

    path = recording["path"]
    label = recording["label"]

    poses = np.load(path)

    if poses.ndim != 3 or poses.shape[1:] != (17, 3):

        print(
            f"Skipping invalid file: {path}"
        )

        return []

    num_frames = len(poses)

    if num_frames < WINDOW_FRAMES:

        # Very short recording:
        # use the whole recording and let resampling
        # expand it to the required sequence length.

        feature_sequence = build_temporal_features(
            poses,
            target_length=SEQUENCE_LENGTH,
            sample_rate=SAMPLE_RATE,
        )

        return [
            (
                feature_sequence.astype(np.float32),
                label,
            )
        ]

    samples = []

    for start in range(
        0,
        num_frames - WINDOW_FRAMES + 1,
        WINDOW_STRIDE,
    ):

        end = start + WINDOW_FRAMES

        window = poses[start:end]

        features = build_temporal_features(
            window,
            target_length=SEQUENCE_LENGTH,
            sample_rate=SAMPLE_RATE,
        )

        samples.append(
            (
                features.astype(np.float32),
                label,
            )
        )

    return samples


def build_samples(recordings):

    samples = []

    for i, recording in enumerate(recordings, 1):

        class_name = recording["class_name"]
        path = recording["path"]

        windows = recording_to_windows(recording)

        samples.extend(windows)

        print(
            f"[{i}/{len(recordings)}] "
            f"{class_name:10s} "
            f"{path.name:30s} "
            f"-> {len(windows):3d} windows"
        )

    return samples


# ============================================================
# Class weights
# ============================================================

def calculate_class_weights(samples, num_classes):

    counts = np.zeros(num_classes, dtype=np.float32)

    for _, label in samples:
        counts[label] += 1

    print()
    print("Training window counts:")

    for i, count in enumerate(counts):
        print(f"  class {i}: {int(count)}")

    total = counts.sum()

    weights = np.zeros(num_classes, dtype=np.float32)

    for i in range(num_classes):

        if counts[i] > 0:
            weights[i] = total / (
                num_classes * counts[i]
            )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# Training
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(features)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total,
    )


# ============================================================
# Validation
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)

            loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    return (
        total_loss / total,
        correct / total,
        np.array(all_labels),
        np.array(all_predictions),
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed()

    print()
    print("======================================")
    print("       TEMPORAL POSE TRAINER")
    print("======================================")
    print()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    classes = discover_classes()

    print()
    print("Classes:")

    for i, class_name in enumerate(classes):
        print(f"  {i}: {class_name}")

    print()

    # --------------------------------------------------------
    # Recordings
    # --------------------------------------------------------

    recordings = discover_recordings(classes)

    print(
        f"Total recordings: {len(recordings)}"
    )

    # --------------------------------------------------------
    # Split recordings
    # --------------------------------------------------------

    train_recordings, val_recordings = split_recordings(
        recordings
    )

    print()
    print(
        f"Training recordings:   {len(train_recordings)}"
    )

    print(
        f"Validation recordings: {len(val_recordings)}"
    )

    # --------------------------------------------------------
    # Build temporal samples
    # --------------------------------------------------------

    print()
    print("Building training windows...")
    print()

    train_samples = build_samples(
        train_recordings
    )

    print()
    print(
        f"Training samples: {len(train_samples)}"
    )

    print()
    print("Building validation windows...")
    print()

    val_samples = build_samples(
        val_recordings
    )

    print()
    print(
        f"Validation samples: {len(val_samples)}"
    )

    if not train_samples:
        raise RuntimeError(
            "No training samples were created."
        )

    if not val_samples:
        raise RuntimeError(
            "No validation samples were created."
        )

    # --------------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------------

    train_dataset = TemporalPoseDataset(
        train_samples
    )

    val_dataset = TemporalPoseDataset(
        val_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = PoseSequenceGRU(
        input_size=FEATURES_PER_FRAME,
        hidden_size=HIDDEN_SIZE,
        num_classes=len(classes),
        num_layers=NUM_LAYERS,
    ).to(device)

    print()
    print("Model:")
    print(model)

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_samples,
        len(classes),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("Training...")
    print()

    best_val_accuracy = -1.0

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        (
            val_loss,
            val_accuracy,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.3f} | "
            f"lr {current_lr:.6f}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "input_size": FEATURES_PER_FRAME,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "sequence_length": SEQUENCE_LENGTH,
                "sample_rate": SAMPLE_RATE,
            }

            torch.save(
                checkpoint,
                MODEL_PATH,
            )

            print(
                f"  Saved best model -> {MODEL_PATH}"
            )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    (
        val_loss,
        val_accuracy,
        labels,
        predictions,
    ) = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    print()
    print("======================================")
    print("       FINAL VALIDATION")
    print("======================================")
    print()

    print(
        f"Validation loss: {val_loss:.4f}"
    )

    print(
        f"Validation accuracy: {val_accuracy:.3f}"
    )

    print()
    print("Classification report:")
    print()

    print(
        classification_report(
            labels,
            predictions,
            labels=np.arange(len(classes)),
            target_names=classes,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print()

    print(
        confusion_matrix(
            labels,
            predictions,
            labels=np.arange(len(classes)),
        )
    )

    print()
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    print(f"Saved: {MODEL_PATH}")
    print()


if __name__ == "__main__":
    main()