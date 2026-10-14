"""
recognize_gui.py

GUI version of temporal pose-sequence recognition.

Features:
    - File selector for video input
    - Save location query
    - Progress bar
    - Embedded video display (toggleable)
    - Quick start for live webcam feed
    - Optional screen capture
"""

from __future__ import annotations

import argparse
import os
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ultralytics import YOLO

try:
    import tkinter as tk
    import tkinter.ttk
    from tkinter import filedialog, messagebox
    from PIL import Image, ImageTk
except ImportError:
    print("ERROR: tkinter and Pillow are required for GUI mode.")
    print("Install with: pip install pillow")
    exit(1)

try:
    import mss
except ImportError:
    mss = None

from temporal_features import (
    normalize_pose,
    build_temporal_features,
    SEQUENCE_LENGTH,
    SAMPLE_RATE,
)


# ============================================================
# CONFIGURATION
# ============================================================

POSE_MODEL = "yolov8n-pose.pt"
CLASSIFIER = "pose_sequence_model.pt"
CONFIDENCE = 0.5
IMG_SIZE = 640
WINDOW_SECONDS = 1.0
MAX_HISTORY_SECONDS = 3.0
OUTPUT_SMOOTHING = 3
MAX_MISSING_SECONDS = 3.0


# ============================================================
# DISPLAY
# ============================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.65
LABEL_THICKNESS = 2
INFO_SCALE = 0.45
INFO_THICKNESS = 1
SKELETON_THICKNESS = 2


# ============================================================
# TEMPORAL MODEL (same as before)
# ============================================================

class PoseSequenceGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        output, hidden = self.gru(x)
        features = output[:, -1, :]
        return self.classifier(features)


def load_classifier(path):
    print("Loading temporal classifier...")
    checkpoint = torch.load(path, map_location="cpu")
    classes = checkpoint["classes"]
    input_size = checkpoint.get("input_size", 85)
    hidden_size = checkpoint.get("hidden_size", 64)
    num_layers = checkpoint.get("num_layers", 1)
    
    model = PoseSequenceGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=len(classes),
        num_layers=num_layers,
    )
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Classes: {', '.join(classes)}")
    print(f"Input size: {input_size}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    
    return model, classes


# ============================================================
# PERSON STATE (same as before)
# ============================================================

class PersonState:
    def __init__(self, max_history_seconds, sample_fps):
        max_frames = max(30, int(max_history_seconds * sample_fps))
        self.poses = deque(maxlen=max_frames)
        self.timestamps = deque(maxlen=max_frames)
        self.predictions = deque(maxlen=OUTPUT_SMOOTHING)
        self.confidences = deque(maxlen=OUTPUT_SMOOTHING)
        self.stable_label = "..."
        self.stable_confidence = 0.0
        self.last_seen = 0.0

    def add_pose(self, pose, timestamp):
        self.poses.append(pose)
        self.timestamps.append(timestamp)
        self.last_seen = timestamp

    def ready(self, timestamp):
        if len(self.poses) < 2:
            return False
        elapsed = timestamp - self.timestamps[0]
        return elapsed >= WINDOW_SECONDS

    def get_recent_poses(self, timestamp):
        cutoff = timestamp - WINDOW_SECONDS
        selected = [
            pose for pose, pose_time in zip(self.poses, self.timestamps)
            if pose_time >= cutoff
        ]
        return selected

    def update_prediction(self, prediction, confidence, classes):
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        
        if not self.predictions:
            return
        
        counts = np.bincount(list(self.predictions), minlength=len(classes))
        candidate = int(np.argmax(counts))
        
        candidate_confidences = [
            conf for pred, conf in zip(self.predictions, self.confidences)
            if pred == candidate
        ]
        
        average_confidence = float(np.mean(candidate_confidences))
        self.stable_label = classes[candidate]
        self.stable_confidence = average_confidence


@torch.no_grad()
def classify_person(state, timestamp, model, classes):
    if not state.ready(timestamp):
        return None, None
    
    recent_poses = state.get_recent_poses(timestamp)
    if len(recent_poses) < 2:
        return None, None
    
    poses = np.stack(recent_poses, axis=0)
    features = build_temporal_features(poses, target_length=SEQUENCE_LENGTH, sample_rate=SAMPLE_RATE)
    tensor = torch.from_numpy(features).unsqueeze(0)
    
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    prediction = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[prediction].item())
    label = classes[prediction]
    
    return label, confidence


def analyze_frame(frame, pose_model, classifier, classes, person_states, frame_timestamp, source_fps):
    results = pose_model.track(
        frame, persist=True, tracker="bytetrack.yaml",
        imgsz=IMG_SIZE, conf=CONFIDENCE, verbose=False,
    )
    result = results[0]
    annotated = result.plot(kpt_line=True, kpt_radius=3, line_width=SKELETON_THICKNESS)
    
    if result.keypoints is None or result.boxes is None or len(result.keypoints) == 0:
        return annotated
    
    people = result.keypoints.data.cpu().numpy()
    
    if result.boxes.is_track:
        track_ids = result.boxes.id.int().cpu().tolist()
    else:
        track_ids = list(range(len(people)))
    
    for index, (person, track_id) in enumerate(zip(people, track_ids)):
        normalized = normalize_pose(person).reshape(17, 3)
        
        if track_id not in person_states:
            person_states[track_id] = PersonState(
                max_history_seconds=MAX_HISTORY_SECONDS,
                sample_fps=source_fps,
            )
        
        state = person_states[track_id]
        state.add_pose(normalized, frame_timestamp)
        
        raw_label, confidence = classify_person(state, frame_timestamp, classifier, classes)
        
        if raw_label is not None:
            prediction_index = classes.index(raw_label)
            state.update_prediction(prediction_index, confidence, classes)
        
        box = result.boxes[index]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        if state.stable_label == "...":
            label = f"ID {track_id}: WARMING UP"
            info = f"{len(state.poses)} poses"
        else:
            label = f"ID {track_id}: {state.stable_label.upper()}"
            info = f"confidence: {state.stable_confidence:.2f}"
        
        label_size, _ = cv2.getTextSize(label, FONT, LABEL_SCALE, LABEL_THICKNESS)
        info_size, _ = cv2.getTextSize(info, FONT, INFO_SCALE, INFO_THICKNESS)
        
        box_width = max(label_size[0], info_size[0]) + 12
        box_height = label_size[1] + info_size[1] + 16
        
        label_x = max(int(x1), 0)
        label_y = max(int(y1) - 8, box_height + 4)
        
        cv2.rectangle(annotated, (label_x, label_y - box_height), (label_x + box_width, label_y), (0, 0, 0), -1)
        cv2.putText(annotated, label, (label_x + 6, label_y - info_size[1] - 6), FONT, LABEL_SCALE, (0, 255, 0), LABEL_THICKNESS, cv2.LINE_AA)
        cv2.putText(annotated, info, (label_x + 6, label_y - 5), FONT, INFO_SCALE, (220, 220, 220), INFO_THICKNESS, cv2.LINE_AA)
    
    return annotated


def cleanup_person_states(person_states, current_time):
    expired = [track_id for track_id, state in person_states.items() if current_time - state.last_seen > MAX_MISSING_SECONDS]
    for track_id in expired:
        del person_states[track_id]


# ============================================================
# GUI APPLICATION
# ============================================================

class PoseRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Sequence Recognition")
        self.root.geometry("900x700")
        
        # State
        self.processing = False
        self.show_video = tk.BooleanVar(value=True)
        self.stop_event = threading.Event()
        
        # Load models
        print("Loading models...")
        self.pose_model = YOLO(POSE_MODEL)
        self.classifier, self.classes = load_classifier(CLASSIFIER)
        print("Models loaded.")
        
        self.setup_ui()
    
    def setup_ui(self):
        # Top control panel
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack(fill=tk.X, padx=10)
        
        tk.Button(control_frame, text="Select Video", command=self.select_video, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Live Feed (Webcam)", command=self.start_webcam, width=20).pack(side=tk.LEFT, padx=5)
        
        if mss is not None:
            tk.Button(control_frame, text="Screen Capture", command=self.start_screen_capture, width=15).pack(side=tk.LEFT, padx=5)
        
        tk.Checkbutton(control_frame, text="Show Video", variable=self.show_video).pack(side=tk.RIGHT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var, font=("Arial", 10)).pack(pady=5)
        
        # Video display
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    
    def select_video(self):
        if self.processing:
            messagebox.showwarning("Warning", "Already processing!")
            return
        
        video_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if not video_path:
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save Processed Video As",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
            initialfile=f"{Path(video_path).stem}_processed.mp4"
        )
        
        if not save_path:
            return
        
        self.start_processing(video_path, save_path)
    
    def start_webcam(self):
        if self.processing:
            messagebox.showwarning("Warning", "Already processing!")
            return
        
        self.start_processing(0, None, is_webcam=True)
    
    def start_screen_capture(self):
        if self.processing:
            messagebox.showwarning("Warning", "Already processing!")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save Screen Recording As",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
            initialfile="screen_recording_processed.mp4"
        )
        
        if not save_path:
            return
        
        self.start_processing("screen", save_path, is_screen=True)
    
    def start_processing(self, source, save_path, is_webcam=False, is_screen=False):
        self.processing = True
        self.stop_event.clear()
        self.progress_var.set(0)
        
        thread = threading.Thread(
            target=self.process_video,
            args=(source, save_path, is_webcam, is_screen),
            daemon=True
        )
        thread.start()
    
    def process_video(self, source, save_path, is_webcam, is_screen):
        try:
            if is_screen:
                sct = mss.mss()
                monitor = sct.monitors[1]
                cap = None
                fps = 30.0
                total_frames = 0
                width = monitor["width"]
                height = monitor["height"]
            elif is_webcam:
                cap = cv2.VideoCapture(source)
                sct = None
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                cap = cv2.VideoCapture(source)
                sct = None
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            writer = None
            if save_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            
            person_states = {}
            frame_number = 0
            
            while not self.stop_event.is_set():
                if is_screen:
                    screenshot = sct.grab(monitor)
                    frame = np.asarray(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    frame_timestamp = time.perf_counter()
                    ok = True
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_timestamp = frame_number / fps
                
                frame_number += 1
                
                annotated = analyze_frame(
                    frame, self.pose_model, self.classifier, self.classes,
                    person_states, frame_timestamp, fps
                )
                
                cleanup_person_states(person_states, frame_timestamp)
                
                if writer:
                    writer.write(annotated)
                
                # Update display
                if self.show_video.get():
                    self.update_display(annotated)
                
                # Update progress
                if total_frames > 0:
                    progress = (frame_number / total_frames) * 100
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    self.root.after(0, lambda f=frame_number, t=total_frames: self.status_var.set(f"Processing: {f}/{t} frames"))
                else:
                    self.root.after(0, lambda f=frame_number: self.status_var.set(f"Processing: {f} frames"))
            
            # Cleanup
            if cap:
                cap.release()
            if sct:
                sct.close()
            if writer:
                writer.release()
            
            self.root.after(0, lambda: self.status_var.set("Finished!"))
            self.root.after(0, lambda: self.progress_var.set(100))
            
            if save_path:
                self.root.after(0, lambda: messagebox.showinfo("Done", f"Saved to:\n{save_path}"))
        
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n{str(e)}"))
        
        finally:
            self.processing = False
    
    def update_display(self, frame):
        # Resize frame to fit display
        display_width = 800
        display_height = 600
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_image)
        self.root.after(0, lambda: self.video_label.configure(image=photo))
        self.root.after(0, lambda: setattr(self, '_last_photo', photo))  # Keep reference


def main():
    root = tk.Tk()
    app = PoseRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()