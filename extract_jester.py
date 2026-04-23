"""
extract_jester.py
────────────────────────────────────────────────────────────────
Extracts dynamic gesture sequences from the Jester dataset.

The Jester dataset uses numbered folders (1/, 2/, 3/ ...) with
a CSV file mapping folder IDs to class labels.

Usage
-----
  python extract_jester.py --jester_dir path/to/jester/ --csv path/to/labels.csv

The script will:
  1. Read the CSV to find video IDs for your 4 gesture classes
  2. Extract MediaPipe landmark sequences from those videos
  3. Save to data/dynamic/ ready for training

Expected CSV format (any of these work):
  video_id, label
  id, label
  (first col = folder number, second col = class name)
"""

import os
import sys
import csv
import argparse
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, ".")
from preprocessing.feature_extractor import (
    DYNAMIC_LABELS,
    normalize_landmarks,
    landmarks_to_flat,
)

SAVE_DIR        = "data/dynamic"
MODEL_PATH      = "models/hand_landmarker.task"
SEQUENCE_LENGTH = 30
N_FEATURES      = 63
STRIDE          = 5    # extract a sequence every N frames
MAX_PER_CLASS   = 500  # cap sequences per class to avoid imbalance

os.makedirs(SAVE_DIR, exist_ok=True)

# Jester class names → your gesture IDs
JESTER_MAP = {
    "swiping left":                    0,
    "swipe left":                      0,
    "swiping right":                   1,
    "swipe right":                     1,
    "zooming in with full hand":       2,
    "zoom in":                         2,
    "zooming in with two fingers":     2,
    "zooming out with full hand":      3,
    "zoom out":                        3,
    "zooming out with two fingers":    3,
}

# ── MediaPipe setup ───────────────────────────────────────────────────────────
BaseOptions           = mp_python.BaseOptions
HandLandmarker        = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode     = mp_vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
)
detector = HandLandmarker.create_from_options(options)

def next_index(gesture_id, label):
    prefix = f"{gesture_id}_{label}_"
    return sum(1 for f in os.listdir(SAVE_DIR)
               if f.startswith(prefix) and f.endswith(".npy"))

def extract_landmarks(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(img)
    if not result.hand_landmarks:
        return None
    pts = normalize_landmarks(result.hand_landmarks[0])
    return np.array(landmarks_to_flat(pts), dtype=np.float32)

def process_video_folder(folder_path, gesture_id, idx):
    """Process a Jester video folder (images named 00001.jpg etc.)"""
    label  = DYNAMIC_LABELS[gesture_id]
    frames = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if len(frames) < SEQUENCE_LENGTH:
        return 0, idx

    landmarks = []
    for fname in frames:
        frame = cv2.imread(os.path.join(folder_path, fname))
        if frame is None:
            landmarks.append(None)
            continue
        landmarks.append(extract_landmarks(frame))

    saved = 0
    i = 0
    while i + SEQUENCE_LENGTH <= len(landmarks):
        window = landmarks[i: i + SEQUENCE_LENGTH]
        if all(lm is not None for lm in window):
            seq  = np.array(window, dtype=np.float32)
            path = os.path.join(SAVE_DIR, f"{gesture_id}_{label}_{idx}.npy")
            np.save(path, seq)
            idx   += 1
            saved += 1
        i += STRIDE
    return saved, idx

def process_video_file(video_path, gesture_id, idx):
    """Process a standard video file (.mp4, .avi etc.)"""
    label  = DYNAMIC_LABELS[gesture_id]
    cap    = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, idx

    landmarks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks.append(extract_landmarks(frame))
    cap.release()

    if len(landmarks) < SEQUENCE_LENGTH:
        return 0, idx

    saved = 0
    i = 0
    while i + SEQUENCE_LENGTH <= len(landmarks):
        window = landmarks[i: i + SEQUENCE_LENGTH]
        if all(lm is not None for lm in window):
            seq  = np.array(window, dtype=np.float32)
            path = os.path.join(SAVE_DIR, f"{gesture_id}_{label}_{idx}.npy")
            np.save(path, seq)
            idx   += 1
            saved += 1
        i += STRIDE
    return saved, idx

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--jester_dir", required=True,
                    help="Path to Jester folder containing numbered subfolders")
parser.add_argument("--csv", required=True,
                    help="Path to the Jester CSV file mapping IDs to labels "
                         "(e.g. jester-v1-train.csv or labels.csv)")
parser.add_argument("--max", type=int, default=MAX_PER_CLASS,
                    help=f"Max sequences to extract per class (default {MAX_PER_CLASS})")
args = parser.parse_args()

# ── Read CSV and build ID → gesture_id mapping ────────────────────────────────
print(f"\nReading {args.csv} ...")
id_to_gesture = {}

with open(args.csv, newline="", encoding="utf-8") as f:
    # Auto-detect delimiter
    sample  = f.read(2048); f.seek(0)
    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    reader  = csv.reader(f, dialect)
    header  = next(reader, None)
    print(f"  Columns detected: {header}")

    for row in reader:
        if len(row) < 2:
            continue
        video_id  = row[0].strip()
        label_raw = row[1].strip().lower()
        gid       = JESTER_MAP.get(label_raw)
        if gid is not None:
            id_to_gesture[video_id] = gid

print(f"  Found {len(id_to_gesture)} matching video IDs across 4 gesture classes")
for gid, lbl in DYNAMIC_LABELS.items():
    count = sum(1 for v in id_to_gesture.values() if v == gid)
    print(f"    [{gid}] {lbl:<20} {count} videos")

# ── Process videos ─────────────────────────────────────────────────────────────
print(f"\nProcessing videos from {args.jester_dir} ...")
saved_per_class = {gid: 0 for gid in DYNAMIC_LABELS}
idx_per_class   = {gid: next_index(gid, lbl) for gid, lbl in DYNAMIC_LABELS.items()}
skipped         = 0

entries = list(id_to_gesture.items())
for i, (video_id, gesture_id) in enumerate(entries):
    if saved_per_class[gesture_id] >= args.max:
        continue

    label    = DYNAMIC_LABELS[gesture_id]
    item_dir = os.path.join(args.jester_dir, str(video_id))

    # Progress print every 50 videos
    if i % 50 == 0:
        print(f"  [{i}/{len(entries)}]  " +
              "  ".join(f"{DYNAMIC_LABELS[g]}: {saved_per_class[g]}"
                        for g in DYNAMIC_LABELS))

    if os.path.isdir(item_dir):
        # Jester image-folder format
        n, idx_per_class[gesture_id] = process_video_folder(
            item_dir, gesture_id, idx_per_class[gesture_id]
        )
    else:
        # Try as video file with common extensions
        found = False
        for ext in (".mp4", ".avi", ".mov", ".webm"):
            vpath = os.path.join(args.jester_dir, str(video_id) + ext)
            if os.path.exists(vpath):
                n, idx_per_class[gesture_id] = process_video_file(
                    vpath, gesture_id, idx_per_class[gesture_id]
                )
                found = True
                break
        if not found:
            skipped += 1
            continue

    saved_per_class[gesture_id] += n

detector.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n✅  Done.")
for gid, lbl in DYNAMIC_LABELS.items():
    total = next_index(gid, lbl)
    print(f"  [{gid}] {lbl:<20} +{saved_per_class[gid]} new  →  {total} total")
if skipped:
    print(f"  ({skipped} video IDs skipped — folder/file not found)")
print("\nNext steps:")
print("  python prepare_dynamic_data.py")
print("  python models/train_dynamic.py")
print("  python models/evaluate.py")
