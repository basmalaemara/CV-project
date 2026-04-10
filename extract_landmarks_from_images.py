"""
extract_landmarks_from_images.py  (MediaPipe Tasks API — v0.10+)
────────────────────────────────────────────────────────────────
1. Auto-downloads the ASL dataset via kagglehub
2. Runs MediaPipe on every image to extract 21 hand landmarks
3. Saves normalized landmark CSVs ready for training

Output:
  data/static/<class_id>_<label>.csv   (one per letter A-Z)
  data/asl_class_map.csv               (id → label mapping)
"""

import cv2
import os
import sys
import csv
import glob
import random
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import kagglehub

sys.path.insert(0, ".")
from preprocessing.feature_extractor import normalize_landmarks, landmarks_to_flat

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR      = "data/static"
MODEL_PATH    = "models/hand_landmarker.task"
MAX_PER_CLASS = 500      # cap per letter — keeps training fast and balanced

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Step 1: Download dataset ──────────────────────────────────────────────────
print("📦  Downloading dataset via kagglehub...")
print("    (first run may take a few minutes — it caches afterwards)\n")
dataset_path = kagglehub.dataset_download("vignonantoine/combinedasldatasets")
print(f"\n✅  Dataset path: {dataset_path}\n")

# ── Step 2: Find class folders ────────────────────────────────────────────────
# The dataset may have the images directly under dataset_path or one level deeper
# Search for any folder that contains images
def find_image_root(base):
    """Walk until we find folders that contain image files."""
    for root, dirs, files in os.walk(base):
        images = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if images and dirs == []:   # leaf folder with images
            return os.path.dirname(root)
        if dirs and any(
            any(f.lower().endswith((".jpg",".jpeg",".png")) for f in os.listdir(os.path.join(root, d)))
            for d in dirs[:3]
        ):
            return root
    return base

image_root = find_image_root(dataset_path)
print(f"📁  Image root detected: {image_root}")

class_folders = sorted([
    d for d in os.listdir(image_root)
    if os.path.isdir(os.path.join(image_root, d))
])

if not class_folders:
    print(f"❌  No class subfolders found under '{image_root}'")
    print("    Contents:", os.listdir(image_root)[:10])
    sys.exit(1)

label2id = {lbl: i for i, lbl in enumerate(class_folders)}
print(f"🔠  Found {len(class_folders)} classes: {class_folders}\n")

# ── Step 3: Write CSV headers ─────────────────────────────────────────────────
HEADER = ["gesture_id", "label"] + [f"f{i}" for i in range(63)]
for lbl in class_folders:
    path = os.path.join(SAVE_DIR, f"{label2id[lbl]}_{lbl}.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)

# ── Step 4: Set up MediaPipe ──────────────────────────────────────────────────
BaseOptions           = mp_python.BaseOptions
HandLandmarker        = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode     = mp_vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
detector = HandLandmarker.create_from_options(options)

# ── Step 5: Process all images ────────────────────────────────────────────────
total_saved  = 0
total_missed = 0

print("🔍  Extracting landmarks from images...\n")

for lbl in class_folders:
    folder   = os.path.join(image_root, lbl)
    gid      = label2id[lbl]
    out_path = os.path.join(SAVE_DIR, f"{gid}_{lbl}.csv")

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp",
                "*.JPG", "*.JPEG", "*.PNG", "*.BMP"):
        image_paths.extend(glob.glob(os.path.join(folder, ext)))

    if not image_paths:
        print(f"  [{lbl}] ⚠️  No images found — skipping.")
        continue

    if len(image_paths) > MAX_PER_CLASS:
        random.seed(42)
        image_paths = random.sample(image_paths, MAX_PER_CLASS)

    saved  = 0
    missed = 0

    for img_path in image_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            missed += 1
            continue

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        try:
            result = detector.detect(mp_image)
        except Exception:
            missed += 1
            continue

        if not result.hand_landmarks:
            missed += 1
            continue

        lms  = result.hand_landmarks[0]
        pts  = normalize_landmarks(lms)
        flat = landmarks_to_flat(pts)

        row = [gid, lbl] + flat
        with open(out_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        saved += 1

    total_saved  += saved
    total_missed += missed
    pct = saved / max(len(image_paths), 1) * 100
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"  [{lbl}] |{bar}| {saved:>4} saved  {missed:>4} missed  ({pct:.0f}%)")

detector.close()

# ── Step 6: Save class map ────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
with open("data/asl_class_map.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label"])
    for lbl, gid in label2id.items():
        w.writerow([gid, lbl])

print(f"\n{'='*55}")
print(f"✅  Landmark extraction complete!")
print(f"    Total saved  : {total_saved:,} samples")
print(f"    Total missed : {total_missed:,} (no hand detected in image)")
print(f"    Classes      : {len(class_folders)}")
print(f"    CSVs written : data/static/")
print(f"\n👉  Next step: python models/train_static.py")
