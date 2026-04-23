"""
extract_from_video.py
────────────────────────────────────────────────────────────────
Extract dynamic gesture sequences from video files.

Usage
-----
  python extract_from_video.py --video path/to/video.mp4 --gesture 0
  python extract_from_video.py --video path/to/video.mp4 --gesture 1
  python extract_from_video.py --folder path/to/folder/  --gesture 2

Gesture IDs
-----------
  0 = swipe_left
  1 = swipe_right
  2 = zoom_in
  3 = zoom_out

How it works
------------
  - Scans every 30-frame window in the video
  - If a hand is detected in all 30 frames → saves as one .npy sequence
  - Shows a preview window so you can verify before saving
  - Saved directly to data/dynamic/ with auto-incremented filenames

Tips
----
  - --webcam mode: just keep doing the gesture continuously, press Q when done
  - 30 seconds of continuous gesturing = ~50 sequences extracted automatically
  - No need to press SPACE for each sequence — sliding window does it all
  - Works with phone videos too: copy to PC then use --video
"""

import cv2
import os
import sys
import glob
import argparse
import numpy as np
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
STRIDE          = 10   # slide window every N frames (lower = more sequences extracted)

os.makedirs(SAVE_DIR, exist_ok=True)

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

# ── Helpers ───────────────────────────────────────────────────────────────────
def next_index(gesture_id, label):
    prefix = f"{gesture_id}_{label}_"
    return sum(1 for f in os.listdir(SAVE_DIR)
               if f.startswith(prefix) and f.endswith(".npy"))

def extract_landmarks(frame):
    """Return flat (63,) landmark array or None if no hand detected."""
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(img)
    if not result.hand_landmarks:
        return None
    pts  = normalize_landmarks(result.hand_landmarks[0])
    return np.array(landmarks_to_flat(pts), dtype=np.float32)

def process_video(video_path, gesture_id):
    label   = DYNAMIC_LABELS[gesture_id]
    cap     = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"\n  Video: {os.path.basename(video_path)}")
    print(f"  Frames: {total_frames}  |  FPS: {fps:.0f}  |  "
          f"Duration: {total_frames/fps:.1f}s")

    # Read all frames with hand landmarks
    all_landmarks = []
    frame_idx     = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        lm    = extract_landmarks(frame)
        all_landmarks.append(lm)   # None if no hand
        frame_idx += 1
    cap.release()

    # Slide a 30-frame window, save windows where hand visible in all frames
    saved   = 0
    idx     = next_index(gesture_id, label)
    i       = 0
    while i + SEQUENCE_LENGTH <= len(all_landmarks):
        window = all_landmarks[i: i + SEQUENCE_LENGTH]
        if all(lm is not None for lm in window):
            seq  = np.array(window, dtype=np.float32)   # (30, 63)
            path = os.path.join(SAVE_DIR, f"{gesture_id}_{label}_{idx}.npy")
            np.save(path, seq)
            idx   += 1
            saved += 1
        i += STRIDE

    print(f"  Extracted {saved} sequences  →  data/dynamic/")
    return saved

# ── CLI ───────────────────────────────────────────────────────────────────────
def record_from_webcam(gesture_id):
    """Record directly from laptop webcam using sliding window extraction."""
    label  = DYNAMIC_LABELS[gesture_id]
    cap    = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"\n  Webcam recording for gesture [{gesture_id}]: {label.upper()}")
    print("  Keep doing the gesture continuously. Press Q to stop.\n")

    buffer    = []   # rolling buffer of landmark arrays
    saved     = 0
    idx       = next_index(gesture_id, label)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        lm     = extract_landmarks(frame)

        # Rolling buffer — keep last SEQUENCE_LENGTH frames
        buffer.append(lm)
        if len(buffer) > SEQUENCE_LENGTH:
            buffer.pop(0)

        # Every STRIDE frames, check if buffer is full and all hands detected
        if frame_num % STRIDE == 0 and len(buffer) == SEQUENCE_LENGTH:
            if all(x is not None for x in buffer):
                seq  = np.array(buffer, dtype=np.float32)
                path = os.path.join(SAVE_DIR, f"{gesture_id}_{label}_{idx}.npy")
                np.save(path, seq)
                idx   += 1
                saved += 1

        # UI overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (20, 10, 10), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        hand_ok  = lm is not None
        buf_full = len(buffer) == SEQUENCE_LENGTH and all(x is not None for x in buffer)
        dot_col  = (50, 230, 100) if hand_ok else (60, 60, 200)

        cv2.putText(frame, f"Gesture: {label.upper().replace('_',' ')}",
                    (12, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(frame, f"Saved: {saved} sequences   Buffer: {len(buffer)}/{SEQUENCE_LENGTH}",
                    (12, 68), cv2.FONT_HERSHEY_DUPLEX, 0.65,
                    (50, 230, 100) if buf_full else (180, 180, 180), 2)
        cv2.circle(frame, (w - 30, 30), 10, dot_col, -1)
        cv2.putText(frame, "Hand OK" if hand_ok else "No hand!",
                    (w - 110, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Progress bar filling up as buffer fills
        frac  = len([x for x in buffer if x is not None]) / SEQUENCE_LENGTH
        bar_w = int((w - 20) * frac)
        cv2.rectangle(frame, (10, h - 20), (w - 10, h - 6), (40, 30, 30), -1)
        cv2.rectangle(frame, (10, h - 20), (10 + bar_w, h - 6),
                      (50, 220, 120) if buf_full else (80, 80, 255), -1)

        cv2.imshow(f"Recording — {label}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()
    return saved


parser = argparse.ArgumentParser(description="Extract gesture sequences from webcam or video files")
parser.add_argument("--video",   type=str,  help="Path to a single video file")
parser.add_argument("--folder",  type=str,  help="Path to a folder of video files")
parser.add_argument("--webcam",  action="store_true", help="Record live from laptop webcam")
parser.add_argument("--gesture", type=int,  required=True,
                    choices=list(DYNAMIC_LABELS.keys()),
                    help="Gesture ID: " + ", ".join(
                        f"{k}={v}" for k, v in DYNAMIC_LABELS.items()))
args = parser.parse_args()

if not args.video and not args.folder and not args.webcam:
    parser.error("Provide --video, --folder, or --webcam")

gesture_id = args.gesture
print(f"\n=== Extracting sequences for gesture [{gesture_id}]: "
      f"{DYNAMIC_LABELS[gesture_id]} ===")

total_saved = 0
if args.webcam:
    total_saved = record_from_webcam(gesture_id)
else:
    video_files = []
    if args.video:
        video_files = [args.video]
    elif args.folder:
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MOV", "*.MP4"):
            video_files += glob.glob(os.path.join(args.folder, ext))
        video_files.sort()
        print(f"  Found {len(video_files)} video files in {args.folder}")
    for vf in video_files:
        total_saved += process_video(vf, gesture_id)

detector.close()

print(f"\n✅  Total sequences extracted: {total_saved}")
print(f"   Gesture [{gesture_id}] {DYNAMIC_LABELS[gesture_id]} now has "
      f"{next_index(gesture_id, DYNAMIC_LABELS[gesture_id])} sequences total")
print("\nRun prepare_dynamic_data.py again to re-split and re-augment with the new data.")
