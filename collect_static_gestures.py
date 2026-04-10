"""
collect_static_gestures.py  (MediaPipe Tasks API — v0.10+)
────────────────────────────────────────────────────────────────
Collect hand-landmark data for STATIC gestures via webcam.

Controls
--------
  [0-7]  Select gesture class
  SPACE  Toggle recording ON / OFF
  Q      Quit
"""

import cv2
import csv
import os
import sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

sys.path.insert(0, ".")
from preprocessing.feature_extractor import (
    GESTURE_LABELS,
    normalize_landmarks,
    landmarks_to_flat,
)

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR   = "data/static"
MODEL_PATH = "models/hand_landmarker.task"
SAMPLES_PER_GESTURE = 300
os.makedirs(SAVE_DIR, exist_ok=True)

# ── MediaPipe Tasks setup ─────────────────────────────────────────────────────
BaseOptions   = mp_python.BaseOptions
HandLandmarker       = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode    = mp_vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = HandLandmarker.create_from_options(options)

# ── Hand connection pairs for drawing ────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_landmarks_manual(frame, landmarks, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 200, 120), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 4, (0, 160, 80), 2)

# ── CSV setup ─────────────────────────────────────────────────────────────────
HEADER = (
    ["gesture_id"]
    + [f"f{i}" for i in range(63)]
)
for gid, label in GESTURE_LABELS.items():
    path = os.path.join(SAVE_DIR, f"{gid}_{label}.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)

def count_existing(gid, label):
    path = os.path.join(SAVE_DIR, f"{gid}_{label}.csv")
    try:
        return max(0, sum(1 for _ in open(path)) - 1)
    except FileNotFoundError:
        return 0

count = {gid: count_existing(gid, lbl) for gid, lbl in GESTURE_LABELS.items()}

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

current_gesture = 0
recording = False

print("\n=== Static Gesture Data Collection ===")
print("Controls: [0-7] select class | [SPACE] toggle record | [Q] quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ── Run detection ─────────────────────────────────────────────────────────
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = detector.detect(mp_image)

    # ── Dark header ───────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    label_name = GESTURE_LABELS[current_gesture]
    rec_color  = (50, 230, 100) if recording else (80, 80, 255)
    rec_text   = "● REC" if recording else "■ PAUSED — press SPACE"

    cv2.putText(frame, f"Gesture [{current_gesture}]: {label_name.upper().replace('_',' ')}",
                (12, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(frame, f"{rec_text}   Samples: {count[current_gesture]}/{SAMPLES_PER_GESTURE}",
                (12, 72), cv2.FONT_HERSHEY_DUPLEX, 0.72, rec_color, 2)

    # Class pills
    pill_h = 82 // len(GESTURE_LABELS)
    for gid, glbl in GESTURE_LABELS.items():
        done     = count[gid] >= SAMPLES_PER_GESTURE
        pill_col = (0, 180, 60) if done else ((50, 180, 80) if gid == current_gesture else (50, 50, 70))
        y0 = 8 + gid * pill_h
        cv2.rectangle(frame, (w - 210, y0), (w - 8, y0 + pill_h - 2), pill_col, -1)
        tick = "✓ " if done else f"{gid}: "
        cv2.putText(frame, tick + glbl[:10], (w - 205, y0 + pill_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    # ── Landmarks + recording ─────────────────────────────────────────────────
    hand_detected = result.hand_landmarks and len(result.hand_landmarks) > 0
    if hand_detected:
        lms = result.hand_landmarks[0]
        draw_landmarks_manual(frame, lms, h, w)

        if recording and count[current_gesture] < SAMPLES_PER_GESTURE:
            pts  = normalize_landmarks(lms)
            flat = landmarks_to_flat(pts)
            row  = [current_gesture] + flat
            lbl  = GESTURE_LABELS[current_gesture]
            path = os.path.join(SAVE_DIR, f"{current_gesture}_{lbl}.csv")
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            count[current_gesture] += 1

        if recording and count[current_gesture] >= SAMPLES_PER_GESTURE:
            recording = False
            print(f"✅  '{GESTURE_LABELS[current_gesture]}' complete! ({SAMPLES_PER_GESTURE} samples)")

    # Hand status
    dot_col = (50, 230, 100) if hand_detected else (60, 60, 200)
    cv2.circle(frame, (20, 116), 8, dot_col, -1)
    cv2.putText(frame, "Hand detected" if hand_detected else "No hand — show your hand!",
                (34, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    # Progress bar
    done_frac = min(count[current_gesture] / SAMPLES_PER_GESTURE, 1.0)
    bar_w     = int((w - 20) * done_frac)
    cv2.rectangle(frame, (10, h - 22), (w - 10, h - 8), (40, 40, 60), -1)
    cv2.rectangle(frame, (10, h - 22), (10 + bar_w, h - 8), (50, 220, 120), -1)
    cv2.putText(frame, f"{int(done_frac*100)}%", (w // 2 - 15, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("Data Collection — Static Gestures", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        recording = not recording
        if recording:
            print(f"🔴 Recording '{GESTURE_LABELS[current_gesture]}'...")
        else:
            print("   Paused.")
    elif chr(key).isdigit() and int(chr(key)) in GESTURE_LABELS:
        current_gesture = int(chr(key))
        recording = False

cap.release()
cv2.destroyAllWindows()
detector.close()

print("\n=== Final Count ===")
for gid, lbl in GESTURE_LABELS.items():
    status = "✅" if count[gid] >= SAMPLES_PER_GESTURE else f"{count[gid]}/{SAMPLES_PER_GESTURE}"
    print(f"  [{gid}] {lbl}: {status}")
