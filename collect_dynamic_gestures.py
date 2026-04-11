"""
collect_dynamic_gestures.py  (MediaPipe Tasks API — v0.10+)
────────────────────────────────────────────────────────────────
Collect hand-landmark SEQUENCES for DYNAMIC gestures.

Controls
--------
  [0-3]  Select gesture class
  SPACE  Start one recording (30-frame sequence)
  Q      Quit
"""

import cv2
import os
import sys
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

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR         = "data/dynamic"
MODEL_PATH       = "models/hand_landmarker.task"
SEQUENCE_LENGTH  = 30
SAMPLES_PER_GESTURE = 20
os.makedirs(SAVE_DIR, exist_ok=True)

# ── MediaPipe Tasks setup ─────────────────────────────────────────────────────
BaseOptions          = mp_python.BaseOptions
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

# ── Drawing helpers ───────────────────────────────────────────────────────────
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
        cv2.line(frame, pts[a], pts[b], (120, 80, 255), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 4, (80, 40, 200), 2)

# ── Count existing samples ────────────────────────────────────────────────────
def count_existing(gid, label):
    prefix = f"{gid}_{label}_"
    return sum(1 for f in os.listdir(SAVE_DIR) if f.startswith(prefix) and f.endswith(".npy"))

sample_count = {gid: count_existing(gid, lbl) for gid, lbl in DYNAMIC_LABELS.items()}

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

gesture_id = 0
sequence   = []
collecting = False

print("\n=== Dynamic Gesture Data Collection ===")
print("Controls: [0-3] select class | [SPACE] start one recording | [Q] quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = detector.detect(mp_image)

    # ── Header ────────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (30, 15, 15), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    label_name = DYNAMIC_LABELS[gesture_id]
    rec_color  = (50, 230, 100) if collecting else (80, 80, 255)

    cv2.putText(frame, f"Gesture [{gesture_id}]: {label_name.upper().replace('_',' ')}",
                (12, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(frame,
                f"Samples: {sample_count[gesture_id]}/{SAMPLES_PER_GESTURE}   "
                f"Frames captured: {len(sequence)}/{SEQUENCE_LENGTH}",
                (12, 72), cv2.FONT_HERSHEY_DUPLEX, 0.65, rec_color, 2)

    # Sidebar pills
    for gid_p, glbl in DYNAMIC_LABELS.items():
        done     = sample_count[gid_p] >= SAMPLES_PER_GESTURE
        pill_col = (0, 180, 60) if done else ((80, 50, 200) if gid_p == gesture_id else (50, 50, 70))
        y0 = 8 + gid_p * 22
        cv2.rectangle(frame, (w - 210, y0), (w - 8, y0 + 20), pill_col, -1)
        tick = "✓ " if done else f"{gid_p}: "
        cv2.putText(frame, tick + glbl, (w - 205, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    # ── Detection + recording ─────────────────────────────────────────────────
    hand_detected = result.hand_landmarks and len(result.hand_landmarks) > 0
    if hand_detected:
        lms = result.hand_landmarks[0]
        draw_landmarks_manual(frame, lms, h, w)

        if collecting:
            pts  = normalize_landmarks(lms)
            flat = landmarks_to_flat(pts)
            sequence.append(flat)

            if len(sequence) >= SEQUENCE_LENGTH:
                arr  = np.array(sequence[:SEQUENCE_LENGTH], dtype=np.float32)
                path = os.path.join(
                    SAVE_DIR,
                    f"{gesture_id}_{DYNAMIC_LABELS[gesture_id]}_{sample_count[gesture_id]}.npy"
                )
                np.save(path, arr)
                sample_count[gesture_id] += 1
                print(f"  ✅ Saved sample {sample_count[gesture_id]} for '{DYNAMIC_LABELS[gesture_id]}'")
                sequence   = []
                collecting = False

    # Hand status dot
    dot_col = (50, 230, 100) if hand_detected else (60, 60, 200)
    cv2.circle(frame, (20, 116), 8, dot_col, -1)
    hint = "Hand detected" if hand_detected else "No hand — show your hand!"
    cv2.putText(frame, hint, (34, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    # Recording progress bar
    frac  = len(sequence) / SEQUENCE_LENGTH if collecting else 0
    bar_w = int((w - 20) * frac)
    cv2.rectangle(frame, (10, h - 22), (w - 10, h - 8), (40, 30, 30), -1)
    cv2.rectangle(frame, (10, h - 22), (10 + bar_w, h - 8), (80, 80, 255), -1)

    if not collecting:
        cv2.putText(frame, "Press SPACE to record one gesture sequence",
                    (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

    cv2.imshow("Data Collection — Dynamic Gestures", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" ") and not collecting:
        sequence   = []
        collecting = True
        print(f"\n🎬 GO — perform '{DYNAMIC_LABELS[gesture_id]}' now!")
    elif chr(key).isdigit() and int(chr(key)) in DYNAMIC_LABELS:
        gesture_id = int(chr(key))
        collecting = False
        sequence   = []

cap.release()
cv2.destroyAllWindows()
detector.close()

print("\n=== Collection Summary ===")
for gid, lbl in DYNAMIC_LABELS.items():
    status = "✅" if sample_count[gid] >= SAMPLES_PER_GESTURE else f"{sample_count[gid]}/{SAMPLES_PER_GESTURE}"
    print(f"  [{gid}] {lbl}: {status}")
