"""
collect_dynamic_gestures.py  (MediaPipe Tasks API — v0.10+)
────────────────────────────────────────────────────────────────
Collect hand-landmark SEQUENCES for DYNAMIC gestures.

Controls
--------
  [0-3]  Select gesture class
  SPACE  Start one recording (3-second preview → 30-frame capture)
  Q      Quit

Resume mode: existing .npy files are counted automatically; only the
remaining sequences needed to reach TARGET_PER_GESTURE are collected.
"""

import cv2
import os
import sys
import time
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
SAVE_DIR            = "data/dynamic"
MODEL_PATH          = "models/hand_landmarker.task"
SEQUENCE_LENGTH     = 30
TARGET_PER_GESTURE  = 200   # target sequences per gesture
PREVIEW_SECONDS     = 3     # countdown before each recording
os.makedirs(SAVE_DIR, exist_ok=True)

# ── MediaPipe Tasks setup ─────────────────────────────────────────────────────
BaseOptions           = mp_python.BaseOptions
HandLandmarker        = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode     = mp_vision.RunningMode

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

# ── Resume helpers ────────────────────────────────────────────────────────────
def count_existing(gid, label):
    prefix = f"{gid}_{label}_"
    return sum(1 for f in os.listdir(SAVE_DIR)
               if f.startswith(prefix) and f.endswith(".npy"))

def next_save_index(gid, label):
    """Return the lowest non-conflicting integer index for a new file."""
    prefix = f"{gid}_{label}_"
    used = set()
    for f in os.listdir(SAVE_DIR):
        if f.startswith(prefix) and f.endswith(".npy"):
            try:
                used.add(int(f[len(prefix):-4]))
            except ValueError:
                pass
    idx = 0
    while idx in used:
        idx += 1
    return idx

# ── Resume: count what we already have ───────────────────────────────────────
sample_count = {gid: count_existing(gid, lbl) for gid, lbl in DYNAMIC_LABELS.items()}

print("\n=== Dynamic Gesture Data Collection (Resume Mode) ===")
print(f"Target: {TARGET_PER_GESTURE} sequences per gesture\n")
for gid, lbl in DYNAMIC_LABELS.items():
    have  = sample_count[gid]
    need  = max(0, TARGET_PER_GESTURE - have)
    print(f"  [{gid}] {lbl}: {have} existing  →  {need} still needed")
print("\nControls: [0-3] select class | [SPACE] start recording | [Q] quit\n")

# ── State machine ─────────────────────────────────────────────────────────────
# States: "idle" | "preview" | "collecting"
STATE        = "idle"
preview_end  = 0.0
sequence     = []

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

gesture_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    now   = time.time()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = detector.detect(mp_image)

    # ── Transition: preview → collecting ─────────────────────────────────────
    if STATE == "preview" and now >= preview_end:
        STATE    = "collecting"
        sequence = []
        print(f"\n  GO — perform '{DYNAMIC_LABELS[gesture_id]}' now!")

    # ── Header bar ───────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (30, 15, 15), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    label_name = DYNAMIC_LABELS[gesture_id]
    have       = sample_count[gesture_id]
    need       = max(0, TARGET_PER_GESTURE - have)
    rec_color  = (50, 230, 100) if STATE == "collecting" else (80, 80, 255)

    cv2.putText(frame, f"Gesture [{gesture_id}]: {label_name.upper().replace('_',' ')}",
                (12, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(frame,
                f"Collected: {have} / {TARGET_PER_GESTURE}   "
                f"Remaining: {need}   "
                f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
                (12, 72), cv2.FONT_HERSHEY_DUPLEX, 0.65, rec_color, 2)

    # ── Sidebar gesture pills ─────────────────────────────────────────────────
    for gid_p, glbl in DYNAMIC_LABELS.items():
        cnt      = sample_count[gid_p]
        done     = cnt >= TARGET_PER_GESTURE
        pill_col = (0, 180, 60) if done else ((80, 50, 200) if gid_p == gesture_id else (50, 50, 70))
        y0 = 8 + gid_p * 22
        cv2.rectangle(frame, (w - 230, y0), (w - 8, y0 + 20), pill_col, -1)
        tick = "OK" if done else f"{cnt:>3}/{TARGET_PER_GESTURE}"
        cv2.putText(frame, f"{gid_p}: {glbl}  {tick}", (w - 225, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    # ── Collection progress bar (per gesture) ────────────────────────────────
    bar_y0, bar_y1 = h - 44, h - 30
    coll_frac = min(have / TARGET_PER_GESTURE, 1.0)
    coll_w    = int((w - 20) * coll_frac)
    cv2.rectangle(frame, (10, bar_y0), (w - 10, bar_y1), (40, 30, 30), -1)
    coll_color = (0, 200, 80) if coll_frac >= 1.0 else (200, 150, 30)
    cv2.rectangle(frame, (10, bar_y0), (10 + coll_w, bar_y1), coll_color, -1)
    cv2.putText(frame, f"Progress: {have}/{TARGET_PER_GESTURE}",
                (14, bar_y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    # ── Recording progress bar (per frame) ───────────────────────────────────
    rec_y0, rec_y1 = h - 22, h - 8
    frame_frac = len(sequence) / SEQUENCE_LENGTH if STATE == "collecting" else 0
    frame_w    = int((w - 20) * frame_frac)
    cv2.rectangle(frame, (10, rec_y0), (w - 10, rec_y1), (40, 30, 30), -1)
    cv2.rectangle(frame, (10, rec_y0), (10 + frame_w, rec_y1), (80, 80, 255), -1)

    # ── Detection + landmark drawing ─────────────────────────────────────────
    hand_detected = result.hand_landmarks and len(result.hand_landmarks) > 0
    if hand_detected:
        lms = result.hand_landmarks[0]
        draw_landmarks_manual(frame, lms, h, w)

        if STATE == "collecting":
            flat = [coord for lm in lms for coord in (lm.x, lm.y, lm.z)]
            sequence.append(flat)

            if len(sequence) >= SEQUENCE_LENGTH:
                arr = np.array(sequence[:SEQUENCE_LENGTH], dtype=np.float32)
                idx  = next_save_index(gesture_id, label_name)
                path = os.path.join(SAVE_DIR, f"{gesture_id}_{label_name}_{idx}.npy")
                np.save(path, arr)
                sample_count[gesture_id] += 1
                new_have = sample_count[gesture_id]
                bar_str  = "[" + "#" * (new_have * 20 // TARGET_PER_GESTURE) + \
                           "." * (20 - new_have * 20 // TARGET_PER_GESTURE) + "]"
                print(f"  Saved #{idx}  {label_name}  {bar_str}  {new_have}/{TARGET_PER_GESTURE}")
                sequence = []
                STATE    = "idle"

    # ── Hand status ───────────────────────────────────────────────────────────
    dot_col = (50, 230, 100) if hand_detected else (60, 60, 200)
    cv2.circle(frame, (20, 116), 8, dot_col, -1)
    hint = "Hand detected" if hand_detected else "No hand — show your hand!"
    cv2.putText(frame, hint, (34, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    # ── Preview countdown overlay ─────────────────────────────────────────────
    if STATE == "preview":
        secs_left = max(0, preview_end - now)
        alpha     = 0.45
        overlay2  = frame.copy()
        cv2.rectangle(overlay2, (w//2 - 220, h//2 - 60), (w//2 + 220, h//2 + 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, f"Get ready: {secs_left:.1f}s",
                    (w//2 - 180, h//2 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 220, 255), 2)
        cv2.putText(frame, f"Perform: {label_name.upper().replace('_',' ')}",
                    (w//2 - 180, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # ── Idle hint ─────────────────────────────────────────────────────────────
    if STATE == "idle":
        if need == 0:
            cv2.putText(frame, "Gesture complete! Switch with [0-3]",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 80), 1)
        else:
            cv2.putText(frame, "Press SPACE to record  |  [0-3] switch gesture",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

    cv2.imshow("Data Collection — Dynamic Gestures", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" ") and STATE == "idle":
        if sample_count[gesture_id] >= TARGET_PER_GESTURE:
            print(f"  '{label_name}' already has {TARGET_PER_GESTURE} sequences. Switch gesture with [0-3].")
        else:
            STATE       = "preview"
            preview_end = now + PREVIEW_SECONDS
            print(f"\n  Preparing '{label_name}' — get into position...")
    elif chr(key).isdigit() and int(chr(key)) in DYNAMIC_LABELS:
        gesture_id = int(chr(key))
        STATE      = "idle"
        sequence   = []

cap.release()
cv2.destroyAllWindows()
detector.close()

print("\n=== Collection Summary ===")
for gid, lbl in DYNAMIC_LABELS.items():
    cnt    = sample_count[gid]
    status = "DONE" if cnt >= TARGET_PER_GESTURE else f"{cnt}/{TARGET_PER_GESTURE}"
    bar    = "[" + "#" * (cnt * 20 // TARGET_PER_GESTURE) + \
             "." * (20 - min(cnt, TARGET_PER_GESTURE) * 20 // TARGET_PER_GESTURE) + "]"
    print(f"  [{gid}] {lbl}: {bar} {status}")
