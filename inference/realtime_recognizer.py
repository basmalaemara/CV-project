"""
realtime_recognizer.py  (MediaPipe Tasks API — v0.10+)
────────────────────────────────────────────────────────────────
Real-time hand gesture recognition — live webcam demo.

Requires:
  models/hand_landmarker.task
  models/keypoint_classifier.keras
  models/point_history_classifier.keras   (optional)

Press Q to quit.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import tensorflow as tf
from collections import deque
import time
import os
import sys

sys.path.insert(0, ".")
from preprocessing.feature_extractor import (
    GESTURE_LABELS,
    DYNAMIC_LABELS,
    normalize_landmarks,
    landmarks_to_flat,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH              = "models/hand_landmarker.task"
CONFIDENCE_STATIC       = 0.82
CONFIDENCE_DYNAMIC      = 0.85
SEQUENCE_LENGTH         = 30
HISTORY_MAX             = 8

# ── Load Keras models ─────────────────────────────────────────────────────────
def load_model_safe(path):
    if not os.path.exists(path):
        print(f"⚠️  Not found: {path}  (skipping)")
        return None
    return tf.keras.models.load_model(path)

static_model  = load_model_safe("models/keypoint_classifier.keras")
dynamic_model = load_model_safe("models/point_history_classifier.keras")

static_labels  = (np.load("models/static_class_labels.npy")
                  if os.path.exists("models/static_class_labels.npy")
                  else list(GESTURE_LABELS.values()))
dynamic_labels = (np.load("models/dynamic_class_labels.npy")
                  if os.path.exists("models/dynamic_class_labels.npy")
                  else list(DYNAMIC_LABELS.values()))

# ── MediaPipe Tasks ───────────────────────────────────────────────────────────
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

# ── Drawing ───────────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

ACCENT_GREEN  = (80, 230, 130)
ACCENT_BLUE   = (255, 160, 60)

def draw_hand(frame, landmarks, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 200, 120), 2)
    for i, (x, y) in enumerate(pts):
        r = 6 if i == 0 else 4
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), r, (0, 160, 80), 2)

def draw_bar(img, x, y, bw, conf, color):
    cv2.rectangle(img, (x, y), (x + bw, y + 10), (40, 40, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(bw * conf), y + 10), color, -1)

def blend_rect(frame, pt1, pt2, color, alpha=0.75):
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_ui(frame, s_label, s_conf, d_label, d_conf, history, fps, hand_ok):
    h, w = frame.shape[:2]

    # Top panel
    blend_rect(frame, (0, 0), (w, 110), (16, 16, 28), alpha=0.82)

    # FPS badge
    fps_col = ACCENT_GREEN if fps > 24 else (40, 40, 220)
    blend_rect(frame, (w - 112, 8), (w - 8, 44), fps_col, alpha=0.9)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 107, 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.70, (10, 10, 20), 2)

    # Hand dot
    cv2.circle(frame, (20, 20), 8, ACCENT_GREEN if hand_ok else (60, 60, 180), -1)
    cv2.putText(frame, "Hand detected" if hand_ok else "No hand",
                (34, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    # Static row
    cv2.putText(frame, "STATIC:", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 200, 160), 1)
    label_display = s_label.replace("_", " ") if s_label != "—" else "—"
    cv2.putText(frame, label_display, (85, 58),
                cv2.FONT_HERSHEY_DUPLEX, 0.82, ACCENT_GREEN, 2)
    draw_bar(frame, 12, 64, 260, s_conf, ACCENT_GREEN)
    cv2.putText(frame, f"{s_conf:.0%}", (278, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, ACCENT_GREEN, 1)

    # Dynamic row
    cv2.putText(frame, "DYNAMIC:", (12, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 160, 220), 1)
    dlabel_display = d_label.replace("_", " ") if d_label != "—" else "—"
    cv2.putText(frame, dlabel_display, (95, 94),
                cv2.FONT_HERSHEY_DUPLEX, 0.74, ACCENT_BLUE, 2)
    draw_bar(frame, 12, 98, 260, d_conf, ACCENT_BLUE)
    cv2.putText(frame, f"{d_conf:.0%}", (278, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, ACCENT_BLUE, 1)

    # History sidebar
    if history:
        ph = 24 + len(history) * 26 + 8
        blend_rect(frame, (w - 215, 50), (w - 6, 50 + ph), (20, 20, 36), alpha=0.85)
        cv2.putText(frame, "Recent:", (w - 208, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 220), 1)
        for i, hl in enumerate(list(history)):
            fade = max(100, 255 - i * 25)
            col  = (int(200*fade/255), int(200*fade/255), int(255*fade/255))
            cv2.putText(frame, f"• {hl}", (w - 208, 92 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1)

    # Bottom hint
    cv2.putText(frame, "Q — quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 80, 100), 1)

    return frame

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_buffer    = deque(maxlen=SEQUENCE_LENGTH)
gesture_history = deque(maxlen=HISTORY_MAX)

s_label, s_conf = "—", 0.0
d_label, d_conf = "—", 0.0
prev_time = time.time()

print("\n✅  Recognition started. Press Q to quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result    = detector.detect(mp_image)

    hand_ok = bool(result.hand_landmarks)

    if hand_ok:
        lms = result.hand_landmarks[0]
        draw_hand(frame, lms, h, w)

        pts  = normalize_landmarks(lms)
        flat = landmarks_to_flat(pts)

        # Static
        if static_model is not None:
            probs = static_model.predict(
                np.array([flat], dtype=np.float32), verbose=0
            )[0]
            if probs.max() >= CONFIDENCE_STATIC:
                new_lbl = str(static_labels[np.argmax(probs)]).replace("_", " ")
                if new_lbl != s_label:
                    gesture_history.appendleft(new_lbl)
                s_label = new_lbl
                s_conf  = float(probs.max())
            else:
                s_label, s_conf = "—", 0.0

        # Dynamic
        frame_buffer.append(flat)
        if dynamic_model is not None and len(frame_buffer) == SEQUENCE_LENGTH:
            seq    = np.array([list(frame_buffer)], dtype=np.float32)
            dprobs = dynamic_model.predict(seq, verbose=0)[0]
            if dprobs.max() >= CONFIDENCE_DYNAMIC:
                new_dyn = str(dynamic_labels[np.argmax(dprobs)]).replace("_", " ")
                if new_dyn != d_label:
                    gesture_history.appendleft(f"[dyn] {new_dyn}")
                d_label = new_dyn
                d_conf  = float(dprobs.max())
            else:
                d_label, d_conf = "—", 0.0
    else:
        s_label, s_conf = "—", 0.0
        d_label, d_conf = "—", 0.0

    now       = time.time()
    fps       = 1.0 / (now - prev_time + 1e-9)
    prev_time = now

    draw_ui(frame, s_label, s_conf, d_label, d_conf, gesture_history, fps, hand_ok)
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("Session ended.")
