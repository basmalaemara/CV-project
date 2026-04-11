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
import string
import time
import os
import sys

sys.path.insert(0, ".")
from preprocessing.feature_extractor import (
    GESTURE_LABELS,
    DYNAMIC_LABELS,
    normalize_landmarks,
    landmarks_to_flat,
    extract_advanced_features,
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
static_model  = load_model_safe("models/keypoint_classifier.keras")

static_labels  = (np.load("models/static_class_labels.npy")
                  if os.path.exists("models/static_class_labels.npy")
                  else list(GESTURE_LABELS.values()))
static_labels  = (np.load("models/static_class_labels.npy")
                  if os.path.exists("models/static_class_labels.npy")
                  else list(GESTURE_LABELS.values()))

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

def draw_ui(frame, s_label, s_conf, sentence_built, history, fps, hand_ok, current_mode, conflict_pending=None):
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

    # Static prediction row
    cv2.putText(frame, "CURRENT DETECT:", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 200, 160), 1)
    label_display = s_label.replace("_", " ") if s_label != "-" else "-"
    cv2.putText(frame, label_display.upper(), (145, 58),
                cv2.FONT_HERSHEY_DUPLEX, 0.82, ACCENT_GREEN, 2)
    draw_bar(frame, 12, 64, 260, s_conf, ACCENT_GREEN)
    cv2.putText(frame, f"{s_conf:.0%}", (278, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, ACCENT_GREEN, 1)

    # Conflict Popup
    if conflict_pending is not None:
        blend_rect(frame, (w//2 - 250, h//2 - 100), (w//2 + 250, h//2 + 80), (30, 20, 120), alpha=0.95)
        cv2.rectangle(frame, (w//2 - 250, h//2 - 100), (w//2 + 250, h//2 + 80), (100, 100, 255), 2)
        cv2.putText(frame, "AMBIGUOUS SIGN DETECTED!", (w//2 - 180, h//2 - 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 180, 255), 2)
        cv2.putText(frame, f"Did you mean '{conflict_pending[0].upper()}' or '{conflict_pending[1].upper()}'?",
                    (w//2 - 220, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Press the '{conflict_pending[0].upper()}' or '{conflict_pending[1].upper()}' key to confirm.",
                    (w//2 - 230, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

    # Sentence Box at bottom
    blend_rect(frame, (0, h - 80), (w, h), (30, 20, 40), alpha=0.95)
    
    # Mode indicator
    mode_text = f"MODE: [{current_mode}]   (Press '1': Letters  '2': Numbers  '3': All)"
    cv2.putText(frame, mode_text, (15, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)

    cv2.putText(frame, "SENTENCE:", (15, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 200), 1)
    cv2.putText(frame, sentence_built, (15, h - 5),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 230, 120), 2)


    # History sidebar
    if history:
        ph = 24 + len(history) * 26 + 8
        blend_rect(frame, (w - 215, 50), (w - 6, 50 + ph), (20, 20, 36), alpha=0.85)
        cv2.putText(frame, "Recent:", (w - 208, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 220), 1)
        for i, hl in enumerate(list(history)):
            fade = max(100, 255 - i * 25)
            col  = (int(200*fade/255), int(200*fade/255), int(255*fade/255))
            cv2.putText(frame, f"> {hl}", (w - 208, 92 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1)

    # Controls hint
    cv2.putText(frame, "Hold sign = Type | '5' = Space | 'Thumbs Down' = Backspace | 'Fist' = Clear", (10, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)

    return frame

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_buffer    = deque(maxlen=SEQUENCE_LENGTH)
gesture_history = deque(maxlen=HISTORY_MAX)

sentence_built = ""
frames_held = 0
last_raw = ""
s_label, s_conf = "-", 0.0
prev_time = time.time()
current_mode = "ALL"  # Options: "ALL", "LETTERS", "NUMBERS"
conflict_pending = None

CONFLICT_MAP = {
    "1": "d", "d": "1",
    "0": "o", "o": "0",
    "2": "v", "v": "2",
    "6": "w", "w": "6",
    "9": "f", "f": "9"
}

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

        # Word Builder Logic
        if static_model is not None:
            features = extract_advanced_features(flat)
            features_tnsr = tf.convert_to_tensor([features], dtype=tf.float32)
            probs = static_model(features_tnsr, training=False)[0].numpy()
            
            # --- MODE FILTERING ---
            # Automatically zero-out the confidence of elements we want to ignore
            for i, class_name in enumerate(static_labels):
                lbl_lower = str(class_name).lower()
                if current_mode == "LETTERS" and lbl_lower in "0123456789":
                    probs[i] = 0.0
                elif current_mode == "NUMBERS" and lbl_lower in string.ascii_lowercase and len(lbl_lower) == 1:
                    probs[i] = 0.0

            if probs.max() >= CONFIDENCE_STATIC:
                raw_lbl = str(static_labels[np.argmax(probs)]).replace("_", " ")
                s_conf  = float(probs.max())
                s_label = raw_lbl
                
                if raw_lbl == last_raw:
                    frames_held += 1
                else:
                    frames_held = 0
                    last_raw = raw_lbl
                
                # If held steady for 6 frames (~0.4 seconds), register the character!
                if frames_held == 6 and conflict_pending is None:
                    lbl_lower = raw_lbl.lower()
                    
                    if current_mode == "ALL" and lbl_lower in CONFLICT_MAP:
                        # Enter conflict resolution state
                        conflict_pending = (lbl_lower, CONFLICT_MAP[lbl_lower])
                        frames_held = -30  # Prevent re-triggering while resolving
                    else:
                        if raw_lbl in ["open hand", "open hand / 5", "5"]:
                            sentence_built += " "
                        elif raw_lbl == "thumbs down":
                            sentence_built = sentence_built[:-1]
                        elif raw_lbl in ["fist", "fist / a / s"]:
                            sentence_built = ""
                        else:
                            letter = raw_lbl.split(" / ")[0].lower()
                            if letter == "thumbs up":
                                sentence_built += "👍" 
                            else:
                                sentence_built += letter
                        
                        gesture_history.appendleft(raw_lbl)
                        frames_held = -5
            else:
                s_label, s_conf = "-", 0.0
                frames_held = 0
    else:
        s_label, s_conf = "-", 0.0
        frames_held = 0

    now       = time.time()
    fps       = 1.0 / (now - prev_time + 1e-9)
    prev_time = now

    draw_ui(frame, s_label, s_conf, sentence_built, gesture_history, fps, hand_ok, current_mode, conflict_pending)
    cv2.imshow("Hand Gesture Recognition", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if conflict_pending is not None:
        if key != 255: # A key was pressed
            c1, c2 = conflict_pending
            key_char = chr(key).lower()
            if key_char == c1:
                sentence_built += c1
                conflict_pending = None
            elif key_char == c2:
                sentence_built += c2
                conflict_pending = None
    else:
        if key == ord("q"):
            break
        elif key == ord("1"):
            current_mode = "LETTERS"
        elif key == ord("2"):
            current_mode = "NUMBERS"
        elif key == ord("3"):
            current_mode = "ALL"

cap.release()
cv2.destroyAllWindows()
detector.close()
print("Session ended.")
