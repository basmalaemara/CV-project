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
CONFIDENCE_STATIC       = 0.60 # Lowered for easier detection
CONFIDENCE_DYNAMIC      = 0.50 # Lowered for better responsiveness
SEQUENCE_LENGTH         = 30
HISTORY_MAX             = 8
HOLD_THRESHOLD          = 2.0  # Seconds to hold (Very Deliberate)
TYPE_COOLDOWN           = 1.5  # Seconds between repeats

# ── Load Keras models ─────────────────────────────────────────────────────────
def load_model_safe(path):
    if not os.path.exists(path):
        print(f"⚠️  Not found: {path}  (skipping)")
        return None
    return tf.keras.models.load_model(path)

static_model  = load_model_safe("models/keypoint_classifier.keras")
dynamic_model  = load_model_safe("models/point_history_classifier.keras")

static_labels  = (np.load("models/static_class_labels.npy")
                  if os.path.exists("models/static_class_labels.npy")
                  else list(GESTURE_LABELS.values()))

dynamic_labels  = (np.load("models/dynamic_class_labels.npy")
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

# ── Modern Color Palette ──────────────────────────────────────────────────────
ACCENT_GREEN = (100, 255, 140)   # Soft Neon Green
ACCENT_BLUE  = (255, 180, 100)   # Cyber Blue
ACCENT_PINK  = (200, 120, 255)   # Vibrant Pink
BG_DARK      = (22, 18, 28)      # Deep Navy/Purple
PANEL_COLOR  = (40, 35, 50)      # Glass Panel
TEXT_WHITE   = (245, 245, 255)
GOLD_SHIMMER = (100, 220, 255)

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

def draw_bar(frame, x, y, bw, conf, color):
    cv2.rectangle(frame, (x, y), (x + bw, y + 10), (40, 40, 60), -1)
    cv2.rectangle(frame, (x, y), (x + int(bw * conf), y + 10), color, -1)

def blend_rect(frame, p1, p2, col, alpha=0.5):
    """Solid rect for ultra-performance on all machines"""
    cv2.rectangle(frame, p1, p2, col, -1)

def draw_ui(frame, s_label, s_conf, d_label, d_conf, sentence_built, history, fps, hand_ok, current_mode, elapsed, conflict_pending=None):
    h, w = frame.shape[:2]

    # Pre-calculate UI values
    hold_p = min(1.0, elapsed / HOLD_THRESHOLD) if s_label != "-" else 0
    hold_col = ACCENT_GREEN if hold_p >= 1.0 else (255, 120, 100)

    # 1. LUXURY HEADER (Frosted Glass Effect)
    blend_rect(frame, (8, 8), (w - 8, 115), BG_DARK, alpha=0.7)
    cv2.rectangle(frame, (8, 8), (w - 8, 115), (70, 60, 90), 1)

    # Status Badge
    sh = " ACTIVE" if hand_ok else " SEARCHING..."
    sc = ACCENT_GREEN if hand_ok else (100, 100, 255)
    cv2.circle(frame, (35, 38), 7, sc, -1)
    cv2.putText(frame, f"SYSTEM {sh}", (55, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_WHITE, 1)

    # FPS with mini-glow
    cv2.putText(frame, f"CORE SPEED: {fps:.0f}", (w - 180, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 180), 1)

    # --- CENTER: MAIN DETECTOR (Static) ---
    cp = w // 2
    cv2.putText(frame, "PRIMARY GESTURE", (cp - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 140, 180), 1)
    
    label_txt = s_label.replace("_", " ").upper() if s_label != "-" else "---"
    (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
    cv2.putText(frame, label_txt, (cp - tw // 2, 95), cv2.FONT_HERSHEY_DUPLEX, 1.2, ACCENT_GREEN, 3)
    
    # Progress Ring/Bar for Hold
    draw_bar(frame, cp - 120, 105, 240, hold_p, hold_col)

    # --- SIDES: INTELLIGENCE PANELS ---
    # Left Side: Stat
    cv2.putText(frame, "INTENT INFO", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 130, 150), 1)
    cv2.putText(frame, f"CONF: {s_conf:.0%}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT_GREEN, 1)

    # Right Side: Dynamic
    if d_label and d_label != "-":
        cv2.putText(frame, "COMMAND DETECTED", (w - 230, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ACCENT_BLUE, 1)
        d_disp = d_label.replace("_", " ").upper()
        cv2.putText(frame, d_disp, (w - 230, 100), cv2.FONT_HERSHEY_DUPLEX, 0.6, ACCENT_BLUE, 2)

    # 2. SENTENCE DASHBOARD (Bottom)
    blend_rect(frame, (8, h - 100), (w - 8, h - 8), (35, 30, 45), alpha=0.9)
    cv2.rectangle(frame, (8, h - 100), (w - 8, h - 8), (100, 90, 120), 1)

    # Mode Pill
    m_col = ACCENT_PINK if current_mode == "ALL" else ACCENT_BLUE
    blend_rect(frame, (25, h - 90), (120, h - 65), m_col, alpha=0.9)
    cv2.putText(frame, current_mode, (38, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BG_DARK, 2)

    # Main Sentence
    display_sent = sentence_built if sentence_built else "Ready to communicate..."
    sent_col = TEXT_WHITE if sentence_built else (120, 110, 140)
    cv2.putText(frame, display_sent, (25, h - 25), cv2.FONT_HERSHEY_DUPLEX, 1.3, sent_col, 2)

    # 3. CONFLICT OVERLAY
    if conflict_pending is not None:
        blend_rect(frame, (0, 0), (w, h), (10, 5, 20), alpha=0.7) # Darken screen
        blend_rect(frame, (w//2 - 260, h//2 - 90), (w//2 + 260, h//2 + 90), BG_DARK, alpha=0.95)
        cv2.rectangle(frame, (w//2 - 260, h//2 - 90), (w//2 + 260, h//2 + 90), ACCENT_PINK, 2)
        
        cv2.putText(frame, "RESOLUTION REQUIRED", (w//2 - 130, h//2 - 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, ACCENT_PINK, 2)
        
        if len(conflict_pending) == 3:
            msg = "Pick: 5, C, or SPACE?"
            cv2.putText(frame, msg, (w//2 - 210, h//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, TEXT_WHITE, 3)
            cv2.putText(frame, "Press '5' for 5, 'c' for C, or [Space] for Space.", (w//2 - 210, h//2 + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        else:
            msg = f"Pick: {conflict_pending[0].upper()} or {conflict_pending[1].upper()}?"
            cv2.putText(frame, msg, (w//2 - 190, h//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, TEXT_WHITE, 3)
            cv2.putText(frame, "Press the corresponding key now.", (w//2 - 160, h//2 + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # 4. RECENT TOASTS (Sidebar)
    if history:
        for i, hl in enumerate(list(history)[:5]):
            y_pos = 140 + i * 45
            blend_rect(frame, (w - 180, y_pos), (w - 15, y_pos + 35), PANEL_COLOR, alpha=0.9)
            cv2.putText(frame, str(hl).upper(), (w - 165, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_WHITE, 1)

    return frame

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_buffer    = deque(maxlen=SEQUENCE_LENGTH)
gesture_history = deque(maxlen=HISTORY_MAX)

sentence_built = ""
last_raw = ""
gesture_start_time = time.time()
last_type_time = 0
s_label, s_conf = "-", 0.0
d_label, d_conf = "-", 0.0
prev_time = time.time()
current_mode = "ALL"  # Options: "ALL", "LETTERS", "NUMBERS"
conflict_pending = None
hand_lost_time = 0
dynamic_cooldown = 0
frame_count = 0
wrist_history = deque(maxlen=5) # 5 frames for extreme speed

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
        hand_lost_time = time.time()
        lms = result.hand_landmarks[0]
        draw_hand(frame, lms, h, w)

        pts  = normalize_landmarks(lms)
        flat = landmarks_to_flat(pts)

        # --- MOTION-LOCK ENGINE ---
        wrist_history.append((lms[0].x, lms[0].y))
        
        # Calculate instant velocity to block static errors
        moving_fast = False
        if len(wrist_history) >= 2:
            vx = abs(wrist_history[-1][0] - wrist_history[-2][0])
            vy = abs(wrist_history[-1][1] - wrist_history[-2][1])
            if vx > 0.04 or vy > 0.04: # Hand is in flight
                moving_fast = True

        if len(wrist_history) == 5 and time.time() > dynamic_cooldown:
            dx = wrist_history[-1][0] - wrist_history[0][0]
            dy = wrist_history[-1][1] - wrist_history[0][1]
            
            # TRIGGER ACTIONS
            if abs(dx) > 0.15 and abs(dy) < 0.12: # Horizontal
                dynamic_cooldown = time.time() + 0.6
                if dx < 0: # Left
                    sentence_built = sentence_built[:-1]
                    d_label, d_conf = "BACKSPACE", 1.0
                    gesture_history.appendleft("BACKSPACE")
                else:      # Right
                    sentence_built += " "
                    d_label, d_conf = "SPACE", 1.0
                    gesture_history.appendleft("SPACE")
                wrist_history.clear()
                moving_fast = True
            elif abs(dy) > 0.15 and abs(dx) < 0.12: # Vertical
                dynamic_cooldown = time.time() + 0.6
                if dy > 0: # Down
                    sentence_built = ""
                    d_label, d_conf = "CLEAR ALL", 1.0
                    gesture_history.appendleft("CLEAR ALL")
                else:      # Up (Extra shortcut for Space)
                    sentence_built += " "
                    d_label, d_conf = "SPACE", 1.0
                    gesture_history.appendleft("SPACE")
                wrist_history.clear()
                moving_fast = True

        # --- STATIC CLASSIFIER (Only if not moving) ---
        if moving_fast:
            s_label, s_conf = "MOVING...", 0.0
            gesture_start_time = time.time() # Reset timer while moving
        elif static_model is not None:
            features = extract_advanced_features(flat)
            features_tnsr = tf.convert_to_tensor([features], dtype=tf.float32)
            probs = static_model(features_tnsr, training=False)[0].numpy()
            
            # --- MODE FILTERING ---
            filtered_probs = probs.copy()
            
            for i, class_name in enumerate(static_labels):
                lbl_lower = str(class_name).lower()
                is_num = lbl_lower.isdigit()
                is_letter = (len(lbl_lower) == 1 and lbl_lower.isalpha())
                is_control = lbl_lower in ["thumbs_down", "fist", "open_hand"]
                
                if current_mode == "NUMBERS":
                    allowed = is_num or is_control or (lbl_lower in CONFLICT_MAP and CONFLICT_MAP[lbl_lower].isdigit())
                    if not allowed:
                        filtered_probs[i] = 0.0
                elif current_mode == "LETTERS":
                    allowed = is_letter or is_control or (lbl_lower in CONFLICT_MAP and CONFLICT_MAP[lbl_lower].isalpha())
                    if not allowed:
                        filtered_probs[i] = 0.0

            best_idx = np.argmax(filtered_probs)
            raw_lbl_tentative = str(static_labels[best_idx]).lower()
            
            s_conf = float(probs[best_idx])
            if raw_lbl_tentative in CONFLICT_MAP:
                try:
                    other_lbl = CONFLICT_MAP[raw_lbl_tentative]
                    other_idx = list(static_labels).index(other_lbl)
                    s_conf += float(probs[other_idx])
                except ValueError:
                    pass

            # --- HAND PHYSICS ENFORCER (Stop C vs 5 Confusion) ---
            # Calculate spread between Index Tip (8) and Pinky Tip (20)
            p_index = lms[8]
            p_pinky = lms[20]
            spread = np.sqrt((p_index.x - p_pinky.x)**2 + (p_index.y - p_pinky.y)**2)
            
            # If spread is large, it MUST be 5 or Space, NOT a curved C
            is_spread_open = spread > 0.16 # Sharpened threshold (was 0.14)
            
            if is_spread_open and raw_lbl_tentative.lower() == "c":
                raw_lbl = "5" # Force to 5 if hand is open
                lbl_lower = "5"
            elif not is_spread_open and raw_lbl_tentative.lower() in ["5", "open hand"]:
                raw_lbl = "C" # Force to C if hand is curved
                lbl_lower = "c"
            else:
                raw_lbl = str(static_labels[best_idx]).replace("_", " ")
                lbl_lower = raw_lbl.lower()
                
                if current_mode == "LETTERS" and lbl_lower in CONFLICT_MAP and lbl_lower.isdigit():
                    raw_lbl = CONFLICT_MAP[lbl_lower].upper()
                    lbl_lower = raw_lbl.lower()
                elif current_mode == "NUMBERS" and lbl_lower in CONFLICT_MAP and lbl_lower.isalpha():
                    raw_lbl = CONFLICT_MAP[lbl_lower].upper()
                    lbl_lower = raw_lbl.lower()

                s_label = raw_lbl
                
                if raw_lbl != last_raw:
                    gesture_start_time = time.time()
                    last_raw = raw_lbl
                
                elapsed = time.time() - gesture_start_time
                
                if elapsed >= HOLD_THRESHOLD and (time.time() - last_type_time) >= TYPE_COOLDOWN and conflict_pending is None and time.time() > dynamic_cooldown:
                    lbl_lower = raw_lbl.lower()
                    last_type_time = time.time()
                    
                    # Removed 5/C/Space conflict popup to keep them separate
                    if current_mode == "ALL" and lbl_lower in CONFLICT_MAP:
                        conflict_pending = (lbl_lower, CONFLICT_MAP[lbl_lower])
                    else:
                        if raw_lbl in ["5", "open hand"]:
                            sentence_built += "5"
                        elif raw_lbl == "c":
                            sentence_built += "c"
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

    else:
        # No hand detected
        s_label, s_conf = "-", 0.0
        gesture_start_time = time.time()
        d_label, d_conf = "-", 0.0
        # Only clear the buffer if hand is gone for more than 0.5 seconds
        if time.time() - hand_lost_time > 0.5:
            frame_buffer.clear()

    now       = time.time()
    fps       = 1.0 / (now - prev_time + 1e-9)
    prev_time = now

    elapsed_val = time.time() - gesture_start_time if hand_ok and s_label != "-" else 0
    draw_ui(frame, s_label, s_conf, d_label, d_conf, sentence_built, gesture_history, fps, hand_ok, current_mode, elapsed_val, conflict_pending)
    cv2.imshow("Hand Gesture Recognition", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if conflict_pending is not None:
        if key != 255: # A key was pressed
            if conflict_pending == ('5', 'c', 'space'):
                if key == ord('5'):
                    sentence_built += "5"
                    gesture_history.appendleft("5")
                    conflict_pending = None
                elif key == ord('c'):
                    sentence_built += "C"
                    gesture_history.appendleft("C")
                    conflict_pending = None
                elif key == ord(' '):
                    sentence_built += " "
                    gesture_history.appendleft("SPACE")
                    conflict_pending = None
            else:
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
