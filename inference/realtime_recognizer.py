"""
realtime_recognizer.py  (MediaPipe Tasks API — v0.10+)
────────────────────────────────────────────────────────────────
Real-time hand gesture recognition — Luxury UI Edition.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info and warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress oneDNN custom operations info

import logging
logging.getLogger('absl').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import mediapipe as mp
import PIL.Image as PImage, PIL.ImageDraw as PDraw, PIL.ImageFont as PFont

# Load Windows Native Emoji Font System
try:
    EMOJI_FONT = PFont.truetype("seguiemj.ttf", 52)
except:
    EMOJI_FONT = PFont.load_default()
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import tensorflow as tf
from collections import deque
import time
import math
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
CONFIDENCE_STATIC       = 0.60 
CONFIDENCE_DYNAMIC      = 0.50 
SEQUENCE_LENGTH         = 30
HISTORY_MAX             = 8
HOLD_THRESHOLD          = 1.2  
TYPE_COOLDOWN           = 0.8  

# ── Load Models ───────────────────────────────────────────────────────────────
def load_model_safe(path):
    if not os.path.exists(path): return None
    return tf.keras.models.load_model(path)

static_model  = load_model_safe("models/keypoint_classifier.keras")
dynamic_model  = load_model_safe("models/point_history_classifier.keras")
static_labels = (np.load("models/static_class_labels.npy") if os.path.exists("models/static_class_labels.npy") else list(GESTURE_LABELS.values()))

# ── MediaPipe ─────────────────────────────────────────────────────────────────
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ── Aesthetics ────────────────────────────────────────────────────────────────
ACCENT_GREEN = (100, 255, 140); ACCENT_BLUE = (255, 180, 100); ACCENT_PINK = (200, 120, 255)
BG_DARK = (22, 18, 28); TEXT_WHITE = (245, 245, 255)

def blend_rect(frame, p1, p2, col, alpha=0.5):
    overlay = frame.copy()
    cv2.rectangle(overlay, p1, p2, col, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_bar(frame, x, y, bw, conf, color):
    cv2.rectangle(frame, (x, y), (x + bw, y + 10), (40, 40, 60), -1)
    cv2.rectangle(frame, (x, y), (x + int(bw * conf), y + 10), color, -1)

def draw_hand(frame, landmarks, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for (a, b) in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]:
        cv2.line(frame, pts[a], pts[b], (80, 200, 120), 2)
    for i, (x, y) in enumerate(pts): cv2.circle(frame, (x, y), 5 if i==0 else 3, (255, 255, 255), -1)

def dist(p1, p2): return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# ── Logic Helpers ─────────────────────────────────────────────────────────────
def get_physics_gesture(lms, mode="ALL"):
    if not lms: return None
    base = lms[0]; palm = dist(base, lms[9])
    
    # Robust Extension Check
    t = dist(lms[4], lms[17]) > palm * 0.8 or dist(lms[4], lms[5]) > palm * 0.5
    i = dist(lms[8], base) > dist(lms[6], base) * 1.1
    m = dist(lms[12], base) > dist(lms[10], base) * 1.1
    r = dist(lms[16], base) > dist(lms[14], base) * 1.1
    p = dist(lms[20], base) > dist(lms[18], base) * 1.1
    
    hand_upright = lms[0].y > lms[9].y - (palm * 0.4)

    # Physics Distinguish Two-Finger Gestures (U, V, K, 2, 3)
    if hand_upright and i and m and not r and not p:
        if lms[4].y < lms[9].y and dist(lms[4], lms[10]) < palm * 0.6:
            if mode in ["LETTERS", "ALL"]: return "k"
        elif t:
            if mode in ["NUMBERS", "ALL"]: return "3"
        else:
            if dist(lms[8], lms[12]) > palm * 0.35:
                if mode in ["NUMBERS", "ALL"]: return "2"
                elif mode == "LETTERS": return "v"
            else:
                if mode in ["LETTERS", "ALL"]: return "u"

    # Physics Distinguish One-Finger (1, D, Z)
    if hand_upright and i and not m and not r and not p and not t:
        if mode == "ALL": return "1"
        elif mode == "NUMBERS": return "1"
        elif mode == "LETTERS": return "d"

    # Priority Number Overrides
    if mode in ["NUMBERS", "ALL"] and hand_upright:
        if i and m and r and p and t: return "5"
        if i and m and r and p and not t: return "4"
        
        # 6-9 (ASL Pinched - Other fingers MUST be extended!)
        jt = palm * 0.30
        if dist(lms[4], lms[8]) < jt and m and r and p: return "9"
        if dist(lms[4], lms[12]) < jt and i and r and p: return "8"
        if dist(lms[4], lms[16]) < jt and i and m and p: return "7"
        if dist(lms[4], lms[20]) < jt and i and m and r: return "6"

    # Phrases (Special Physics-based)
    if i and p and not m and not r and t and lms[8].y < lms[6].y: return "I LOVE YOU!"
    def is_curled(tip, pip, base): return dist(tip, base) < dist(pip, base)
    if is_curled(lms[8],lms[6],base) and is_curled(lms[12],lms[10],base) and is_curled(lms[16],lms[14],base) and is_curled(lms[20],lms[18],base):
        if lms[4].y < lms[5].y and dist(lms[4],lms[8]) > palm*0.5: return "THUMBS UP!"
    if t and p and not i and not m and not r: return "CALL ME!"
    return None

def get_multi_hand_gesture(lms_list):
    if len(lms_list) < 2: return None
    h1, h2 = lms_list[0], lms_list[1]
    if dist(h1[4], h2[4]) < 0.1 and dist(h1[8], h2[8]) < 0.1: return "HEART!"
    if dist(h1[0], h2[0]) < 0.15: return "APPLAUSE!"
    return None

def draw_onboarding_ui(frame, step, hello_triggered, hello_time, anim_timer, active_zone=None, zone_timer=0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,h), (15, 10, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    cw, ch = 800, 400
    cx, cy = w//2, h//2
    # Card background
    cv2.rectangle(frame, (cx-cw//2, cy-ch//2), (cx+cw//2, cy+ch//2), (30, 25, 40), -1)
    cv2.rectangle(frame, (cx-cw//2, cy-ch//2), (cx+cw//2, cy+ch//2), ACCENT_PINK, 2)
    
    if step == 4:
        dt = time.time() - anim_timer
        scale = 1.0 + dt*1.5
        cv2.putText(frame, "DONE! LET'S GO!", (cx-380, cy+50), cv2.FONT_HERSHEY_DUPLEX, scale, ACCENT_GREEN, 5)
        return frame

    title = f"STEP {step+1}/4: "
    text1, text2 = "", ""
    
    if step == 0:
        title += "WELCOME"
        text1 = "Show an Open Hand (5) to say Hello!"
        if hello_triggered:
            dt = time.time() - hello_time
            scale = 2.0 + 0.3 * math.sin(dt * 15)
            r = int(120 + 130 * math.sin(dt*8))
            g = 255
            b = int(150 + 100 * math.cos(dt*8))
            cv2.putText(frame, "HELLOOO!", (cx-140, cy+50), cv2.FONT_HERSHEY_DUPLEX, scale, (b,g,r), 4)
        else:
            cv2.putText(frame, "[WAITING FOR GESTURE...]", (cx-200, cy+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140,140,160), 2)
    elif step == 1:
        title += "CONTROLS"
        text1 = "Swipe LEFT (move hand fast right-to-left) to Backspace."
        text2 = "Swipe RIGHT to Space. Swipe DOWN to Clear."
    elif step == 2:
        title += "MODES"
        text1 = "Press 1 (Letters) or 2 (Numbers) to switch modes."
        text2 = "Press 3 for ALL mode. Identical signs pop up a menu!"
    elif step == 3:
        title += "ALL SET!"
        text1 = "You are ready to use the system."
        text2 = "Press ESC at any time to safely Exit the app."
        
    cv2.putText(frame, title, (cx-cw//2+40, cy-ch//2+60), cv2.FONT_HERSHEY_DUPLEX, 1.2, ACCENT_PINK, 2)
    cv2.putText(frame, text1, (cx-cw//2+40, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_WHITE, 2)
    if text2: cv2.putText(frame, text2, (cx-cw//2+40, cy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,220), 2)
        
    # ProgressBar
    bw = 600
    cv2.rectangle(frame, (cx-bw//2, cy+ch//2-40), (cx+bw//2, cy+ch//2-30), (50,50,70), -1)
    progress_w = int(bw*(step/3)) if step <= 3 else bw
    if progress_w > 0:
        cv2.rectangle(frame, (cx-bw//2, cy+ch//2-40), (cx-bw//2 + progress_w, cy+ch//2-30), ACCENT_GREEN, -1)
        
    # Hover Next Button
    if step > 0 and step < 4:
        nx, ny = cx + cw//2 - 180, cy + ch//2 - 90
        is_hover = (active_zone == "NEXT_ONBOARD")
        col = ACCENT_GREEN if is_hover else (80,70,90)
        cv2.rectangle(frame, (nx, ny), (nx+150, ny+50), col, 3)
        cv2.putText(frame, "NEXT ->", (nx+25, ny+35), cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT_WHITE, 2)
        if is_hover: draw_bar(frame, nx+10, ny+50, 130, min(1.0, (time.time()-zone_timer)/0.8), ACCENT_GREEN)
        
    return frame

def draw_ui(frame, s_label, s_conf, d_label, d_conf, sentence_built, history, fps, hand_ok, current_mode, elapsed, conflict_pending=None, is_paused=False, pending_system_control=False, active_zone=None, zone_timer=0):
    h, w = frame.shape[:2]; hold_p = min(1.0, elapsed / HOLD_THRESHOLD) if s_label not in ["-", "MOVING...", "SELECT OPTION...", "PAUSED"] else 0
    hold_col = ACCENT_GREEN if hold_p >= 1.0 else (255, 120, 100)
    blend_rect(frame, (8, 8), (w - 8, 115), BG_DARK, alpha=0.8); cv2.rectangle(frame, (8, 8), (w - 8, 115), (70, 60, 90), 1)
    sc = ACCENT_GREEN if hand_ok else (100, 100, 255); cv2.circle(frame, (35, 38), 7, sc, -1)
    cv2.putText(frame, f"SYSTEM {'ACTIVE' if hand_ok else 'SEARCHING...'}", (55, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_WHITE, 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 180), 1)
    cp = w // 2; cv2.putText(frame, "PRIMARY GESTURE", (cp - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 140, 180), 1)
    disp_lbl = s_label.replace("_", " ").upper() if s_label != "-" else "---"
    (tw, th), _ = cv2.getTextSize(disp_lbl, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
    cv2.putText(frame, disp_lbl, (cp - tw // 2, 95), cv2.FONT_HERSHEY_DUPLEX, 1.2, ACCENT_GREEN if s_conf > 0.15 else (140,140,140), 3); draw_bar(frame, cp - 120, 105, 240, hold_p, hold_col)
    cv2.putText(frame, "INTENT INFO", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 130, 150), 1)
    cv2.putText(frame, f"CONF: {s_conf:.0%}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT_GREEN, 1)
    if d_label and d_label != "-":
        cv2.putText(frame, "COMMAND DETECTED", (w - 230, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ACCENT_BLUE, 1)
        cv2.putText(frame, d_label.upper(), (w - 230, 100), cv2.FONT_HERSHEY_DUPLEX, 0.6, ACCENT_BLUE, 2)
    blend_rect(frame, (8, h - 100), (w - 8, h - 8), (35, 30, 45), alpha=0.9); cv2.rectangle(frame, (8, h - 100), (w - 8, h - 8), (100, 90, 120), 1)
    m_col = ACCENT_PINK if current_mode == "ALL" else ACCENT_BLUE; blend_rect(frame, (25, h - 90), (120, h - 65), m_col, alpha=0.9)
    cv2.putText(frame, current_mode, (38, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20,20,20), 2)
    img_pil = PImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = PDraw.Draw(img_pil)
    draw.text((25, h - 85), (sentence_built if sentence_built else "Ready..."), font=EMOJI_FONT, fill=(255, 245, 245))
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    bar_y = h - 130; blend_rect(frame, (8, bar_y), (w - 8, bar_y + 25), (30, 25, 35), alpha=0.8); ctrl_x1, ctrl_x2 = 15, 145
    is_hover = (active_zone == "SYSTEM_TRIGGER"); blend_rect(frame, (ctrl_x1+2, bar_y), (ctrl_x2-2, bar_y+25), ACCENT_PINK if is_hover else (40,35,50), 0.9)
    cv2.putText(frame, "[0] SYSTEM", (ctrl_x1+10, bar_y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACCENT_PINK, 2)
    for i, name in enumerate(["LETTERS", "NUMBERS", "ALL"]):
        x = 180 + i*180; cv2.putText(frame, f"[{i+1}] {name}", (x, bar_y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACCENT_GREEN if current_mode==name else (150,140,160), 2 if current_mode==name else 1)
    if history:
        for i, hl in enumerate(list(history)[:5]):
            y_pos = 140 + i * 45; blend_rect(frame, (w-180, y_pos), (w-15, y_pos+35), (40,35,50), alpha=0.9)
            cv2.putText(frame, str(hl).upper(), (w-165, y_pos+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_WHITE, 1)
            
    ui_active = (pending_system_control or conflict_pending or pending_destructive_action)
    
    if is_paused and not pending_system_control:
        blend_rect(frame, (w//2-100, h//2-40), (w//2+100, h//2+10), (20,20,40), 0.8)
        cv2.putText(frame, "SYSTEM PAUSED", (w//2-85, h//2-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, ACCENT_PINK, 2)
    elif pending_system_control:
        blend_rect(frame, (0,0), (w,h), (10,5,20), 0.6); opts = ["RESUME", "PAUSE", "CANCEL"]; zw = w // 3
        for i, opt in enumerate(opts):
            x1, x2 = i*zw, (i+1)*zw
            # Ensure safe zone identifier assignment
            z_id = opt if opt == "CANCEL" else f"SYS_{opt}"
            is_act = (active_zone == z_id)
            cv2.rectangle(frame, (x1+20, 150), (x2-20, h-180), (100,100,255) if opt=="CANCEL" else (ACCENT_GREEN if is_act else (80,70,90)), 2)
            cv2.putText(frame, opt, (x1+zw//2-40, h//2), cv2.FONT_HERSHEY_DUPLEX, 0.9, TEXT_WHITE, 2); 
            if is_act: draw_bar(frame, x1+50, h-220, zw-100, min(1.0, (time.time()-zone_timer)/1.2), (100,100,255) if opt=="CANCEL" else ACCENT_GREEN)
    elif conflict_pending:
        blend_rect(frame, (0,0), (w,h), (10,5,20), 0.6)
        c_opts = list(conflict_pending) + ["CANCEL"]
        zw = w // len(c_opts)
        for i, opt in enumerate(c_opts):
            x1, x2 = i*zw, (i+1)*zw
            opt_str = str(opt).upper()
            z_id = "CANCEL" if opt == "CANCEL" else f"OPT_{i}"
            is_act = (active_zone == z_id)
            col = (100,100,255) if opt == "CANCEL" else (ACCENT_GREEN if is_act else (80,70,90))
            cv2.rectangle(frame, (x1+20, 150), (x2-20, h-180), col, 2)
            cv2.putText(frame, opt_str, (x1+zw//2-40, h//2), cv2.FONT_HERSHEY_DUPLEX, 0.9, TEXT_WHITE, 2); 
            if is_act: draw_bar(frame, x1+50, h-220, zw-100, min(1.0, (time.time()-zone_timer)/1.2), col)
    elif pending_destructive_action:
        blend_rect(frame, (0,0), (w,h), (10,5,20), 0.6); cv2.putText(frame, f"CONFIRM {pending_destructive_action}?", (w//2-180, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, ACCENT_BLUE, 2)
        c_opts = ["YES", "NO", "CANCEL"]
        zw = w // 3
        for i, z in enumerate(c_opts):
            x1, x2 = i*zw, (i+1)*zw; is_act = (active_zone == z)
            col = ACCENT_GREEN if (z=="YES" and is_act) else ((100,100,255) if is_act else (40,40,40))
            cv2.rectangle(frame, (x1+20, 150), (x2-20, h-180), col, 2); cv2.putText(frame, z, (x1+zw//2-40, h//2), cv2.FONT_HERSHEY_DUPLEX, 0.9, TEXT_WHITE, 2)
            if is_act: draw_bar(frame, x1+50, h-220, zw-100, min(1.0, (time.time()-zone_timer)/1.2), col)
    return frame

# ── Main ──────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0); cap.set(3, 1280); cap.set(4, 720)
sentence_built, last_raw, prev_time = "", "", time.time()
gesture_start_time, last_type_time, dynamic_cooldown = time.time(), 0, 0
current_mode, conflict_pending, hand_lost_time = "ALL", None, 0
wrist_history = deque(maxlen=8); gesture_history = deque(maxlen=HISTORY_MAX)
pending_destructive_action, zone_timer, active_zone, ui_cooldown, is_paused, pending_system_control = None, 0, None, 0, False, False
s_label, s_conf, d_label, d_conf = "-", 0.0, "-", 0.0

app_state = "ONBOARDING"
onboarding_step = 0
onboarding_timer = time.time()
hello_triggered = False
hello_trigger_time = 0

running = True

# INTELLIGENT CONFLICT MAP
CONFLICT_MAP = {
    "f": ["f", "9"], "9": ["f", "9"],
    "v": ["v", "2", "Victory!"], "2": ["v", "2", "Victory!"],
    "w": ["w", "6"], "6": ["6", "w"],
    "b": ["b", "4"], "4": ["b", "4"],
    "5": ["5", "Hello!"], "0": ["0", "o"], "o": ["0", "o"],
    "1": ["1", "d", "z"], "d": ["1", "d", "z"], "z": ["1", "d", "z"]
}
LETTERS_CONFLICT_MAP = {
    "d": ["d", "z"], "z": ["d", "z"]
}
PHYSICS_CONFLICTS = {"I LOVE YOU!":("I Love You 🤟 "), "THUMBS UP!":("Right On 👍 "), "CALL ME!":("Call Me 🤙 ")}

while cap.isOpened() and running:
    ret, frame = cap.read(); 
    if not ret: break
    frame = cv2.flip(frame, 1); h, w = frame.shape[:2]
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    hand_ok = bool(result.hand_landmarks)
    ui_active = (pending_system_control or conflict_pending or pending_destructive_action)
    
    if hand_ok:
        hand_lost_time = time.time(); lms = result.hand_landmarks[0]; draw_hand(frame, lms, h, w)
        wrist_history.append((lms[0].x, lms[0].y)); m_gest = get_multi_hand_gesture(result.hand_landmarks)
        moving_fast = False
        
        if len(wrist_history) == 8:
            vx, vy = sum(abs(wrist_history[i][0]-wrist_history[i-1][0]) for i in range(1,8))/7, sum(abs(wrist_history[i][1]-wrist_history[i-1][1]) for i in range(1,8))/7
            if vx > 0.035 or vy > 0.035: moving_fast = True
            
            if time.time() > dynamic_cooldown and not (is_paused or ui_active) and app_state == "MAIN":
                dx, dy = wrist_history[-1][0]-wrist_history[0][0], wrist_history[-1][1]-wrist_history[0][1]
                if dx < -0.15: 
                    pending_destructive_action, ui_cooldown, dynamic_cooldown, d_label = "BACKSPACE", time.time()+1.2, time.time()+1.0, "BACKSPACE?"
                elif dx > 0.15: 
                    sentence_built += " "
                    dynamic_cooldown, d_label = time.time()+1.0, "SPACE"
                elif dy > 0.25: 
                    pending_destructive_action, ui_cooldown, dynamic_cooldown, d_label = "CLEAR ALL", time.time()+1.2, time.time()+1.0, "CLEAR ALL?"
        
        if m_gest: s_label, s_conf = m_gest, 1.0
        elif is_paused or ui_active:
            s_label, s_conf = ("SELECT OPTION..." if ui_active else "PAUSED"), 1.0
        else:
            if moving_fast: s_label = "MOVING..."
            else:
                p_lbl = get_physics_gesture(lms, mode=current_mode)
                probs = static_model(tf.convert_to_tensor([extract_advanced_features(landmarks_to_flat(normalize_landmarks(lms)))], dtype=tf.float32))[0].numpy()
                
                best_idx, best_p = -1, -1
                if current_mode == "ALL":
                    best_idx, best_p = np.argmax(probs), np.max(probs)
                else:
                    f_idx, f_p = -1, -1
                    for i, lbl in enumerate(static_labels):
                        slbl = str(lbl).lower()
                        valid = (current_mode == "NUMBERS" and slbl.isdigit()) or (current_mode == "LETTERS" and not slbl.isdigit())
                        if valid and probs[i] > f_p: f_idx, f_p = i, probs[i]
                    if f_idx != -1: best_idx, best_p = f_idx, f_p

                if p_lbl:
                    s_label, s_conf = p_lbl, 1.0
                elif best_idx != -1:
                    s_label, s_conf = str(static_labels[best_idx]), float(best_p)
                else:
                    s_label, s_conf = "-", 0.0
        
        hx, hy = int(lms[8].x * w), int(lms[8].y * h)
        if app_state == "ONBOARDING":
            if onboarding_step > 0 and onboarding_step < 4:
                cx, cy, cw, ch = w//2, h//2, 800, 400
                nx, ny, nw, nh = cx + cw//2 - 180, cy + ch//2 - 90, 150, 50
                # Added hard 2.0s cooldown lock between cards
                if time.time() > dynamic_cooldown:
                    if nx - 100 < hx < nx+nw + 100 and ny - 100 < hy < ny+nh + 100:
                        if active_zone != "NEXT_ONBOARD": active_zone, zone_timer = "NEXT_ONBOARD", time.time()
                        if time.time() - zone_timer > 1.0:
                            onboarding_step += 1
                            dynamic_cooldown = time.time() + 2.0
                            onboarding_timer = time.time()
                            active_zone = None
                    else: active_zone = None
                else: active_zone = None
        else:
            if ui_active:
                if pending_system_control:
                    opts = ["RESUME", "PAUSE", "CANCEL"]
                    idx = min(2, int(hx // (w//3)))
                    z = "CANCEL" if opts[idx] == "CANCEL" else f"SYS_{opts[idx]}"
                    if active_zone != z: active_zone, zone_timer = z, time.time()
                    if time.time()-zone_timer > 1.2:
                        if z == "SYS_PAUSE": is_paused = True
                        elif z == "SYS_RESUME": is_paused = False
                        pending_system_control, active_zone = False, None
                elif conflict_pending and time.time() > ui_cooldown:
                    c_opts = list(conflict_pending) + ["CANCEL"]
                    idx = min(len(c_opts)-1, int(hx // (w//len(c_opts))))
                    z = "CANCEL" if c_opts[idx] == "CANCEL" else f"OPT_{idx}"
                    if active_zone != z: active_zone, zone_timer = z, time.time()
                    if time.time()-zone_timer > 1.2:
                        res = c_opts[idx]
                        if res != "CANCEL":
                            append_str = "Hello! 👋 " if res == "Hello!" else (res if len(res)==1 else res+" ")
                            sentence_built += append_str
                        conflict_pending, active_zone = None, None
                elif pending_destructive_action and time.time() > ui_cooldown:
                    c_opts = ["YES", "NO", "CANCEL"]
                    idx = min(2, int(hx // (w//3))); z = c_opts[idx]
                    if active_zone != z: active_zone, zone_timer = z, time.time()
                    if time.time()-zone_timer > 1.2:
                        if z=="YES":
                            if pending_destructive_action == "BACKSPACE":
                                if sentence_built.endswith(" "):
                                    sentence_built = sentence_built.rstrip().rpartition(' ')[0]
                                    if sentence_built: sentence_built += " "
                                    else: sentence_built = ""
                                else:
                                    sentence_built = sentence_built[:-1]
                            elif pending_destructive_action == "CLEAR ALL":
                                sentence_built = ""
                            elif pending_destructive_action == "EXIT APP":
                                running = False
                            d_label = "-"
                        pending_destructive_action, active_zone = None, None
            elif hy > h - 160:
                if hx < 200:
                    if active_zone != "SYSTEM_TRIGGER": active_zone, zone_timer = "SYSTEM_TRIGGER", time.time()
                    if time.time()-zone_timer > 0.8: pending_system_control, active_zone = True, None
                else: active_zone = None
            else: active_zone = None
    else:
        s_label, s_conf = "-", 0.0

    # Typing Logic Lock
    if app_state == "MAIN":
        if pending_destructive_action is None and s_label != last_raw: gesture_start_time, last_raw = time.time(), s_label
        elapsed = time.time()-gesture_start_time if hand_ok and s_label not in ["-", "MOVING...", "SELECT OPTION...", "PAUSED"] else 0
        if not (is_paused or ui_active) and elapsed >= HOLD_THRESHOLD and time.time()-last_type_time > TYPE_COOLDOWN and not conflict_pending:
            clean_lbl = str(s_label).lower()
            if current_mode == "ALL" and clean_lbl in CONFLICT_MAP:
                conflict_pending, ui_cooldown = CONFLICT_MAP[clean_lbl], time.time()+1.2
            elif current_mode == "LETTERS" and clean_lbl in LETTERS_CONFLICT_MAP:
                conflict_pending, ui_cooldown = LETTERS_CONFLICT_MAP[clean_lbl], time.time()+1.2
            else:
                gesture_history.appendleft(s_label)
                if s_label in PHYSICS_CONFLICTS: sentence_built += PHYSICS_CONFLICTS[s_label]
                elif "HEART" in s_label: sentence_built += "💖 "
                elif "APPLAUSE" in s_label: sentence_built += "👏 "
                elif s_label != "-" and "MOVING" not in s_label: sentence_built += s_label.lower()
                last_type_time = time.time()
    else:
        elapsed = 0

    fps = 1.0 / (time.time() - prev_time + 1e-9); prev_time = time.time()
    
    if app_state == "ONBOARDING":
        if onboarding_step == 0 and s_label == "5" and not hello_triggered:
            hello_triggered = True
            hello_trigger_time = time.time()
        if onboarding_step == 0 and hello_triggered and time.time() - hello_trigger_time > 2.0:
            onboarding_step = 1
            hello_triggered = False
            dynamic_cooldown = time.time() + 1.0
            
        if onboarding_step == 4 and time.time() - onboarding_timer > 1.8:
            app_state = "MAIN"
            gesture_start_time = time.time()
            
        frame = draw_onboarding_ui(frame, onboarding_step, hello_triggered, hello_trigger_time, onboarding_timer, active_zone, zone_timer)
        cv2.imshow("Hand Gesture Recognition", frame)
    else:
        cv2.imshow("Hand Gesture Recognition", draw_ui(frame, s_label, s_conf, d_label, d_conf, sentence_built, gesture_history, fps, hand_ok, current_mode, elapsed, conflict_pending, is_paused, pending_system_control, active_zone, zone_timer))
        
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): running = False
    elif k == 27: # ESC KEY
        if ui_active: pending_system_control, conflict_pending, pending_destructive_action, active_zone = False, None, None, None
        else: pending_destructive_action, ui_cooldown, dynamic_cooldown, d_label = "EXIT APP", time.time()+1.2, time.time()+0.7, "EXIT?"
    elif k in [ord("1"), ord("2"), ord("3")]: current_mode = ["LETTERS","NUMBERS","ALL"][int(chr(k))-1]
    elif k == ord("0"): pending_system_control = True
cap.release(); cv2.destroyAllWindows(); detector.close()
