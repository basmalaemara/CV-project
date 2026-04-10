# Hand Gesture Recognition

A real-time Computer Vision system that detects and classifies hand gestures using MediaPipe and deep learning (MLP + LSTM).

## Gesture Vocabulary

### Static (pose)
| ID | Gesture | Label |
|----|---------|-------|
| 0 | ✋ Open hand | `open_hand` |
| 1 | ✊ Fist | `fist` |
| 2 | 👍 Thumbs up | `thumbs_up` |
| 3 | 👎 Thumbs down | `thumbs_down` |
| 4 | ✌️ Peace | `peace` |
| 5 | 👌 OK | `ok` |
| 6 | ☝️ Pointing up | `pointing_up` |
| 7 | 👇 Pointing down | `pointing_down` |

### Dynamic (motion)
| ID | Gesture | Label |
|----|---------|-------|
| 0 | ← Swipe left | `swipe_left` |
| 1 | → Swipe right | `swipe_right` |
| 2 | 🔍+ Zoom in | `zoom_in` |
| 3 | 🔍− Zoom out | `zoom_out` |

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage — Step by Step

```powershell
# 1. Collect static gesture data (300 samples per gesture)
python collect_static_gestures.py

# 2. Collect dynamic gesture data (100 sequences per gesture)
python collect_dynamic_gestures.py

# 3. Train static MLP model
python models/train_static.py

# 4. Train dynamic LSTM model
python models/train_dynamic.py

# 5. Evaluate both models
python models/evaluate.py

# 6. Run live demo!
python inference/realtime_recognizer.py
```

## Architecture

```
Webcam → MediaPipe (21 landmarks × 3D)
              │
    ┌─────────┴──────────┐
    ▼                    ▼
  MLP (63,)         LSTM (30, 63)
  Static gestures   Dynamic gestures
    │                    │
    └─────────┬──────────┘
              ▼
         Gesture label + confidence
              │
         Overlay UI (OpenCV)
```

## Project Structure

```
project/
├── data/
│   ├── static/                  # Landmark CSVs
│   └── dynamic/                 # Sequence .npy files
├── preprocessing/
│   └── feature_extractor.py     # Shared normalization utilities
├── models/
│   ├── train_static.py
│   ├── train_dynamic.py
│   ├── evaluate.py
│   ├── keypoint_classifier.keras
│   └── point_history_classifier.keras
├── inference/
│   └── realtime_recognizer.py   ← Main demo
├── docs/figures/                # Confusion matrices + training curves
├── collect_static_gestures.py
├── collect_dynamic_gestures.py
└── requirements.txt
```

## Expected Results

| Model | Accuracy | Speed |
|-------|----------|-------|
| Static MLP | > 95% | ~1 ms/frame |
| Dynamic LSTM | > 90% | ~5 ms/frame |
| Full pipeline | — | > 25 FPS on CPU |
