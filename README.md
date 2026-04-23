# Hand Gesture Recognition

A real-time Computer Vision system that detects and classifies hand gestures using MediaPipe and deep learning (Residual MLP + BiLSTM with Attention).

## Gesture Vocabulary

### Static (pose) — 37 classes
| ID | Gesture | Label |
|----|---------|-------|
| 0–9 | Digits | `0` – `9` |
| a–z | ASL Alphabet | `a` – `z` |
| open_hand | Open hand | `open_hand` |
| fist | Fist | `fist` |
| thumbs_up | Thumbs up | `thumbs_up` |
| thumbs_down | Thumbs down | `thumbs_down` |
| peace | Peace / Victory | `peace` |
| ok | OK | `ok` |
| pointing_up | Pointing up | `pointing_up` |
| pointing_down | Pointing down | `pointing_down` |

### Dynamic (motion) — 4 classes
| ID | Gesture | Label | Action |
|----|---------|-------|--------|
| 0 | Swipe left | `swipe_left` | Backspace (delete last character) |
| 1 | Swipe right | `swipe_right` | Add space |
| 2 | Zoom in | `zoom_in` | Increase sentence text size |
| 3 | Zoom out | `zoom_out` | Decrease sentence text size |

> **Swipe down** (hand moved downward) triggers **Clear All** via the physics engine.

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or use the automated setup script:

```powershell
.\setup_env.ps1
```

---

## Usage — Step by Step

```powershell
# 1. Collect static gesture data
python scripts/collect_static_gestures.py

# 2. Collect dynamic gesture data
python scripts/collect_dynamic_gestures.py

# 3. Split dynamic data into train/test (prevents data leakage)
python scripts/prepare_dynamic_data.py

# 4. Train static Residual MLP
python models/train_static.py

# 5. Train dynamic BiLSTM + Attention
python models/train_dynamic.py

# 6. Evaluate both models
python models/evaluate.py

# 7. Run model comparison against baselines
python models/compare_static.py
python models/compare_dynamic.py

# 8. Run live demo
python inference/realtime_recognizer.py
```

---

## Architecture

```
Webcam → MediaPipe Hand Landmarker (21 landmarks × 3D)
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
  Residual MLP                BiLSTM + Attention
  Input: 273-dim               Input: (30, 63)
  (63 raw + 210 pairwise       raw landmark sequences
   distances)                  30 frames × 63 coords
  Static gestures              Dynamic gestures
          │                        │
          └───────────┬────────────┘
                      ▼
              Gesture label + confidence
                      │
              Physics engine overlay
              (special phrases, conflict
               resolution, multi-hand)
                      │
              Real-time UI (OpenCV + PIL)
```

---

## Project Structure

```
project/
├── data/
│   ├── static/                    # Landmark CSVs (37 gesture classes)
│   ├── dynamic/                   # Training sequences (.npy, raw coords)
│   └── dynamic_test/              # Held-out test sequences (no leakage)
├── preprocessing/
│   ├── __init__.py
│   └── feature_extractor.py       # Feature extraction + label maps
├── models/
│   ├── train_static.py            # Residual MLP trainer
│   ├── train_dynamic.py           # BiLSTM+Attention trainer
│   ├── evaluate.py                # Full evaluation + confusion matrices
│   ├── compare_static.py          # Baseline comparison (6 models)
│   ├── compare_dynamic.py         # Baseline comparison (5 architectures)
│   ├── keypoint_classifier.keras  # Trained static model
│   └── point_history_classifier.keras  # Trained dynamic model
├── inference/
│   └── realtime_recognizer.py     # Live demo (main entry point)
├── docs/
│   ├── evaluation_report.txt
│   ├── comparison_static.txt
│   ├── comparison_dynamic.txt
│   └── figures/                   # All plots (PNG)
├── scripts/
│   ├── collect_static_gestures.py
│   ├── collect_dynamic_gestures.py
│   ├── prepare_dynamic_data.py
│   ├── augment_dynamic_gestures.py
│   ├── extract_from_video.py
│   ├── extract_jester.py
│   └── extract_landmarks_from_images.py
├── requirements.txt
└── setup_env.ps1
```

---

## Results

### Model Accuracy

| Model | Test Accuracy | CV Accuracy |
|-------|--------------|-------------|
| Static Residual MLP (37 classes) | **99.88%** | 99.87% ± 0.09% |
| Dynamic BiLSTM + Attention (4 classes) | **100.00%** | 100.00% ± 0.00% |
| Full pipeline | — | > 25 FPS on CPU |

### Comparison Against Baselines

**Static (37 classes, 2,461 test samples):**

| Model | Accuracy |
|-------|----------|
| **Residual MLP + pairwise features (ours)** | **99.88%** |
| Random Forest (200 trees) | 95.57% |
| SVM (RBF kernel) | 95.25% |
| K-Nearest Neighbours | 94.80% |
| Logistic Regression | 94.35% |
| Shallow MLP (no residuals) | 94.11% |

**Dynamic (4 classes):**

| Model | Accuracy |
|-------|----------|
| **BiLSTM + Attention (ours)** | **100.00%** |
| Flat MLP (no temporal order) | 99.93% |
| Simple LSTM (unidirectional) | 97.30% |
| Bidirectional GRU | 96.63% |
| 1D CNN | 92.13% |

> Comparison figures saved in `docs/figures/`.
