"""
augment_dynamic_gestures.py
────────────────────────────────────────────────────────────────
Augments existing dynamic gesture sequences to generate more
training data without manual collection.

Each original sequence → 10 augmented variants:
  1. Gaussian noise
  2. Horizontal mirror (flip x-axis)
  3. Time warp — slow  (stretch motion to 1.3x speed)
  4. Time warp — fast  (compress motion to 0.7x speed)
  5. Scale variation
  6. 2D rotation
  7. Noise + mirror
  8. Scale + noise
  9. Time warp + noise
 10. Mirror + scale

Run once before train_dynamic.py.
"""

import os
import sys
import numpy as np

sys.path.insert(0, ".")
from preprocessing.feature_extractor import DYNAMIC_LABELS

SAVE_DIR        = "data/dynamic"
SEQUENCE_LENGTH = 30   # frames per sequence
N_FEATURES      = 63   # 21 landmarks × 3 coords (x, y, z)

# x coords: indices 0, 3, 6, ..., 60  (every 3rd starting at 0)
# y coords: indices 1, 4, 7, ..., 61
X_INDICES = np.arange(0, N_FEATURES, 3)
Y_INDICES = np.arange(1, N_FEATURES, 3)

# ── Augmentation functions ────────────────────────────────────────────────────

def add_noise(seq, scale=0.02):
    return seq + np.random.normal(0, scale, seq.shape).astype(np.float32)

def mirror(seq):
    """Flip hand horizontally by negating all x-coordinates."""
    s = seq.copy()
    s[:, X_INDICES] *= -1
    return s

def time_warp(seq, factor):
    """Stretch (factor>1) or compress (factor<1) the motion temporally."""
    n = SEQUENCE_LENGTH
    warped_len  = max(5, int(n * factor))
    orig_times  = np.linspace(0, 1, n)
    warp_times  = np.linspace(0, 1, warped_len)
    final_times = np.linspace(0, 1, n)

    result = np.zeros_like(seq)
    for i in range(N_FEATURES):
        warped          = np.interp(warp_times, orig_times, seq[:, i])
        result[:, i]    = np.interp(final_times, warp_times, warped)
    return result.astype(np.float32)

def scale(seq, low=0.85, high=1.15):
    factor = np.random.uniform(low, high)
    return (seq * factor).astype(np.float32)

def rotate2d(seq, max_angle_deg=15):
    """Rotate landmarks in the XY plane."""
    angle   = np.random.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    s = seq.copy()
    x = seq[:, X_INDICES]
    y = seq[:, Y_INDICES]
    s[:, X_INDICES] = x * cos_a - y * sin_a
    s[:, Y_INDICES] = x * sin_a + y * cos_a
    return s.astype(np.float32)

# ── Load existing sequences ───────────────────────────────────────────────────

def load_sequences(gesture_id, label):
    prefix = f"{gesture_id}_{label}_"
    files  = sorted(f for f in os.listdir(SAVE_DIR)
                    if f.startswith(prefix) and f.endswith(".npy"))
    seqs = []
    for fname in files:
        arr = np.load(os.path.join(SAVE_DIR, fname))
        if arr.shape == (SEQUENCE_LENGTH, N_FEATURES):
            seqs.append(arr)
    return seqs

def next_index(gesture_id, label):
    prefix = f"{gesture_id}_{label}_"
    existing = [f for f in os.listdir(SAVE_DIR)
                if f.startswith(prefix) and f.endswith(".npy")]
    return len(existing)

def save_seq(seq, gesture_id, label, idx):
    path = os.path.join(SAVE_DIR, f"{gesture_id}_{label}_{idx}.npy")
    np.save(path, seq.astype(np.float32))

# ── Augmentation pipeline per sequence ───────────────────────────────────────

def augment_one(seq):
    """Return 10 augmented variants of a single sequence."""
    return [
        add_noise(seq),
        mirror(seq),
        time_warp(seq, 1.3),
        time_warp(seq, 0.7),
        scale(seq),
        rotate2d(seq),
        mirror(add_noise(seq)),
        add_noise(scale(seq)),
        add_noise(time_warp(seq, np.random.uniform(0.8, 1.2))),
        scale(mirror(seq)),
    ]

# ── Main ──────────────────────────────────────────────────────────────────────

print("\n=== Dynamic Gesture Augmentation ===\n")
total_generated = 0

for gid, label in DYNAMIC_LABELS.items():
    originals = load_sequences(gid, label)
    if not originals:
        print(f"  [{gid}] {label:<15} — no sequences found, skipping")
        continue

    idx = next_index(gid, label)
    generated = 0

    for seq in originals:
        for aug_seq in augment_one(seq):
            save_seq(aug_seq, gid, label, idx)
            idx       += 1
            generated += 1

    total_generated += generated
    print(f"  [{gid}] {label:<15} {len(originals):>3} original → +{generated} augmented "
          f"= {len(originals) + generated} total")

print(f"\n✅  Done. Generated {total_generated} new sequences.")
print(f"    Total sequences in data/dynamic/: "
      f"{sum(1 for f in os.listdir(SAVE_DIR) if f.endswith('.npy'))}")
print("\nNext step: python models/train_dynamic.py")
