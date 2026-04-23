"""
prepare_dynamic_data.py
────────────────────────────────────────────────────────────────
Fixes the train/test data leakage in dynamic gesture evaluation.

What this script does:
  1. Keeps only the original sequences (removes augmented files)
  2. Splits originals → 80% train / 20% test (per class, stratified)
  3. Copies test originals to data/dynamic_test/ (never touched again)
  4. Augments only the training originals (30x) into data/dynamic/

After running this:
  - data/dynamic/      → training data (originals + augmented)
  - data/dynamic_test/ → test data (pristine originals only)

Then run:
  python models/train_dynamic.py
  python models/evaluate.py
"""

import os
import sys
import shutil
import numpy as np

sys.path.insert(0, ".")
from preprocessing.feature_extractor import DYNAMIC_LABELS

TRAIN_DIR   = "data/dynamic"
TEST_DIR    = "data/dynamic_test"
N_ORIG      = 21          # original sequences per class (before augmentation)
TEST_RATIO  = 0.20        # 20% held out for testing

SEQUENCE_LENGTH = 30
N_FEATURES      = 63
X_INDICES       = np.arange(0, N_FEATURES, 3)
Y_INDICES       = np.arange(1, N_FEATURES, 3)

os.makedirs(TEST_DIR, exist_ok=True)

# ── Augmentation functions ────────────────────────────────────────────────────

def add_noise(seq, scale=0.02):
    return seq + np.random.normal(0, scale, seq.shape).astype(np.float32)

def mirror(seq):
    s = seq.copy(); s[:, X_INDICES] *= -1; return s

def time_warp(seq, factor):
    n           = SEQUENCE_LENGTH
    warped_len  = max(5, int(n * factor))
    orig_t      = np.linspace(0, 1, n)
    warp_t      = np.linspace(0, 1, warped_len)
    final_t     = np.linspace(0, 1, n)
    result      = np.zeros_like(seq)
    for i in range(N_FEATURES):
        warped        = np.interp(warp_t, orig_t, seq[:, i])
        result[:, i]  = np.interp(final_t, warp_t, warped)
    return result.astype(np.float32)

def scale_seq(seq, low=0.85, high=1.15):
    return (seq * np.random.uniform(low, high)).astype(np.float32)

def rotate2d(seq, max_deg=15):
    a               = np.random.uniform(-max_deg, max_deg) * np.pi / 180
    cos_a, sin_a    = np.cos(a), np.sin(a)
    s               = seq.copy()
    x, y            = seq[:, X_INDICES], seq[:, Y_INDICES]
    s[:, X_INDICES] = x * cos_a - y * sin_a
    s[:, Y_INDICES] = x * sin_a + y * cos_a
    return s.astype(np.float32)

def augment_one(seq):
    """Return 30 augmented variants of a single sequence."""
    return [
        # ── Single transforms ──────────────────────────────────────────────
        add_noise(seq, 0.01),
        add_noise(seq, 0.02),
        add_noise(seq, 0.04),
        mirror(seq),
        time_warp(seq, 0.6),
        time_warp(seq, 0.75),
        time_warp(seq, 0.9),
        time_warp(seq, 1.1),
        time_warp(seq, 1.25),
        time_warp(seq, 1.5),
        scale_seq(seq, 0.75, 0.90),
        scale_seq(seq, 0.90, 1.10),
        scale_seq(seq, 1.10, 1.30),
        rotate2d(seq, max_deg=8),
        rotate2d(seq, max_deg=20),
        # ── Two-way combinations ───────────────────────────────────────────
        mirror(add_noise(seq, 0.02)),
        mirror(time_warp(seq, 0.75)),
        mirror(time_warp(seq, 1.25)),
        mirror(scale_seq(seq)),
        add_noise(scale_seq(seq), 0.02),
        add_noise(time_warp(seq, 0.8),  0.01),
        add_noise(time_warp(seq, 1.2),  0.01),
        rotate2d(add_noise(seq, 0.02), max_deg=10),
        rotate2d(scale_seq(seq),       max_deg=10),
        scale_seq(time_warp(seq, 0.85), 0.90, 1.10),
        scale_seq(time_warp(seq, 1.15), 0.90, 1.10),
        # ── Three-way combinations ─────────────────────────────────────────
        mirror(add_noise(scale_seq(seq),       0.02)),
        mirror(add_noise(time_warp(seq, 0.8),  0.02)),
        rotate2d(add_noise(scale_seq(seq),     0.01), max_deg=8),
        add_noise(scale_seq(time_warp(seq, np.random.uniform(0.7, 1.3))), 0.02),
    ]

# ════════════════════════════════════════════════════════════════════════════
# Step 1 — Remove augmented files, keep only originals (idx 0..N_ORIG-1)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 1: Removing augmented files ===")
removed = 0
for gid, label in DYNAMIC_LABELS.items():
    prefix = f"{gid}_{label}_"
    for fname in os.listdir(TRAIN_DIR):
        if not (fname.startswith(prefix) and fname.endswith(".npy")):
            continue
        idx = int(fname.replace(prefix, "").replace(".npy", ""))
        if idx >= N_ORIG:
            os.remove(os.path.join(TRAIN_DIR, fname))
            removed += 1
print(f"  Removed {removed} augmented files.")

# ════════════════════════════════════════════════════════════════════════════
# Step 2 — Split originals into train / test per class
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 2: Splitting originals into train / test ===")

train_files = {gid: [] for gid in DYNAMIC_LABELS}
test_files  = {gid: [] for gid in DYNAMIC_LABELS}

for gid, label in DYNAMIC_LABELS.items():
    prefix  = f"{gid}_{label}_"
    originals = sorted(
        f for f in os.listdir(TRAIN_DIR)
        if f.startswith(prefix) and f.endswith(".npy")
    )[:N_ORIG]

    n_test  = max(1, round(len(originals) * TEST_RATIO))
    n_train = len(originals) - n_test

    # Last n_test files → test (deterministic, no randomness needed for 20 seqs)
    train_files[gid] = originals[:n_train]
    test_files[gid]  = originals[n_train:]

    print(f"  [{gid}] {label:<15}  {len(originals)} originals  ->  "
          f"train: {n_train}  |  test: {n_test}")

# ════════════════════════════════════════════════════════════════════════════
# Step 3 — Move test files to data/dynamic_test/ (remove from training dir)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 3: Moving test originals to data/dynamic_test/ ===")

# Clear old test directory first
for f in os.listdir(TEST_DIR):
    if f.endswith(".npy"):
        os.remove(os.path.join(TEST_DIR, f))

for gid, files in test_files.items():
    for fname in files:
        shutil.move(
            os.path.join(TRAIN_DIR, fname),
            os.path.join(TEST_DIR, fname),
        )
    print(f"  [{gid}] moved {len(files)} test files (removed from training dir)")

# ════════════════════════════════════════════════════════════════════════════
# Step 4 — Augment only training originals (10x each)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Step 4: Augmenting training originals ===")
total_aug = 0

for gid, label in DYNAMIC_LABELS.items():
    # Next available index = N_ORIG (originals occupy 0..N_ORIG-1)
    idx       = N_ORIG
    generated = 0

    for fname in train_files[gid]:
        seq = np.load(os.path.join(TRAIN_DIR, fname))
        if seq.shape != (SEQUENCE_LENGTH, N_FEATURES):
            continue
        for aug_seq in augment_one(seq):
            path = os.path.join(TRAIN_DIR, f"{gid}_{label}_{idx}.npy")
            np.save(path, aug_seq.astype(np.float32))
            idx       += 1
            generated += 1

    total_aug += generated
    n_train    = len(train_files[gid])
    print(f"  [{gid}] {label:<15}  {n_train} train originals  ->  "
          f"+{generated} augmented  =  {n_train + generated} total training seqs")

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
total_train = sum(1 for f in os.listdir(TRAIN_DIR) if f.endswith(".npy"))
total_test  = sum(1 for f in os.listdir(TEST_DIR)  if f.endswith(".npy"))

print("\nDone.")
print(f"    data/dynamic/      -> {total_train} sequences (training)")
print(f"    data/dynamic_test/ -> {total_test} sequences (testing, pristine)")
print("\nNext steps:")
print("  python models/train_dynamic.py")
print("  python models/evaluate.py")
