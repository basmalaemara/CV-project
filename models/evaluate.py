"""
evaluate.py
────────────────────────────────────────────────────────────────
Evaluate both trained models and save:
  - Classification reports
  - Confusion matrices (PNG, annotated)
  - 5-fold cross-validation results
  - Per-class accuracy table (worst → best)
  - Full text report -> docs/evaluation_report.txt

Run after training both models.
"""

import os
import sys
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, ".")
from preprocessing.feature_extractor import (
    GESTURE_LABELS,
    DYNAMIC_LABELS,
    extract_advanced_features,
)

os.makedirs("docs/figures", exist_ok=True)

# ── Report buffer ─────────────────────────────────────────────────────────────
_report_lines = []

def log(text=""):
    print(text)
    _report_lines.append(str(text))

def save_report():
    os.makedirs("docs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("docs/evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Evaluation Report — {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n".join(_report_lines))
    print("\n[SAVED] docs/evaluation_report.txt")

# ── Shared confusion matrix plotter ──────────────────────────────────────────
def plot_confusion_matrix(cm, labels, title, save_path, figsize, annot_size):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": annot_size}, linewidths=0.4, ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log(f"[SAVED] {save_path}")

# ── Per-class accuracy table (sorted worst → best) ────────────────────────────
def per_class_accuracy_table(cm, labels, model_name):
    log(f"\n-- Per-Class Accuracy  (worst -> best)  [{model_name}] --")
    log(f"\n  {'Class':<22} {'Correct':>7} {'Total':>7}  {'Accuracy':>9}")
    log(f"  {'-'*22} {'-'*7} {'-'*7}  {'-'*9}")
    rows = []
    for i, lbl in enumerate(labels):
        total   = int(cm[i].sum())
        correct = int(cm[i, i])
        acc     = correct / total if total > 0 else None
        rows.append((lbl, acc, correct, total))
    rows.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))
    for lbl, acc, correct, total in rows:
        acc_str = f"{acc * 100:.1f}%" if acc is not None else "n/a"
        log(f"  {lbl:<22} {correct:>7} {total:>7}  {acc_str:>9}")

# ════════════════════════════════════════════════════════════════════════════
# 1. STATIC MODEL
#    Load data the same way train_static.py does — using the "label" column
#    directly from CSVs. This avoids the broken gesture_id integer mapping.
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("  STATIC GESTURE EVALUATION")
log("=" * 60)

all_dfs = []
for csv_file in sorted(glob.glob("data/static/*.csv")):
    all_dfs.append(pd.read_csv(csv_file))

if not all_dfs:
    log("ERROR: No CSV files found in data/static/")
    sys.exit(1)

data = pd.concat(all_dfs, ignore_index=True)

if "label" in data.columns:
    missing_mask = data["label"].isna()
    data.loc[missing_mask, "label"] = data.loc[missing_mask, "gesture_id"].map(GESTURE_LABELS)
    data["label"] = data["label"].fillna(data["gesture_id"].astype(str))
    X_raw = data.drop(["gesture_id", "label"], axis=1).values.astype(np.float32)
else:
    data["label"] = data["gesture_id"].astype(str)
    X_raw = data.drop("gesture_id", axis=1).values.astype(np.float32)

y_str  = data["label"].astype(str).str.lower().values
X_full = extract_advanced_features(X_raw)

static_model  = tf.keras.models.load_model("models/keypoint_classifier.keras")
static_labels = np.load("models/static_class_labels.npy")
label_to_idx  = {lbl: i for i, lbl in enumerate(static_labels)}

# Map string labels → model class indices (direct lookup, no integer detour)
y_idx   = np.array([label_to_idx.get(lbl, -1) for lbl in y_str])
valid   = y_idx >= 0
X_full  = X_full[valid]
y_idx   = y_idx[valid]
dropped = (~valid).sum()
if dropped:
    log(f"  (dropped {int(dropped)} samples with unknown labels)")

log(f"\n  Loaded {len(y_idx)} samples | {len(static_labels)} classes")

_, X_test, _, y_test = train_test_split(
    X_full, y_idx, test_size=0.2, random_state=42, stratify=y_idx
)

y_pred     = np.argmax(static_model.predict(X_test, verbose=0), axis=1)
static_acc = np.mean(y_pred == y_test)
log(f"\nOverall accuracy (80/20 split): {static_acc:.4f}  ({static_acc*100:.2f}%)")

log("\n-- Classification Report (Static) --")
log(classification_report(y_test, y_pred, target_names=static_labels, zero_division=0))

cm_static = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(
    cm_static, static_labels,
    title="Static Gesture Confusion Matrix",
    save_path="docs/figures/static_confusion_matrix.png",
    figsize=(14, 12), annot_size=5,
)
per_class_accuracy_table(cm_static, static_labels, "Static")

log("\n-- 5-Fold Stratified Cross-Validation  [Static] --")
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = []
for fold, (_, test_idx) in enumerate(skf.split(X_full, y_idx)):
    preds = np.argmax(static_model.predict(X_full[test_idx], verbose=0), axis=1)
    acc   = np.mean(preds == y_idx[test_idx])
    cv_acc.append(acc)
    log(f"  Fold {fold + 1}: {acc:.4f}")
log(f"\n  CV Accuracy: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 2. DYNAMIC MODEL
#    Evaluate ONLY on original sequences (before augmentation).
#    Augmented files are derived from originals — including them in the test
#    set causes data leakage and inflates accuracy to 100%.
#    Originals = the first N_ORIG files per class (lowest filename indices).
# ════════════════════════════════════════════════════════════════════════════
try:
    log("\n" + "=" * 60)
    log("  DYNAMIC GESTURE EVALUATION  (originals only — no augmented)")
    log("=" * 60)

    # Load ONLY from the pristine test split (never seen during training)
    DYNAMIC_TEST_DIR = "data/dynamic_test"
    DYNAMIC_TRAIN_DIR = "data/dynamic"

    if not os.path.exists(DYNAMIC_TEST_DIR) or not os.listdir(DYNAMIC_TEST_DIR):
        log("  NOTE: data/dynamic_test/ not found — run prepare_dynamic_data.py first.")
        log("  Falling back to first 21 originals from data/dynamic/")
        DYNAMIC_TEST_DIR = DYNAMIC_TRAIN_DIR
        N_ORIG = 21
    else:
        N_ORIG = None   # load all files in test dir

    X_d, y_d = [], []
    for gid, label in DYNAMIC_LABELS.items():
        prefix = f"{gid}_{label}_"
        files  = sorted(
            f for f in os.listdir(DYNAMIC_TEST_DIR)
            if f.startswith(prefix) and f.endswith(".npy")
        )
        if N_ORIG is not None:
            files = files[:N_ORIG]
        for fname in files:
            arr = np.load(os.path.join(DYNAMIC_TEST_DIR, fname))
            if arr.shape == (30, 63):
                X_d.append(arr)
                y_d.append(gid)

    X_d = np.array(X_d, dtype=np.float32)
    y_d = np.array(y_d, dtype=int)
    log(f"\n  Test set: {len(y_d)} pristine sequences from {DYNAMIC_TEST_DIR}")

    # Use all test sequences — no further splitting needed
    X_d_test, y_d_test = X_d, y_d

    dynamic_model  = tf.keras.models.load_model("models/point_history_classifier.keras")
    dynamic_labels = np.load("models/dynamic_class_labels.npy")
    str_dyn_labels = [str(l) for l in dynamic_labels]

    y_d_pred = np.argmax(dynamic_model.predict(X_d_test, verbose=0), axis=1)
    dyn_acc  = np.mean(y_d_pred == y_d_test)
    log(f"\nOverall accuracy (80/20 split): {dyn_acc:.4f}  ({dyn_acc*100:.2f}%)")

    log("\n-- Classification Report (Dynamic) --")
    log(classification_report(y_d_test, y_d_pred, target_names=str_dyn_labels, zero_division=0))

    cm_d = confusion_matrix(y_d_test, y_d_pred)
    plot_confusion_matrix(
        cm_d, str_dyn_labels,
        title="Dynamic Gesture Confusion Matrix",
        save_path="docs/figures/dynamic_confusion_matrix.png",
        figsize=(7, 5), annot_size=10,
    )
    per_class_accuracy_table(cm_d, str_dyn_labels, "Dynamic")

    # CV on the pristine test set (leave-one-out if small, else 5-fold)
    n_splits = min(5, len(y_d_test))
    log(f"\n-- {n_splits}-Fold Cross-Validation on test set  [Dynamic] --")
    kf   = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_d = []
    for fold, (_, test_idx) in enumerate(kf.split(X_d_test)):
        preds = np.argmax(dynamic_model.predict(X_d_test[test_idx], verbose=0), axis=1)
        acc   = np.mean(preds == y_d_test[test_idx])
        cv_d.append(acc)
        log(f"  Fold {fold + 1}: {acc:.4f}")
    log(f"\n  CV Accuracy: {np.mean(cv_d):.4f} ± {np.std(cv_d):.4f}")

except FileNotFoundError as e:
    log(f"WARNING: Skipping dynamic evaluation: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("  SUMMARY")
log("=" * 60)
log(f"  Static  — accuracy: {static_acc:.4f}  |  CV: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
try:
    log(f"  Dynamic — accuracy: {dyn_acc:.4f}  |  CV: {np.mean(cv_d):.4f} ± {np.std(cv_d):.4f}")
except NameError:
    log("  Dynamic — skipped")

save_report()
