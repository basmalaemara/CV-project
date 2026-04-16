"""
evaluate.py
────────────────────────────────────────────────────────────────
Evaluate both trained models and save:
  - Classification reports
  - Confusion matrices (PNG)

Run after training both models.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, ".")
from preprocessing.feature_extractor import (
    GESTURE_LABELS,
    DYNAMIC_LABELS,
    load_static_dataset,
    load_dynamic_dataset,
    extract_advanced_features,
)

os.makedirs("docs/figures", exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# 1. STATIC MODEL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  STATIC GESTURE EVALUATION")
print("=" * 55)

X, y = load_static_dataset()
_, X_test_raw, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Extract 210 advanced features (total 273) to match model expectations
X_test = extract_advanced_features(X_test_raw)

static_model = tf.keras.models.load_model("models/keypoint_classifier.keras")
static_labels = np.load("models/static_class_labels.npy")

# --- Ground Truth Mapping ---
# The model was trained using LabelBinarizer on the 'label' string column.
# We need to map our numeric gestureIDs to those alphabetical indices.
from preprocessing.feature_extractor import GESTURE_LABELS
id_to_label = GESTURE_LABELS
label_to_idx = {label: i for i, label in enumerate(static_labels)}

# Map ground truth IDs to string labels, then to categorical indices
y_test_mapped = []
for gid in y_test:
    lbl_str = str(id_to_label.get(gid, gid)).lower()
    idx = label_to_idx.get(lbl_str, 0)
    y_test_mapped.append(idx)
y_test_mapped = np.array(y_test_mapped)

y_pred = np.argmax(static_model.predict(X_test, verbose=0), axis=1)
print("\n" + classification_report(y_test_mapped, y_pred, target_names=static_labels))

cm = confusion_matrix(y_test_mapped, y_pred)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=False, cmap="Greens", xticklabels=static_labels, yticklabels=static_labels)
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.title("Static Gesture Confusion Matrix")
plt.tight_layout()
plt.savefig("docs/figures/static_confusion_matrix.png")
print("\nSaved -> docs/figures/static_confusion_matrix.png")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 2. DYNAMIC MODEL
# ════════════════════════════════════════════════════════════════════════════
try:
    print("\n" + "=" * 55)
    print("  DYNAMIC GESTURE EVALUATION")
    print("=" * 55)

    X_d, y_d = load_dynamic_dataset()
    _, X_d_test, _, y_d_test = train_test_split(
        X_d, y_d, test_size=0.2, random_state=42, stratify=y_d
    )

    dynamic_model = tf.keras.models.load_model("models/point_history_classifier.keras")
    dynamic_labels = np.load("models/dynamic_class_labels.npy")

    y_d_pred = np.argmax(dynamic_model.predict(X_d_test, verbose=0), axis=1)
    str_dynamic_labels = [str(l) for l in dynamic_labels]
    print("\n" + classification_report(y_d_test, y_d_pred, target_names=str_dynamic_labels))

    cm_d = confusion_matrix(y_d_test, y_d_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm_d, annot=True, fmt="d",
                xticklabels=str_dynamic_labels, yticklabels=str_dynamic_labels,
                cmap="Purples", linewidths=0.5, ax=ax)
    ax.set_title("Dynamic Gesture — Confusion Matrix", fontsize=14, pad=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig("docs/figures/dynamic_confusion_matrix.png", dpi=150)
    print("[SUCCESS] Saved -> docs/figures/dynamic_confusion_matrix.png")
    plt.close()

except FileNotFoundError as e:
    print(f"⚠️  Skipping dynamic evaluation: {e}")
