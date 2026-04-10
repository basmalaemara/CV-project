"""
train_static.py  — works for both custom gestures AND ASL dataset
────────────────────────────────────────────────────────────────
Auto-detects all classes from the CSVs in data/static/.

Output
------
  models/keypoint_classifier.keras
  models/static_class_labels.npy
  docs/figures/static_training_curves.png
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

os.makedirs("models", exist_ok=True)
os.makedirs("docs/figures", exist_ok=True)

# ── Load data — auto-detect classes ──────────────────────────────────────────
print("Loading data from data/static/ ...")
all_dfs = []
for csv_file in sorted(glob.glob("data/static/*.csv")):
    df = pd.read_csv(csv_file)
    all_dfs.append(df)

if not all_dfs:
    print("❌  No CSV files found in data/static/")
    print("    Run extract_landmarks_from_images.py (Kaggle) or collect_static_gestures.py first.")
    sys.exit(1)

data = pd.concat(all_dfs, ignore_index=True)

# Support both 'label' column (ASL dataset) and numeric gesture_id (custom)
if "label" in data.columns:
    # ASL dataset — use string labels directly
    X = data.drop(["gesture_id", "label"], axis=1).values.astype(np.float32)
    y_raw = data["label"].values
else:
    # Custom collected data — map integer id to string
    X = data.drop("gesture_id", axis=1).values.astype(np.float32)
    y_raw = data["gesture_id"].astype(str).values

lb    = LabelBinarizer()
y_ohe = lb.fit_transform(y_raw)

print(f"Dataset: {X.shape[0]} samples  |  {len(lb.classes_)} classes: {list(lb.classes_)}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y_ohe, test_size=0.2, random_state=42, stratify=y_raw
)
print(f"Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}\n")

# ── Build MLP ─────────────────────────────────────────────────────────────────
n_classes = y_ohe.shape[1]

model = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.35),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.30),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.20),
    layers.Dense(n_classes, activation="softmax"),
], name="KeypointClassifier")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=7, verbose=1),
        callbacks.ModelCheckpoint("models/keypoint_classifier_best.keras",
                                  save_best_only=True, monitor="val_accuracy"),
    ],
    verbose=1,
)

# ── Save ──────────────────────────────────────────────────────────────────────
model.save("models/keypoint_classifier.keras")
np.save("models/static_class_labels.npy", lb.classes_)
print(f"\n✅  Model saved  →  models/keypoint_classifier.keras")
print(f"    Classes ({len(lb.classes_)}): {list(lb.classes_)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle(f"MLP Training — {len(lb.classes_)} classes", fontsize=14)

axes[0].plot(history.history["loss"],     label="train", color="#4C9BE8")
axes[0].plot(history.history["val_loss"], label="val",   color="#E8844C", linestyle="--")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history["accuracy"],     label="train", color="#4CE87A")
axes[1].plot(history.history["val_accuracy"], label="val",   color="#E84C6B", linestyle="--")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("docs/figures/static_training_curves.png", dpi=150)
print("📊  Training curves → docs/figures/static_training_curves.png")
plt.show()
