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
from preprocessing.feature_extractor import extract_advanced_features

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
    from preprocessing.feature_extractor import GESTURE_LABELS
    missing_mask = data["label"].isna()
    data.loc[missing_mask, "label"] = data.loc[missing_mask, "gesture_id"].map(GESTURE_LABELS)
    data["label"] = data["label"].fillna(data["gesture_id"].astype(str))
    X_raw = data.drop(["gesture_id", "label"], axis=1).values.astype(np.float32)
else:
    data["label"] = data["gesture_id"].astype(str)
    X_raw = data.drop("gesture_id", axis=1).values.astype(np.float32)

# Removed class grouping so the user can keep numbers and letters distinct!
# They will rely on making distinct poses during data collection to differentiate them.
y_raw_list = data["label"].astype(str).str.lower().values

# ── Step 3: Data Augmentation (Rotation, Jitter, Noise, Scaling) ──────────
X_aug_list = [X_raw]
y_final_list = [y_raw_list]

for _ in range(8): # Increased from 6 to 8 for more power!
    # 1. Random Rotation (+/- 15 degrees) around the Z-axis 
    # (Since landmarks are centered at wrist, simple 2D rotation on XY works great)
    angle = np.random.uniform(-15, 15) * (np.pi / 180.0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    # Reshape to (N, 21, 3) to apply rotation to each point
    pts = X_raw.reshape(-1, 21, 3)
    rotated_pts = pts.copy()
    rotated_pts[:, :, 0] = pts[:, :, 0] * cos_a - pts[:, :, 1] * sin_a
    rotated_pts[:, :, 1] = pts[:, :, 0] * sin_a + pts[:, :, 1] * cos_a
    X_rot = rotated_pts.reshape(X_raw.shape)

    # 2. 1.8% random gaussian noise
    noise = np.random.normal(loc=0.0, scale=0.018, size=X_rot.shape)
    
    # 3. Random size scaling between 85% and 115%
    scale = np.random.uniform(0.85, 1.15, size=(X_rot.shape[0], 1))
    
    noisy_scaled_X = (X_rot + noise) * scale
    X_aug_list.append(noisy_scaled_X)
    y_final_list.append(y_raw_list)

X_augmented = np.vstack(X_aug_list).astype(np.float32)
y_raw = np.concatenate(y_final_list)

# Now calculate angles and distances on the massive augmented dataset!
X = extract_advanced_features(X_augmented)


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
    layers.Input(shape=(273,)),
    layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.20),
    layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.20),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.15),
    layers.Dense(n_classes, activation="softmax")
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
