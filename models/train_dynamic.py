"""
train_dynamic.py
────────────────────────────────────────────────────────────────
Train an LSTM classifier on dynamic hand-gesture sequences.

Output
------
  models/point_history_classifier.keras
  models/dynamic_class_labels.npy
  docs/figures/dynamic_training_curves.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from preprocessing.feature_extractor import load_dynamic_dataset, DYNAMIC_LABELS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

os.makedirs("models", exist_ok=True)
os.makedirs("docs/figures", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
X, y = load_dynamic_dataset()    # (N, 30, 63)

lb = LabelBinarizer()
y_ohe = lb.fit_transform(y)

# Handle very small datasets (e.g. during early collection)
test_size = 0.2 if len(X) > 10 else 0.1
try:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_ohe, test_size=test_size, random_state=42, stratify=y_ohe if len(X) > 10 else None
    )
except Exception:
    # If stratification still fails (e.g. single sample class), just split randomly
    X_train, X_val, y_train, y_val = train_test_split(X, y_ohe, test_size=test_size, random_state=42)

print(f"Dataset Size: {len(X)} | Train: {X_train.shape}  |  Val: {X_val.shape}")

# ── Build LSTM ────────────────────────────────────────────────────────────────
model = models.Sequential(
    [
        layers.Input(shape=(30, 63)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(y_ohe.shape[1], activation="softmax"),
    ],
    name="PointHistoryClassifier",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=16,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_accuracy", patience=20,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=10, verbose=1),
        callbacks.ModelCheckpoint("models/point_history_classifier_best.keras",
                                  save_best_only=True, monitor="val_accuracy"),
    ],
    verbose=1,
)

# ── Save ──────────────────────────────────────────────────────────────────────
model.save("models/point_history_classifier.keras")
np.save("models/dynamic_class_labels.npy", lb.classes_)
print("\n✅  Dynamic model saved → models/point_history_classifier.keras")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Dynamic Gesture LSTM — Training History", fontsize=14)

axes[0].plot(history.history["loss"], label="train loss", color="#4C9BE8")
axes[0].plot(history.history["val_loss"], label="val loss",  color="#E8844C", linestyle="--")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history["accuracy"], label="train acc", color="#4CE87A")
axes[1].plot(history.history["val_accuracy"], label="val acc", color="#E84C6B", linestyle="--")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("docs/figures/dynamic_training_curves.png", dpi=150)
print("📊  Training curves saved → docs/figures/dynamic_training_curves.png")
plt.show()
