"""
train_dynamic.py
────────────────────────────────────────────────────────────────
Train a Bidirectional LSTM classifier on dynamic hand-gesture sequences.

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

# ── Data Augmentation for Dynamic Sequences ───────────────────────────────────
X_aug_list = [X]
y_aug_list = [y]

for _ in range(8):
    # 1. Time warping: randomly speed up/slow down parts of the sequence
    X_warped = X.copy()
    noise = np.random.normal(0, 0.015, size=X.shape)
    X_warped = X_warped + noise
    
    # 2. Random scaling
    scale = np.random.uniform(0.85, 1.15, size=(X.shape[0], 1, 1))
    X_warped = X_warped * scale
    
    X_aug_list.append(X_warped)
    y_aug_list.append(y)

# 3. Reverse time sequences (play gesture backwards)
X_reversed = X[:, ::-1, :]
X_aug_list.append(X_reversed)
y_aug_list.append(y)

X_all = np.vstack(X_aug_list).astype(np.float32)
y_all = np.concatenate(y_aug_list)

lb = LabelBinarizer()
y_ohe = lb.fit_transform(y_all)

# Handle very small datasets (e.g. during early collection)
test_size = 0.2 if len(X_all) > 10 else 0.1
try:
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_ohe, test_size=test_size, random_state=42, stratify=y_all
    )
except Exception:
    # If stratification still fails (e.g. single sample class), just split randomly
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_ohe, test_size=test_size, random_state=42)

print(f"Dataset Size: {len(X_all)} (original: {len(X)}) | Train: {X_train.shape}  |  Val: {X_val.shape}")

# ── Build Bidirectional LSTM with Attention ──────────────────────────────────
inp = layers.Input(shape=(30, 63))

# Attention via GlobalAveragePooling (no Lambda — serialization-safe)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
x = layers.Dropout(0.3)(x)
x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
x = layers.Dropout(0.3)(x)

# Weighted pooling via learned attention gate
attn = layers.Dense(1, activation='sigmoid')(x)
x = layers.Multiply()([x, attn])
x = layers.GlobalAveragePooling1D()(x)

x = layers.Dense(64, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu")(x)
out = layers.Dense(y_ohe.shape[1], activation="softmax")(x)

model = models.Model(inputs=inp, outputs=out, name="PointHistoryClassifier_v2")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_accuracy", patience=25,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=12, verbose=1),
        callbacks.ModelCheckpoint("models/point_history_classifier_best.keras",
                                  save_best_only=True, monitor="val_accuracy"),
    ],
    verbose=1,
)

# ── Save ──────────────────────────────────────────────────────────────────────
model.save("models/point_history_classifier.keras")
np.save("models/dynamic_class_labels.npy", lb.classes_)

best_val = max(history.history["val_accuracy"])
best_train = max(history.history["accuracy"])
print(f"\n[DONE] Dynamic model saved -> models/point_history_classifier.keras")
print(f"       Best Train Acc: {best_train:.4f}")
print(f"       Best Val   Acc: {best_val:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle(f"Dynamic Gesture BiLSTM+Attention | Best Val Acc: {best_val:.2%}", fontsize=14)

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
print("[SAVED] Training curves -> docs/figures/dynamic_training_curves.png")
plt.close()
