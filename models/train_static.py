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
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

os.makedirs("models", exist_ok=True)
os.makedirs("docs/figures", exist_ok=True)

                                                                               
print("Loading data from data/static/ ...")
all_dfs = []
for csv_file in sorted(glob.glob("data/static/*.csv")):
    df = pd.read_csv(csv_file)
    all_dfs.append(df)

if not all_dfs:
    print("No CSV files found in data/static/")
    print("    Run extract_landmarks_from_images.py (Kaggle) or collect_static_gestures.py first.")
    sys.exit(1)

data = pd.concat(all_dfs, ignore_index=True)

                                                                           
if "label" in data.columns:
    from preprocessing.feature_extractor import GESTURE_LABELS
    missing_mask = data["label"].isna()
    data.loc[missing_mask, "label"] = data.loc[missing_mask, "gesture_id"].map(GESTURE_LABELS)
    data["label"] = data["label"].fillna(data["gesture_id"].astype(str))
    X_raw = data.drop(["gesture_id", "label"], axis=1).values.astype(np.float32)
else:
    data["label"] = data["gesture_id"].astype(str)
    X_raw = data.drop("gesture_id", axis=1).values.astype(np.float32)

                                                                           
                                                                                       
y_raw_list = data["label"].astype(str).str.lower().values

                                                                               
print("\n--- Class Distribution (original data) ---")
unique_raw, counts_raw = np.unique(y_raw_list, return_counts=True)
underrepresented = set()
for lbl, cnt in sorted(zip(unique_raw, counts_raw), key=lambda x: x[1]):
    flag = " ⚠️  (< 250)" if cnt < 250 else ""
    print(f"  {lbl:<20} {cnt:>5}{flag}")
    if cnt < 250:
        underrepresented.add(lbl)
print(f"\n  Total: {len(y_raw_list)} samples | {len(unique_raw)} classes")

                                                                              
X_aug_list = [X_raw]
y_final_list = [y_raw_list]

for i in range(12):                                               
                                                          
    angle = np.random.uniform(-20, 20) * (np.pi / 180.0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    pts = X_raw.reshape(-1, 21, 3)
    rotated_pts = pts.copy()
    rotated_pts[:, :, 0] = pts[:, :, 0] * cos_a - pts[:, :, 1] * sin_a
    rotated_pts[:, :, 1] = pts[:, :, 0] * sin_a + pts[:, :, 1] * cos_a
    X_rot = rotated_pts.reshape(X_raw.shape)

                                                      
    noise = np.random.normal(loc=0.0, scale=0.020, size=X_rot.shape)
    
                                                 
    scale = np.random.uniform(0.80, 1.20, size=(X_rot.shape[0], 1))
    
    noisy_scaled_X = (X_rot + noise) * scale
    X_aug_list.append(noisy_scaled_X)
    y_final_list.append(y_raw_list)

                                                                   
pts_mirror = X_raw.reshape(-1, 21, 3).copy()
pts_mirror[:, :, 0] = -pts_mirror[:, :, 0]                      
X_aug_list.append(pts_mirror.reshape(X_raw.shape))
y_final_list.append(y_raw_list)

X_augmented = np.vstack(X_aug_list).astype(np.float32)
y_raw = np.concatenate(y_final_list)

                                                                      
X = extract_advanced_features(X_augmented)

lb    = LabelBinarizer()
y_ohe = lb.fit_transform(y_raw)

print(f"Dataset: {X.shape[0]} samples  |  {len(lb.classes_)} classes: {list(lb.classes_)}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y_ohe, test_size=0.2, random_state=42, stratify=y_raw
)
print(f"Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}\n")

y_train_idx = np.argmax(y_train, axis=1)

                                                                                
if SMOTE_AVAILABLE and underrepresented:
    under_indices    = {i for i, l in enumerate(lb.classes_) if l in underrepresented}
    unique_tr, counts_tr = np.unique(y_train_idx, return_counts=True)
    median_count     = int(np.median(counts_tr))
    sampling_strategy = {
        i: max(median_count, counts_tr[list(unique_tr).index(i)])
        for i in under_indices if i in unique_tr
    }
    smote            = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)
    X_train, y_train_idx = smote.fit_resample(X_train, y_train_idx)
    y_train          = lb.transform(lb.classes_[y_train_idx])
    print(f"  SMOTE applied — training set: {len(y_train_idx)} samples")

                                                                                
cw_arr         = compute_class_weight("balanced", classes=np.arange(y_train.shape[1]),
                                      y=np.argmax(y_train, axis=1))
class_weight_dict = dict(enumerate(cw_arr))

                                                                               
n_classes = y_ohe.shape[1]
n_features = X.shape[1]

inp = layers.Input(shape=(n_features,))

         
x = layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0003))(inp)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)

         
x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0003))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)

                
res = layers.Dense(128, activation="relu")(x)
res = layers.BatchNormalization()(res)
res = layers.Dropout(0.15)(res)
res = layers.Dense(256, activation="relu")(res)                       
x = layers.Add()([x, res])                   

         
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

         
x = layers.Dense(64, activation="relu")(x)
x = layers.BatchNormalization()(x)

out = layers.Dense(n_classes, activation="softmax")(x)

model = models.Model(inputs=inp, outputs=out, name="KeypointClassifier_v2")

                                                     
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=150 * (X_train.shape[0] // 64),
    alpha=1e-5
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),                    
    metrics=["accuracy"],
)
model.summary()

                                                                                
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_accuracy", patience=20,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint("models/keypoint_classifier_best.keras",
                                  save_best_only=True, monitor="val_accuracy"),
    ],
    verbose=1,
)

                                                                               
print("\n--- Per-Class Accuracy on Validation Set (worst → best) ---")
y_val_idx  = np.argmax(y_val, axis=1)
y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
rows = []
for i, lbl in enumerate(lb.classes_):
    mask  = y_val_idx == i
    total = mask.sum()
    acc   = (y_val_pred[mask] == i).sum() / total if total > 0 else None
    rows.append((lbl, acc, total))
for lbl, acc, total in sorted(rows, key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0)):
    acc_str = f"{acc*100:.1f}%" if acc is not None else "n/a"
    flag    = " ⚠️" if acc is not None and acc < 0.85 else ""
    print(f"  {lbl:<20} {acc_str:>7}  (n={total}){flag}")

                                                                                
model.save("models/keypoint_classifier.keras")
np.save("models/static_class_labels.npy", lb.classes_)

best_val = max(history.history["val_accuracy"])
best_train = max(history.history["accuracy"])
print(f"\n[DONE] Model saved -> models/keypoint_classifier.keras")
print(f"       Classes ({len(lb.classes_)}): {list(lb.classes_)}")
print(f"       Best Train Acc: {best_train:.4f}")
print(f"       Best Val   Acc: {best_val:.4f}")

                                                                                
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle(f"MLP Training - {len(lb.classes_)} classes | Best Val Acc: {best_val:.2%}", fontsize=14)

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
print("[SAVED] Training curves -> docs/figures/static_training_curves.png")
plt.close()
