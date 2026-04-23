import pandas as pd
import numpy as np
import os
import glob


GESTURE_LABELS = {
    0: "open_hand",
    1: "fist",
    2: "thumbs_up",
    3: "thumbs_down",
    4: "peace",
    5: "ok",
    6: "pointing_up",
    7: "pointing_down",
}

DYNAMIC_LABELS = {
    0: "swipe_left",
    1: "swipe_right",
    2: "zoom_in",
    3: "zoom_out",
}


def normalize_landmarks(hand_landmarks):
    """
    Normalize 21 hand landmarks (NormalizedLandmark list) relative to wrist.
    Returns list of (x, y, z) tuples, scaled so max_val = 1.
    Works with the new mediapipe.tasks NormalizedLandmark objects.
    """
    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
    wx, wy, wz = pts[0]
    pts = [(x - wx, y - wy, z - wz) for x, y, z in pts]
    max_val = max(abs(v) for pt in pts for v in pt) + 1e-6
    pts = [(x / max_val, y / max_val, z / max_val) for x, y, z in pts]
    return pts


def landmarks_to_flat(pts):
    """Flatten list of (x, y, z) tuples to a 1D list of 63 floats."""
    return [v for pt in pts for v in pt]


def extract_advanced_features(X_flat):
    """
    Given an (N, 63) array or a simple list of 63 coordinates,
    computes 210 pairwise distances to build a rich (N, 273) shape descriptor.
    This makes classification extremely accurate!
    """
    is_1d = False
    if isinstance(X_flat, list):
        X_flat = np.array([X_flat], dtype=np.float32)
        is_1d = True
    elif X_flat.ndim == 1:
        X_flat = X_flat.reshape(1, -1)
        is_1d = True
        
    N = X_flat.shape[0]
    pts = X_flat.reshape((N, 21, 3))
    
    features = [X_flat]
    
                                                                        
                                                    
    diff = pts[:, :, np.newaxis, :] - pts[:, np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)                     
    
                                                       
    i_indices, j_indices = np.triu_indices(21, k=1)
    distances = dist_matrix[:, i_indices, j_indices]                  
    
    features.append(distances)
    res = np.hstack(features)
    
    return res[0] if is_1d else res


def load_static_dataset(data_dir="data/static"):
    """Load all static gesture CSVs into X (N, 63) and y (N,) arrays."""
    all_dfs = []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "Run collect_static_gestures.py first."
        )
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
    data = pd.concat(all_dfs, ignore_index=True)
    
                                                
    to_drop = ["gesture_id", "label"]
    existing_to_drop = [c for c in to_drop if c in data.columns]
    X = data.drop(existing_to_drop, axis=1).values.astype(np.float32)
    
                              
    y = data["gesture_id"].values.astype(int)
    
    print(f"[static] Loaded {X.shape[0]} samples across {len(set(y))} classes.")
    return X, y


def load_dynamic_dataset(data_dir="data/dynamic"):
    """Load all dynamic gesture .npy files into X (N, 30, 63) and y (N,) arrays."""
    X, y = [], []
    npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files found in '{data_dir}'. "
            "Run collect_dynamic_gestures.py first."
        )
    for npy_file in npy_files:
        arr = np.load(npy_file)                            
        gesture_id = int(os.path.basename(npy_file).split("_")[0])
        X.append(arr)
        y.append(gesture_id)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=int)
    print(f"[dynamic] Loaded {X.shape[0]} sequences, shape {X.shape}.")
    return X, y
