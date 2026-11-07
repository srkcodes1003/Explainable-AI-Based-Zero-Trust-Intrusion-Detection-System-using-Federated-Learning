"""
Training module for RandomForestClassifier.
Comparison baseline model for Intrusion Detection benchmarking.
Configured for resource-constrained (8 GB RAM) execution.
"""

import sys
import time
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    TRAIN_DATA_FILE,
    VAL_DATA_FILE,
    RF_MODEL_FILE,
    SAVED_MODELS_DIR,
    MODELS_DIR,
    RF_CONFIG,
)


def train_random_forest_model():
    """
    Train RandomForestClassifier on the processed training set.
    Records accurate training time and prediction latency.
    """
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST CLASSIFIER (BASELINE COMPARISON MODEL)")
    print("=" * 60)

    # Load preprocessed datasets
    df_train = pd.read_csv(TRAIN_DATA_FILE)
    df_val = pd.read_csv(VAL_DATA_FILE)

    X_train = df_train.drop(columns=["Label"]).to_numpy(dtype=np.float32)
    y_train = df_train["Label"].to_numpy(dtype=np.int32)

    X_val = df_val.drop(columns=["Label"]).to_numpy(dtype=np.float32)
    y_val = df_val["Label"].to_numpy(dtype=np.int32)

    print(f"[*] Training records: {len(X_train):,}, Validation records: {len(X_val):,}")

    clf = RandomForestClassifier(
        n_estimators=RF_CONFIG["n_estimators"],
        max_depth=RF_CONFIG["max_depth"],
        n_jobs=RF_CONFIG["n_jobs"],
        random_state=RF_CONFIG["random_state"],
        class_weight="balanced_subsample",  # Handles minority attack classes in trees
    )

    t0 = time.perf_counter()
    print("[*] Training Random Forest with 50 estimators...")
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t_pred0 = time.perf_counter()
    y_val_pred = clf.predict(X_val)
    pred_time = time.perf_counter() - t_pred0

    val_acc = float(np.mean(y_val_pred == y_val))
    print(f"[+] Random Forest Training Complete in: {train_time:.3f}s")
    print(f"[+] Prediction Time (Val {len(X_val)} rows): {pred_time:.4f}s ({pred_time/len(X_val)*1000:.3f} ms/sample)")
    print(f"[+] Validation Accuracy: {val_acc * 100:.2f}%")

    # Save model to both requested locations
    for path in [RF_MODEL_FILE, MODELS_DIR / "random_forest_model.pkl"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, path)
    print(f"[+] Saved trained model to {RF_MODEL_FILE}")

    return clf, train_time, pred_time


if __name__ == "__main__":
    train_random_forest_model()

