"""
Training module for SGDClassifier (Stochastic Gradient Descent).
Primary lightweight model for centralized benchmarking and Flower federated aggregation.
Optimized for 8 GB RAM and supports incremental partial_fit.
"""

import sys
import time
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    TRAIN_DATA_FILE,
    VAL_DATA_FILE,
    SGD_MODEL_FILE,
    SAVED_MODELS_DIR,
    MODELS_DIR,
    SGD_CONFIG,
    LABEL_ENCODER_FILE,
)


def train_sgd_model(use_partial_fit: bool = False, batch_size: int = 5000):
    """
    Train SGDClassifier on the processed training set.
    Records accurate training time and prediction latency.
    """
    print("\n" + "=" * 60)
    print("TRAINING SGDCLASSIFIER (PRIMARY FEDERATED-COMPATIBLE MODEL)")
    print("=" * 60)

    # Load preprocessed datasets
    df_train = pd.read_csv(TRAIN_DATA_FILE)
    df_val = pd.read_csv(VAL_DATA_FILE)

    X_train = df_train.drop(columns=["Label"]).to_numpy(dtype=np.float32)
    y_train = df_train["Label"].to_numpy(dtype=np.int32)

    X_val = df_val.drop(columns=["Label"]).to_numpy(dtype=np.float32)
    y_val = df_val["Label"].to_numpy(dtype=np.int32)

    all_classes = np.unique(y_train)
    print(f"[*] Training records: {len(X_train):,}, Validation records: {len(X_val):,}")
    print(f"[*] Target Classes: {all_classes.tolist()}")

    # Initialize SGDClassifier with calibrated log_loss (logistic regression)
    clf = SGDClassifier(
        loss=SGD_CONFIG["loss"],
        penalty=SGD_CONFIG["penalty"],
        alpha=SGD_CONFIG["alpha"],
        max_iter=SGD_CONFIG["max_iter"],
        tol=SGD_CONFIG["tol"],
        learning_rate=SGD_CONFIG["learning_rate"],
        random_state=SGD_CONFIG["random_state"],
        class_weight="balanced",  # Critical for minority attack classes like Web Attack
    )

    t0 = time.perf_counter()
    if use_partial_fit:
        print(f"[*] Training via incremental partial_fit (Batch Size: {batch_size:,})...")
        num_batches = int(np.ceil(len(X_train) / batch_size))
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min(len(X_train), (b + 1) * batch_size)
            X_b = X_train[start_idx:end_idx]
            y_b = y_train[start_idx:end_idx]
            clf.partial_fit(X_b, y_b, classes=all_classes)
    else:
        print("[*] Training SGDClassifier across full training partition...")
        clf.fit(X_train, y_train)

    train_time = time.perf_counter() - t0

    # Measure prediction time on validation set
    t_pred0 = time.perf_counter()
    y_val_pred = clf.predict(X_val)
    pred_time = time.perf_counter() - t_pred0

    val_acc = float(np.mean(y_val_pred == y_val))
    print(f"[+] SGD Training Complete in: {train_time:.3f}s")
    print(f"[+] Prediction Time (Val {len(X_val)} rows): {pred_time:.4f}s ({pred_time/len(X_val)*1000:.3f} ms/sample)")
    print(f"[+] Validation Accuracy: {val_acc * 100:.2f}%")

    # Save model to both requested locations
    for path in [SGD_MODEL_FILE, MODELS_DIR / "sgd_model.pkl"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, path)
    print(f"[+] Saved trained model to {SGD_MODEL_FILE}")

    return clf, train_time, pred_time


if __name__ == "__main__":
    train_sgd_model()

