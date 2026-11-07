"""
Phase 4 Machine Learning Intrusion Detection Tests.
Verifies model persistence, prediction interfaces, evaluation metrics, and chart outputs.
"""

import pytest
import joblib
import pandas as pd
import numpy as np
from config.settings import (
    SGD_MODEL_FILE,
    RF_MODEL_FILE,
    RESULTS_DIR,
    TEST_DATA_FILE,
)


def test_models_exist():
    """Verify both trained model artifacts exist on disk."""
    assert SGD_MODEL_FILE.exists(), f"Missing SGD model: {SGD_MODEL_FILE}"
    assert RF_MODEL_FILE.exists(), f"Missing RF model: {RF_MODEL_FILE}"


def test_model_predictions():
    """Verify models can load and predict on unseen samples."""
    sgd = joblib.load(SGD_MODEL_FILE)
    rf = joblib.load(RF_MODEL_FILE)

    df_test = pd.read_csv(TEST_DATA_FILE, nrows=20)
    X = df_test.drop(columns=["Label"]).to_numpy(dtype=np.float32)

    pred_sgd = sgd.predict(X)
    pred_rf = rf.predict(X)

    assert len(pred_sgd) == 20
    assert len(pred_rf) == 20
    assert np.all(np.isin(pred_sgd, [0, 1, 2, 3]))
    assert np.all(np.isin(pred_rf, [0, 1, 2, 3]))


def test_evaluation_outputs_exist():
    """Verify evaluation reports and Matplotlib figures exist."""
    comp_csv = RESULTS_DIR / "model_comparison.csv"
    rep_txt = RESULTS_DIR / "classification_report.txt"
    cm_png = RESULTS_DIR / "confusion_matrix.png"
    perf_png = RESULTS_DIR / "performance_metrics.png"

    assert comp_csv.exists() and comp_csv.stat().st_size > 0
    assert rep_txt.exists() and rep_txt.stat().st_size > 0
    assert cm_png.exists() and cm_png.stat().st_size > 1024
    assert perf_png.exists() and perf_png.stat().st_size > 1024

