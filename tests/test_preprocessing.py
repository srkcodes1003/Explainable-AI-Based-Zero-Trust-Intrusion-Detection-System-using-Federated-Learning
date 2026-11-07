"""
Phase 3 Data Preprocessing & Leakage Prevention Test Suite.
Verifies processed partitions, missing/infinite value handling, artifact persistence, and feature consistency.
"""

import pytest
import joblib
import pandas as pd
import numpy as np
from config.settings import (
    TRAIN_DATA_FILE,
    VAL_DATA_FILE,
    TEST_DATA_FILE,
    SCALER_FILE,
    LABEL_ENCODER_FILE,
    METADATA_FILE,
)


def test_processed_files_exist():
    """Verify all expected preprocessed CSVs and metadata exist on disk."""
    assert TRAIN_DATA_FILE.exists(), f"Missing {TRAIN_DATA_FILE}"
    assert VAL_DATA_FILE.exists(), f"Missing {VAL_DATA_FILE}"
    assert TEST_DATA_FILE.exists(), f"Missing {TEST_DATA_FILE}"
    assert METADATA_FILE.exists(), f"Missing {METADATA_FILE}"


def test_processed_data_integrity():
    """Verify clean datasets: no NaNs, no Infs, correct numeric types."""
    df_train = pd.read_csv(TRAIN_DATA_FILE, nrows=5000)
    df_test = pd.read_csv(TEST_DATA_FILE, nrows=2000)
    
    # Check dimensions
    assert df_train.shape[1] == df_test.shape[1], "Train and Test feature counts must match."
    assert "Label" in df_train.columns
    assert "Label" in df_test.columns

    # No NaN or Inf
    assert not df_train.isna().any().any(), "Train partition contains unexpected NaN values."
    assert not df_test.isna().any().any(), "Test partition contains unexpected NaN values."

    numeric_cols = [c for c in df_train.columns if c != "Label"]
    assert np.all(np.isfinite(df_train[numeric_cols].to_numpy())), "Train partition contains infinite values."
    assert np.all(np.isfinite(df_test[numeric_cols].to_numpy())), "Test partition contains infinite values."


def test_scaler_and_encoder_artifacts():
    """Verify fitted scaler and label encoder load and function properly."""
    assert SCALER_FILE.exists(), f"Missing scaler at {SCALER_FILE}"
    assert LABEL_ENCODER_FILE.exists(), f"Missing encoder at {LABEL_ENCODER_FILE}"

    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(LABEL_ENCODER_FILE)

    assert hasattr(scaler, "mean_"), "Loaded scaler was not properly fitted."
    assert len(encoder.classes_) >= 2, "Label encoder must have at least 2 target classes."
    assert "BENIGN" in encoder.classes_

