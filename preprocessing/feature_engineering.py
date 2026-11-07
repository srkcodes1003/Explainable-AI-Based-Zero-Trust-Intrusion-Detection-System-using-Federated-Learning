"""
Feature Engineering & Selection module for Network Intrusion Detection.
Handles constant feature detection, zero-variance pruning, and non-informative column removal.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

# Columns that represent identifiers or metadata rather than statistical network flow features
NON_INFORMATIVE_CANDIDATES = [
    "flow id",
    "source ip",
    "src ip",
    "destination ip",
    "dst ip",
    "timestamp",
    "simillarhttp",
    "unnamed: 0",
]


def identify_non_informative_columns(columns: List[str]) -> List[str]:
    """Find identifier or metadata columns that cause overfitting or data leakage."""
    drop_cols = []
    for col in columns:
        col_lower = col.strip().lower()
        for cand in NON_INFORMATIVE_CANDIDATES:
            if cand == col_lower or cand in col_lower:
                drop_cols.append(col)
                break
    return list(set(drop_cols))


def detect_constant_features(df_train: pd.DataFrame, label_col: str, threshold: float = 1e-8) -> List[str]:
    """
    Detect constant or quasi-constant features on the training set ONLY.
    Features with variance <= threshold across training flows carry zero predictive information.
    """
    feature_cols = [c for c in df_train.columns if c != label_col]
    variances = df_train[feature_cols].var(numeric_only=True)
    constant_features = variances[variances <= threshold].index.tolist()
    return constant_features

