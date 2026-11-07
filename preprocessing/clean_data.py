"""
Data cleaning utilities for CIC-IDS2017 network traffic flows.
Handles column standardization, duplicate columns, infinite values, and label normalization.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from config.settings import LABEL_COLUMN_CANDIDATES, ATTACK_MAPPING, BENIGN_LABEL


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names:
    - Strip leading/trailing whitespaces.
    - Remove duplicate columns by keeping the first occurrence.
    """
    df = df.copy()
    cleaned_cols = [c.strip() for c in df.columns]
    df.columns = cleaned_cols
    
    # Identify and drop duplicate column names if any
    unique_cols = []
    seen = set()
    for col in df.columns:
        if col not in seen:
            unique_cols.append(col)
            seen.add(col)
            
    df = df.loc[:, unique_cols]
    return df


def normalize_label_value(label_str: str) -> str:
    """
    Normalize raw attack label strings:
    - Removes non-ascii characters (e.g. replacement characters in 'Web Attack – Brute Force').
    - Standardizes case and aligns with ATTACK_MAPPING.
    """
    if pd.isna(label_str):
        return BENIGN_LABEL
        
    s = str(label_str).strip()
    # Normalize varied dash characters and unicode noise
    s_clean = re.sub(r"[^\x00-\x7F]+", " - ", s)
    s_clean = re.sub(r"\s+", " ", s_clean).strip()
    s_lower = s_clean.lower()
    
    # Check against known mapping
    for pattern, family in ATTACK_MAPPING.items():
        if pattern.lower() in s_lower or s_lower == pattern.lower():
            return family
            
    # If starting with Web Attack
    if "web attack" in s_lower:
        return "Web Attack"
    if "dos" in s_lower:
        return "DoS"
    if "ddos" in s_lower:
        return "DDoS"
    if "portscan" in s_lower or "port scan" in s_lower:
        return "PortScan"
    if "bot" in s_lower:
        return "Bot"
    if "infiltr" in s_lower:
        return "Infiltration"
    if "patator" in s_lower:
        return "Brute Force"
    if "benign" in s_lower:
        return BENIGN_LABEL
        
    return s_clean.upper()


def sanitize_numerical_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Sanitize dataframe:
    - Replaces +inf and -inf with np.nan.
    - Ensures numeric columns are cast to float32 for RAM efficiency.
    - Preserves the target label column as string.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != label_col]
    
    # Force convert to numeric, invalid parsing becomes NaN
    for c in feature_cols:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    # Replace positive and negative infinity with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # Cast to float32 to reduce memory footprint by 50% compared to float64
    df[feature_cols] = df[feature_cols].astype(np.float32)
    return df

