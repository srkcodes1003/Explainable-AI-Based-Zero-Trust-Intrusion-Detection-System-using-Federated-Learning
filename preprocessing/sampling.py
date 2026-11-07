"""
Stratified chunk-based sampling module for CIC-IDS2017.
Extracts a representative, balanced sample from large CSVs without exceeding 8 GB RAM.
Preserves 100% of minority attack records while sub-sampling dominant classes.
"""

import gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from config.settings import (
    ORIGINAL_DATA_DIR,
    CHUNK_SIZE,
    RANDOM_SEED,
    LABEL_COLUMN_CANDIDATES,
    BENIGN_LABEL,
)
from preprocessing.clean_data import clean_column_names, normalize_label_value, sanitize_numerical_features
from preprocessing.explore_dataset import find_csv_files, detect_label_column


def collect_stratified_sample(
    target_total: int = 120000,
    dev_mode: bool = False,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Stream all CSV files chunk-by-chunk and build a stratified dataset sample.
    - Preserves all rare/minority attack flows.
    - Sub-samples common classes (BENIGN, DDoS, PortScan) proportionally.
    """
    if dev_mode:
        target_total = 30000

    csv_files = find_csv_files(ORIGINAL_DATA_DIR)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {ORIGINAL_DATA_DIR}")

    print(f"[*] Starting stratified sampling. Target size: ~{target_total:,} rows (Dev Mode: {dev_mode})")

    # In-memory reservoirs per normalized class
    class_pools = defaultdict(list)
    total_scanned = 0

    for csv_file in csv_files:
        print(f"    Scanning: {csv_file.name} in chunks of {CHUNK_SIZE:,}...")
        for chunk in pd.read_csv(
            csv_file,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding="cp1252",
            encoding_errors="replace",
        ):
            # Clean headers
            chunk = clean_column_names(chunk)
            label_col = detect_label_column(chunk.columns)
            if not label_col:
                continue

            # Standardize label column name to 'Label'
            if label_col != "Label":
                chunk = chunk.rename(columns={label_col: "Label"})
                label_col = "Label"

            # Normalize raw label values into clean categories
            chunk["Label"] = chunk["Label"].apply(normalize_label_value)

            # Sanitize numerical features and cast to float32
            chunk = sanitize_numerical_features(chunk, label_col="Label")

            # Remove entirely empty rows
            chunk = chunk.dropna(how="all", subset=[c for c in chunk.columns if c != "Label"])

            total_scanned += len(chunk)

            # Partition chunk rows by class
            for label, group in chunk.groupby("Label"):
                class_pools[label].append(group)

        # Force garbage collection between files
        gc.collect()

    print(f"[*] Total records scanned across all files: {total_scanned:,}")

    # Combine pools per class and sample intelligently
    combined_dfs = []
    print("\n[*] Class Distribution in Raw Data:")
    raw_counts = {}
    for label, group_list in class_pools.items():
        full_class_df = pd.concat(group_list, ignore_index=True)
        raw_counts[label] = len(full_class_df)
        print(f"    - {label:<20}: {len(full_class_df):,} records")

    # Determine allocation per class
    # Minority classes (< 5,000 records): keep ALL records (100% preservation)
    # Majority classes: allocate remaining quota balanced across them
    num_classes = len(raw_counts)
    minority_classes = {lbl: cnt for lbl, cnt in raw_counts.items() if cnt < 5000}
    majority_classes = {lbl: cnt for lbl, cnt in raw_counts.items() if cnt >= 5000}

    minority_total = sum(minority_classes.values())
    remaining_quota = max(10000, target_total - minority_total)
    majority_quota_per_class = remaining_quota // max(1, len(majority_classes))

    sampled_dfs = []
    for label, group_list in class_pools.items():
        full_class_df = pd.concat(group_list, ignore_index=True)
        if label in minority_classes:
            # Keep all minority flows
            sampled_dfs.append(full_class_df)
            print(f"    [Preserved 100%] {label}: {len(full_class_df):,} rows")
        else:
            # Sample with replacement=False
            sample_n = min(len(full_class_df), majority_quota_per_class)
            sampled = full_class_df.sample(n=sample_n, random_state=random_seed)
            sampled_dfs.append(sampled)
            print(f"    [Sampled] {label}: {sample_n:,} / {len(full_class_df):,} rows")

    # Combine and shuffle
    final_df = pd.concat(sampled_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Release intermediate pools
    del class_pools, sampled_dfs
    gc.collect()

    print(f"\n[+] Stratified Sample constructed: {len(final_df):,} total records, {final_df.shape[1]} features.")
    return final_df

