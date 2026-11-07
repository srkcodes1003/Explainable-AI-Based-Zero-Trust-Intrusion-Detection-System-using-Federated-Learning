"""
Main Preprocessing Pipeline for CIC-IDS2017 Intrusion Detection System.
Guarantees zero data leakage by splitting datasets BEFORE fitting any scaler or imputer.
Optimized for 8 GB RAM.
"""

import sys
import gc
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    TRAIN_DATA_FILE,
    VAL_DATA_FILE,
    TEST_DATA_FILE,
    METADATA_FILE,
    SCALER_FILE,
    LABEL_ENCODER_FILE,
    FEATURE_NAMES_FILE,
    REPORTS_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    VAL_RATIO,
    TEST_RATIO,
)
from preprocessing.sampling import collect_stratified_sample
from preprocessing.feature_engineering import (
    identify_non_informative_columns,
    detect_constant_features,
)


def run_preprocessing_pipeline(
    target_total: int = 120000,
    dev_mode: bool = False,
    random_seed: int = RANDOM_SEED,
):
    """
    Execute end-to-end preprocessing pipeline:
    1. Stratified sampling across files
    2. Drop non-informative columns
    3. Strict Train / Validation / Test split
    4. Detect constant features on Train ONLY
    5. Fit SimpleImputer on Train ONLY, transform Train, Val, Test
    6. Fit StandardScaler on Train ONLY, transform Train, Val, Test
    7. Fit LabelEncoder on Train, encode Train, Val, Test
    8. Save processed datasets, fitted artifacts, and comprehensive report
    """
    print("\n" + "=" * 70)
    print("PHASE 3: CIC-IDS2017 PREPROCESSING & FEATURE PIPELINE")
    print("=" * 70)

    # Step 1: Collect clean stratified sample
    raw_df = collect_stratified_sample(
        target_total=target_total,
        dev_mode=dev_mode,
        random_seed=random_seed,
    )

    # Step 2: Remove non-informative / identifier columns
    drop_candidates = identify_non_informative_columns(list(raw_df.columns))
    if drop_candidates:
        print(f"[*] Dropping {len(drop_candidates)} non-informative identifier columns: {drop_candidates}")
        raw_df = raw_df.drop(columns=drop_candidates, errors="ignore")

    # Step 3: Strict Data Leakage Prevention - Split BEFORE fitting any transformation
    # 70% Train, 10% Validation, 20% Test
    print("\n[*] Splitting dataset into Train (70%), Validation (10%), Test (20%)...")
    
    # Check minimum class size for stratified split
    class_counts = raw_df["Label"].value_counts()
    print("    Class distribution before split:")
    for lbl, count in class_counts.items():
        print(f"      - {lbl:<20}: {count:,}")

    # Use stratified split on classes with >= 5 records
    stratify_col = raw_df["Label"] if class_counts.min() >= 5 else None

    df_train_val, df_test = train_test_split(
        raw_df,
        test_size=TEST_RATIO,
        random_state=random_seed,
        stratify=stratify_col,
    )

    val_relative_ratio = VAL_RATIO / (1.0 - TEST_RATIO)  # 0.10 / 0.80 = 0.125
    stratify_train_val = df_train_val["Label"] if df_train_val["Label"].value_counts().min() >= 5 else None

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_relative_ratio,
        random_state=random_seed,
        stratify=stratify_train_val,
    )

    # Free memory
    del raw_df, df_train_val
    gc.collect()

    print(f"    Train size:      {len(df_train):,} rows")
    print(f"    Validation size: {len(df_val):,} rows")
    print(f"    Test size:       {len(df_test):,} rows")

    # Separate X (features) and y (target)
    label_col = "Label"
    feature_cols = [c for c in df_train.columns if c != label_col]

    X_train = df_train[feature_cols].copy()
    y_train = df_train[label_col].copy()

    X_val = df_val[feature_cols].copy()
    y_val = df_val[label_col].copy()

    X_test = df_test[feature_cols].copy()
    y_test = df_test[label_col].copy()

    # Step 4: Detect constant features on TRAIN ONLY
    constant_features = detect_constant_features(df_train, label_col=label_col)
    if constant_features:
        print(f"[*] Detected {len(constant_features)} constant features on Train set. Pruning:")
        for cf in constant_features[:10]:
            print(f"      - {cf}")
        if len(constant_features) > 10:
            print(f"      ... and {len(constant_features) - 10} more.")

        X_train = X_train.drop(columns=constant_features)
        X_val = X_val.drop(columns=constant_features)
        X_test = X_test.drop(columns=constant_features)
        feature_cols = [c for c in feature_cols if c not in constant_features]

    # Step 5: Missing value imputation - Fit on TRAIN ONLY
    print("[*] Fitting SimpleImputer (median) on Train set only...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # Step 6: Feature Standardization - Fit on TRAIN ONLY
    print("[*] Fitting StandardScaler on Train set only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Step 7: Label Encoding
    print("[*] Encoding target labels consistently across sets...")
    label_encoder = LabelEncoder()
    # Fit encoder on train labels (ensure all classes in val/test exist in train)
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Handle rare unseen classes safely
    known_classes = set(label_encoder.classes_)
    y_val_clean = y_val.apply(lambda x: x if x in known_classes else "BENIGN")
    y_test_clean = y_test.apply(lambda x: x if x in known_classes else "BENIGN")
    y_val_encoded = label_encoder.transform(y_val_clean)
    y_test_encoded = label_encoder.transform(y_test_clean)

    label_mapping = {int(idx): str(cls_name) for idx, cls_name in enumerate(label_encoder.classes_)}
    print("    Label Mapping:")
    for idx, cls_name in label_mapping.items():
        print(f"      [{idx}] -> {cls_name}")

    # Reconstruct processed DataFrames
    df_train_proc = pd.DataFrame(X_train_scaled, columns=feature_cols, dtype=np.float32)
    df_train_proc["Label"] = y_train_encoded.astype(np.int32)

    df_val_proc = pd.DataFrame(X_val_scaled, columns=feature_cols, dtype=np.float32)
    df_val_proc["Label"] = y_val_encoded.astype(np.int32)

    df_test_proc = pd.DataFrame(X_test_scaled, columns=feature_cols, dtype=np.float32)
    df_test_proc["Label"] = y_test_encoded.astype(np.int32)

    # Step 8: Save Processed Datasets
    print(f"\n[*] Saving processed datasets to: {TRAIN_DATA_FILE.parent}...")
    TRAIN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_train_proc.to_csv(TRAIN_DATA_FILE, index=False)
    df_val_proc.to_csv(VAL_DATA_FILE, index=False)
    df_test_proc.to_csv(TEST_DATA_FILE, index=False)
    print(f"    - Train dataset saved:      {TRAIN_DATA_FILE} ({len(df_train_proc):,} rows)")
    print(f"    - Validation dataset saved: {VAL_DATA_FILE} ({len(df_val_proc):,} rows)")
    print(f"    - Test dataset saved:       {TEST_DATA_FILE} ({len(df_test_proc):,} rows)")

    # Step 9: Save Fitted Preprocessing Artifacts
    # Save both in models/saved/ and models/ directly as requested
    for target_scaler_file in [SCALER_FILE, MODELS_DIR / "scaler.pkl"]:
        joblib.dump(scaler, target_scaler_file)
    for target_encoder_file in [LABEL_ENCODER_FILE, MODELS_DIR / "label_encoder.pkl"]:
        joblib.dump(label_encoder, target_encoder_file)
    
    with open(FEATURE_NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    metadata = {
        "num_features": len(feature_cols),
        "features": feature_cols,
        "constant_features_removed": constant_features,
        "label_mapping": label_mapping,
        "train_rows": len(df_train_proc),
        "val_rows": len(df_val_proc),
        "test_rows": len(df_test_proc),
        "random_seed": random_seed,
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Step 10: Generate Comprehensive Preprocessing Report
    generate_preprocessing_report(
        df_train_proc,
        df_val_proc,
        df_test_proc,
        feature_cols,
        constant_features,
        label_mapping,
    )

    print("\n[+] Phase 3 Preprocessing Pipeline executed successfully.")
    return df_train_proc, df_val_proc, df_test_proc


def generate_preprocessing_report(
    train_df, val_df, test_df, features, constant_dropped, label_map
):
    """Save preprocessing report to both requested report locations."""
    lines = []
    lines.append("=" * 80)
    lines.append("CIC-IDS2017 DATA PREPROCESSING & LEAKAGE PREVENTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Final Retained Features Count: {len(features)}")
    lines.append(f"Constant/Zero-Variance Features Removed: {len(constant_dropped)}")
    if constant_dropped:
        lines.append("  Removed Features: " + ", ".join(constant_dropped))
    lines.append("-" * 80)
    lines.append("DATASET PARTITION SIZES:")
    lines.append(f"  Training Set:   {len(train_df):,} records ({round(len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100, 1)}%)")
    lines.append(f"  Validation Set: {len(val_df):,} records ({round(len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100, 1)}%)")
    lines.append(f"  Test Set:       {len(test_df):,} records ({round(len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100, 1)}%)")
    lines.append(f"  Total Processed: {len(train_df)+len(val_df)+len(test_df):,} records")
    lines.append("-" * 80)
    lines.append("LABEL MAPPING & CLASS DISTRIBUTION ACROSS PARTITIONS:")
    lines.append(f"{'Class ID':<10} | {'Class Name':<22} | {'Train':<12} | {'Val':<10} | {'Test'}")
    lines.append("-" * 80)
    
    train_counts = train_df["Label"].value_counts().to_dict()
    val_counts = val_df["Label"].value_counts().to_dict()
    test_counts = test_df["Label"].value_counts().to_dict()

    for idx, name in label_map.items():
        tr = train_counts.get(idx, 0)
        va = val_counts.get(idx, 0)
        te = test_counts.get(idx, 0)
        lines.append(f"{idx:<10} | {name:<22} | {tr:<12,} | {va:<10,} | {te:<10,}")

    lines.append("=" * 80)
    lines.append("DATA LEAKAGE PREVENTION AUDIT:")
    lines.append("1. Split Timing: Partitioned train/val/test BEFORE fitting imputer and scaler.")
    lines.append("2. Imputation: SimpleImputer(median) fit strictly on train_df.")
    lines.append("3. Scaling: StandardScaler fit strictly on train_df.")
    lines.append("4. Feature Selection: Zero-variance detection computed on train_df only.")
    lines.append("5. Reproducibility: Pinned seed used across all splits and sampling steps.")
    lines.append("=" * 80)

    report_content = "\n".join(lines)
    
    # Save to both requested locations
    for report_path in [
        RESULTS_DIR / "preprocessing_report.txt",
        REPORTS_DIR / "preprocessing_report.txt",
    ]:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

    print(f"[+] Preprocessing report saved to {RESULTS_DIR / 'preprocessing_report.txt'}")


if __name__ == "__main__":
    run_preprocessing_pipeline()

