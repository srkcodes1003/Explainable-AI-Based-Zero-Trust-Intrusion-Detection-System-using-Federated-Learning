"""
Central configuration module for Explainable AI-Based Zero Trust IDS.
Contains all dataset paths, model hyperparameters, federated learning settings,
and hardware optimization thresholds.
"""

from pathlib import Path

# Base Paths (Relative to project root for portability across systems)
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
ORIGINAL_DATA_DIR = DATASET_DIR / "original"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
CLIENTS_DATA_DIR = DATASET_DIR / "clients"

MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = RESULTS_DIR / "reports"
GRAPHS_DIR = RESULTS_DIR / "graphs"

# Ensure all critical runtime directories exist
for directory in [
    ORIGINAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    CLIENTS_DATA_DIR,
    SAVED_MODELS_DIR,
    REPORTS_DIR,
    GRAPHS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Processed Dataset Files
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"
VAL_DATA_FILE = PROCESSED_DATA_DIR / "validation.csv"
METADATA_FILE = PROCESSED_DATA_DIR / "preprocessing_metadata.json"

# Saved Preprocessing Artifacts
SCALER_FILE = SAVED_MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_FILE = SAVED_MODELS_DIR / "label_encoder.pkl"
FEATURE_NAMES_FILE = SAVED_MODELS_DIR / "feature_names.json"

# Saved Model Artifacts
SGD_MODEL_FILE = SAVED_MODELS_DIR / "sgd_model.pkl"
RF_MODEL_FILE = SAVED_MODELS_DIR / "random_forest_model.pkl"
FEDERATED_MODEL_FILE = SAVED_MODELS_DIR / "federated_global_model.pkl"

# Sampling & Hardware Optimization (Strictly tuned for 8 GB RAM)
CHUNK_SIZE = 10000
DEV_SAMPLE_SIZE = 30000
TRAIN_TARGET_SIZE = 100000
EVAL_TARGET_SIZE = 20000
VAL_RATIO = 0.10
TEST_RATIO = 0.20
RANDOM_SEED = 42

# Label Standardization
LABEL_COLUMN_CANDIDATES = ["Label", "label", " Label", "Label ", "class", "Class"]
BENIGN_LABEL = "BENIGN"

# Attack Family Aggregation Mapping (Normalizes subtle naming variations)
ATTACK_MAPPING = {
    "benign": "BENIGN",
    "dos": "DoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "ddos": "DDoS",
    "portscan": "PortScan",
    "bot": "Bot",
    "infiltration": "Infiltration",
    "web attack – brute force": "Web Attack",
    "web attack – xss": "Web Attack",
    "web attack – sql injection": "Web Attack",
    "web attack - brute force": "Web Attack",
    "web attack - xss": "Web Attack",
    "web attack - sql injection": "Web Attack",
    "ftp-patator": "Brute Force",
    "ssh-patator": "Brute Force",
    "heartbleed": "Heartbleed",
}

# Machine Learning Hyperparameters
SGD_CONFIG = {
    "loss": "log_loss",  # Provides calibrated probability outputs for Zero Trust
    "penalty": "l2",
    "alpha": 1e-4,
    "max_iter": 1000,
    "tol": 1e-3,
    "learning_rate": "constant",
    "eta0": 0.05,
    "random_state": RANDOM_SEED,
}

RF_CONFIG = {
    "n_estimators": 50,  # Kept lightweight for 8 GB RAM
    "max_depth": 15,
    "n_jobs": 2,         # Limit concurrency to avoid CPU/RAM starvation
    "random_state": RANDOM_SEED,
}

# Federated Learning Configuration
FEDERATED_CONFIG = {
    "num_clients": 3,
    "rounds": 5,
    "local_epochs": 1,
    "batch_size": 128,
    "server_address": "127.0.0.1:8080",
    "distribution_strategy": "non_iid",  # Supports 'iid' and 'non_iid'
    "alpha_dirichlet": 0.5,             # Heterogeneity degree for Non-IID Dirichlet split
}

