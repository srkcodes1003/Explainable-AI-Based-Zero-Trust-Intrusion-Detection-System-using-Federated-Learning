# DEVELOPMENT LOG: Explainable AI-Based Zero Trust IDS Using Federated Learning

## Project Metadata
- **Project Title:** Explainable AI-Based Zero Trust Intrusion Detection System Using Federated Learning
- **Developer/Role:** Senior Python Developer, ML Engineer & Cybersecurity Architect
- **Environment:** Windows 11 (64-bit), Python 3.12.10 (tested & compatible with 3.11 specifications), Virtual Environment (`venv`), 8 GB RAM target
- **Dataset:** CIC-IDS2017 MachineLearningCSV

---

## Phase 1: Environment Setup
- **Objective:** Verify Python installation, create `venv`, define and install compatible pinned dependencies, and verify all imports.
- **Actions Taken:**
  - Workspace scanned. Located primary Python installation at `C:\Users\Hp\AppData\Local\Programs\Python\Python312\python.exe` (Python 3.12.10 64-bit).
  - Created isolated virtual environment `venv` in project directory.
  - Defined initial dependencies, resolved and installed binary wheels for all components:
    - `numpy==2.0.2`
    - `pandas==2.2.3`
    - `scikit-learn==1.5.2`
    - `imbalanced-learn==0.12.4`
    - `joblib==1.4.2`
    - `flwr==1.15.2`
    - `shap==0.46.0`
    - `matplotlib==3.9.4`
    - `seaborn==0.13.2`
    - `flask==3.0.3`
    - `pytest==8.3.5`
  - Created automated diagnostic tool `check_environment.py`.
  - Generated diagnostic report at `results/reports/environment_report.txt`.
  - Created `tests/test_environment.py` asserting Python version, Windows platform, and import/serialization of all core modules.
  - Ran pytest: 7 passed in 25.72s.
  - Pinned exact tested versions in `requirements.txt`.
  - Created central `config/settings.py` managing all global constants, filepaths, and 8 GB RAM optimization limits.
- **Test Results:** 100% passed (7/7 tests).
- **Status:** **Phase 1 Completed Successfully.**

---

## Phase 2: Dataset Collection & Exploration
- **Objective:** Dynamically discover CSV files under `dataset/original/`, analyze dimensions, class distributions, nulls, infinities, and memory requirements.
- **Actions Taken:**
  - Implemented `dataset/download_dataset.py` with multi-source verified download capability.
  - Obtained genuine CIC-IDS2017 flow data files (`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`, `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`, `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`).
  - Implemented `preprocessing/explore_dataset.py` with dynamic file discovery, chunked reading (10,000 rows/chunk), whitespace-tolerant label detection, and memory profiling.
  - Discovered 3 CSV files comprising **682,578 actual network flow records** across 79 features.
  - Identified:
    - Normal (BENIGN): 393,441 records (57.64%)
    - Malicious (Attacks): 289,137 records (42.36%)
    - Attack categories: PortScan (158,930), DDoS (128,027), Web Attack Brute Force (1,507), Web Attack XSS (652), Web Attack SQL Injection (21).
    - Missing values (39), Infinite values (1,041).
  - Generated comprehensive report at `results/reports/dataset_exploration_report.txt`.
  - Created and ran `tests/test_exploration.py`: 3 passed in 7.35s.
- **Status:** **Phase 2 Completed Successfully.**

---

## Phase 3: Data Preprocessing
- **Objective:** Chunk-based reading, column cleanup, infinity/missing handling, label encoding, no-leakage train/val/test splitting, StandardScaler fitting, metadata export.
- **Actions Taken:**
  - Implemented `preprocessing/clean_data.py`: whitespace stripping, column deduplication, unicode dash normalization, `float32/float64` type handling.
  - Implemented `preprocessing/feature_engineering.py`: constant feature detection (variance <= 1e-8) and non-informative column filtering.
  - Implemented `preprocessing/sampling.py`: chunk-based stratified sampling with 100% preservation of minority class `Web Attack` (2,180 records).
  - Implemented `preprocessing/preprocess_pipeline.py`:
    - Strict leakage prevention: partitioned dataset into Train (83,999 rows / 70%), Validation (12,000 rows / 10%), Test (24,000 rows / 20%) **before** fitting transformations.
    - Detected and pruned 10 constant features on the training set only.
    - Fitted `SimpleImputer(median)` and `StandardScaler` strictly on the training set.
    - Saved `train.csv`, `validation.csv`, `test.csv` in `dataset/processed/`.
    - Persisted `scaler.pkl` and `label_encoder.pkl` in `models/saved/` and `models/`.
    - Saved metadata in `dataset/processed/preprocessing_metadata.json`.
    - Generated report in `results/preprocessing_report.txt` and `results/reports/preprocessing_report.txt`.
  - Created and ran `tests/test_preprocessing.py`: 3 passed in 8.12s.
- **Status:** **Phase 3 Completed Successfully.**

---

## Phase 4: Machine Learning Intrusion Detection
- **Objective:** Train SGDClassifier (online/lightweight) and RandomForestClassifier (baseline comparison), evaluate with precision, recall, F1, confusion matrix, and performance graphs.
- **Actions Taken:**
  - Implemented `models/train_sgd.py`: trained SGDClassifier with `log_loss`, balanced class weights, convergence tracking.
    - Training time: 7.14s, Validation Accuracy: 97.95%, Pred time: 0.002 ms/sample.
  - Implemented `models/train_random_forest.py`: trained RandomForestClassifier (50 estimators, max depth 15, balanced subsample, 2 CPU workers).
    - Training time: 16.33s, Validation Accuracy: 99.92%, Pred time: 0.009 ms/sample.
  - Implemented `models/evaluate.py`: evaluated on independent 24,000-sample Test set:
    - SGDClassifier: 98.08% Accuracy, 0.9144 Macro Precision, 0.9637 Macro Recall, 0.9353 Macro F1, 4.72% FPR, 0.001 ms/sample inference latency.
    - Random Forest: 99.94% Accuracy, 0.9974 Macro Precision, 0.9958 Macro Recall, 0.9966 Macro F1, 0.04% FPR, 0.021 ms/sample inference latency.
  - Generated and saved artifacts:
    - `results/model_comparison.csv`
    - `results/classification_report.txt`
    - `results/confusion_matrix.png`
    - `results/performance_metrics.png`
  - Created and ran `tests/test_models.py`: 3 passed in 13.58s.
- **Status:** **Phase 4 Completed Successfully.**

---

## Phase 5: Federated Learning
- **Objective:** Flower-based 3-client simulated federated learning system with FedAvg, weight serialization, non-IID/IID evaluation, and global model benchmarking.
- **Actions Taken:**
  - Implemented `federated_learning/partition_data.py`: partitioned `train.csv` into 3 decentralized client files:
    - Client 1: Organization A (Campus Network) - 29,579 records
    - Client 2: Organization B (Cloud Cluster) - 47,897 records
    - Client 3: Organization C (Branch Edge) - 6,523 records
  - Implemented `federated_learning/client.py`: created `IDSFlowerClient` inheriting from `flwr.client.NumPyClient` with serialized weight exchanges (`coef_`, `intercept_`) and partial_fit batching.
  - Implemented `federated_learning/strategy.py`: customized Flower `FedAvg` with global evaluation against independent 24,000-sample test set.
  - Implemented `federated_learning/train_federated.py`: executed 5 communication rounds across 3 client organizations.
    - Global Test Accuracy converged from 93.73% (Round 1) to **99.01%** (Round 5).
    - Macro F1 reached **0.9602**, Cross-Entropy Loss reduced to **0.1087**.
    - Outperformed single centralized training baseline (98.08%) while guaranteeing zero raw flow data exposure.
  - Generated and saved artifacts:
    - `models/saved/federated_global_model.pkl`
    - `results/reports/federated_learning_report.txt`
    - `results/reports/communication_rounds.csv`
    - `results/reports/federated_vs_centralized.csv`
    - `results/graphs/federated_accuracy_curve.png`
    - `results/graphs/federated_vs_centralized.png`
  - Created and ran `tests/test_federated_learning.py`: 4 passed in 14.54s.
  - Integrated `main.py` CLI supporting `--phase explore`, `--phase preprocess`, `--phase train`, `--phase federated`, and `--phase all`.
  - Ran full test suite (`pytest tests/ -v`): 20 passed in 31.82s.
- **Status:** **Phase 5 Completed Successfully.**

---

## Complete Test Suite Status
- Total automated tests: 20
- Passed: 20 (100%)
- Failed: 0
- Execution duration: 31.82 seconds
