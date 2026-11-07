# Explainable AI-Based Zero Trust Intrusion Detection System Using Federated Learning

A modular, privacy-preserving, explainable Machine Learning Intrusion Detection System (IDS) engineered for enterprise Zero Trust perimeters, trained collaboratively across distributed organizations using Flower Federated Learning.

Designed and implemented for an academic **B.Tech Computer Science Major Project**, strictly optimized for standard Windows machines with 8 GB RAM.

---

## 1. System Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │          Central Aggregation Server          │
                  │              (Flower FedAvg)                 │
                  └──────┬────────────────┬───────────────┬──────┘
                         │                │               │
        Global Weights   │                │               │  Global Weights
         Distribution    │                │               │   Distribution
                         ▼                ▼               ▼
                 ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
                 │ Client Org A  │ │ Client Org B  │ │ Client Org C  │
                 │(Campus Network│ │(Cloud Cluster)│ │ (Branch Edge) │
                 └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                         │                 │                 │
                  Local Gradient    Local Gradient    Local Gradient
                     Updates           Updates           Updates
                         │                 │                 │
                         └─────────────────┴─────────────────┘
                                           │
                                           ▼
                                ┌─────────────────────┐
                                │ Global Evaluation   │
                                │ (Test Set: 24,000)  │
                                └─────────────────────┘
```

### Core Tenets:
1. **Zero Raw Data Sharing:** Raw network packets and flow logs never leave client institutional boundaries. Only serialized mathematical weight vectors (`coef_` and `intercept_`) are exchanged.
2. **Zero Data Leakage:** Preprocessing transformations (median imputation and feature scaling) are fitted strictly on the training partition before distribution.
3. **8 GB RAM Optimization:** Chunk-based reading (5,000–10,000 records), `float32/float64` precision optimization, and non-daemonized Flower aggregation ensure zero memory starvation.
4. **Minority Class Preservation:** Stratified sampling preserves 100% of scarce attack categories (e.g. Web Attacks).

---

## 2. Project Directory Structure

```text
Explainable_AI_Zero_Trust_IDS/
│
├── config/
│   ├── __init__.py
│   └── settings.py                   # Central configuration & hyperparameters
│
├── dataset/
│   ├── original/                     # Raw CIC-IDS2017 CSV files
│   ├── processed/                    # Preprocessed train, validation, test splits
│   │   ├── train.csv                 # 83,999 flow records (70%)
│   │   ├── validation.csv            # 12,000 flow records (10%)
│   │   └── test.csv                  # 24,000 flow records (20%)
│   ├── clients/                      # Decentralized client partitions
│   │   ├── client_1.csv              # Organization A (Campus Network)
│   │   ├── client_2.csv              # Organization B (Cloud Cluster)
│   │   ├── client_3.csv              # Organization C (Branch Edge)
│   │   └── partition_metadata.json
│   └── download_dataset.py           # Automated academic mirror downloader
│
├── preprocessing/
│   ├── __init__.py
│   ├── explore_dataset.py            # Dynamic CSV profiling & statistics
│   ├── clean_data.py                 # Header cleanup, deduplication, float casting
│   ├── feature_engineering.py        # Constant feature & ID column removal
│   ├── sampling.py                   # Stratified reservoir sampling
│   └── preprocess_pipeline.py        # Zero-leakage transformation pipeline
│
├── models/
│   ├── __init__.py
│   ├── train_sgd.py                  # Primary federated-compatible SGDClassifier
│   ├── train_random_forest.py        # Baseline comparison Random Forest
│   ├── evaluate.py                   # Metrics calculation & graph generation
│   └── saved/                        # Joblib persisted models & scalers
│       ├── sgd_model.pkl
│       ├── random_forest_model.pkl
│       ├── federated_global_model.pkl
│       ├── scaler.pkl
│       └── label_encoder.pkl
│
├── federated_learning/
│   ├── __init__.py
│   ├── client.py                     # Flower IDSFlowerClient (NumPyClient)
│   ├── strategy.py                   # Custom FedAvg & global server evaluation
│   ├── partition_data.py             # IID & Dirichlet Non-IID partitioner
│   ├── server.py                     # Central aggregation server entrypoint
│   └── train_federated.py            # Federated training engine
│
├── results/
│   ├── reports/                      # Text & CSV analytical reports
│   │   ├── dataset_exploration_report.txt
│   │   ├── preprocessing_report.txt
│   │   ├── classification_report.txt
│   │   ├── model_comparison.csv
│   │   ├── communication_rounds.csv
│   │   └── federated_vs_centralized.csv
│   └── graphs/                       # Publication-ready visualizations
│       ├── confusion_matrix.png
│       ├── performance_metrics.png
│       ├── federated_accuracy_curve.png
│       └── federated_vs_centralized.png
│
├── tests/                            # Automated test suite (20 unit & integration tests)
│   ├── test_environment.py
│   ├── test_exploration.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_federated_learning.py
│
├── check_environment.py              # Environment diagnostic script
├── main.py                           # Master CLI entrypoint
├── requirements.txt                  # Pinned tested dependency versions
├── DEVELOPMENT_LOG.md                # Development trajectory and audit trail
└── README.md                         # Project documentation
```

---

## 3. Installation & Setup (Windows 10/11)

### Step 1: Open PowerShell
Open Windows PowerShell in the project directory:
```powershell
cd "c:\Users\Hp\Desktop\Major Project (Project)"
```

### Step 2: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Verify Environment & Dependencies
```powershell
python check_environment.py
```
*Expected output:* `Status: Environment setup completed successfully.`

---

## 4. Execution Workflow

The project is structured with a unified command-line interface via `main.py`:

### Phase 2: Explore Dataset
Dynamically identifies all CSV files, counts normal/attack records, detects nulls, infinities, and estimated memory:
```powershell
python main.py --phase explore
```

### Phase 3: Preprocess Dataset
Performs chunk-based cleaning, stratified sampling, zero-leakage splitting, zero-variance pruning, and standardization:
```powershell
python main.py --phase preprocess
```

### Phase 4: Train Machine Learning Baseline Models
Trains the primary `SGDClassifier` and baseline `RandomForestClassifier`, evaluating both on the independent 24,000-sample test set:
```powershell
python main.py --phase train
```

### Phase 5: Run Flower Federated Learning
Simulates 3 decentralized client organizations collaborating via Flower `FedAvg`:
```powershell
python main.py --phase federated --rounds 5 --clients 3
```

### Run Entire Pipeline End-to-End
```powershell
python main.py --phase all
```

### Run Automated Quality Assurance Test Suite
Executes all 20 tests across all 5 phases:
```powershell
python -m pytest tests/ -v
```

---

## 5. Experimental Results

All reported metrics are generated from actual execution on genuine CIC-IDS2017 network traffic flows:

### 1. Centralized Machine Learning Benchmarks (Independent Test Set: 24,000 flows)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | False Positive Rate | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SGDClassifier** (Log Loss) | 98.08% | 0.9144 | 0.9637 | 0.9353 | 4.72% | 0.001 ms/sample |
| **Random Forest** (50 Trees) | 99.94% | 0.9974 | 0.9958 | 0.9966 | 0.04% | 0.021 ms/sample |

### 2. Federated Learning Convergence (Flower FedAvg across 3 Clients)

| Communication Round | Participating Clients | Global Test Accuracy | Global Macro F1 | Global Cross-Entropy Loss |
| :---: | :---: | :---: | :---: | :---: |
| **Round 1** | 3 / 3 (100%) | 93.73% | 0.8257 | 0.2177 |
| **Round 2** | 3 / 3 (100%) | 98.14% | 0.9344 | 0.1728 |
| **Round 3** | 3 / 3 (100%) | 98.11% | 0.9204 | 0.1346 |
| **Round 4** | 3 / 3 (100%) | 97.28% | 0.9134 | 0.1354 |
| **Round 5** | 3 / 3 (100%) | **99.01%** | **0.9602** | **0.1087** |

### 3. Centralized vs Federated Learning Paradigm Comparison

| Paradigm | Data Privacy | Raw Data Transferred | Test Accuracy | Macro F1 |
| :--- | :--- | :--- | :---: | :---: |
| **Centralized Baseline** | None (Vulnerable to central breach) | Full dataset (83,999 records) | 98.08% | 0.9353 |
| **Federated Learning (FedAvg)** | **High (Zero raw data exposure)** | **0 records (Parameter vectors only)** | **99.01%** | **0.9602** |

---

## 6. Future Project Extensions

The modular codebase is architected with placeholder packages for seamless Phase 6–8 integration:

### Phase 6: Explainable AI (`explainable_ai/`)
- Compute SHAP (SHapley Additive exPlanations) values on federated model predictions.
- Identify top contributing network features (e.g., `Destination Port`, `Flow Duration`, `Packet Length Mean`) per detection.

### Phase 7: Zero Trust Security Engine (`zero_trust/`)
- Contextual risk scoring based on prediction confidence and anomaly distance.
- Dynamic session throttling, multi-factor re-authentication triggers, and perimeter enforcement.

### Phase 8: Interactive Monitoring Dashboard (`backend/` & `frontend/`)
- Lightweight Flask REST API.
- Live traffic visualization, attack telemetry, and real-time federated client node status.

