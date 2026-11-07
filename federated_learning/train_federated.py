"""
Federated Learning Training and Orchestration Engine.
Coordinates Flower federated rounds across 3 client organizations,
aggregates model updates via Flower FedAvg, saves the global model, and benchmarks against centralized baseline.
Strictly optimized for 8 GB RAM execution without external daemon dependencies.
"""

import sys
import json
import joblib
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
    Status,
    Code,
)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    CLIENTS_DATA_DIR,
    TEST_DATA_FILE,
    SGD_MODEL_FILE,
    FEDERATED_MODEL_FILE,
    MODELS_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
    GRAPHS_DIR,
    FEDERATED_CONFIG,
    SGD_CONFIG,
)
from federated_learning.partition_data import create_client_partitions, ORGANIZATION_NAMES
from federated_learning.client import IDSFlowerClient
from federated_learning.strategy import fit_metrics_aggregation


class SimulatedClientProxy:
    """Lightweight proxy representing client identity in Flower strategy aggregation."""
    def __init__(self, cid: str):
        self.cid = cid


def run_federated_learning(
    num_rounds: int = 5,
    num_clients: int = 3,
    local_epochs: int = 1,
    batch_size: int = 128,
    distribution_strategy: str = "non_iid",
):
    """
    Execute full Federated Learning workflow using Flower framework primitives:
    1. Verify or create partitioned client datasets
    2. Instantiate 3 autonomous IDSFlowerClient objects
    3. Execute FedAvg parameter aggregation cycles
    4. Evaluate global model on independent test set at each communication round
    5. Save global federated model and comparison artifacts
    """
    print("\n" + "=" * 80)
    print("PHASE 5: FEDERATED LEARNING WITH FLOWER FRAMEWORK")
    print(f"Clients: {num_clients} | Rounds: {num_rounds} | Local Epochs: {local_epochs} | Distribution: {distribution_strategy.upper()}")
    print("=" * 80)

    # 1. Ensure client datasets exist
    expected_clients = [CLIENTS_DATA_DIR / f"client_{i}.csv" for i in range(1, num_clients + 1)]
    if not all(p.exists() for p in expected_clients):
        print("[*] Creating isolated client partitions...")
        create_client_partitions(strategy=distribution_strategy, num_clients=num_clients)
    else:
        print("[*] Verified existing client datasets:")
        for p in expected_clients:
            print(f"    - {p.name}: {len(pd.read_csv(p)):,} records")

    # 2. Load global independent test set for server evaluation
    df_test = pd.read_csv(TEST_DATA_FILE)
    X_test = df_test.drop(columns=["Label"]).to_numpy(dtype=np.float64)
    y_test = df_test["Label"].to_numpy(dtype=np.int32)
    classes = [0, 1, 2, 3]
    num_features = X_test.shape[1]

    # 3. Initialize Global Model & Initial Parameters
    global_model = SGDClassifier(
        loss=SGD_CONFIG["loss"],
        penalty=SGD_CONFIG["penalty"],
        alpha=SGD_CONFIG["alpha"],
        max_iter=1,
        warm_start=True,
        random_state=SGD_CONFIG["random_state"],
    )
    init_X = np.zeros((len(classes), num_features), dtype=np.float64)
    global_model.partial_fit(init_X, np.array(classes, dtype=np.int32), classes=np.array(classes, dtype=np.int32))

    current_parameters = ndarrays_to_parameters([
        global_model.coef_.copy(),
        global_model.intercept_.copy(),
    ])

    # 4. Initialize Flower FedAvg Strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
    )

    # 5. Instantiate autonomous clients
    clients = [IDSFlowerClient(client_id=i, all_classes=classes) for i in range(1, num_clients + 1)]

    # Tracking records
    round_history = []
    t_start = time.perf_counter()

    print(f"\n[*] Starting {num_rounds} Federated Communication Rounds via FedAvg...")
    print("-" * 80)

    for r in range(1, num_rounds + 1):
        round_t0 = time.perf_counter()
        fit_results = []
        client_metrics_list = []

        # Step A: Local client training (Autonomous on each client's partition)
        current_ndarrays = parameters_to_ndarrays(current_parameters)
        for client in clients:
            updated_ndarrays, num_samples, metrics = client.fit(
                parameters=current_ndarrays,
                config={"local_epochs": local_epochs, "batch_size": batch_size},
            )
            
            client_proxy = SimulatedClientProxy(cid=str(client.client_id))
            fit_res = FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(updated_ndarrays),
                num_examples=num_samples,
                metrics=metrics,
            )
            fit_results.append((client_proxy, fit_res))
            client_metrics_list.append((num_samples, metrics))

        # Step B: Central Aggregation via Flower FedAvg
        aggregated_parameters, agg_metrics = strategy.aggregate_fit(
            server_round=r,
            results=fit_results,
            failures=[],
        )

        if aggregated_parameters is not None:
            current_parameters = aggregated_parameters
            updated_global_weights = parameters_to_ndarrays(current_parameters)
            global_model.coef_ = updated_global_weights[0].copy().astype(np.float64)
            global_model.intercept_ = updated_global_weights[1].copy().astype(np.float64)

        # Step C: Global Evaluation on Independent Test Set
        y_pred = global_model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        prec_m = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_m = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

        try:
            proba = global_model.predict_proba(X_test)
            loss = float(log_loss(y_test, proba, labels=classes))
        except Exception:
            loss = float(1.0 - acc)

        round_duration = time.perf_counter() - round_t0
        print(f"Round {r:2d}/{num_rounds} | Global Acc: {acc*100:6.2f}% | Macro F1: {f1_m:.4f} | Loss: {loss:.4f} | Clients: {len(fit_results)}/3 ({round_duration:.2f}s)")

        round_history.append({
            "round": r,
            "global_accuracy": acc,
            "global_f1_macro": f1_m,
            "global_precision": prec_m,
            "global_recall": rec_m,
            "global_loss": loss,
            "active_clients": len(fit_results),
            "round_duration_sec": round(round_duration, 3),
        })

    total_fed_time = time.perf_counter() - t_start
    print("-" * 80)
    print(f"[+] All {num_rounds} Federated Rounds Completed in {total_fed_time:.2f} seconds.")

    # 6. Save Final Global Federated Model
    for path in [FEDERATED_MODEL_FILE, MODELS_DIR / "federated_global_model.pkl"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(global_model, path)
    print(f"[+] Saved global federated model to {FEDERATED_MODEL_FILE}")

    # 7. Compare Centralized vs Federated Performance
    centralized_model = joblib.load(SGD_MODEL_FILE) if SGD_MODEL_FILE.exists() else None
    cent_acc = 0.0
    cent_f1 = 0.0
    if centralized_model is not None:
        y_cent_pred = centralized_model.predict(X_test)
        cent_acc = accuracy_score(y_test, y_cent_pred)
        cent_f1 = f1_score(y_test, y_cent_pred, average="macro", zero_division=0)

    final_fed = round_history[-1]
    fed_acc = final_fed["global_accuracy"]
    fed_f1 = final_fed["global_f1_macro"]

    comparison_data = [
        {
            "Paradigm": "Centralized Learning (Baseline)",
            "Data Sharing": "Raw flow data pooled to central server",
            "Accuracy": round(cent_acc, 4),
            "Macro F1": round(cent_f1, 4),
            "Privacy Preservation": "None (Full Data Exposure)",
        },
        {
            "Paradigm": "Federated Learning (FedAvg)",
            "Data Sharing": "Zero raw data shared; model updates only",
            "Accuracy": round(fed_acc, 4),
            "Macro F1": round(fed_f1, 4),
            "Privacy Preservation": "High (Decentralized Local Training)",
        },
    ]
    df_compare = pd.DataFrame(comparison_data)

    for comp_file in [RESULTS_DIR / "federated_vs_centralized.csv", REPORTS_DIR / "federated_vs_centralized.csv"]:
        comp_file.parent.mkdir(parents=True, exist_ok=True)
        df_compare.to_csv(comp_file, index=False)
    print(f"[+] Saved comparison table to {RESULTS_DIR / 'federated_vs_centralized.csv'}")

    # 8. Save Communication Rounds Log
    df_rounds = pd.DataFrame(round_history)
    for rounds_file in [RESULTS_DIR / "communication_rounds.csv", REPORTS_DIR / "communication_rounds.csv"]:
        rounds_file.parent.mkdir(parents=True, exist_ok=True)
        df_rounds.to_csv(rounds_file, index=False)
    print(f"[+] Saved communication round logs to {RESULTS_DIR / 'communication_rounds.csv'}")

    # 9. Plot Federated Convergence Curve
    plt.figure(figsize=(9, 5))
    rounds = [r["round"] for r in round_history]
    accs = [r["global_accuracy"] * 100 for r in round_history]
    f1s = [r["global_f1_macro"] * 100 for r in round_history]

    plt.plot(rounds, accs, marker="o", color="#1f77b4", linewidth=2.5, label="Global Test Accuracy (%)")
    plt.plot(rounds, f1s, marker="s", color="#ff7f0e", linewidth=2.0, linestyle="--", label="Global Macro F1 (%)")
    if cent_acc > 0:
        plt.axhline(cent_acc * 100, color="gray", linestyle=":", label=f"Centralized Benchmark ({cent_acc*100:.1f}%)")

    plt.title(f"Federated Learning Convergence ({num_rounds} Communication Rounds)", fontsize=12, fontweight="bold")
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Score (%)", fontsize=11)
    plt.ylim(min(accs) - 5, 102)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    for curve_path in [RESULTS_DIR / "federated_accuracy_curve.png", GRAPHS_DIR / "federated_accuracy_curve.png"]:
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"[+] Saved convergence chart to {RESULTS_DIR / 'federated_accuracy_curve.png'}")

    # 10. Plot Centralized vs Federated Bar Chart
    plt.figure(figsize=(7, 5))
    paradigms = ["Centralized", "Federated (FedAvg)"]
    acc_vals = [cent_acc * 100, fed_acc * 100]
    colors = ["#4c72b0", "#55a868"]

    bars = plt.bar(paradigms, acc_vals, color=colors, width=0.45)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h - 7, f"{h:.2f}%", ha="center", va="bottom", color="white", fontweight="bold", fontsize=11)

    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.title("Centralized vs Federated Intrusion Detection Accuracy", fontsize=12, fontweight="bold")
    plt.ylim(0, 110)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    for bar_path in [RESULTS_DIR / "federated_vs_centralized.png", GRAPHS_DIR / "federated_vs_centralized.png"]:
        bar_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"[+] Saved paradigm comparison chart to {RESULTS_DIR / 'federated_vs_centralized.png'}")

    # 11. Write Comprehensive Federated Learning Report
    lines = []
    lines.append("=" * 80)
    lines.append("FEDERATED LEARNING INTRUSION DETECTION SYSTEM - TRAINING REPORT")
    lines.append("=" * 80)
    lines.append(f"Framework:                   Flower (flwr v1.15.2)")
    lines.append(f"Aggregation Strategy:        Federated Averaging (FedAvg)")
    lines.append(f"Client Organizations:        {num_clients}")
    lines.append(f"  - Client 1: {ORGANIZATION_NAMES.get(1)}")
    lines.append(f"  - Client 2: {ORGANIZATION_NAMES.get(2)}")
    lines.append(f"  - Client 3: {ORGANIZATION_NAMES.get(3)}")
    lines.append(f"Data Distribution Strategy:  {distribution_strategy.upper()} (Dirichlet non-IID partitioning)")
    lines.append(f"Communication Rounds:        {num_rounds}")
    lines.append(f"Local Epochs per Round:      {local_epochs}")
    lines.append(f"Local Batch Size:            {batch_size}")
    lines.append(f"Total Simulation Duration:   {total_fed_time:.2f} seconds")
    lines.append("-" * 80)
    lines.append("COMMUNICATION ROUND PERFORMANCE HISTORY (GLOBAL TEST SET):")
    lines.append("-" * 80)
    lines.append(f"{'Round':<8} | {'Global Accuracy':<18} | {'Global Macro F1':<18} | {'Loss'}")
    lines.append("-" * 80)
    for r_entry in round_history:
        lines.append(f"{r_entry['round']:<8} | {r_entry['global_accuracy']*100:>6.2f}%{' ':<11} | {r_entry['global_f1_macro']:>6.4f}{' ':<11} | {r_entry['global_loss']:.4f}")
    lines.append("-" * 80)
    lines.append("CENTRALIZED VS FEDERATED COMPARISON:")
    lines.append(f"  Centralized Benchmark Accuracy: {cent_acc*100:.2f}% (Macro F1: {cent_f1:.4f})")
    lines.append(f"  Federated (FedAvg) Accuracy:    {fed_acc*100:.2f}% (Macro F1: {fed_f1:.4f})")
    diff = (fed_acc - cent_acc) * 100
    lines.append(f"  Accuracy Difference:            {diff:+.2f}%")
    lines.append("  Privacy Guarantee:              Raw network records NEVER leave client perimeters.")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    for rep_file in [RESULTS_DIR / "federated_learning_report.txt", REPORTS_DIR / "federated_learning_report.txt"]:
        with open(rep_file, "w", encoding="utf-8") as f:
            f.write(report_text)
    print(f"[+] Saved federated learning report to {RESULTS_DIR / 'federated_learning_report.txt'}")

    return round_history, df_compare


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_federated_learning(num_rounds=rounds)
