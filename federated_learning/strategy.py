"""
Federated Learning Aggregation Strategy and Server Evaluation.
Implements FedAvg with server-side evaluation against the global test set.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
import flwr as fl
from flwr.common import Parameters, Scalar, parameters_to_ndarrays

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import TEST_DATA_FILE, SGD_CONFIG


def get_evaluate_fn(all_classes: List[int] = None) -> Callable:
    """
    Generate evaluation function for Flower server.
    Evaluates the aggregated global parameters at every communication round
    using the independent global test partition.
    """
    classes = all_classes or [0, 1, 2, 3]
    df_test = pd.read_csv(TEST_DATA_FILE)
    X_test = df_test.drop(columns=["Label"]).to_numpy(dtype=np.float32)
    y_test = df_test["Label"].to_numpy(dtype=np.int32)

    eval_model = SGDClassifier(
        loss=SGD_CONFIG["loss"],
        penalty=SGD_CONFIG["penalty"],
        alpha=SGD_CONFIG["alpha"],
        max_iter=1,
        warm_start=True,
    )
    # Initialize shapes
    init_X = np.zeros((len(classes), X_test.shape[1]), dtype=np.float32)
    eval_model.partial_fit(init_X, np.array(classes, dtype=np.int32), classes=np.array(classes, dtype=np.int32))

    def evaluate(
        server_round: int,
        parameters: Parameters,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Convert Parameters to NumPy ndarrays
        ndarrays = parameters_to_ndarrays(parameters)
        eval_model.coef_ = ndarrays[0].copy().astype(np.float64)
        eval_model.intercept_ = ndarrays[1].copy().astype(np.float64)

        y_pred = eval_model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        prec_m = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_m = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

        try:
            proba = eval_model.predict_proba(X_test)
            loss = float(log_loss(y_test, proba, labels=classes))
        except Exception:
            loss = float(1.0 - acc)

        print(f"\n[Round {server_round} Global Evaluation] Accuracy: {acc*100:.2f}% | Loss: {loss:.4f} | F1: {f1_m:.4f}")

        return loss, {
            "accuracy": acc,
            "f1_macro": f1_m,
            "precision_macro": prec_m,
            "recall_macro": rec_m,
        }

    return evaluate


def fit_metrics_aggregation(fit_metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate training metrics from all participating client organizations."""
    total_samples = sum(num_samples for num_samples, _ in fit_metrics)
    if total_samples == 0:
        return {}

    weighted_acc = sum(num_samples * m.get("local_accuracy", 0.0) for num_samples, m in fit_metrics) / total_samples
    weighted_loss = sum(num_samples * m.get("local_loss", 0.0) for num_samples, m in fit_metrics) / total_samples

    return {
        "client_weighted_accuracy": float(weighted_acc),
        "client_weighted_loss": float(weighted_loss),
        "active_clients_count": len(fit_metrics),
    }

