"""
Phase 5 Federated Learning Test Suite.
Verifies client data isolation, Flower NumPyClient operations, FedAvg parameter aggregation,
and global model evaluation artifacts.
"""

import pytest
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import (
    CLIENTS_DATA_DIR,
    FEDERATED_MODEL_FILE,
    TEST_DATA_FILE,
    RESULTS_DIR,
)
from federated_learning.client import IDSFlowerClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.common import FitRes, Status, Code


def test_client_partitions_exist():
    """Verify 3 distinct client datasets exist with non-empty samples."""
    for i in range(1, 4):
        c_path = CLIENTS_DATA_DIR / f"client_{i}.csv"
        assert c_path.exists(), f"Client {i} dataset missing: {c_path}"
        df = pd.read_csv(c_path, nrows=50)
        assert len(df) > 0
        assert "Label" in df.columns


def test_client_fit_and_parameter_exchange():
    """Verify IDSFlowerClient can load, get/set parameters, and perform local training."""
    client = IDSFlowerClient(client_id=1, all_classes=[0, 1, 2, 3])
    initial_params = client.get_parameters(config={})
    assert len(initial_params) == 2, "Model parameters must contain [coef_, intercept_]"

    # Test local fit for 1 epoch on mini-batch
    updated_params, num_samples, metrics = client.fit(
        parameters=initial_params,
        config={"local_epochs": 1, "batch_size": 64},
    )

    assert len(updated_params) == 2
    assert num_samples > 0
    assert "local_accuracy" in metrics
    assert "local_loss" in metrics


def test_flower_fedavg_aggregation():
    """Verify Flower FedAvg accurately aggregates model weight arrays."""
    strategy = FedAvg()

    w1 = [np.array([[2.0, 4.0]], dtype=np.float64), np.array([1.0], dtype=np.float64)]
    w2 = [np.array([[4.0, 6.0]], dtype=np.float64), np.array([3.0], dtype=np.float64)]

    p1 = ndarrays_to_parameters(w1)
    p2 = ndarrays_to_parameters(w2)

    res1 = FitRes(status=Status(code=Code.OK, message=""), parameters=p1, num_examples=100, metrics={})
    res2 = FitRes(status=Status(code=Code.OK, message=""), parameters=p2, num_examples=100, metrics={})

    class DummyProxy: pass
    results = [(DummyProxy(), res1), (DummyProxy(), res2)]

    aggregated_p, _ = strategy.aggregate_fit(1, results, failures=[])
    agg_weights = parameters_to_ndarrays(aggregated_p)

    # Average of w1 and w2 (equal num_examples=100)
    expected_coef = np.array([[3.0, 5.0]])
    expected_intercept = np.array([2.0])

    assert np.allclose(agg_weights[0], expected_coef)
    assert np.allclose(agg_weights[1], expected_intercept)


def test_global_federated_model_and_reports():
    """Verify saved global federated model and reporting deliverables exist."""
    assert FEDERATED_MODEL_FILE.exists(), f"Missing global model: {FEDERATED_MODEL_FILE}"
    model = joblib.load(FEDERATED_MODEL_FILE)

    df_test = pd.read_csv(TEST_DATA_FILE, nrows=10)
    X = df_test.drop(columns=["Label"]).to_numpy(dtype=np.float64)
    preds = model.predict(X)
    assert len(preds) == 10

    # Deliverables verification
    assert (RESULTS_DIR / "federated_vs_centralized.csv").exists()
    assert (RESULTS_DIR / "communication_rounds.csv").exists()
    assert (RESULTS_DIR / "federated_accuracy_curve.png").exists()
    assert (RESULTS_DIR / "federated_vs_centralized.png").exists()
    assert (RESULTS_DIR / "federated_learning_report.txt").exists()

