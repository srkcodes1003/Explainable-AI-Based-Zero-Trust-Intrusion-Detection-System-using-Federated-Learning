"""
Flower NumPyClient implementation for Intrusion Detection.
Represents an autonomous client organization training locally without sharing raw flow data.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import flwr as fl

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import CLIENTS_DATA_DIR, TRAIN_DATA_FILE, SGD_CONFIG


def get_shared_class_weights(classes: List[int]) -> Dict[int, float]:
    """Compute shared balanced class weights from training partition."""
    if TRAIN_DATA_FILE.exists():
        df_tr = pd.read_csv(TRAIN_DATA_FILE, usecols=["Label"])
        y_tr = df_tr["Label"].to_numpy()
        w = compute_class_weight("balanced", classes=np.array(classes), y=y_tr)
        return {int(c): float(weight) for c, weight in zip(classes, w)}
    return {c: 1.0 for c in classes}


class IDSFlowerClient(fl.client.NumPyClient):
    """
    Simulated Federated Client Organization.
    Trains SGDClassifier on private local network traffic logs.
    Communicates strictly through serialized model parameters (coef_ and intercept_).
    """

    def __init__(self, client_id: int, all_classes: List[int] = None):
        self.client_id = int(client_id)
        self.data_file = CLIENTS_DATA_DIR / f"client_{self.client_id}.csv"
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Client dataset not found at {self.data_file}")

        # Load private local training data
        df = pd.read_csv(self.data_file)
        self.X_local = df.drop(columns=["Label"]).to_numpy(dtype=np.float64)
        self.y_local = df["Label"].to_numpy(dtype=np.int32)
        
        # Enforce consistent global class list across all clients
        self.all_classes = all_classes or [0, 1, 2, 3]
        self.num_classes = len(self.all_classes)
        self.num_features = self.X_local.shape[1]

        # Shared precomputed class weights for partial_fit compatibility
        self.class_weight = get_shared_class_weights(self.all_classes)

        # Initialize local SGDClassifier with precomputed class weights
        self.model = SGDClassifier(
            loss=SGD_CONFIG["loss"],
            penalty=SGD_CONFIG["penalty"],
            alpha=SGD_CONFIG["alpha"],
            max_iter=1,
            warm_start=True,
            learning_rate=SGD_CONFIG["learning_rate"],
            eta0=SGD_CONFIG.get("eta0", 0.05),
            random_state=SGD_CONFIG["random_state"] + self.client_id,
            class_weight=self.class_weight,
        )

        # Initialize coefficients and intercepts with all global classes
        init_X = np.zeros((self.num_classes, self.num_features), dtype=np.float64)
        init_y = np.array(self.all_classes, dtype=np.int32)
        self.model.partial_fit(init_X, init_y, classes=np.array(self.all_classes, dtype=np.int32))

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Extract and return model weights: [coef_, intercept_]."""
        return [self.model.coef_.copy(), self.model.intercept_.copy()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Inject global parameters into local model."""
        if len(parameters) >= 2:
            self.model.coef_ = parameters[0].copy().astype(np.float64)
            self.model.intercept_ = parameters[1].copy().astype(np.float64)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train local model for the configured number of local epochs.
        Never transmits raw data — only returns updated parameter weights.
        """
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 128))

        n_samples = len(self.X_local)
        indices = np.arange(n_samples)

        for epoch in range(local_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                X_batch = self.X_local[batch_idx]
                y_batch = self.y_local[batch_idx]
                self.model.partial_fit(
                    X_batch,
                    y_batch,
                    classes=np.array(self.all_classes, dtype=np.int32),
                )

        # Compute local post-fit metrics
        y_pred = self.model.predict(self.X_local)
        acc = float(accuracy_score(self.y_local, y_pred))
        
        try:
            proba = self.model.predict_proba(self.X_local)
            loss = float(log_loss(self.y_local, proba, labels=self.all_classes))
        except Exception:
            loss = float(1.0 - acc)

        metrics = {
            "client_id": self.client_id,
            "local_accuracy": acc,
            "local_loss": loss,
            "samples": n_samples,
        }

        return self.get_parameters(config={}), n_samples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """Evaluate received global parameters on local private data."""
        self.set_parameters(parameters)
        y_pred = self.model.predict(self.X_local)
        acc = float(accuracy_score(self.y_local, y_pred))
        
        try:
            proba = self.model.predict_proba(self.X_local)
            loss = float(log_loss(self.y_local, proba, labels=self.all_classes))
        except Exception:
            loss = float(1.0 - acc)

        return loss, len(self.X_local), {"accuracy": acc, "loss": loss}


def create_client(cid: str) -> IDSFlowerClient:
    """Helper factory for Flower simulation."""
    client_id = int(cid) + 1  # 0-indexed string -> 1-indexed client id
    return IDSFlowerClient(client_id=client_id)
