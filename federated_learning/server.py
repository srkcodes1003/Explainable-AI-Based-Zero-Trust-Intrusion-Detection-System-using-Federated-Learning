"""
Centralized Aggregation Server for Federated Learning IDS.
Wraps strategy configuration and provides direct server initialization.
"""

import sys
from pathlib import Path
import flwr as fl

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from federated_learning.strategy import get_evaluate_fn, fit_metrics_aggregation
from federated_learning.train_federated import run_federated_learning


def start_aggregation_server(num_rounds: int = 5):
    """Launch Flower server aggregation workflow."""
    print("[*] Initializing Central Federated Aggregation Server...")
    return run_federated_learning(num_rounds=num_rounds)


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    start_aggregation_server(num_rounds=rounds)

