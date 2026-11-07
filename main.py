"""
Master CLI Entrypoint for Explainable AI-Based Zero Trust IDS Using Federated Learning.
Supports phase-based execution matching the documented workflow.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from preprocessing.explore_dataset import explore_dataset
from preprocessing.preprocess_pipeline import run_preprocessing_pipeline
from models.train_sgd import train_sgd_model
from models.train_random_forest import train_random_forest_model
from models.evaluate import evaluate_models
from federated_learning.train_federated import run_federated_learning


def main():
    parser = argparse.ArgumentParser(
        description="Explainable AI-Based Zero Trust Intrusion Detection System Using Federated Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["explore", "preprocess", "train", "federated", "all"],
        help="Pipeline phase to execute: explore, preprocess, train, federated, or all",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of federated communication rounds (for --phase federated)",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=3,
        help="Number of simulated federated client organizations",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development sampling mode (~30k rows)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("EXPLAINABLE AI ZERO TRUST IDS USING FEDERATED LEARNING - CLI ENGINE")
    print(f"Executing Phase: {args.phase.upper()}")
    print("=" * 80)

    if args.phase in ["explore", "all"]:
        print("\n>>> EXECUTING PHASE 2: DATASET EXPLORATION")
        explore_dataset()

    if args.phase in ["preprocess", "all"]:
        print("\n>>> EXECUTING PHASE 3: DATA PREPROCESSING")
        run_preprocessing_pipeline(dev_mode=args.dev)

    if args.phase in ["train", "all"]:
        print("\n>>> EXECUTING PHASE 4: MACHINE LEARNING MODEL TRAINING & EVALUATION")
        train_sgd_model()
        train_random_forest_model()
        evaluate_models()

    if args.phase in ["federated", "all"]:
        print("\n>>> EXECUTING PHASE 5: FLOWER FEDERATED LEARNING")
        run_federated_learning(num_rounds=args.rounds, num_clients=args.clients)

    print("\n" + "=" * 80)
    print(f"Phase '{args.phase}' completed successfully.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

