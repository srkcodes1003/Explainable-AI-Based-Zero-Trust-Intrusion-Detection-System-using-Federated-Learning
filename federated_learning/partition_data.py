"""
Data Partitioning Module for Federated Learning.
Splits the centralized training dataset across 3 distinct client organizations:
- Client 1: Organization A
- Client 2: Organization B
- Client 3: Organization C

Supports both IID and Non-IID distributions with reproducible random seeds.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    TRAIN_DATA_FILE,
    CLIENTS_DATA_DIR,
    RANDOM_SEED,
    FEDERATED_CONFIG,
)

ORGANIZATION_NAMES = {
    1: "Organization A (Enterprise Campus Network)",
    2: "Organization B (Cloud Infrastructure)",
    3: "Organization C (Branch Office / Edge Perimeter)",
}


def partition_iid(df: pd.DataFrame, num_clients: int, random_seed: int = RANDOM_SEED):
    """Uniformly partition dataset across clients with identical class distributions."""
    shuffled_df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    client_dfs = np.array_split(shuffled_df, num_clients)
    return [pd.DataFrame(c_df) for c_df in client_dfs]


def partition_non_iid_dirichlet(
    df: pd.DataFrame,
    num_clients: int,
    alpha: float = 0.5,
    random_seed: int = RANDOM_SEED,
):
    """
    Partition dataset across clients using Dirichlet distribution to simulate realistic Non-IID skew.
    Each client organization encounters differing volumes and proportions of specific attack vectors.
    """
    np.random.seed(random_seed)
    client_indices = [[] for _ in range(num_clients)]
    labels = df["Label"].to_numpy()
    unique_classes = np.unique(labels)

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        
        # Dirichlet proportions for this class
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Ensure proportions sum to 1 and allocate indices
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
        
        assigned_splits = np.split(cls_indices, splits)
        for i in range(num_clients):
            client_indices[i].extend(assigned_splits[i])

    client_dfs = []
    for i in range(num_clients):
        c_idx = client_indices[i]
        np.random.shuffle(c_idx)
        c_df = df.iloc[c_idx].reset_index(drop=True)
        client_dfs.append(c_df)

    return client_dfs


def create_client_partitions(
    strategy: str = "non_iid",
    num_clients: int = 3,
    random_seed: int = RANDOM_SEED,
):
    """
    Generate isolated client datasets and save to dataset/clients/client_{id}.csv.
    Enforces privacy by physically separating client datasets into distinct files.
    """
    print("\n" + "=" * 70)
    print("FEDERATED DATA PARTITIONING ACROSS 3 CLIENT ORGANIZATIONS")
    print("=" * 70)

    if not TRAIN_DATA_FILE.exists():
        raise FileNotFoundError(f"Training data not found at {TRAIN_DATA_FILE}")

    df_train = pd.read_csv(TRAIN_DATA_FILE)
    CLIENTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Input Data: {TRAIN_DATA_FILE} ({len(df_train):,} rows)")
    print(f"[*] Partitioning Strategy: {strategy.upper()} (Seed: {random_seed})")

    if strategy.lower() == "iid":
        client_partitions = partition_iid(df_train, num_clients, random_seed)
    else:
        alpha = FEDERATED_CONFIG.get("alpha_dirichlet", 0.5)
        client_partitions = partition_non_iid_dirichlet(df_train, num_clients, alpha=alpha, random_seed=random_seed)

    partition_summary = {}

    for client_id, c_df in enumerate(client_partitions, 1):
        c_path = CLIENTS_DATA_DIR / f"client_{client_id}.csv"
        c_df.to_csv(c_path, index=False)
        
        counts = c_df["Label"].value_counts().to_dict()
        org_name = ORGANIZATION_NAMES.get(client_id, f"Client {client_id}")
        
        partition_summary[f"client_{client_id}"] = {
            "organization": org_name,
            "file_path": str(c_path),
            "total_records": len(c_df),
            "class_distribution": {int(k): int(v) for k, v in counts.items()},
        }

        print(f"\n[+] Created Client {client_id}: {org_name}")
        print(f"    Path:    {c_path}")
        print(f"    Records: {len(c_df):,}")
        print(f"    Classes: {counts}")

    meta_file = CLIENTS_DATA_DIR / "partition_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(partition_summary, f, indent=2)
    print(f"\n[+] Saved partition metadata to {meta_file}")

    return client_partitions, partition_summary


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "non_iid"
    create_client_partitions(strategy=strategy)

