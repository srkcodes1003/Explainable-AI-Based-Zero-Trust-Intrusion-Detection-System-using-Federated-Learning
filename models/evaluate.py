"""
Model Evaluation and Benchmarking Module for Intrusion Detection.
Generates comprehensive evaluation metrics, confusion matrices, comparison tables, and plots.
"""

import sys
import time
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    TEST_DATA_FILE,
    SGD_MODEL_FILE,
    RF_MODEL_FILE,
    LABEL_ENCODER_FILE,
    RESULTS_DIR,
    REPORTS_DIR,
    GRAPHS_DIR,
)


def calculate_false_positive_rate(cm: np.ndarray, benign_idx: int = 0) -> float:
    """
    Calculate the False Positive Rate (FPR):
    FPR = FP / (FP + TN), where Positive = Attack, Negative = BENIGN.
    A benign traffic record classified as an attack is a False Positive.
    """
    total_samples = cm.sum()
    # True Negatives: Benign correctly classified as Benign
    tn = cm[benign_idx, benign_idx]
    # False Positives: Benign incorrectly classified as any attack
    fp = cm[benign_idx, :].sum() - tn
    if (fp + tn) == 0:
        return 0.0
    return float(fp / (fp + tn))


def evaluate_models():
    """
    Evaluate SGDClassifier and RandomForestClassifier on the independent Test partition.
    Produces actual metrics, comparison CSV, text reports, and Matplotlib visual figures.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: MODEL EVALUATION & BENCHMARKING ON INDEPENDENT TEST SET")
    print("=" * 70)

    # 1. Load Test Dataset
    if not TEST_DATA_FILE.exists():
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_FILE}")

    df_test = pd.read_csv(TEST_DATA_FILE)
    X_test = df_test.drop(columns=["Label"]).to_numpy(dtype=np.float32)
    y_test = df_test["Label"].to_numpy(dtype=np.int32)

    # 2. Load Label Encoder
    encoder = joblib.load(LABEL_ENCODER_FILE)
    class_names = [str(c) for c in encoder.classes_]
    benign_idx = list(encoder.classes_).index("BENIGN") if "BENIGN" in encoder.classes_ else 0

    print(f"[*] Loaded Test Set: {len(X_test):,} flows, {len(class_names)} classes: {class_names}")

    # 3. Load Models
    models = {}
    if SGD_MODEL_FILE.exists():
        models["SGDClassifier (Federated Primary)"] = joblib.load(SGD_MODEL_FILE)
    else:
        print(f"[-] Warning: {SGD_MODEL_FILE} not found.")

    if RF_MODEL_FILE.exists():
        models["Random Forest (Baseline Comparison)"] = joblib.load(RF_MODEL_FILE)
    else:
        print(f"[-] Warning: {RF_MODEL_FILE} not found.")

    if not models:
        raise RuntimeError("No trained model files found for evaluation.")

    comparison_records = []
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CIC-IDS2017 INTRUSION DETECTION SYSTEM - MODEL EVALUATION REPORT")
    report_lines.append(f"Evaluated on Independent Test Partition: {TEST_DATA_FILE} ({len(X_test):,} samples)")
    report_lines.append("=" * 80)

    confusion_matrices = {}

    for model_name, model in models.items():
        print(f"\n[*] Evaluating: {model_name}...")
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_latency = time.perf_counter() - t0

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[model_name] = cm
        fpr = calculate_false_positive_rate(cm, benign_idx=benign_idx)

        print(f"    - Accuracy:        {acc * 100:.2f}%")
        print(f"    - Macro Precision: {prec_macro:.4f}")
        print(f"    - Macro Recall:    {rec_macro:.4f}")
        print(f"    - Macro F1-Score:  {f1_macro:.4f}")
        print(f"    - Weighted F1:     {f1_weighted:.4f}")
        print(f"    - False Pos Rate:  {fpr * 100:.2f}%")
        print(f"    - Total Pred Time: {pred_latency:.4f}s ({pred_latency / len(X_test) * 1000:.3f} ms/sample)")

        comparison_records.append({
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Macro Precision": round(prec_macro, 4),
            "Macro Recall": round(rec_macro, 4),
            "Macro F1-Score": round(f1_macro, 4),
            "Weighted F1-Score": round(f1_weighted, 4),
            "False Positive Rate": round(fpr, 4),
            "Inference Time (s)": round(pred_latency, 4),
        })

        clf_rep = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
        report_lines.append(f"\nMODEL: {model_name}")
        report_lines.append("-" * 60)
        report_lines.append(f"Accuracy:            {acc * 100:.2f}%")
        report_lines.append(f"Macro Precision:     {prec_macro:.4f}")
        report_lines.append(f"Macro Recall:        {rec_macro:.4f}")
        report_lines.append(f"Macro F1-Score:      {f1_macro:.4f}")
        report_lines.append(f"Weighted F1-Score:   {f1_weighted:.4f}")
        report_lines.append(f"False Positive Rate: {fpr * 100:.2f}%")
        report_lines.append(f"Inference Latency:   {pred_latency / len(X_test) * 1000:.3f} ms/flow")
        report_lines.append("\nPer-Class Classification Report:")
        report_lines.append(clf_rep)
        report_lines.append("=" * 80)

    # 4. Save Model Comparison Table
    df_comp = pd.DataFrame(comparison_records)
    comp_file = RESULTS_DIR / "model_comparison.csv"
    comp_rep_file = REPORTS_DIR / "model_comparison.csv"
    df_comp.to_csv(comp_file, index=False)
    df_comp.to_csv(comp_rep_file, index=False)
    print(f"\n[+] Saved comparison metrics table to {comp_file}")

    # 5. Save Classification Report Text
    rep_file = RESULTS_DIR / "classification_report.txt"
    rep_rep_file = REPORTS_DIR / "classification_report.txt"
    with open(rep_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    with open(rep_rep_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[+] Saved detailed classification reports to {rep_file}")

    # 6. Generate and Save Confusion Matrix Plot
    plt.figure(figsize=(14, 5))
    for i, (m_name, cm) in enumerate(confusion_matrices.items(), 1):
        plt.subplot(1, len(confusion_matrices), i)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"Confusion Matrix: {m_name.split(' ')[0]}", fontsize=11, fontweight="bold")
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")

    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    cm_graph_path = GRAPHS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.savefig(cm_graph_path, dpi=200)
    plt.close()
    print(f"[+] Saved confusion matrix visualizations to {cm_path}")

    # 7. Generate and Save Performance Metrics Chart
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    model_names_list = list(df_comp["Model"].values)
    for idx, m_row in df_comp.iterrows():
        offset = (idx - 0.5) * width
        vals = [m_row[m] for m in metrics_to_plot]
        plt.bar(x + offset, vals, width, label=m_row["Model"].split(" ")[0])

    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.title("Performance Comparison: SGDClassifier vs Random Forest", fontsize=13, fontweight="bold")
    plt.xticks(x, metrics_to_plot, fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    plt.tight_layout()

    perf_path = RESULTS_DIR / "performance_metrics.png"
    perf_graph_path = GRAPHS_DIR / "performance_metrics.png"
    plt.savefig(perf_path, dpi=200)
    plt.savefig(perf_graph_path, dpi=200)
    plt.close()
    print(f"[+] Saved performance comparison chart to {perf_path}")

    print("\n[+] Phase 4 Model Evaluation successfully completed.")
    return df_comp


if __name__ == "__main__":
    evaluate_models()

