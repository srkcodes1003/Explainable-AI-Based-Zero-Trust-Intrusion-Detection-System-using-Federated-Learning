"""
Phase 2: Dataset Collection & Dynamic Exploration Module.
Discovers and profiles CIC-IDS2017 CSV files dynamically without loading entire files into memory.
Strictly optimized for 8 GB RAM using chunk-based processing.
"""

import sys
import gc
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    ORIGINAL_DATA_DIR,
    REPORTS_DIR,
    CHUNK_SIZE,
    LABEL_COLUMN_CANDIDATES,
    BENIGN_LABEL,
)


def find_csv_files(directory: Path):
    """
    Dynamically discover all CSV files in directory and any subdirectories.
    Never relies on hardcoded filenames.
    """
    if not directory.exists():
        return []
    return sorted(list(directory.glob("*.csv")) + list(directory.glob("*/*.csv")))


def detect_label_column(columns):
    """
    Dynamically identify the target label column from column names,
    handling leading/trailing whitespace variations in CIC-IDS2017.
    """
    clean_map = {col.strip(): col for col in columns}
    for candidate in LABEL_COLUMN_CANDIDATES:
        c_clean = candidate.strip()
        if c_clean in clean_map:
            return clean_map[c_clean]
    # Fallback: search for any column containing 'label' (case-insensitive)
    for col in columns:
        if "label" in col.strip().lower():
            return col
    return None


def explore_single_csv(file_path: Path, chunk_size: int = CHUNK_SIZE):
    """
    Inspect a single CSV file in chunks to minimize memory consumption.
    Returns aggregated metrics without holding the full dataframe in RAM.
    """
    print(f"[*] Scanning: {file_path.name}")
    total_rows = 0
    column_names = []
    label_col = None
    label_counts = Counter()
    missing_count = 0
    infinite_count = 0
    total_bytes = 0

    first_chunk = True

    try:
        for chunk in pd.read_csv(
            file_path,
            chunksize=chunk_size,
            low_memory=False,
            encoding="cp1252",
            encoding_errors="replace",
        ):
            # Standardize column headers by stripping whitespace
            if first_chunk:
                column_names = list(chunk.columns)
                label_col = detect_label_column(column_names)
                first_chunk = False

            total_rows += len(chunk)
            total_bytes += chunk.memory_usage(deep=True).sum()

            # Inspect label distributions if label column was detected
            if label_col and label_col in chunk.columns:
                cleaned_labels = chunk[label_col].astype(str).str.strip()
                label_counts.update(cleaned_labels.value_counts().to_dict())

            # Detect missing values
            missing_count += int(chunk.isna().sum().sum())

            # Detect infinite values across numeric columns
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                inf_mask = np.isinf(chunk[numeric_cols].to_numpy())
                infinite_count += int(np.sum(inf_mask))

        # Memory cleanup
        gc.collect()

        return {
            "status": "success",
            "file_name": file_path.name,
            "file_path": str(file_path),
            "rows": total_rows,
            "columns_count": len(column_names),
            "columns": column_names,
            "label_col": label_col,
            "label_counts": dict(label_counts),
            "missing_count": missing_count,
            "infinite_count": infinite_count,
            "estimated_ram_mb": round(total_bytes / (1024 * 1024), 2),
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
        }

    except Exception as e:
        return {
            "status": "error",
            "file_name": file_path.name,
            "file_path": str(file_path),
            "error": str(e),
        }


def generate_exploration_report(csv_files, results, output_file: Path):
    """
    Format all findings into a structured report and save to disk.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 80)
    lines.append("CIC-IDS2017 DATASET EXPLORATION & PROFILING REPORT")
    lines.append("=" * 80)
    lines.append(f"Directory Scanned: {ORIGINAL_DATA_DIR}")
    lines.append(f"Total CSV Files Detected: {len(csv_files)}")
    lines.append("=" * 80)

    if not csv_files:
        lines.append("")
        lines.append("STATUS: NO DATASET FILES FOUND")
        lines.append("-" * 80)
        lines.append("The original CIC-IDS2017 CSV files are not currently present in:")
        lines.append(f"  {ORIGINAL_DATA_DIR}")
        lines.append("")
        lines.append("DATASET DOWNLOAD INSTRUCTIONS:")
        lines.append("1. Download the official 'GeneratedLabelledFlows.zip' or 'MachineLearningCSV.zip' from:")
        lines.append("   https://www.unb.ca/cic/datasets/ids-2017.html")
        lines.append("2. Extract the archive and copy the CSV files into:")
        lines.append(f"   {ORIGINAL_DATA_DIR}")
        lines.append("3. Expected CSV files include:")
        lines.append("   - Tuesday-WorkingHours.pcap_ISCX.csv")
        lines.append("   - Wednesday-workingHours.pcap_ISCX.csv")
        lines.append("   - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
        lines.append("   - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
        lines.append("   - Friday-WorkingHours-Morning.pcap_ISCX.csv")
        lines.append("   - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
        lines.append("   - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        lines.append("")
        lines.append("NOTE: As per project guidelines, synthetic network traffic is not fabricated.")
        lines.append("Place the official files in the folder and re-run this exploration step.")
        lines.append("=" * 80)
    else:
        grand_total_rows = sum(r.get("rows", 0) for r in results if r["status"] == "success")
        grand_total_size_mb = sum(r.get("file_size_mb", 0) for r in results if r["status"] == "success")
        grand_total_missing = sum(r.get("missing_count", 0) for r in results if r["status"] == "success")
        grand_total_infinite = sum(r.get("infinite_count", 0) for r in results if r["status"] == "success")

        combined_labels = Counter()
        for r in results:
            if r["status"] == "success":
                combined_labels.update(r.get("label_counts", {}))

        benign_count = combined_labels.get(BENIGN_LABEL, 0)
        attack_count = sum(count for lbl, count in combined_labels.items() if lbl != BENIGN_LABEL)

        lines.append(f"Cumulative Records (Rows): {grand_total_rows:,}")
        lines.append(f"Total On-Disk Size:         {grand_total_size_mb:,.2f} MB")
        lines.append(f"Total Missing (NaN) Values: {grand_total_missing:,}")
        lines.append(f"Total Infinite Values:      {grand_total_infinite:,}")
        lines.append(f"Normal (BENIGN) Records:    {benign_count:,} ({round(benign_count / max(1, grand_total_rows) * 100, 2)}%)")
        lines.append(f"Malicious Records:          {attack_count:,} ({round(attack_count / max(1, grand_total_rows) * 100, 2)}%)")
        lines.append("-" * 80)
        lines.append("ATTACK CATEGORY BREAKDOWN:")
        lines.append("-" * 80)
        lines.append(f"{'Attack Class':<35} | {'Count':<12} | {'Percentage'}")
        lines.append("-" * 80)
        for label, count in combined_labels.most_common():
            pct = round(count / max(1, grand_total_rows) * 100, 3)
            lines.append(f"{label:<35} | {count:<12,} | {pct:>7.3f}%")

        lines.append("=" * 80)
        lines.append("PER-FILE SUMMARY:")
        lines.append("=" * 80)
        for r in results:
            if r["status"] == "success":
                lines.append(f"File: {r['file_name']}")
                lines.append(f"  Rows: {r['rows']:,} | Columns: {r['columns_count']} | Disk: {r['file_size_mb']} MB")
                lines.append(f"  Target Label Column: '{r['label_col']}'")
                lines.append(f"  Missing: {r['missing_count']:,} | Infinite: {r['infinite_count']:,}")
                lines.append("  Detected Classes:")
                for lbl, c in sorted(r["label_counts"].items(), key=lambda x: -x[1]):
                    lines.append(f"    - {lbl}: {c:,}")
                lines.append("-" * 40)
            else:
                lines.append(f"File: {r['file_name']} [FAILED: {r.get('error')}]")
        lines.append("=" * 80)

    report_text = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)
    print(f"\n[+] Dataset exploration report saved to: {output_file}")


def explore_dataset():
    """Main execution function for Phase 2 exploration."""
    csv_files = find_csv_files(ORIGINAL_DATA_DIR)
    results = []
    for csv_file in csv_files:
        res = explore_single_csv(csv_file)
        results.append(res)

    report_path = REPORTS_DIR / "dataset_exploration_report.txt"
    generate_exploration_report(csv_files, results, report_path)
    return len(csv_files), results


if __name__ == "__main__":
    count, _ = explore_dataset()
    sys.exit(0)

