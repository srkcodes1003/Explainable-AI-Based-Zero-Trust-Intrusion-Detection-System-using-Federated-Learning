"""
Phase 2 Dataset Exploration Test.
Verifies dynamic CSV discovery, column identification, and exploration logic.
"""

import pytest
from pathlib import Path
from preprocessing.explore_dataset import find_csv_files, detect_label_column, explore_single_csv
from config.settings import ORIGINAL_DATA_DIR, REPORTS_DIR

def test_find_csv_files():
    """Verify CSV files are detected dynamically."""
    csvs = find_csv_files(ORIGINAL_DATA_DIR)
    assert len(csvs) >= 1, "At least one CSV file should be detected in dataset/original/"

def test_detect_label_column():
    """Verify dynamic label column detection handles leading/trailing spaces."""
    cols_with_space = ["Destination Port", " Flow Duration", " Label"]
    assert detect_label_column(cols_with_space) == " Label"
    
    cols_clean = ["Destination Port", "Flow Duration", "Label"]
    assert detect_label_column(cols_clean) == "Label"

def test_single_csv_profiling():
    """Verify single CSV profiling returns valid counts and statistics."""
    csvs = find_csv_files(ORIGINAL_DATA_DIR)
    assert len(csvs) > 0
    first_file = csvs[0]
    metrics = explore_single_csv(first_file, chunk_size=5000)
    assert metrics["status"] == "success"
    assert metrics["rows"] > 0
    assert metrics["columns_count"] > 10
    assert "BENIGN" in metrics["label_counts"]

