"""
Phase 1 Environment & Dependency Verification Test.
Verifies Python version, OS platform, and critical library imports.
"""

import sys
import platform
import pytest

def test_python_version():
    """Verify 64-bit architecture and Python version."""
    assert sys.maxsize > 2**32, "Python must be 64-bit to handle numerical arrays."
    major, minor = sys.version_info.major, sys.version_info.minor
    # Supports Python 3.11 and 3.12
    assert (major == 3 and minor in (11, 12)), (
        f"Expected Python 3.11 or 3.12, found Python {major}.{minor}.{sys.version_info.micro}"
    )

def test_os_platform():
    """Verify operating system is Windows."""
    assert platform.system() == "Windows", f"Expected Windows OS, detected: {platform.system()}"

def test_core_numerical_imports():
    """Verify NumPy and Pandas imports and basic functionality."""
    import numpy as np
    import pandas as pd
    arr = np.array([1, 2, 3], dtype=np.float32)
    df = pd.DataFrame({"feature": arr})
    assert len(df) == 3
    assert df["feature"].dtype == np.float32

def test_machine_learning_imports():
    """Verify Scikit-learn, Imbalanced-learn and Joblib imports."""
    import sklearn
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import imblearn
    import joblib

    clf = SGDClassifier()
    scaler = StandardScaler()
    assert clf is not None
    assert scaler is not None

def test_federated_learning_imports():
    """Verify Flower (flwr) imports."""
    import flwr
    from flwr.common import (
        ndarrays_to_parameters,
        parameters_to_ndarrays,
        Parameters,
    )
    import numpy as np

    # Test parameter serialization
    test_weights = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]
    params = ndarrays_to_parameters(test_weights)
    restored = parameters_to_ndarrays(params)
    assert np.allclose(test_weights[0], restored[0])

def test_explainable_ai_imports():
    """Verify SHAP import."""
    import shap
    assert shap.__version__ is not None

def test_visualization_and_api_imports():
    """Verify Matplotlib, Seaborn, and Flask imports."""
    import matplotlib
    import seaborn
    import flask
    assert matplotlib.__version__ is not None
    assert seaborn.__version__ is not None
    assert flask.__version__ is not None

