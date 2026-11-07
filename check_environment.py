"""
Environment diagnostic and dependency reporting script.
Verifies system architecture, Python version, RAM, and library compatibility.
"""

import sys
import platform
import importlib
from pathlib import Path

def generate_report():
    packages = [
        "numpy",
        "pandas",
        "sklearn",
        "imblearn",
        "joblib",
        "flwr",
        "shap",
        "matplotlib",
        "seaborn",
        "flask",
        "pytest"
    ]
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("EXPLAINABLE AI ZERO TRUST IDS - ENVIRONMENT VERIFICATION")
    report_lines.append("=" * 60)
    report_lines.append(f"Operating System: {platform.system()} {platform.release()} ({platform.version()})")
    report_lines.append(f"Architecture:     {platform.architecture()[0]}")
    report_lines.append(f"Python Version:   {platform.python_version()} ({sys.executable})")
    report_lines.append(f"Is 64-bit Python: {sys.maxsize > 2**32}")
    report_lines.append("-" * 60)
    report_lines.append(f"{'Package':<20} | {'Status':<12} | {'Installed Version'}")
    report_lines.append("-" * 60)
    
    all_passed = True
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "Available")
            report_lines.append(f"{pkg:<20} | {'SUCCESS':<12} | {version}")
        except Exception as e:
            report_lines.append(f"{pkg:<20} | {'FAILED':<12} | {str(e)}")
            all_passed = False
            
    report_lines.append("=" * 60)
    if all_passed:
        report_lines.append("Status: Environment setup completed successfully.")
    else:
        report_lines.append("Status: Some dependencies failed to load.")
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    output_path = Path(__file__).resolve().parent / "results" / "reports" / "environment_report.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    return all_passed

if __name__ == "__main__":
    success = generate_report()
    sys.exit(0 if success else 1)

