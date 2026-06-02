from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_latest_datasets(project_root: Path) -> Dict[str, Any]:
    """Load datasets from multiple possible locations (in priority order)"""
    
    # Try paths in order of preference
    possible_merged_paths = [
        project_root / "DATA_AFTER_PREPROCESSING" / "dataset_merged.csv",
        project_root / "DIAGNOSTIC_PIPELINE" / "outputs" / "latest" / "dataset_merged.csv",
        project_root / "pipeline_execution" / "outputs" / "latest" / "dataset_merged.csv",
    ]
    
    possible_diagnostic_paths = [
        project_root / "DIAGNOSTIC_PIPELINE" / "outputs" / "latest" / "diagnostic_report.csv",
        project_root / "pipeline_execution" / "outputs" / "latest" / "diagnostic_report.csv",
    ]
    
    possible_config_paths = [
        project_root / "DIAGNOSTIC_PIPELINE" / "outputs" / "latest" / "config.yaml",
        project_root / "pipeline_execution" / "outputs" / "latest" / "config.yaml",
    ]
    
    # Find first existing merged dataset
    merged_path = None
    for path in possible_merged_paths:
        if path.exists():
            merged_path = path
            break
    
    # Find first existing diagnostic report
    diagnostics_path = None
    for path in possible_diagnostic_paths:
        if path.exists():
            diagnostics_path = path
            break
    
    # Find first existing config
    config_path = None
    for path in possible_config_paths:
        if path.exists():
            config_path = path
            break

    datasets: Dict[str, Any] = {
        "merged": pd.read_csv(merged_path, encoding="utf-8") if merged_path and merged_path.exists() else None,
        "diagnostics": pd.read_csv(diagnostics_path, encoding="utf-8") if diagnostics_path and diagnostics_path.exists() else None,
        "config": config_path.read_text(encoding="utf-8") if config_path and config_path.exists() else None,
    }
    return datasets
