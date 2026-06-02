from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.data_loader import load_latest_datasets
from components.metrics import render_metric_grid
from utils.ui import set_page_config, load_css


APP_TITLE = "WorldBank Interactive Dashboard"
ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    set_page_config()
    load_css(ROOT_DIR / "STREAMLIT_APP" / "assets" / "styles.css")
    
    st.title(APP_TITLE)
    st.caption("Interactive dashboard for Phase 1 diagnostics, data exploration, and visualization.")

    datasets = load_latest_datasets(ROOT_DIR)
    merged_df = datasets.get("merged")
    diagnostics = datasets.get("diagnostics")

    if merged_df is None:
        st.error("""
        ❌ No dataset found! 
        
        Please ensure one of these files exists:
        - `DATA_AFTER_PREPROCESSING/dataset_merged.csv`
        - `DIAGNOSTIC_PIPELINE/outputs/latest/dataset_merged.csv`
        
        To generate the data, run:
        ```bash
        cd DIAGNOSTIC_PIPELINE
        python run_pipeline.py full --scenario default
        ```
        """)
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    render_metric_grid(
        [
            (col1, "Rows", f"{len(merged_df):,}"),
            (col2, "Columns", f"{len(merged_df.columns):,}"),
            (col3, "Missing values", f"{int(merged_df.isna().sum().sum()):,}"),
            (col4, "Status", "Phase 1 Complete"),
        ]
    )

    st.subheader("Data Preview")
    st.dataframe(merged_df.head(50), width='stretch')

    if diagnostics is not None:
        st.subheader("Diagnostics Summary")
        st.dataframe(diagnostics, width='stretch')

if __name__ == "__main__":
    main()
