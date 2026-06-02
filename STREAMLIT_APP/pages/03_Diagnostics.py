from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.data_loader import load_latest_datasets
from services.diagnostics_service import render_diagnostics_view
from utils.ui import set_page_config, load_css


ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    set_page_config()
    load_css(ROOT_DIR / "streamlit_app" / "assets" / "styles.css")
    
    st.title("📊 Diagnostics & Analysis")
    
    datasets = load_latest_datasets(ROOT_DIR)
    diagnostics = datasets.get("diagnostics")

    if diagnostics is None:
        st.warning("Diagnostic report not found.")
        return

    st.subheader("Diagnostic Report")
    st.markdown("""
    Báo cáo chẩn đoán toàn diện về chất lượng dữ liệu, phân phối, và các phát hiện outlier.
    """)
    render_diagnostics_view(diagnostics)


if __name__ == "__main__":
    main()
