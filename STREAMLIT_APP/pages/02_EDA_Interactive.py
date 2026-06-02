from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.data_loader import load_latest_datasets
from services.eda_service import render_eda_panel
from utils.ui import set_page_config, load_css


ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    set_page_config()
    load_css(ROOT_DIR / "streamlit_app" / "assets" / "styles.css")
    
    st.title("🔍 Interactive EDA")
    datasets = load_latest_datasets(ROOT_DIR)
    df = datasets.get("merged")

    if df is None:
        st.warning("Merged dataset not found.")
        return

    st.markdown("Tạo các biểu đồ interactif được cá nhân hóa:")
    render_eda_panel(df)


if __name__ == "__main__":
    main()
