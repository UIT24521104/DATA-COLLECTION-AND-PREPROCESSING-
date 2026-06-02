from __future__ import annotations

from pathlib import Path

import streamlit as st

from services.data_loader import load_latest_datasets
from components.filters import render_basic_filters, apply_basic_filters
from utils.ui import set_page_config, load_css


ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    set_page_config()
    load_css(ROOT_DIR / "streamlit_app" / "assets" / "styles.css")
    
    st.title("Data Explorer")
    datasets = load_latest_datasets(ROOT_DIR)
    df = datasets.get("merged")

    if df is None:
        st.warning("Merged dataset not found.")
        return

    filtered_df, filter_state = render_basic_filters(df)
    final_df = apply_basic_filters(filtered_df, filter_state)

    st.caption("Query and filter the merged dataset interactively.")
    st.dataframe(final_df, width='stretch')


if __name__ == "__main__":
    main()
