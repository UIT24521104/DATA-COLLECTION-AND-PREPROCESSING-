from __future__ import annotations

from pathlib import Path

import streamlit as st


def load_css(file_path: Path) -> None:
    """Load a CSS file into the Streamlit app."""
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def set_page_config() -> None:
    """Set Streamlit page configuration."""
    st.set_page_config(
        page_title="World Bank Data Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
