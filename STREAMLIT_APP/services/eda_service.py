from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from components.charts import render_chart_controls, render_distribution_chart, render_correlation_chart


def render_eda_panel(df: pd.DataFrame) -> None:
    numeric_cols: List[str] = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for EDA.")
        return

    selected_cols = st.multiselect("Choose variables", numeric_cols, default=numeric_cols[:4])
    if not selected_cols:
        st.info("Select at least one variable.")
        return

    chart_type = render_chart_controls()
    if chart_type == "distribution":
        render_distribution_chart(df, selected_cols)
    elif chart_type == "correlation":
        render_correlation_chart(df, selected_cols)
