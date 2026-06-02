from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_chart_controls() -> str:
    return st.radio("Chart type", ["distribution", "correlation"], horizontal=True)


def render_distribution_chart(df: pd.DataFrame, cols: List[str]) -> None:
    choice = st.selectbox("Distribution chart", ["histogram", "box"], index=0)
    
    num_features = len(cols)
    cols_per_row = 2
    num_rows = (num_features + cols_per_row - 1) // cols_per_row
    
    if choice == "histogram":
        # Display multiple histograms for all selected columns
        for row in range(num_rows):
            cols_row = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                col_num = row * cols_per_row + col_idx
                if col_num < num_features:
                    with cols_row[col_idx]:
                        fig = px.histogram(df, x=cols[col_num], nbins=30, title=f"Histogram: {cols[col_num]}")
                        st.plotly_chart(fig, use_container_width=True)
    else:
        # Display multiple boxplots for all selected columns
        for row in range(num_rows):
            cols_row = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                col_num = row * cols_per_row + col_idx
                if col_num < num_features:
                    with cols_row[col_idx]:
                        fig = px.box(df, y=cols[col_num], points="outliers", title=cols[col_num])
                        st.plotly_chart(fig, use_container_width=True)


def render_correlation_chart(df: pd.DataFrame, cols: List[str]) -> None:
    corr = df[cols].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmin=-1, zmax=1))
    st.plotly_chart(fig, width='stretch')
