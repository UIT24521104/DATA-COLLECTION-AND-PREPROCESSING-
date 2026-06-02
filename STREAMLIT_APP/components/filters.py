from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import streamlit as st


def render_basic_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    numeric_cols = [col for col in df.select_dtypes(include="number").columns.tolist() if col != "Year"]
    country_col = "Country" if "Country" in df.columns else None
    year_col = "Year" if "Year" in df.columns else None

    filters: Dict[str, object] = {}

    if country_col is not None:
        countries = sorted(df[country_col].dropna().astype(str).unique().tolist())
        filters["countries"] = st.multiselect("Country", countries, default=countries[:5] if len(countries) >= 5 else countries)

    if year_col is not None:
        years = sorted(df[year_col].dropna().astype(int).unique().tolist())
        if years:
            filters["year_range"] = st.slider("Year range", min_value=min(years), max_value=max(years), value=(min(years), max(years)))

    filters["numeric_cols"] = st.multiselect("Numeric columns", numeric_cols, default=numeric_cols[:5])
    return df, filters


def apply_basic_filters(df: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    result = df.copy()

    if "countries" in filters and "Country" in result.columns and filters["countries"]:
        result = result[result["Country"].astype(str).isin(filters["countries"])]

    if "year_range" in filters and "Year" in result.columns and filters["year_range"]:
        start_year, end_year = filters["year_range"]
        result = result[result["Year"].between(start_year, end_year)]

    if "numeric_cols" in filters and filters["numeric_cols"]:
        keep_cols = [col for col in ["Country", "Year"] if col in result.columns] + list(filters["numeric_cols"])
        result = result.loc[:, list(dict.fromkeys(keep_cols))]

    return result
