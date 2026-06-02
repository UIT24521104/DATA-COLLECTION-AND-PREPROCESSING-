from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_diagnostics_view(diagnostics: Any) -> None:
    if hasattr(diagnostics, "head"):
        st.dataframe(diagnostics, width='stretch')
        return

    if isinstance(diagnostics, Dict):
        st.json(diagnostics)
    else:
        st.write(diagnostics)
