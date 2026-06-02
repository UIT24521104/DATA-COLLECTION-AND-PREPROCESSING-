from __future__ import annotations

from typing import Iterable, Tuple

import streamlit as st


def render_metric_grid(items: Iterable[Tuple[st.delta_generator.DeltaGenerator, str, str]]) -> None:
    for container, label, value in items:
        with container:
            st.metric(label, value)
