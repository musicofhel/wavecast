"""Reusable metric display components."""

from __future__ import annotations

import streamlit as st

from dashboard.config import VERDICT_COLORS


def metric_row(metrics: list[tuple[str, str, str | None]]):
    """Display a row of st.metric cards.

    metrics: list of (label, value, delta)
    """
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics, strict=False):
        with col:
            st.metric(label, value, delta=delta)


def verdict_badge(verdict: str) -> str:
    """Return colored HTML badge for an experiment verdict."""
    color = VERDICT_COLORS.get(verdict, "#64748b")
    return (
        f'<span style="background-color:{color};padding:2px 8px;'
        f'border-radius:4px;color:white;font-weight:bold;font-size:0.8em">'
        f"{verdict}</span>"
    )
