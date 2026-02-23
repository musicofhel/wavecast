"""Reusable Plotly chart builders."""

from __future__ import annotations

import plotly.graph_objects as go

from dashboard.config import COLORS

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    margin=dict(l=40, r=20, t=30, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)


def equity_curve(dates_all, pnl_all, dates_a2i, pnl_a2i, height=400):
    """Equity curve with All (gray) and A2i (blue) traces."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates_all,
            y=pnl_all,
            mode="lines",
            name="All trades",
            line=dict(color=COLORS["all"], width=1.5),
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates_a2i,
            y=pnl_a2i,
            mode="lines",
            name="A2i (production)",
            line=dict(color=COLORS["a2i"], width=2.5),
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
    fig.update_layout(height=height, yaxis_title="Cumulative Return", **_LAYOUT_DEFAULTS)
    return fig


def rolling_accuracy(
    dates_all,
    acc_all,
    dates_a2i,
    acc_a2i,
    ref_a2i=0.667,
    ref_all=0.622,
    height=350,
):
    """Rolling accuracy with reference lines."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates_all,
            y=acc_all,
            mode="lines",
            name="All trades",
            line=dict(color=COLORS["all"], width=1.5),
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates_a2i,
            y=acc_a2i,
            mode="lines",
            name="A2i",
            line=dict(color=COLORS["a2i"], width=2.5),
        )
    )

    fig.add_hline(
        y=ref_a2i,
        line_dash="dash",
        line_color=COLORS["accent"],
        annotation_text=f"A2i 2026 ref ({ref_a2i:.1%})",
    )
    fig.add_hline(
        y=ref_all,
        line_dash="dash",
        line_color=COLORS["neutral"],
        annotation_text=f"Unfiltered ref ({ref_all:.1%})",
    )

    fig.update_layout(
        height=height,
        yaxis=dict(range=[0.40, 0.85], title="Accuracy"),
        **_LAYOUT_DEFAULTS,
    )
    return fig


def bar_chart(
    x: list,
    y: list,
    highlight_x: str | None = None,
    title: str = "",
    y_title: str = "",
    height: int = 350,
):
    """Simple bar chart with one bar highlighted in blue."""
    colors = [
        COLORS["a2i"] if xi == highlight_x else COLORS["neutral"] for xi in x
    ]
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        height=height,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig
