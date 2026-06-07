"""Reusable Plotly chart builders."""

from __future__ import annotations

from collections.abc import Sequence

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


_BIN_LABELS = ["Strong Down", "Down", "Flat", "Up", "Strong Up"]
_BIN_COLORS = ["#dc2626", "#f87171", "#94a3b8", "#4ade80", "#16a34a"]


def softmax_bar_chart(
    probs: Sequence[float],
    bin_midpoints: Sequence[float],
    title: str = "Softmax Forecast",
    height: int = 300,
) -> go.Figure:
    """5-bar chart showing softmax probability distribution over return quantile bins."""
    annotations = [f"{m * 100:+.2f}%" for m in bin_midpoints]
    text_labels = [f"{p:.1%}" for p in probs]

    fig = go.Figure(
        go.Bar(
            x=_BIN_LABELS,
            y=list(probs),
            marker_color=_BIN_COLORS,
            text=text_labels,
            textposition="outside",
            hovertext=[f"{lbl}: {ann}" for lbl, ann in zip(_BIN_LABELS, annotations, strict=True)],
            hoverinfo="text+y",
        )
    )

    # Annotate bin midpoint returns below each bar
    for i, ann in enumerate(annotations):
        fig.add_annotation(
            x=_BIN_LABELS[i],
            y=-0.02,
            text=ann,
            showarrow=False,
            font=dict(size=10, color="#94a3b8"),
            yanchor="top",
        )

    fig.update_layout(
        title=title,
        yaxis=dict(title="Probability", range=[0, max(probs) * 1.3 + 0.05]),
        height=height,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=50),
    )
    return fig


def ticker_ranking_bars(
    ticker_data: list[dict],
    selected_ticker: str,
    metric_key: str = "sharpe_with_costs",
    title: str = "All Tickers Ranked by Sharpe (with costs)",
    height: int = 500,
) -> go.Figure:
    """Horizontal bar chart ranking all tickers by a metric. Selected ticker highlighted."""
    sorted_data = sorted(ticker_data, key=lambda d: d.get(metric_key, 0))
    names = [d["name"] for d in sorted_data]
    values = [d.get(metric_key, 0) for d in sorted_data]
    colors = [
        COLORS["a2i"] if n == selected_ticker else COLORS["neutral"] for n in names
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Sharpe (with costs)",
        height=height,
        template="plotly_dark",
        margin=dict(l=60, r=40, t=40, b=30),
    )
    return fig
