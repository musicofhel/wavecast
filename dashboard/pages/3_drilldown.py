"""Page 3: Per-ticker drilldown of forward test predictions."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.config import COLORS, SECTOR_FOR_TICKER, TICKERS, fmt_pct, fmt_pnl  # noqa: E402
from dashboard.data.loader import load_forward_predictions  # noqa: E402


def render():
    df = load_forward_predictions()

    if df is None or df.empty:
        st.info("No forward test data yet.")
        return

    ticker = st.selectbox("Ticker", TICKERS, index=0)
    sector = SECTOR_FOR_TICKER.get(ticker, "unknown")
    st.caption(f"Sector: {sector}")

    ticker_df = df[df["ticker"] == ticker].copy()
    if ticker_df.empty:
        st.warning(f"No predictions for {ticker}.")
        return

    resolved = ticker_df[ticker_df["resolved_at"].notna()]
    a2i_resolved = resolved[resolved.get("a2i_trade", False) == True]  # noqa: E712

    # Metric row
    n_preds = len(ticker_df)
    n_a2i = len(a2i_resolved)

    acc = None
    if len(resolved) > 0 and "correct" in resolved.columns:
        acc = resolved["correct"].sum() / len(resolved)

    a2i_acc = None
    if n_a2i > 0:
        a2i_acc = a2i_resolved["correct"].sum() / n_a2i

    cum_pnl = 0.0
    dir_resolved = resolved[resolved["predicted_direction"] != 0]
    if not dir_resolved.empty:
        cum_pnl = (dir_resolved["predicted_direction"] * dir_resolved["actual_return"]).sum()

    cols = st.columns(5)
    with cols[0]:
        st.metric("Predictions", n_preds)
    with cols[1]:
        st.metric("A2i Trades", n_a2i)
    with cols[2]:
        st.metric("Accuracy", fmt_pct(acc) if acc is not None else "\u2014")
    with cols[3]:
        st.metric("A2i Accuracy", fmt_pct(a2i_acc) if a2i_acc is not None else "\u2014")
    with cols[4]:
        st.metric("Cum PnL", fmt_pnl(cum_pnl))

    # Prediction timeline scatter
    if len(resolved) > 1:
        st.subheader("Prediction Timeline")
        _render_timeline(resolved)

    # Magnitude distribution
    if "predicted_magnitude" in ticker_df.columns:
        mags = ticker_df["predicted_magnitude"].dropna()
        if len(mags) > 0:
            st.subheader("Magnitude Distribution")
            _render_magnitude_hist(ticker_df)

    # Full predictions table
    st.subheader(f"All Predictions ({len(ticker_df)})")
    _render_table(ticker_df.sort_values("timestamp", ascending=False))


def _render_timeline(resolved: pd.DataFrame):
    """Scatter: x=timestamp, y=actual_return, color=correct/wrong, size=magnitude."""
    resolved = resolved.sort_values("resolved_at").copy()

    # Separate correct and wrong
    correct = resolved[resolved["correct"] == True]  # noqa: E712
    wrong = resolved[resolved["correct"] == False]  # noqa: E712

    fig = go.Figure()

    for subset, name, color in [
        (correct, "Correct", COLORS["up"]),
        (wrong, "Wrong", COLORS["down"]),
    ]:
        if subset.empty:
            continue

        mag = subset.get("predicted_magnitude")
        sizes = None
        if mag is not None:
            mag_vals = mag.fillna(0.005)
            sizes = (mag_vals / mag_vals.max() * 15 + 5).tolist()

        fig.add_trace(
            go.Scatter(
                x=subset["resolved_at"],
                y=subset["actual_return"],
                mode="markers",
                name=name,
                marker=dict(color=color, size=sizes or 8, opacity=0.8),
                text=[
                    f"Dir: {r.get('predicted_direction')}, "
                    f"Actual: {r.get('actual_return', 0):.4f}"
                    for _, r in subset.iterrows()
                ],
                hoverinfo="text+x",
            )
        )

    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
    fig.update_layout(
        template="plotly_dark",
        height=350,
        yaxis_title="Actual Return",
        margin=dict(l=40, r=20, t=20, b=30),
    )
    st.plotly_chart(fig, width="stretch")


def _render_magnitude_hist(df: pd.DataFrame):
    """Histogram of predicted magnitudes, split by tercile."""
    df = df[df["predicted_magnitude"].notna()].copy()
    if df.empty:
        return

    large = df[df["magnitude_tercile"] == "large"]["predicted_magnitude"]
    rest = df[df["magnitude_tercile"] != "large"]["predicted_magnitude"]

    fig = go.Figure()
    if not rest.empty:
        fig.add_trace(
            go.Histogram(
                x=rest,
                name="Medium/Small",
                marker_color=COLORS["neutral"],
                opacity=0.5,
            )
        )
    if not large.empty:
        fig.add_trace(
            go.Histogram(
                x=large,
                name="Large (A2i)",
                marker_color=COLORS["a2i"],
                opacity=0.8,
            )
        )

    fig.update_layout(
        barmode="overlay",
        template="plotly_dark",
        height=300,
        xaxis_title="Predicted Magnitude (Signal B)",
        margin=dict(l=40, r=20, t=20, b=30),
    )
    st.plotly_chart(fig, width="stretch")


def _render_table(df: pd.DataFrame):
    """Full predictions table for a ticker."""
    rows = []
    for _, r in df.iterrows():
        direction = r.get("predicted_direction", 0)
        dir_sym = {1: "\u2191", -1: "\u2193", 0: "\u2014"}.get(direction, "?")

        actual_dir = r.get("actual_direction")
        if pd.isna(actual_dir) or actual_dir is None:
            actual_sym = "pending"
        else:
            actual_sym = {1: "\u2191", -1: "\u2193", 0: "\u2014"}.get(int(actual_dir), "?")

        correct = r.get("correct")
        correct_sym = "\u2014"
        if correct is not None and not (isinstance(correct, float) and pd.isna(correct)):
            correct_sym = "\u2713" if correct else "\u2717"

        actual_ret = r.get("actual_return")
        ret_str = f"{actual_ret:+.2%}" if actual_ret is not None and not pd.isna(actual_ret) else "\u2014"

        mag = r.get("predicted_magnitude")
        mag_str = f"{mag:.4f}" if mag is not None and not pd.isna(mag) else "\u2014"

        tercile = r.get("magnitude_tercile", "\u2014")
        tercile_str = {"large": "L", "medium": "M", "small": "S"}.get(str(tercile), "\u2014")

        a2i = r.get("a2i_trade", False)

        ts = r.get("timestamp")
        ts_str = ts.strftime("%m-%d %H:%M") if pd.notna(ts) else "\u2014"

        rows.append({
            "Time": ts_str,
            "Dir": dir_sym,
            "Mag": mag_str,
            "Tercile": tercile_str,
            "A2i": "\u2713" if a2i else "",
            "Actual": actual_sym,
            "Result": correct_sym,
            "Return": ret_str,
        })

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


st.title("Ticker Drilldown")
render()
