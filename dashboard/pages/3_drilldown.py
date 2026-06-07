"""Page 3: Per-ticker drilldown — backtest stats, forecast, prediction history, ranking."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.charts import softmax_bar_chart, ticker_ranking_bars  # noqa: E402
from dashboard.config import COLORS, SECTOR_FOR_TICKER, TICKERS, fmt_pct, fmt_sharpe  # noqa: E402
from dashboard.data.loader import (  # noqa: E402
    load_forward_predictions,
    load_pnl_simulation,
    load_ticker_breakdown,
)


def render():
    # --- Data loading ---
    breakdown = load_ticker_breakdown()
    fwd_df = load_forward_predictions()
    pnl_sim = load_pnl_simulation()

    bin_midpoints = None
    if pnl_sim and "meta" in pnl_sim:
        bin_midpoints = pnl_sim["meta"].get("bin_midpoints")

    # Build lookup: ticker name -> backtest dict
    bt_lookup: dict[str, dict] = {}
    bt_list: list[dict] = []
    if breakdown and "per_ticker" in breakdown:
        bt_list = breakdown["per_ticker"]
        for t in bt_list:
            bt_lookup[t["name"]] = t

    # --- Ticker selector ---
    ticker = st.selectbox("Ticker", TICKERS, index=0)
    sector = SECTOR_FOR_TICKER.get(ticker, "unknown")
    st.caption(f"Sector: {sector}")

    # ── 2025 Backtest Performance ──
    bt = bt_lookup.get(ticker)
    if bt:
        st.subheader("2025 Backtest Performance")
        cols = st.columns(5)
        with cols[0]:
            st.metric("Accuracy", fmt_pct(bt.get("econ_dir_accuracy")))
        with cols[1]:
            st.metric("Sharpe (raw)", fmt_sharpe(bt.get("sharpe_raw")))
        with cols[2]:
            st.metric("Sharpe (costs)", fmt_sharpe(bt.get("sharpe_with_costs")))
        with cols[3]:
            st.metric("Transition", fmt_pct(bt.get("transition_accuracy")))
        with cols[4]:
            st.metric("N samples", f"{bt.get('n_valid', 0):,}")

    # ── Latest Forecast (softmax) ──
    if fwd_df is not None and not fwd_df.empty:
        ticker_fwd = fwd_df[fwd_df["ticker"] == ticker].copy()
        if not ticker_fwd.empty:
            # Find most recent prediction with softmax_probs
            with_probs = ticker_fwd[ticker_fwd["softmax_probs"].apply(lambda x: x is not None)]
            if not with_probs.empty and bin_midpoints is not None:
                latest = with_probs.sort_values("timestamp", ascending=False).iloc[0]
                probs = latest["softmax_probs"]
                if isinstance(probs, list) and len(probs) == len(bin_midpoints):
                    st.subheader("Latest Forecast")
                    # Direction annotation
                    argmax_idx = probs.index(max(probs))
                    direction_label = _BIN_LABELS[argmax_idx]
                    confidence = probs[argmax_idx]
                    ts = latest.get("timestamp")
                    ts_str = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else ""
                    st.caption(
                        f"Prediction: **{direction_label}** ({confidence:.1%}) — {ts_str}"
                    )
                    fig = softmax_bar_chart(probs, bin_midpoints)
                    st.plotly_chart(fig, use_container_width=True)

    # ── Prediction History (forward test) ──
    if fwd_df is not None and not fwd_df.empty:
        ticker_fwd = fwd_df[fwd_df["ticker"] == ticker].copy()
        if not ticker_fwd.empty:
            resolved = ticker_fwd[ticker_fwd["resolved_at"].notna()]
            if len(resolved) > 1:
                st.subheader("Prediction Timeline")
                _render_timeline(resolved)

            # Magnitude distribution
            if "predicted_magnitude" in ticker_fwd.columns:
                mags = ticker_fwd["predicted_magnitude"].dropna()
                if len(mags) > 0:
                    st.subheader("Magnitude Distribution")
                    _render_magnitude_hist(ticker_fwd)

            # Full predictions table
            st.subheader(f"All Predictions ({len(ticker_fwd)})")
            _render_table(ticker_fwd.sort_values("timestamp", ascending=False))

    # ── All Tickers Ranked by Sharpe ──
    if bt_list:
        st.subheader("All Tickers Ranked by Sharpe")
        fig = ticker_ranking_bars(bt_list, ticker)
        st.plotly_chart(fig, use_container_width=True)


_BIN_LABELS = ["Strong Down", "Down", "Flat", "Up", "Strong Up"]


def _render_timeline(resolved: pd.DataFrame):
    """Scatter: x=timestamp, y=actual_return, color=correct/wrong, size=magnitude."""
    resolved = resolved.sort_values("resolved_at").copy()

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
    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)


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

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


st.title("Ticker Drilldown")
render()
