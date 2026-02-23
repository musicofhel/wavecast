"""Page 1: Live forward test monitoring."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.charts import equity_curve, rolling_accuracy  # noqa: E402
from dashboard.config import fmt_pct  # noqa: E402
from dashboard.data.loader import load_forward_predictions, load_run_log_tail  # noqa: E402


def render():
    df = load_forward_predictions()

    if df is None or df.empty:
        st.info(
            "No forward test data yet. Run `scripts/run_forward_test.sh` after "
            "market close to start accumulating predictions."
        )
        return

    resolved = df[df["resolved_at"].notna()].copy()
    pending = df[df["resolved_at"].isna()]
    n_total = len(df)
    n_resolved = len(resolved)
    n_pending = len(pending)

    # A2i subset (resolved only)
    a2i_resolved = resolved[resolved.get("a2i_trade", False) == True]  # noqa: E712
    n_a2i = len(a2i_resolved)

    # Compute A2i accuracy
    a2i_acc = None
    if n_a2i > 0 and "correct" in a2i_resolved.columns:
        a2i_correct = a2i_resolved["correct"].sum()
        a2i_acc = a2i_correct / n_a2i

    # Header
    last_ts = df["timestamp"].max()
    st.caption(f"Last prediction: {last_ts}")

    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

    # Metric cards
    cols = st.columns(5)
    with cols[0]:
        st.metric("Total Predictions", n_total)
    with cols[1]:
        st.metric("Resolved", n_resolved)
    with cols[2]:
        st.metric("Pending", n_pending)
    with cols[3]:
        a2i_pct = f"{n_a2i / n_resolved:.0%} of resolved" if n_resolved > 0 else None
        st.metric("A2i Trades", n_a2i, delta=a2i_pct)
    with cols[4]:
        if a2i_acc is not None:
            delta_vs_ref = a2i_acc - 0.667
            st.metric(
                "A2i Accuracy",
                fmt_pct(a2i_acc),
                delta=f"{delta_vs_ref:+.1%} vs 2026 ref",
            )
        else:
            st.metric("A2i Accuracy", "\u2014")

    # Equity curve
    if n_resolved > 1:
        st.subheader("Equity Curve")
        _render_equity_curve(resolved)

        st.subheader("Rolling Accuracy (20-prediction window)")
        _render_rolling_accuracy(resolved)

    # Recent predictions table
    st.subheader("Recent Predictions")
    _render_predictions_table(df.sort_values("timestamp", ascending=False).head(20))

    # Run log
    with st.expander("Run Log (last 20 lines)"):
        log = load_run_log_tail(20)
        if log:
            st.code(log, language="text")
        else:
            st.caption("No run log yet.")


def _render_equity_curve(resolved: pd.DataFrame):
    """Cumulative PnL chart: All trades vs A2i."""
    resolved = resolved.sort_values("resolved_at")

    # All directional trades
    all_dir = resolved[resolved["predicted_direction"] != 0].copy()
    if all_dir.empty:
        st.caption("No directional trades to plot.")
        return

    all_dir["pnl"] = all_dir["predicted_direction"] * all_dir["actual_return"]
    all_dir["cum_pnl"] = all_dir["pnl"].cumsum()

    # A2i trades
    a2i = resolved[resolved.get("a2i_trade", False) == True].copy()  # noqa: E712
    dates_a2i, pnl_a2i = [], []
    if not a2i.empty:
        a2i = a2i.sort_values("resolved_at")
        a2i["pnl"] = a2i["predicted_direction"] * a2i["actual_return"]
        a2i["cum_pnl"] = a2i["pnl"].cumsum()
        dates_a2i = a2i["resolved_at"]
        pnl_a2i = a2i["cum_pnl"]

    fig = equity_curve(all_dir["resolved_at"], all_dir["cum_pnl"], dates_a2i, pnl_a2i)
    st.plotly_chart(fig, width="stretch")


def _render_rolling_accuracy(resolved: pd.DataFrame):
    """Rolling accuracy chart."""
    resolved = resolved.sort_values("resolved_at").copy()
    if "correct" not in resolved.columns:
        return

    resolved["correct_int"] = resolved["correct"].astype(float)

    # All trades rolling
    all_rolling = resolved["correct_int"].rolling(20, min_periods=5).mean()

    # A2i rolling
    a2i = resolved[resolved.get("a2i_trade", False) == True].copy()  # noqa: E712
    dates_a2i, acc_a2i = [], []
    if len(a2i) >= 5:
        a2i_rolling = a2i["correct_int"].rolling(20, min_periods=5).mean()
        dates_a2i = a2i["resolved_at"]
        acc_a2i = a2i_rolling

    fig = rolling_accuracy(
        resolved["resolved_at"], all_rolling, dates_a2i, acc_a2i
    )
    st.plotly_chart(fig, width="stretch")


def _render_predictions_table(df: pd.DataFrame):
    """Show recent predictions as a formatted table."""
    rows = []
    for _, r in df.iterrows():
        direction = r.get("predicted_direction", 0)
        dir_sym = {1: "\u2191", -1: "\u2193", 0: "\u2014"}.get(direction, "?")

        actual_dir = r.get("actual_direction")
        if pd.isna(actual_dir) or actual_dir is None:
            actual_sym = "pending"
        else:
            actual_sym = {1: "\u2191", -1: "\u2193", 0: "\u2014"}.get(
                int(actual_dir), "?"
            )

        correct = r.get("correct")
        correct_sym = (
            "\u2014" if pd.isna(correct) or correct is None
            else "\u2713" if correct else "\u2717"
        )

        actual_ret = r.get("actual_return")
        ret_str = f"{actual_ret:+.2%}" if actual_ret is not None and not pd.isna(actual_ret) else "\u2014"

        mag = r.get("predicted_magnitude")
        mag_str = f"{mag:.4f}" if mag is not None and not pd.isna(mag) else "\u2014"

        tercile = r.get("magnitude_tercile", "\u2014")
        tercile_str = {"large": "L", "medium": "M", "small": "S"}.get(
            str(tercile), "\u2014"
        )

        a2i = r.get("a2i_trade", False)
        a2i_str = "\u2713" if a2i else ""

        ts = r.get("timestamp")
        ts_str = ts.strftime("%m-%d %H:%M") if pd.notna(ts) else "\u2014"

        rows.append(
            {
                "Time": ts_str,
                "Ticker": r.get("ticker", ""),
                "Dir": dir_sym,
                "Mag": mag_str,
                "Tercile": tercile_str,
                "A2i": a2i_str,
                "Actual": actual_sym,
                "Result": correct_sym,
                "Return": ret_str,
            }
        )

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


st.title("Forward Test")
render()
