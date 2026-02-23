"""Page 4: 66-experiment research archive."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.config import COLORS, fmt_pct, fmt_sharpe  # noqa: E402
from dashboard.data.loader import load_baseline_lock, load_experiment_results  # noqa: E402


def render():
    df = load_experiment_results()

    if df is None or df.empty:
        st.info("No experiment results found in audit directories.")
        return

    baseline = load_baseline_lock()

    # Summary stats
    n_total = len(df)
    n_pass = len(df[df["verdict"] == "PASS"])
    n_interesting = len(df[df["verdict"] == "INTERESTING"])
    n_fail = len(df[df["verdict"] == "FAIL"])

    st.caption(
        f"{n_total} experiments | {n_pass} PASS | {n_interesting} INTERESTING | "
        f"{n_fail} FAIL"
    )
    if baseline:
        st.caption(
            f"CE Baseline: econ_dir={fmt_pct(baseline.get('econ_dir'))}, "
            f"transition={fmt_pct(baseline.get('transition'))}, "
            f"Sharpe={fmt_sharpe(baseline.get('sharpe_cost'))}"
        )

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        round_filter = st.selectbox(
            "Round",
            ["All"] + sorted(df["round"].dropna().unique().tolist()),
        )
    with col2:
        verdict_filter = st.selectbox(
            "Verdict",
            ["All"] + sorted(df["verdict"].dropna().unique().tolist()),
        )

    filtered = df.copy()
    if round_filter != "All":
        filtered = filtered[filtered["round"] == round_filter]
    if verdict_filter != "All":
        filtered = filtered[filtered["verdict"] == verdict_filter]

    # Results table
    st.subheader(f"Results ({len(filtered)} experiments)")
    _render_table(filtered)

    # Scatter: econ_dir vs transition
    scatter_df = filtered.dropna(subset=["econ_dir", "transition"])
    if len(scatter_df) > 1:
        st.subheader("Econ Dir vs Transition Accuracy")
        _render_scatter(scatter_df, baseline)

    # Experiment detail on selection
    if not filtered.empty:
        with st.expander("Experiment Detail (click to expand)"):
            exp_name = st.selectbox(
                "Select experiment",
                filtered["name"].tolist(),
            )
            if exp_name:
                row = filtered[filtered["name"] == exp_name].iloc[0]
                st.json(row.dropna().to_dict())


def _render_table(df):
    """Sortable experiment results table."""
    display_cols = ["name", "round", "econ_dir", "transition", "large_move", "flat_pct", "sharpe_cost", "verdict"]
    available_cols = [c for c in display_cols if c in df.columns]
    table_df = df[available_cols].copy()

    # Format percentage columns
    for col in ["econ_dir", "transition", "large_move", "flat_pct"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(lambda x: fmt_pct(x) if x is not None else "\u2014")

    if "sharpe_cost" in table_df.columns:
        table_df["sharpe_cost"] = table_df["sharpe_cost"].apply(
            lambda x: fmt_sharpe(x) if x is not None else "\u2014"
        )

    st.dataframe(table_df, width="stretch", hide_index=True)


def _render_scatter(df, baseline):
    """Scatter plot: econ_dir vs transition, colored by verdict."""
    # Filter to valid Sharpe values for sizing
    plot_df = df.copy()
    if "sharpe_cost" in plot_df.columns:
        plot_df["sharpe_abs"] = plot_df["sharpe_cost"].fillna(0).abs().clip(lower=1)
    else:
        plot_df["sharpe_abs"] = 5

    color_map = {
        "PASS": COLORS["pass"],
        "FAIL": COLORS["fail"],
        "INTERESTING": COLORS["interesting"],
        "MARGINAL": COLORS["marginal"],
    }

    fig = px.scatter(
        plot_df,
        x="econ_dir",
        y="transition",
        color="verdict",
        size="sharpe_abs",
        hover_name="name",
        hover_data=["flat_pct", "large_move", "sharpe_cost"],
        color_discrete_map=color_map,
        template="plotly_dark",
        height=500,
    )

    # Baseline crosshairs
    if baseline:
        bl_ed = baseline.get("econ_dir", 0.636)
        bl_tr = baseline.get("transition", 0.523)
        fig.add_vline(x=bl_ed, line_dash="dash", line_color=COLORS["accent"], opacity=0.5)
        fig.add_hline(y=bl_tr, line_dash="dash", line_color=COLORS["accent"], opacity=0.5)
        fig.add_annotation(
            x=bl_ed,
            y=bl_tr,
            text="CE Baseline",
            showarrow=True,
            arrowhead=2,
            font=dict(color=COLORS["accent"]),
        )

    fig.update_layout(
        xaxis_title="Economic Directional Accuracy",
        yaxis_title="Transition Accuracy",
        margin=dict(l=40, r=20, t=30, b=30),
    )
    st.plotly_chart(fig, width="stretch")


st.title("Experiment Archive")
render()
