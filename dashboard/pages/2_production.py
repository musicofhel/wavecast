"""Page 2: Production system reference — validated A2i performance."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import plotly.figure_factory as ff  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.charts import bar_chart  # noqa: E402
from dashboard.config import fmt_pct, fmt_pnl, fmt_sharpe  # noqa: E402
from dashboard.data.loader import (  # noqa: E402
    load_crosstab,
    load_model_config,
    load_pnl_simulation,
    load_validation,
)


def render():
    val = load_validation()
    pnl = load_pnl_simulation()

    if val is None and pnl is None:
        st.info("No production validation data found.")
        return

    st.caption("Model: d1_augmented_v1 | Trained: 2021-2025 | Filter: Large tercile by Signal B")

    # 2026 OOS vs 2025 Test
    if val is not None:
        st.subheader("2026 OOS vs 2025 Test")
        _render_comparison(val)

    # 8-config bar chart
    if pnl is not None:
        st.subheader("8-Config Sharpe Comparison (2025 Test)")
        _render_config_bars(pnl)

    # A2i vs B2i head-to-head
    if pnl is not None:
        st.subheader("A2i vs B2i Head-to-Head")
        _render_head_to_head(pnl)

    # Reversal filter failure
    if val is not None and "reversal_breakdown" in val:
        _render_reversal_failure(val)

    # Cross-tab heatmap
    crosstab = load_crosstab()
    if crosstab is not None:
        st.subheader("Cross-Tab: Magnitude \u00d7 Transition (2025 Test)")
        _render_crosstab(crosstab)

    # System spec
    with st.expander("System Specification"):
        config = load_model_config()
        if config:
            st.json(config)
        else:
            st.caption("Model config not found.")


def _render_comparison(val: dict):
    """2026 vs 2025 side-by-side metrics."""
    # A2i uses result_2026 (filtered) vs reference_2025 (filtered = B2i)
    v26 = val.get("result_2026", {})
    v25 = val.get("reference_2025", {})

    if not v26 or not v25:
        st.caption("Incomplete validation data.")
        return

    metrics = [
        ("Trades", v26.get("n_trades"), v25.get("n_trades"), lambda x: str(int(x))),
        ("Accuracy", v26.get("accuracy"), v25.get("accuracy"), fmt_pct),
        ("Sharpe", v26.get("sharpe"), v25.get("sharpe"), fmt_sharpe),
        ("Max DD", v26.get("max_drawdown"), v25.get("max_drawdown"), fmt_pnl),
        ("Expectancy", v26.get("expectancy_per_trade"), v25.get("expectancy_per_trade"), fmt_pnl),
        ("Win Rate", v26.get("win_rate"), v25.get("win_rate"), fmt_pct),
        ("W/L Ratio", v26.get("wl_ratio"), v25.get("wl_ratio"), lambda x: f"{x:.2f}" if x else "\u2014"),
    ]

    cols = st.columns(len(metrics))
    for col, (label, v26_val, v25_val, fmt) in zip(cols, metrics, strict=False):
        with col:
            if v26_val is not None and v25_val is not None:
                delta = v26_val - v25_val
                # For max drawdown, more negative is worse
                delta_str = f"{delta:+.3f}" if abs(delta) > 0.01 else f"{delta:+.4f}"
                st.metric(label, fmt(v26_val), delta=delta_str)
            elif v26_val is not None:
                st.metric(label, fmt(v26_val))
            else:
                st.metric(label, "\u2014")

    st.caption("Blue: 2026 OOS (Jan 1 \u2013 Feb 22) | Delta vs 2025 Test")


def _render_config_bars(pnl: dict):
    """8-config Sharpe bar chart."""
    configs_data = pnl.get("configs", {})
    config_order = ["A1i", "A1ii", "A2i", "A2ii", "B1i", "B1ii", "B2i", "B2ii"]
    available = [c for c in config_order if c in configs_data]
    sharpes = [configs_data[c]["sharpe"] for c in available]

    fig = bar_chart(
        available,
        sharpes,
        highlight_x="A2i",
        title="Annualized Sharpe by Trading Configuration",
        y_title="Sharpe Ratio",
    )
    st.plotly_chart(fig, width="stretch")


def _render_head_to_head(pnl: dict):
    """A2i vs B2i comparison table."""
    configs = pnl.get("configs", {})
    a2i = configs.get("A2i", {})
    b2i = configs.get("B2i", {})

    if not a2i or not b2i:
        st.caption("A2i or B2i data not available.")
        return

    rows = [
        ("Trades", a2i.get("n_trades"), b2i.get("n_trades")),
        ("Accuracy", fmt_pct(a2i.get("accuracy")), fmt_pct(b2i.get("accuracy"))),
        ("Sharpe", fmt_sharpe(a2i.get("sharpe")), fmt_sharpe(b2i.get("sharpe"))),
        ("Max DD", fmt_pnl(a2i.get("max_drawdown")), fmt_pnl(b2i.get("max_drawdown"))),
        ("Expectancy", fmt_pnl(a2i.get("expectancy_per_trade")), fmt_pnl(b2i.get("expectancy_per_trade"))),
        ("Win Rate", fmt_pct(a2i.get("win_rate")), fmt_pct(b2i.get("win_rate"))),
    ]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**:blue[A2i (Production)]**")
    with col2:
        st.markdown("**B2i (Reference)**")

    for label, a_val, b_val in rows:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.caption(label)
        with c2:
            st.write(a_val)
        with c3:
            st.write(b_val)


def _render_reversal_failure(val: dict):
    """Reversal filter overfit callout."""
    rb = val["reversal_breakdown"]
    st.warning(
        f"**Reversal Filter Failure (Overfit)**\n\n"
        f"- Reversal accuracy: 78.9% (2025) \u2192 {rb['reversal_accuracy']:.1%} (2026)\n"
        f"- Continuation accuracy: 56.7% (2025) \u2192 {rb['continuation_accuracy']:.1%} (2026)\n"
        f"- Reversal n={rb['reversal_n']}, Continuation n={rb['continuation_n']}\n"
        f"- **Conclusion**: Structure was overfit to test set. Filter rejected."
    )


def _render_crosstab(data: dict):
    """Magnitude x Transition heatmap from CE baseline."""
    ce = data.get("ce_baseline", {})
    if not ce:
        st.caption("No cross-tab data.")
        return

    # Build 2x3 grid: rows=Transition/Continuation, cols=Large/Medium/Small
    cells = [
        ("large_transition", "large_non_transition"),
        ("medium_transition", "medium_non_transition"),
        ("small_transition", "small_non_transition"),
    ]

    z = []
    text = []
    for row_label in ["Transition", "Continuation"]:
        z_row = []
        t_row = []
        for large_key, small_key in cells:
            key = large_key if row_label == "Transition" else small_key
            cell = ce.get(key, {})
            acc = cell.get("accuracy", 0)
            n = cell.get("n_samples", 0)
            z_row.append(acc * 100)
            t_row.append(f"{acc:.1%}<br>n={n:,}")
        z.append(z_row)
        text.append(t_row)

    fig = ff.create_annotated_heatmap(
        z,
        x=["Large", "Medium", "Small"],
        y=["Transition", "Continuation"],
        annotation_text=text,
        colorscale=[[0, "#ef4444"], [0.5, "#1e293b"], [1, "#22c55e"]],
        showscale=True,
    )
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=80, r=20, t=20, b=30))
    st.plotly_chart(fig, width="stretch")


st.title("Production System")
render()
