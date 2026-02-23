"""WaveCast Production Dashboard — landing page.

Streamlit auto-discovers pages/ subdir for multipage navigation.
This file is the default route (/). Shows a brief overview and links
to the four main pages via the sidebar.
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="WaveCast",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.data.loader import load_forward_predictions  # noqa: E402

st.title("WaveCast Dashboard")
st.caption("Production trading system monitor. Use the sidebar to navigate.")

df = load_forward_predictions()

if df is not None and not df.empty:
    resolved = df[df["resolved_at"].notna()]
    pending = df[df["resolved_at"].isna()]

    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Predictions", len(df))
    with cols[1]:
        st.metric("Resolved", len(resolved))
    with cols[2]:
        st.metric("Pending", len(pending))

    last_ts = df["timestamp"].max()
    st.caption(f"Last prediction: {last_ts}")
else:
    st.info(
        "No forward test data yet. Run `scripts/run_forward_test.sh` after "
        "market close to start accumulating predictions."
    )

st.markdown(
    """
**Pages:**
- **Forward Test** — live prediction monitoring, equity curves, rolling accuracy
- **Production System** — validated A2i performance reference, 8-config comparison
- **Ticker Drilldown** — per-ticker prediction history and magnitude analysis
- **Experiment Archive** — 66-experiment research results browser
"""
)
