"""All data loading functions. Reads JSON, JSONL, NPZ from disk."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from dashboard.config import (
    AUDIT_DIR,
    BASELINE_LOCK,
    EXP3_DIR,
    FEATURE_TESTS_DIR,
    MODEL_DIR,
    PNL_SIM_JSON,
    PREDICTIONS_JSONL,
    RUN_LOG,
    VALIDATION_JSON,
)


@st.cache_data(ttl=60)
def load_forward_predictions() -> pd.DataFrame | None:
    """Load forward test predictions from JSONL. TTL=60s for auto-refresh."""
    if not PREDICTIONS_JSONL.exists():
        return None
    text = PREDICTIONS_JSONL.read_text().strip()
    if not text:
        return None
    records = [json.loads(line) for line in text.split("\n") if line.strip()]
    if not records:
        return None
    df = pd.DataFrame(records)
    for col in ["timestamp", "target_timestamp", "resolved_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data
def load_validation() -> dict | None:
    """Load 2026 OOS validation results."""
    if not VALIDATION_JSON.exists():
        return None
    return json.loads(VALIDATION_JSON.read_text())


@st.cache_data
def load_pnl_simulation() -> dict | None:
    """Load 8-config PnL simulation results."""
    if not PNL_SIM_JSON.exists():
        return None
    return json.loads(PNL_SIM_JSON.read_text())


@st.cache_data
def load_baseline_lock() -> dict | None:
    """Load frozen CE baseline reference."""
    if not BASELINE_LOCK.exists():
        return None
    return json.loads(BASELINE_LOCK.read_text())


@st.cache_data
def load_experiment_results() -> pd.DataFrame | None:
    """Load all experiment results from audit directories, normalizing schemas."""
    records = []

    # Phase 12 experiments (exp3)
    if EXP3_DIR.exists():
        for f in sorted(EXP3_DIR.glob("*_results.json")):
            if f.name == "crosstab_results.json":
                continue
            try:
                data = json.loads(f.read_text())
                rec = _normalize_exp3(data, f.name)
                if rec:
                    records.append(rec)
            except (json.JSONDecodeError, KeyError):
                continue

    # Round 1+2 experiments (feature_tests)
    if FEATURE_TESTS_DIR.exists():
        for f in sorted(FEATURE_TESTS_DIR.glob("*_results.json")):
            # Skip validation/diagnostic files
            if f.name in (
                "2026_validation_results.json",
                "three_diagnostics_results.json",
                "stacking_results.json",
            ):
                continue
            try:
                data = json.loads(f.read_text())
                rec = _normalize_feature_test(data, f.name)
                if rec:
                    records.append(rec)
            except (json.JSONDecodeError, KeyError):
                continue

    if not records:
        return None

    return pd.DataFrame(records)


def _normalize_exp3(data: dict, filename: str) -> dict | None:
    """Normalize a Phase 12 experiment result to common schema."""
    name = data.get("name", filename.replace("_results.json", ""))

    # exp3 files have varying structures — extract challenger metrics
    # Pattern: {name_avg: {econ_dir, transition_acc, ...}, baseline_avg: {...}}
    challenger_key = None
    for k in data:
        if k.endswith("_avg") and k != "baseline_avg":
            challenger_key = k
            break

    if challenger_key and isinstance(data[challenger_key], dict):
        m = data[challenger_key]
    else:
        return None

    verdict = data.get("verdict", "")
    # Extract verdict category
    verdict_cat = "FAIL"
    for v in ("PASS", "INTERESTING", "MARGINAL"):
        if v in str(verdict).upper():
            verdict_cat = v
            break

    return {
        "name": name,
        "round": "Phase 12",
        "econ_dir": m.get("econ_dir"),
        "transition": m.get("transition_acc"),
        "large_move": m.get("large_move_acc"),
        "flat_pct": m.get("pred_dist", {}).get("flat"),
        "sharpe_cost": m.get("sharpe_costs"),
        "verdict": verdict_cat,
        "verdict_text": str(verdict),
        "source": filename,
    }


def _normalize_feature_test(data: dict, filename: str) -> dict | None:
    """Normalize a Round 1 or 2 feature test result to common schema."""
    name = data.get("name", filename.replace("_results.json", ""))
    verdict = data.get("verdict", "")

    verdict_cat = "FAIL"
    for v in ("PASS", "INTERESTING", "MARGINAL"):
        if v in str(verdict).upper():
            verdict_cat = v
            break

    # Multiple schema patterns — try each
    econ_dir = transition = large_move = flat_pct = sharpe = None

    # Pattern 1: {baseline_mean: {...}, per_epsilon/per_*: {...}} (label_smoothing style)
    # Use baseline as fallback, prefer challenger if available
    if "baseline_mean" in data:
        bm = data["baseline_mean"]
        econ_dir = bm.get("econ_dir")
        transition = bm.get("transition")
        sharpe = bm.get("sharpe_costs")

    # Pattern 2: {baseline_avg: {...}} + challenger_avg
    if "baseline_avg" in data and isinstance(data["baseline_avg"], dict):
        ba = data["baseline_avg"]
        econ_dir = ba.get("econ_dir", econ_dir)
        transition = ba.get("transition_acc", ba.get("transition", transition))
        large_move = ba.get("large_move_acc", large_move)
        sharpe = ba.get("sharpe_costs", sharpe)
        flat_pct = ba.get("pred_dist", {}).get("flat", flat_pct)

    # Pattern 3: top-level or challenger-specific metrics (prefer over baseline)
    for key in ("challenger_avg", "mean_metrics", "ensemble_avg_metrics"):
        if key in data and isinstance(data[key], dict):
            cm = data[key]
            econ_dir = cm.get("econ_dir", econ_dir)
            transition = cm.get("transition_acc", cm.get("transition", transition))
            large_move = cm.get("large_move_acc", large_move)
            sharpe = cm.get("sharpe_costs", sharpe)
            flat_pct = cm.get("pred_dist", {}).get("flat", flat_pct)
            break

    # Pattern 4: bolt_ceiling style — {ce_baseline: {...}, ceiling: {...}}
    if "ceiling" in data and isinstance(data["ceiling"], dict):
        ce = data.get("ce_baseline", {})
        econ_dir = ce.get("econ_dir", econ_dir)
        sharpe = ce.get("sharpe_costs", sharpe)

    # Pattern 5: direct top-level keys
    econ_dir = data.get("econ_dir", econ_dir)
    transition = data.get("transition", transition)
    sharpe = data.get("sharpe_costs", data.get("sharpe_cost", sharpe))

    return {
        "name": name,
        "round": "Round 1+2",
        "econ_dir": econ_dir,
        "transition": transition,
        "large_move": large_move,
        "flat_pct": flat_pct,
        "sharpe_cost": sharpe,
        "verdict": verdict_cat,
        "verdict_text": str(verdict),
        "source": filename,
    }


@st.cache_data
def load_model_config() -> dict | None:
    """Load production model configuration."""
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text())


@st.cache_data
def load_crosstab() -> dict | None:
    """Load cross-tab (magnitude x transition) results."""
    path = EXP3_DIR / "crosstab_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def load_ticker_breakdown() -> dict | None:
    """Load per-ticker backtest breakdown from D1 audit."""
    path = AUDIT_DIR / "d1_ticker_breakdown.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_run_log_tail(n_lines: int = 20) -> str | None:
    """Load last N lines of the forward test run log."""
    if not RUN_LOG.exists():
        return None
    lines = RUN_LOG.read_text().strip().split("\n")
    return "\n".join(lines[-n_lines:])
