"""Dashboard configuration: paths, constants, color scheme, formatters."""

from pathlib import Path

# Data paths
FORWARD_TEST_DIR = Path.home() / ".wavecast" / "forward_tests" / "d1_forward_v1"
PREDICTIONS_JSONL = FORWARD_TEST_DIR / "predictions.jsonl"
RUN_LOG = FORWARD_TEST_DIR / "run.log"

AUDIT_DIR = Path.home() / ".wavecast" / "audit"
PRODUCTION_DIR = AUDIT_DIR / "production"
VALIDATION_JSON = PRODUCTION_DIR / "2026_validation.json"
PNL_SIM_JSON = PRODUCTION_DIR / "pnl_simulation.json"
BASELINE_PREDICTIONS_NPZ = PRODUCTION_DIR / "ce_baseline_predictions.npz"

EXP3_DIR = AUDIT_DIR / "exp3"
FEATURE_TESTS_DIR = AUDIT_DIR / "feature_tests"
BASELINE_LOCK = Path.home() / "wavecast" / "scripts" / "exp3" / "baseline_lock.json"

MODEL_DIR = Path.home() / ".wavecast" / "models" / "d1_augmented_v1"

# Color scheme (plotly_dark compatible)
COLORS = {
    "up": "#22c55e",
    "down": "#ef4444",
    "neutral": "#64748b",
    "a2i": "#3b82f6",
    "all": "#94a3b8",
    "accent": "#f59e0b",
    "pass": "#22c55e",
    "fail": "#ef4444",
    "marginal": "#f59e0b",
    "interesting": "#8b5cf6",
}

VERDICT_COLORS = {
    "PASS": COLORS["pass"],
    "FAIL": COLORS["fail"],
    "MARGINAL": COLORS["marginal"],
    "INTERESTING": COLORS["interesting"],
}


def fmt_pct(v, decimals=1):
    if v is None:
        return "\u2014"
    return f"{v * 100:.{decimals}f}%" if abs(v) <= 1 else f"{v:.{decimals}f}%"


def fmt_sharpe(v):
    if v is None:
        return "\u2014"
    return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"


def fmt_pnl(v):
    if v is None:
        return "\u2014"
    return f"+{v:.4f}" if v >= 0 else f"{v:.4f}"


TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM", "GS", "BAC",
    "XOM", "CVX", "COP",
    "JNJ", "UNH", "PFE",
    "SPY", "QQQ",
    "GLD", "SLV", "USO", "UNG",
]

SECTORS = {
    "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "finance": ["JPM", "GS", "BAC"],
    "energy": ["XOM", "CVX", "COP"],
    "healthcare": ["JNJ", "UNH", "PFE"],
    "broad_etf": ["SPY", "QQQ"],
    "commodity_etf": ["GLD", "SLV", "USO", "UNG"],
}

SECTOR_FOR_TICKER = {}
for _sector, _tickers in SECTORS.items():
    for _t in _tickers:
        SECTOR_FOR_TICKER[_t] = _sector
