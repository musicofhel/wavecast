"""Backtest grid harness (Phase B / B1).

Runs a grid of {ticker x interval x trading rule} cells through the existing
SignalBacktest framework at fixed round-trip costs, recording per-cell metrics
(including flat-prediction rate and a persistence baseline on the same window)
into a JSONL results ledger.

Trading rules operate on realized returns only (no model needed) except the
``model`` rule, which consumes externally supplied predicted directions.
Economic direction = sign of actual_return; the symbolic token-level metric is
never used here (Phase 7 audit trap).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wavecast.signals.backtest import SignalBacktest
from wavecast.signals.costs import TransactionCostModel
from wavecast.signals.position import PositionSizer
from wavecast.signals.types import SignalSeries, TradingSignal

# House rule: 7bps round-trip cost, split evenly across the two legs.
COST_BPS_ROUND_TRIP = 7.0


def _series(
    directions: NDArray[np.float64],
    timestamps: NDArray[np.datetime64],
    confidence: float = 1.0,
) -> SignalSeries:
    n = len(directions)
    signals = [
        TradingSignal(
            timestamp=timestamps[i],
            direction=int(directions[i]),
            confidence=confidence,
            raw_probability=confidence,
            token_id=-1,
            horizon=1,
        )
        for i in range(n)
    ]
    return SignalSeries(signals=signals)


def _sign(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sign(x).astype(np.float64)


def rule_persistence(
    returns: NDArray[np.float64], timestamps: NDArray[np.datetime64], **params: int
) -> SignalSeries:
    """Predict next direction = sign of last realized return."""
    dirs = np.zeros(len(returns))
    dirs[1:] = _sign(returns[:-1])
    return _series(dirs, timestamps)


def rule_mean_reversion(
    returns: NDArray[np.float64], timestamps: NDArray[np.datetime64], **params: int
) -> SignalSeries:
    """Predict next direction = -sign of last realized return."""
    dirs = np.zeros(len(returns))
    dirs[1:] = -_sign(returns[:-1])
    return _series(dirs, timestamps)


def rule_momentum(
    returns: NDArray[np.float64], timestamps: NDArray[np.datetime64], lookback: int = 3
) -> SignalSeries:
    """Sign of trailing ``lookback``-period cumulative return."""
    dirs = np.zeros(len(returns), dtype=np.float64)
    for i in range(lookback, len(returns)):
        dirs[i] = float(np.sign(np.sum(returns[i - lookback : i])))
    return _series(dirs, timestamps)


def rule_model(
    returns: NDArray[np.float64],
    timestamps: NDArray[np.datetime64],
    predicted_directions: NDArray[np.float64] | None = None,
    confidences: NDArray[np.float64] | None = None,
    **params: int,
) -> SignalSeries:
    """Consume externally supplied model predictions (aligned with returns)."""
    if predicted_directions is None:
        raise ValueError("rule_model requires predicted_directions")
    if len(predicted_directions) != len(returns):
        raise ValueError(
            f"predictions length ({len(predicted_directions)}) != returns ({len(returns)})"
        )
    if confidences is None:
        return _series(_sign(predicted_directions), timestamps)
    return SignalSeries(
        signals=[
            TradingSignal(
                timestamp=timestamps[i],
                direction=int(np.sign(predicted_directions[i])),
                confidence=float(confidences[i]),
                raw_probability=float(confidences[i]),
                token_id=-1,
                horizon=1,
            )
            for i in range(len(returns))
        ]
    )


TRADING_RULES: dict[str, Callable[..., SignalSeries]] = {
    "persistence": rule_persistence,
    "mean_reversion": rule_mean_reversion,
    "momentum": rule_momentum,
    "model": rule_model,
}


@dataclass
class GridCell:
    """One cell of the backtest grid."""

    ticker: str
    interval: str
    rule: str
    params: dict = field(default_factory=dict)

    def key(self) -> str:
        payload = json.dumps(
            {"ticker": self.ticker, "interval": self.interval, "rule": self.rule, "params": self.params},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def cost_model_for(cost_bps_round_trip: float = COST_BPS_ROUND_TRIP) -> TransactionCostModel:
    """Split a round-trip cost in bps evenly over the two legs."""
    per_leg = cost_bps_round_trip / 2.0
    return TransactionCostModel(commission_rate=0.0, spread_bps=per_leg, slippage_bps=0.0)


def run_cell(
    cell: GridCell,
    returns: NDArray[np.float64],
    timestamps: NDArray[np.datetime64],
    predicted_directions: NDArray[np.float64] | None = None,
    confidences: NDArray[np.float64] | None = None,
    cost_bps: float = COST_BPS_ROUND_TRIP,
) -> dict:
    """Run one grid cell; returns a result dict for the results ledger."""
    if cell.rule not in TRADING_RULES:
        raise ValueError(f"unknown rule '{cell.rule}'; known: {sorted(TRADING_RULES)}")
    kwargs = dict(cell.params)
    if cell.rule == "model":
        kwargs["predicted_directions"] = predicted_directions
        kwargs["confidences"] = confidences
    series = TRADING_RULES[cell.rule](returns, timestamps, **kwargs)
    bt = SignalBacktest(
        cost_model=cost_model_for(cost_bps),
        position_sizer=PositionSizer(method="fixed", max_position=1.0),
    )
    result = bt.run(series, returns, timestamps=timestamps)
    directions = series.directions
    flat_rate = float(np.mean(directions == 0))
    return {
        "cell_key": cell.key(),
        "ticker": cell.ticker,
        "interval": cell.interval,
        "rule": cell.rule,
        "params": cell.params,
        "n_bars": len(returns),
        "start": str(timestamps[0]) if len(timestamps) else None,
        "end": str(timestamps[-1]) if len(timestamps) else None,
        "cost_bps_round_trip": cost_bps,
        "sharpe": float(result.metrics.get("sharpe_ratio", 0.0)),
        "total_return": float(result.metrics.get("total_return", 0.0)),
        "expectancy": float(result.metrics.get("expectancy", 0.0)),
        "win_rate": float(result.metrics.get("win_rate", 0.0)),
        "num_trades": int(result.metrics.get("num_trades", 0)),
        "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
        "flat_rate": flat_rate,
    }


def persistence_baseline_sharpe(
    returns: NDArray[np.float64],
    timestamps: NDArray[np.datetime64],
    cost_bps: float = COST_BPS_ROUND_TRIP,
) -> float:
    """Persistence-rule Sharpe on the same window — the required baseline."""
    cell = GridCell(ticker="baseline", interval="baseline", rule="persistence")
    return run_cell(cell, returns, timestamps, cost_bps=cost_bps)["sharpe"]


def run_grid(
    cells: list[GridCell],
    data: dict[tuple[str, str], tuple[NDArray[np.datetime64], NDArray[np.float64]]],
    predictions: dict[tuple[str, str], tuple[NDArray[np.float64], NDArray[np.float64]]] | None = None,
    cost_bps: float = COST_BPS_ROUND_TRIP,
    include_baseline: bool = True,
) -> list[dict]:
    """Run every cell against the matching (timestamps, returns) series.

    Args:
        cells: grid cells to evaluate
        data: keyed by (ticker, interval) -> (timestamps, period returns)
        predictions: optional (ticker, interval) -> (predicted_dirs, confidences),
            required by any ``model`` rule cell
        cost_bps: round-trip transaction cost in basis points
        include_baseline: also record the persistence baseline per data series

    Returns:
        List of result dicts (cells first, then baselines with rule="persistence").
    """
    results: list[dict] = []
    for cell in cells:
        key = (cell.ticker, cell.interval)
        if key not in data:
            raise KeyError(f"no data loaded for {key}")
        timestamps, returns = data[key]
        preds = (predictions or {}).get(key)
        row = run_cell(
            cell,
            returns,
            timestamps,
            predicted_directions=preds[0] if preds else None,
            confidences=preds[1] if preds else None,
            cost_bps=cost_bps,
        )
        if include_baseline and cell.rule != "persistence":
            row["baseline_persistence_sharpe"] = persistence_baseline_sharpe(
                returns, timestamps, cost_bps
            )
        results.append(row)
    return results


def append_results(results: list[dict], path: str | Path = "research/results.jsonl") -> None:
    """Append grid results to the JSONL ledger (creating parent dirs as needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for row in results:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def load_results(path: str | Path = "research/results.jsonl") -> list[dict]:
    """Load all rows from the results ledger."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]
