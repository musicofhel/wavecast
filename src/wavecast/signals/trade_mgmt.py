"""Trade-management variants (Phase B / B4).

Builds on the B1 grid harness: holding-period overlays on a rule's raw
directions, and sizing variants (flat vs volatility-scaled) for the backtest.
The motivation is B1/B2's finding that per-bar flipping is cost-dominated at
7bps — holding a position for N bars amortizes the round-trip cost over more
exposure.

Rule-based only (persistence / mean_reversion focus per the B2 report); no
model required.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.signals.backtest import SignalBacktest
from wavecast.signals.grid import COST_BPS_ROUND_TRIP, GridCell, cost_model_for
from wavecast.signals.position import PositionSizer


def apply_holding(
    directions: NDArray[np.float64], hold: int
) -> NDArray[np.float64]:
    """Hold each raw direction for ``hold`` bars before re-evaluating.

    The raw direction is sampled at bars 0, hold, 2*hold, ... and forward-filled
    in between, so position changes (and their costs) happen at most once per
    ``hold`` bars instead of every bar. hold < 1 raises ValueError; hold == 1 is
    the identity.
    """
    if hold < 1:
        raise ValueError(f"hold must be >= 1, got {hold}")
    d = np.asarray(directions, dtype=np.float64)
    if hold == 1 or len(d) == 0:
        return d.copy()
    out = np.zeros(len(d), dtype=np.float64)
    anchors = np.arange(0, len(d), hold)
    for i, start in enumerate(anchors):
        end = anchors[i + 1] if i + 1 < len(anchors) else len(d)
        out[start:end] = d[start]
    return out


def vol_scaled_confidence(
    returns: NDArray[np.float64],
    lookback: int = 20,
    target_vol: float = 0.01,
) -> NDArray[np.float64]:
    """Confidence in [0, 1] that scales position inversely to trailing vol.

    confidence[i] = clip(target_vol / rolling_std(returns[i-lookback:i]), 0, 1).
    Bars with insufficient history get confidence 0 (flat until lookback bars).
    """
    r = np.asarray(returns, dtype=np.float64)
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    conf = np.zeros(len(r), dtype=np.float64)
    for i in range(len(r)):
        if i < lookback:
            continue
        window = r[i - lookback : i]
        std = float(np.std(window))
        if std > 0:
            conf[i] = min(target_vol / std, 1.0)
    return conf


def run_mgmt_cell(
    cell: GridCell,
    returns: NDArray[np.float64],
    timestamps: NDArray[np.datetime64],
    hold: int = 1,
    sizing: str = "flat",
    vol_lookback: int = 20,
    target_vol: float = 0.01,
    cost_bps: float = COST_BPS_ROUND_TRIP,
) -> dict:
    """Run one cell with trade management applied.

    Args:
        cell: grid cell (rule + params), evaluated via the B1 harness rules
        returns/timestamps: the return series to trade on
        hold: holding period overlay (bars) on the rule's raw directions
        sizing: "flat" (full position) or "vol_scaled" (linear sizer driven by
            inverse-vol confidence)
        vol_lookback/target_vol: parameters of the vol-scaling
        cost_bps: round-trip transaction cost

    Returns:
        Result dict in the B1 ledger schema plus ``hold``, ``sizing`` and
        ``turnover`` (fraction of bars with a position change).
    """
    from wavecast.signals.grid import TRADING_RULES  # local: avoid import cycle
    from wavecast.signals.types import SignalSeries, TradingSignal

    def make_series(dirs: NDArray[np.float64], confs: NDArray[np.float64]) -> SignalSeries:
        return SignalSeries(
            signals=[
                TradingSignal(
                    timestamp=timestamps[i],
                    direction=int(dirs[i]),
                    confidence=float(confs[i]),
                    raw_probability=float(confs[i]),
                    token_id=-1,
                    horizon=1,
                )
                for i in range(len(dirs))
            ]
        )

    if cell.rule not in TRADING_RULES:
        raise ValueError(f"unknown rule '{cell.rule}'; known: {sorted(TRADING_RULES)}")
    if cell.rule == "model":
        raise ValueError("run_mgmt_cell supports rule-based cells only")
    raw = TRADING_RULES[cell.rule](returns, timestamps, **dict(cell.params))
    directions = apply_holding(np.asarray(raw.directions, dtype=np.float64), hold)
    if sizing == "flat":
        sizer = PositionSizer(method="fixed", max_position=1.0)
        signals = make_series(directions, np.ones(len(directions)))
    elif sizing == "vol_scaled":
        conf = vol_scaled_confidence(returns, lookback=vol_lookback, target_vol=target_vol)
        sizer = PositionSizer(method="linear", max_position=1.0)
        signals = make_series(directions, conf)
    else:
        raise ValueError(f"unknown sizing '{sizing}'; known: flat, vol_scaled")

    bt = SignalBacktest(cost_model=cost_model_for(cost_bps), position_sizer=sizer)
    result = bt.run(signals, returns, timestamps=timestamps)
    changes = int(np.sum(np.diff(directions) != 0))
    return {
        "ticker": cell.ticker,
        "interval": cell.interval,
        "rule": cell.rule,
        "params": dict(cell.params),
        "hold": hold,
        "sizing": sizing,
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
        "turnover": changes / max(len(directions) - 1, 1),
        "flat_rate": float(np.mean(directions == 0)),
    }
