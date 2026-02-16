"""Data types for signal generation and backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class TradingSignal:
    """A single trading signal derived from token prediction.

    Attributes:
        timestamp: Signal timestamp
        direction: +1 (long), -1 (short), 0 (flat/no-trade)
        confidence: Signal confidence in [0, 1]
        raw_probability: Raw softmax probability of the predicted token
        token_id: Predicted token ID
        horizon: Prediction horizon (number of steps ahead)
        ticker: Asset ticker symbol
    """
    timestamp: np.datetime64
    direction: int  # +1, -1, 0
    confidence: float
    raw_probability: float
    token_id: int
    horizon: int = 1
    ticker: str = ""


@dataclass
class SignalSeries:
    """Ordered collection of trading signals for one asset and horizon.

    Attributes:
        signals: List of TradingSignal objects
        ticker: Asset ticker symbol
        horizon: Prediction horizon
    """
    signals: list[TradingSignal]
    ticker: str = ""
    horizon: int = 1

    @property
    def directions(self) -> NDArray[np.float64]:
        """Signal directions as array."""
        return np.array([s.direction for s in self.signals], dtype=np.float64)

    @property
    def confidences(self) -> NDArray[np.float64]:
        """Signal confidences as array."""
        return np.array([s.confidence for s in self.signals], dtype=np.float64)

    @property
    def timestamps(self) -> NDArray[np.datetime64]:
        """Signal timestamps as array."""
        return np.array([s.timestamp for s in self.signals], dtype="datetime64[ns]")

    def __len__(self) -> int:
        return len(self.signals)


@dataclass
class TradeRecord:
    """Record of a single executed trade.

    Attributes:
        entry_timestamp: Trade entry time
        exit_timestamp: Trade exit time
        direction: +1 (long) or -1 (short)
        position_size: Position size as fraction of capital
        gross_return: Return before costs
        net_return: Return after costs
        commission: Commission cost
        spread_cost: Spread cost
        slippage_cost: Slippage cost
        confidence: Signal confidence at entry
    """
    entry_timestamp: np.datetime64
    exit_timestamp: np.datetime64
    direction: int
    position_size: float
    gross_return: float
    net_return: float
    commission: float
    spread_cost: float
    slippage_cost: float
    confidence: float


@dataclass
class SignalBacktestResult:
    """Result of a signal-based backtest.

    Attributes:
        returns: Net returns per period
        positions: Position sizes per period
        equity_curve: Equity value over time
        timestamps: Timestamps for each period
        trades: List of individual trade records
        metrics: Dict of computed risk/return metrics
        config: Backtest configuration dict
    """
    returns: NDArray[np.float64]
    positions: NDArray[np.float64]
    equity_curve: NDArray[np.float64]
    timestamps: NDArray[np.datetime64]
    trades: list[TradeRecord] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
