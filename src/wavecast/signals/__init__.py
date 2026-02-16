"""Signal generation and backtesting for WaveCast."""

from wavecast.signals.backtest import SignalBacktest
from wavecast.signals.config import (
    PositionSizingConfig,
    SignalBacktestConfig,
    SignalConfig,
    TransactionCostConfig,
)
from wavecast.signals.costs import TransactionCostModel
from wavecast.signals.generator import SignalGenerator
from wavecast.signals.position import PositionSizer
from wavecast.signals.types import (
    SignalBacktestResult,
    SignalSeries,
    TradeRecord,
    TradingSignal,
)

__all__ = [
    "SignalBacktest",
    "SignalGenerator",
    "PositionSizer",
    "TransactionCostModel",
    "TradingSignal",
    "SignalSeries",
    "TradeRecord",
    "SignalBacktestResult",
    "SignalConfig",
    "PositionSizingConfig",
    "TransactionCostConfig",
    "SignalBacktestConfig",
]
