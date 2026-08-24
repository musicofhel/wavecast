"""Signal generation and backtesting for WaveCast."""

from wavecast.signals.backtest import SignalBacktest
from wavecast.signals.config import (
    PositionSizingConfig,
    SignalBacktestConfig,
    SignalConfig,
    TransactionCostConfig,
)
from wavecast.signals.costs import TransactionCostModel
from wavecast.signals.generator import ReturnSignalGenerator, SignalGenerator
from wavecast.signals.grid import (
    TRADING_RULES,
    GridCell,
    append_results,
    load_results,
    run_cell,
    run_grid,
)
from wavecast.signals.position import PositionSizer
from wavecast.signals.types import (
    SignalBacktestResult,
    SignalSeries,
    TradeRecord,
    TradingSignal,
)

__all__ = [
    "SignalBacktest",
    "GridCell",
    "TRADING_RULES",
    "append_results",
    "load_results",
    "run_cell",
    "run_grid",
    "ReturnSignalGenerator",
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
