"""Configuration classes for signal generation and backtesting."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class SignalConfig(BaseSettings):
    """Signal generation configuration."""

    confidence_threshold: float = 0.0
    calibration_method: str = "none"
    temperature: float = 1.0


class PositionSizingConfig(BaseSettings):
    """Position sizing configuration."""

    method: str = "fixed"
    max_position: float = 1.0
    kelly_fraction: float = 0.5
    min_position: float = 0.0
    lookback: int = 50


class TransactionCostConfig(BaseSettings):
    """Transaction cost configuration."""

    commission_rate: float = 0.001
    spread_bps: float = 2.0
    slippage_bps: float = 1.0


class SignalBacktestConfig(BaseSettings):
    """Full signal backtest configuration."""

    initial_capital: float = 100000.0
    signal: SignalConfig = SignalConfig()
    position_sizing: PositionSizingConfig = PositionSizingConfig()
    costs: TransactionCostConfig = TransactionCostConfig()
