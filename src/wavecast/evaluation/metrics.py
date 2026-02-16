"""Evaluation metrics for forecasting and backtesting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import EvaluationMetrics


def rmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> float:
    """Percentage of correct direction predictions."""
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    return float(np.mean(true_dir == pred_dir))


def hit_rate(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Alias for directional_accuracy."""
    return directional_accuracy(y_true, y_pred)


def sharpe_ratio(
    returns: NDArray[np.float64], risk_free: float = 0.0, periods: int = 252
) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free / periods
    if np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(periods))


def max_drawdown(equity_curve: NDArray[np.float64]) -> float:
    """Maximum drawdown as a positive fraction."""
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / np.where(peak > 0, peak, 1.0)
    return float(np.max(drawdown))


def profit_factor(returns: NDArray[np.float64]) -> float:
    """Sum of positive returns / abs(sum of negative returns)."""
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    if len(losses) == 0 or np.sum(np.abs(losses)) == 0:
        return float("inf") if len(gains) > 0 else 0.0
    return float(np.sum(gains) / np.abs(np.sum(losses)))


def compute_all(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    returns: NDArray[np.float64] | None = None,
    equity_curve: NDArray[np.float64] | None = None,
) -> EvaluationMetrics:
    """Compute all evaluation metrics."""
    if returns is None:
        returns = y_pred  # Use predictions as proxy returns

    if equity_curve is None:
        equity_curve = np.cumprod(1 + returns)

    return EvaluationMetrics(
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        directional_accuracy=directional_accuracy(y_true, y_pred),
        sharpe_ratio=sharpe_ratio(returns),
        max_drawdown=max_drawdown(equity_curve),
        hit_rate=hit_rate(y_true, y_pred),
        profit_factor=profit_factor(returns),
    )
