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


def sortino_ratio(
    returns: NDArray[np.float64], risk_free: float = 0.0, periods: int = 252
) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    excess = returns - risk_free / periods
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(downside) * np.sqrt(periods))


def calmar_ratio(
    returns: NDArray[np.float64], periods: int = 252
) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    equity = np.cumprod(1 + returns)
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    ann_return = float(np.mean(returns) * periods)
    return ann_return / mdd


def value_at_risk(
    returns: NDArray[np.float64], confidence: float = 0.95
) -> float:
    """Historical Value at Risk (VaR) at given confidence level. Returns positive number."""
    if len(returns) == 0:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def conditional_var(
    returns: NDArray[np.float64], confidence: float = 0.95
) -> float:
    """Conditional VaR (Expected Shortfall) — mean of losses beyond VaR. Returns positive number."""
    if len(returns) == 0:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return float(-np.mean(tail))


def win_rate(returns: NDArray[np.float64]) -> float:
    """Fraction of positive returns."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def avg_win_loss_ratio(returns: NDArray[np.float64]) -> float:
    """Average winning trade / average losing trade (absolute values)."""
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    return float(np.mean(wins) / np.mean(np.abs(losses)))


def expectancy(returns: NDArray[np.float64]) -> float:
    """Expected value per trade: win_rate * avg_win - loss_rate * avg_loss."""
    if len(returns) == 0:
        return 0.0
    wr = win_rate(returns)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0
    return wr * avg_win - (1 - wr) * avg_loss


def tail_ratio(returns: NDArray[np.float64]) -> float:
    """Ratio of 95th percentile to absolute 5th percentile."""
    if len(returns) < 20:
        return 0.0
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    if p5 == 0:
        return 0.0
    return float(p95 / abs(p5))
