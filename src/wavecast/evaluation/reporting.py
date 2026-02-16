"""Performance reports and visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import BacktestResult


def generate_report(backtest: BacktestResult) -> str:
    """Generate a text performance report."""
    lines = [
        f"{'=' * 60}",
        f"  Backtest Report: {backtest.model_name}",
        f"  Ticker: {backtest.ticker or 'N/A'}",
        f"{'=' * 60}",
        "",
    ]

    for key, value in backtest.metrics.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            if "accuracy" in key or "return" in key or "drawdown" in key:
                lines.append(f"  {label:<30} {value:>10.2%}")
            else:
                lines.append(f"  {label:<30} {value:>10.4f}")
        else:
            lines.append(f"  {label:<30} {value!s:>10}")

    lines.extend([
        "",
        f"  {'─' * 56}",
        f"  Total Trades: {len(backtest.returns)}",
        f"  Final Equity: {backtest.equity_curve[-1]:,.2f}"
        if len(backtest.equity_curve) > 0
        else "  Final Equity: N/A",
        f"{'=' * 60}",
    ])

    return "\n".join(lines)


def plot_equity_curve(
    backtest: BacktestResult, save_path: Path | None = None
) -> None:
    """Plot equity curve over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(backtest.equity_curve, linewidth=1.5, color="#2196F3")
    ax.set_title(f"Equity Curve — {backtest.model_name}", fontsize=14)
    ax.set_xlabel("Trade")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=backtest.equity_curve[0], color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_returns_distribution(
    backtest: BacktestResult, save_path: Path | None = None
) -> None:
    """Plot returns histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(backtest.returns, bins=50, color="#4CAF50", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    ax.axvline(x=np.mean(backtest.returns), color="blue", linestyle="-", alpha=0.7,
               label=f"Mean: {np.mean(backtest.returns):.4f}")
    ax.set_title(f"Returns Distribution — {backtest.model_name}", fontsize=14)
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_predictions_vs_actual(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    save_path: Path | None = None,
) -> None:
    """Plot predicted vs actual values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.4, s=10, color="#FF9800")
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    axes[0].plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs Actual")
    axes[0].grid(True, alpha=0.3)

    # Time series overlay
    axes[1].plot(y_true, label="Actual", alpha=0.8, linewidth=1)
    axes[1].plot(y_pred, label="Predicted", alpha=0.8, linewidth=1)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Predictions Over Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
