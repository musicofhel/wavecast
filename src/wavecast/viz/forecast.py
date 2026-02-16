"""Forecast visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import ForecastResult


def plot_forecast(
    result: ForecastResult,
    actual: NDArray[np.float64] | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot forecast predictions with optional confidence bands."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(result.predictions))

    if actual is not None:
        ax.plot(actual, label="Actual", color="#2196F3", linewidth=1.5, alpha=0.8)

    ax.plot(x, result.predictions, label="Forecast", color="#FF5722",
            linewidth=1.5, linestyle="--")

    if result.confidence_lower is not None and result.confidence_upper is not None:
        ax.fill_between(
            x, result.confidence_lower, result.confidence_upper,
            alpha=0.15, color="#FF5722", label="95% CI",
        )

    ax.set_title(
        f"Forecast — {result.ticker} ({result.model_name}, h={result.horizon})",
        fontsize=13,
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_predictions_scatter(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    save_path: Path | None = None,
) -> None:
    """Scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#FF9800")

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]

    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title("Actual vs Predicted", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
