"""Fractal analysis visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import HurstResult, MFDFAResult


def plot_hurst(
    result: HurstResult,
    save_path: Path | None = None,
) -> None:
    """Plot Hurst exponent log-log regression."""
    fig, ax = plt.subplots(figsize=(8, 6))

    levels = np.arange(1, len(result.level_variances) + 1)
    log_scales = np.log2(2.0**levels)
    log_vars = np.log2(result.level_variances)

    # Filter out invalid values
    valid = np.isfinite(log_vars)
    log_scales = log_scales[valid]
    log_vars = log_vars[valid]

    ax.scatter(log_scales, log_vars, color="#2196F3", s=60, zorder=5)

    # Regression line
    if len(log_scales) >= 2:
        fit_x = np.linspace(log_scales.min(), log_scales.max(), 100)
        slope = 2 * result.hurst_exponent - 1
        fit_y = slope * fit_x + result.intercept
        ax.plot(fit_x, fit_y, "r--", linewidth=1.5, alpha=0.7,
                label=f"H={result.hurst_exponent:.3f}, R²={result.r_squared:.3f}")

    ax.set_xlabel("log₂(scale)", fontsize=12)
    ax.set_ylabel("log₂(variance)", fontsize=12)
    ax.set_title(
        f"Wavelet-based Hurst Estimation — {result.regime.value}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_mfdfa_spectrum(
    result: MFDFAResult,
    save_path: Path | None = None,
) -> None:
    """Plot MFDFA singularity spectrum f(alpha)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Generalized Hurst H(q)
    ax = axes[0]
    ax.plot(result.q_values, result.hurst_q, "o-", color="#9C27B0",
            markersize=4, linewidth=1.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="H=0.5 (random walk)")
    ax.set_xlabel("q", fontsize=12)
    ax.set_ylabel("H(q)", fontsize=12)
    ax.set_title("Generalized Hurst Exponent", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Singularity spectrum
    ax = axes[1]
    valid = np.isfinite(result.alpha) & np.isfinite(result.f_alpha)
    ax.plot(result.alpha[valid], result.f_alpha[valid], "o-", color="#FF5722",
            markersize=4, linewidth=1.5)
    ax.set_xlabel("α (singularity strength)", fontsize=12)
    ax.set_ylabel("f(α)", fontsize=12)
    ax.set_title(f"Singularity Spectrum (width={result.spectrum_width:.3f})", fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Multifractal DFA Analysis", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_rolling_hurst(
    hurst_values: NDArray[np.float64],
    timestamps: NDArray[np.float64] | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot rolling Hurst exponent over time."""
    fig, ax = plt.subplots(figsize=(14, 5))

    x = timestamps if timestamps is not None else np.arange(len(hurst_values))

    ax.plot(x, hurst_values, color="#2196F3", linewidth=1)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random walk (H=0.5)")
    ax.axhline(y=0.55, color="green", linestyle=":", alpha=0.5, label="Trending threshold")
    ax.axhline(y=0.45, color="orange", linestyle=":", alpha=0.5, label="Mean-reverting threshold")

    ax.fill_between(x, 0.45, 0.55, alpha=0.08, color="gray", label="Random walk zone")

    ax.set_xlabel("Time")
    ax.set_ylabel("Hurst Exponent")
    ax.set_title("Rolling Hurst Exponent", fontsize=13)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
