"""Wavelet scalogram and DWT level visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import WaveletDecomposition


def plot_scalogram(
    coefficients: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    times: NDArray[np.float64] | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot CWT scalogram as a heatmap."""
    fig, ax = plt.subplots(figsize=(14, 6))

    if times is None:
        times = np.arange(coefficients.shape[1], dtype=np.float64)

    im = ax.pcolormesh(
        times, frequencies, np.abs(coefficients),
        cmap="magma", shading="auto",
    )
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Time")
    ax.set_title("CWT Scalogram")
    ax.set_yscale("log")
    fig.colorbar(im, ax=ax, label="Magnitude")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_dwt_levels(
    decomp: WaveletDecomposition,
    save_path: Path | None = None,
) -> None:
    """Plot each DWT level as a subplot."""
    n_plots = decomp.level + 2  # approx + details + original placeholder
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots), sharex=False)

    # Approximation
    axes[0].plot(decomp.approximation, color="#2196F3", linewidth=0.8)
    axes[0].set_title(f"Approximation (cA{decomp.level})", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Detail levels (coarsest to finest)
    for i in range(decomp.level):
        level = decomp.level - i
        coeffs = decomp.detail_at_level(level)
        ax = axes[i + 1]
        ax.plot(coeffs, color="#FF5722", linewidth=0.6, alpha=0.8)
        period = 2**level
        ax.set_title(f"Detail cD{level} (~{period}d period)", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Label bottom axis
    axes[-1].set_xlabel("Coefficient Index")

    fig.suptitle(
        f"DWT Decomposition — {decomp.wavelet}, {decomp.level} levels",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
