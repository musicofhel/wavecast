"""Shapelet pattern visualization."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavecast.core.types import Shapelet


def plot_shapelet(
    shapelet: Shapelet,
    save_path: Path | None = None,
) -> None:
    """Plot a single shapelet."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(shapelet.coefficients, color="#9C27B0", linewidth=1.5)
    ax.fill_between(
        range(len(shapelet.coefficients)),
        shapelet.coefficients,
        alpha=0.2,
        color="#9C27B0",
    )
    ax.set_title(
        f"Shapelet {shapelet.id} — Level {shapelet.wavelet_level}, "
        f"IG={shapelet.information_gain:.4f}, Label={shapelet.label.value}",
        fontsize=11,
    )
    ax.set_xlabel("Index")
    ax.set_ylabel("Coefficient Value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_shapelet_gallery(
    shapelets: list[Shapelet],
    cols: int = 4,
    save_path: Path | None = None,
) -> None:
    """Plot a gallery grid of shapelets."""
    n = len(shapelets)
    if n == 0:
        return

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.asarray(axes).flatten().tolist()

    label_colors = {"up": "#4CAF50", "down": "#F44336", "flat": "#FFC107"}

    for i, shapelet in enumerate(shapelets):
        ax = axes_flat[i]
        color = label_colors.get(shapelet.label.value, "#9C27B0")
        ax.plot(shapelet.coefficients, color=color, linewidth=1.2)
        ax.fill_between(
            range(len(shapelet.coefficients)),
            shapelet.coefficients,
            alpha=0.15,
            color=color,
        )
        ax.set_title(
            f"L{shapelet.wavelet_level} {shapelet.label.value} "
            f"IG={shapelet.information_gain:.3f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    # Hide empty axes
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(f"Shapelet Gallery ({n} patterns)", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
