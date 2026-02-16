"""DTW alignment and distance matrix visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import ShapeletMatch


def plot_dtw_alignment(
    match: ShapeletMatch,
    query: NDArray[np.float64],
    target: NDArray[np.float64],
    save_path: Path | None = None,
) -> None:
    """Plot DTW warping path alignment between query and target."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: aligned series with warping lines
    ax = axes[0]
    ax.plot(query, label="Query", color="#2196F3", linewidth=1.5)
    offset = np.max(np.abs(query)) + np.max(np.abs(target)) + 0.5
    ax.plot(target - offset, label="Target", color="#FF5722", linewidth=1.5)

    for i, j in match.warping_path[::3]:  # Draw every 3rd line to avoid clutter
        if i < len(query) and j < len(target):
            ax.plot(
                [i, j], [query[i], target[j] - offset],
                color="gray", alpha=0.3, linewidth=0.5,
            )

    ax.set_title(f"DTW Alignment (dist={match.distance:.4f})", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: warping path matrix
    ax = axes[1]
    path = np.array(match.warping_path)
    ax.plot(path[:, 0], path[:, 1], color="#9C27B0", linewidth=2)
    ax.plot([0, max(len(query), len(target))],
            [0, max(len(query), len(target))],
            "k--", alpha=0.3)
    ax.set_xlabel("Query Index")
    ax.set_ylabel("Target Index")
    ax.set_title("Warping Path")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_distance_matrix(
    matrix: NDArray[np.float64],
    labels: list[str] | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot a DTW distance matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="DTW Distance")

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

    ax.set_title("Pairwise DTW Distance Matrix")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
