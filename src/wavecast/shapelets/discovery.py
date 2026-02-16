"""W-TSS (Wavelet-domain Thresholded Shapelet Search) algorithm."""

from __future__ import annotations

import uuid

import numpy as np
from numpy.typing import NDArray

from wavecast.core.config import ShapeletConfig
from wavecast.core.exceptions import ShapeletError
from wavecast.core.types import MarketLabel, Shapelet, WaveletDecomposition
from wavecast.shapelets.clustering import deduplicate
from wavecast.shapelets.quality import information_gain


def _extract_contiguous_regions(
    binary: NDArray[np.int_],
    min_length: int,
) -> list[tuple[int, int]]:
    """Find contiguous runs of 1s in a binary array.

    Parameters
    ----------
    binary : 1-D array of 0s and 1s.
    min_length : discard runs shorter than this.

    Returns
    -------
    List of (start, end) index pairs (end is exclusive).
    """
    regions: list[tuple[int, int]] = []
    n = len(binary)
    i = 0
    while i < n:
        if binary[i] == 1:
            start = i
            while i < n and binary[i] == 1:
                i += 1
            if i - start >= min_length:
                regions.append((start, i))
        else:
            i += 1
    return regions


def discover_shapelets(
    decomp: WaveletDecomposition,
    labels: NDArray,
    config: ShapeletConfig | None = None,
) -> list[Shapelet]:
    """Discover discriminative shapelets from a wavelet decomposition.

    Implements the W-TSS algorithm:

    1. For each DWT detail level, standardise the coefficients (z-score),
       apply a binary threshold, extract contiguous regions of significant
       coefficients, score each candidate by information gain, and keep
       the top-k.
    2. Deduplicate via DTW-based greedy clustering.
    3. Assign unique IDs and return sorted by IG descending.

    Parameters
    ----------
    decomp : wavelet decomposition result.
    labels : 1-D array of class labels aligned with the **approximation**
             coefficient length.  When the detail level has a different
             length the labels are truncated or the level is skipped.
    config : optional shapelet discovery settings.

    Returns
    -------
    List of ``Shapelet`` objects, sorted by information gain descending.

    Raises
    ------
    ShapeletError
        If the inputs are invalid.
    """
    if config is None:
        config = ShapeletConfig()

    details = decomp.details  # [cD_n, ..., cD_1]
    if not details:
        raise ShapeletError("Decomposition has no detail levels")

    candidates: list[Shapelet] = []

    for level_idx, coeffs in enumerate(details):
        # level_idx 0 → deepest detail (level = decomp.level), last → level 1
        actual_level = decomp.level - level_idx

        n = len(coeffs)
        if n < config.min_length:
            continue

        # --- 1a. Standardise (z-score) ---
        mean = float(np.mean(coeffs))
        std = float(np.std(coeffs))
        if std == 0:
            continue
        z = (coeffs - mean) / std

        # --- 1b. Binary conversion ---
        binary = (np.abs(z) >= config.z_threshold).astype(np.int_)

        # --- 1c. Extract contiguous runs ---
        regions = _extract_contiguous_regions(binary, config.min_length)
        if not regions:
            continue

        # Align labels to this level's length
        level_labels = labels[:n] if len(labels) >= n else None
        if level_labels is None or len(level_labels) < config.min_length:
            continue

        # --- 1d–e. Filter by variance, score by IG ---
        for start, end in regions:
            region_coeffs = coeffs[start:end]
            region_var = float(np.var(region_coeffs))
            if region_var < config.min_variance:
                continue

            # Score: compute IG using the coefficient values at every position
            # as a sliding-window-distance proxy.  We use a simpler approach —
            # the mean absolute coefficient in the region partitions the labels.
            region_means = np.array([
                float(np.mean(np.abs(coeffs[i : i + len(region_coeffs)])))
                for i in range(n - len(region_coeffs) + 1)
            ])
            truncated_labels = level_labels[: len(region_means)]
            if len(truncated_labels) < 2:
                continue

            # Find best split
            sorted_vals = np.sort(np.unique(region_means))
            best_ig = 0.0
            best_split = 0.0
            for si in range(len(sorted_vals) - 1):
                sp = (sorted_vals[si] + sorted_vals[si + 1]) / 2.0
                ig = information_gain(region_means, truncated_labels, sp)
                if ig > best_ig:
                    best_ig = ig
                    best_split = sp

            if best_ig < config.ig_min:
                continue

            # Determine dominant label above/below split
            above_mask = region_means > best_split
            below_mask = ~above_mask
            above_labels = truncated_labels[above_mask]
            below_labels = truncated_labels[below_mask]

            # Pick the side with fewer samples as the "signal" side
            if len(above_labels) > 0 and len(below_labels) > 0:
                if len(above_labels) <= len(below_labels):
                    signal_labels = above_labels
                else:
                    signal_labels = below_labels
            elif len(above_labels) > 0:
                signal_labels = above_labels
            else:
                signal_labels = below_labels

            if len(signal_labels) == 0:
                continue

            unique_labels, counts = np.unique(signal_labels, return_counts=True)
            dominant_label = unique_labels[np.argmax(counts)]

            # Convert to MarketLabel if it's a string
            try:
                label_enum = MarketLabel(dominant_label)
            except ValueError:
                label_enum = MarketLabel.FLAT

            candidates.append(
                Shapelet(
                    id="",  # assigned after dedup
                    coefficients=region_coeffs.copy(),
                    wavelet_level=actual_level,
                    ticker=decomp.ticker,
                    label=label_enum,
                    information_gain=best_ig,
                    start_index=start,
                    end_index=end,
                    threshold=best_split,
                )
            )

    # --- 1f. Keep top-k per level ---
    by_level: dict[int, list[Shapelet]] = {}
    for c in candidates:
        by_level.setdefault(c.wavelet_level, []).append(c)

    top_candidates: list[Shapelet] = []
    for level in by_level:
        level_cands = sorted(by_level[level], key=lambda s: s.information_gain, reverse=True)
        top_candidates.extend(level_cands[: config.top_k])

    # --- 2. Deduplicate ---
    deduped = deduplicate(top_candidates, threshold=config.cluster_threshold)

    # --- 3. Assign unique IDs ---
    for s in deduped:
        s.id = uuid.uuid4().hex[:8]

    # --- 4. Return sorted by IG descending ---
    deduped.sort(key=lambda s: s.information_gain, reverse=True)
    return deduped
