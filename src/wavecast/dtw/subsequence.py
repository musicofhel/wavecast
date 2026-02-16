"""Subsequence DTW search — find occurrences of a pattern in a long series."""

from __future__ import annotations

import numpy as np
from dtaidistance import dtw
from numpy.typing import NDArray

from wavecast.core.exceptions import DTWError


def subsequence_search(
    query: NDArray[np.float64],
    long_series: NDArray[np.float64],
    window: int = 10,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Sliding-window DTW search for the best matches of *query* in *long_series*.

    Parameters
    ----------
    query : 1-D pattern to search for.
    long_series : 1-D series to search within.
    window : Sakoe-Chiba band width for DTW.
    top_k : number of best matches to return.

    Returns
    -------
    List of ``(start_index, distance)`` tuples sorted by distance ascending.

    Raises
    ------
    DTWError
        If the query is longer than the series.
    """
    m = len(query)
    n = len(long_series)

    if m == 0:
        raise DTWError("Query must be non-empty")
    if m > n:
        raise DTWError(
            f"Query length ({m}) exceeds series length ({n})"
        )

    query_d = query.astype(np.double)
    series_d = long_series.astype(np.double)

    num_windows = n - m + 1
    distances: list[tuple[int, float]] = []

    for i in range(num_windows):
        subseq = series_d[i : i + m]
        d = dtw.distance(
            query_d,
            subseq,
            window=min(window, m - 1),  # window can't exceed series length
            use_pruning=True,
        )
        distances.append((i, float(d)))

    # Sort by distance, then deduplicate overlapping matches
    distances.sort(key=lambda x: x[1])

    # Greedy non-overlapping selection
    selected: list[tuple[int, float]] = []
    used_indices: set[int] = set()

    for start, dist in distances:
        # Check overlap with any already-selected match
        overlaps = False
        for s in used_indices:
            if abs(start - s) < m:
                overlaps = True
                break
        if not overlaps:
            selected.append((start, dist))
            used_indices.add(start)
        if len(selected) >= top_k:
            break

    return selected
