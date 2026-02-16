"""Pairwise DTW distance and similarity matrices."""

from __future__ import annotations

import numpy as np
from dtaidistance import dtw
from numpy.typing import NDArray

from wavecast.core.exceptions import DTWError


def pairwise_dtw_matrix(
    series_list: list[NDArray[np.float64]],
    window: int = 10,
) -> NDArray[np.float64]:
    """Compute a symmetric pairwise DTW distance matrix.

    Uses ``dtaidistance.dtw.distance_matrix_fast`` when available (C
    extension compiled), falling back to the pure-Python variant otherwise.

    Parameters
    ----------
    series_list : list of 1-D arrays (may have different lengths).
    window : Sakoe-Chiba band width.

    Returns
    -------
    n x n symmetric distance matrix (diagonal = 0).

    Raises
    ------
    DTWError
        If fewer than 2 series are provided.
    """
    n = len(series_list)
    if n < 2:
        raise DTWError("Need at least 2 series for a pairwise matrix")

    # dtaidistance expects a list/ndarray of sequences
    prepared = [s.astype(np.double) for s in series_list]

    try:
        dm = dtw.distance_matrix_fast(prepared, window=window, compact=False)
    except Exception:
        # C extension may not be compiled; fall back to pure Python
        dm = dtw.distance_matrix(prepared, window=window, compact=False)

    # dtaidistance returns an ndarray; ensure it's symmetric and float64
    result = np.array(dm, dtype=np.float64)

    # Fill the upper triangle (dtaidistance fills lower tri + diag)
    for i in range(n):
        for j in range(i + 1, n):
            if result[i, j] == 0.0 and result[j, i] != 0.0:
                result[i, j] = result[j, i]
            elif result[j, i] == 0.0 and result[i, j] != 0.0:
                result[j, i] = result[i, j]

    return result


def similarity_matrix(
    distance_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert a distance matrix to a similarity matrix.

    Similarity is defined as ``1 / (1 + d)`` which maps [0, inf) -> (0, 1].

    Parameters
    ----------
    distance_matrix : n x n non-negative distance matrix.

    Returns
    -------
    n x n similarity matrix.
    """
    return 1.0 / (1.0 + distance_matrix)
