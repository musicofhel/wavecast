"""Shape-DTW — descriptor-enhanced Dynamic Time Warping."""

from __future__ import annotations

import numpy as np
from dtaidistance import dtw
from numpy.typing import NDArray

from wavecast.core.exceptions import DTWError


def shape_descriptor(
    series: NDArray[np.float64],
    descriptor: str = "slope",
) -> NDArray[np.float64]:
    """Compute a shape descriptor for a time series.

    Parameters
    ----------
    series : 1-D input array.
    descriptor : one of ``'slope'``, ``'derivative'``, ``'compound'``.
        - ``slope`` : first-order finite differences.
        - ``derivative`` : central differences (endpoints use forward/backward).
        - ``compound`` : concatenation of the raw series and its derivative
          (interleaved: [v0, d0, v1, d1, ...]).

    Returns
    -------
    Transformed 1-D array.

    Raises
    ------
    DTWError
        If the series has fewer than 2 points or the descriptor is unknown.
    """
    if len(series) < 2:
        raise DTWError("Series must have at least 2 points for shape descriptors")

    if descriptor == "slope":
        return np.diff(series)

    elif descriptor == "derivative":
        # Central differences, forward/backward at edges
        d = np.empty_like(series)
        d[0] = series[1] - series[0]
        d[-1] = series[-1] - series[-2]
        d[1:-1] = (series[2:] - series[:-2]) / 2.0
        return d

    elif descriptor == "compound":
        deriv = shape_descriptor(series, "derivative")
        # Interleave: [v0, d0, v1, d1, ...]
        compound = np.empty(len(series) + len(deriv), dtype=np.float64)
        compound[0::2] = series
        compound[1::2] = deriv
        return compound

    else:
        raise DTWError(f"Unknown descriptor '{descriptor}'. Use 'slope', 'derivative', or 'compound'.")


def shape_dtw_distance(
    s1: NDArray[np.float64],
    s2: NDArray[np.float64],
    descriptor: str = "slope",
    window: int = 10,
) -> float:
    """Compute Shape-DTW distance between two series.

    Instead of aligning raw values, this first transforms both series
    using a shape descriptor and then computes DTW on the transformed
    representations.

    Parameters
    ----------
    s1, s2 : 1-D input series.
    descriptor : shape descriptor type (see :func:`shape_descriptor`).
    window : Sakoe-Chiba band width.

    Returns
    -------
    Shape-DTW distance (float).
    """
    if len(s1) < 2 or len(s2) < 2:
        raise DTWError("Both series must have at least 2 points")

    d1 = shape_descriptor(s1, descriptor).astype(np.double)
    d2 = shape_descriptor(s2, descriptor).astype(np.double)

    effective_window = min(window, max(len(d1), len(d2)) - 1)
    if effective_window < 1:
        effective_window = 1

    return float(
        dtw.distance(d1, d2, window=effective_window, use_pruning=True)
    )
