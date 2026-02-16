"""DTW matching of query coefficients against a shapelet library."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from dtaidistance import dtw
from numpy.typing import NDArray

from wavecast.core.config import DTWConfig
from wavecast.core.exceptions import DTWError
from wavecast.core.types import MatchResult, ShapeletMatch
from wavecast.shapelets.library import ShapeletLibrary


def match_single(
    query: NDArray[np.float64],
    target: NDArray[np.float64],
    window: int = 10,
) -> ShapeletMatch:
    """Compute DTW distance and warping path between two series.

    Parameters
    ----------
    query : 1-D query array.
    target : 1-D target array.
    window : Sakoe-Chiba band width.

    Returns
    -------
    ShapeletMatch with distance, normalized distance, and warping path.
    """
    if len(query) == 0 or len(target) == 0:
        raise DTWError("Query and target must be non-empty")

    distance = dtw.distance(
        query.astype(np.double),
        target.astype(np.double),
        window=window,
        use_pruning=True,
    )

    # Warping path via the full DTW matrix
    path = dtw.warping_path(
        query.astype(np.double),
        target.astype(np.double),
        window=window,
    )

    norm_dist = distance / max(len(query), len(target))

    return ShapeletMatch(
        shapelet_id="",
        distance=float(distance),
        normalized_distance=float(norm_dist),
        warping_path=path,
        query_start=0,
        query_end=len(query),
    )


def match_against_library(
    query_coeffs: NDArray[np.float64],
    library: ShapeletLibrary,
    level: int,
    config: DTWConfig | None = None,
) -> MatchResult:
    """Match query coefficients against all shapelets at a given level.

    Parameters
    ----------
    query_coeffs : 1-D wavelet coefficients to match.
    library : shapelet library to search.
    level : wavelet level to filter library shapelets.
    config : optional DTW settings.

    Returns
    -------
    MatchResult with top-k matches sorted by distance ascending.

    Raises
    ------
    DTWError
        If query is empty or no shapelets exist at the given level.
    """
    if config is None:
        config = DTWConfig()

    if len(query_coeffs) == 0:
        raise DTWError("Query coefficients must be non-empty")

    candidates = library.query(level=level)
    if not candidates:
        raise DTWError(f"No shapelets found at level {level}")

    query = query_coeffs.astype(np.double)
    if config.normalize:
        std = float(np.std(query))
        if std > 0:
            query = (query - np.mean(query)) / std

    matches: list[ShapeletMatch] = []
    for shapelet in candidates:
        target = shapelet.coefficients.astype(np.double)
        if config.normalize:
            std_t = float(np.std(target))
            if std_t > 0:
                target = (target - np.mean(target)) / std_t

        distance = dtw.distance(
            query,
            target,
            window=config.window,
            use_pruning=config.use_pruning,
        )

        norm_dist = distance / max(len(query), len(target))

        matches.append(
            ShapeletMatch(
                shapelet_id=shapelet.id,
                distance=float(distance),
                normalized_distance=float(norm_dist),
                warping_path=[],  # skip path for bulk matching (expensive)
                query_start=0,
                query_end=len(query_coeffs),
                shapelet=shapelet,
            )
        )

    # Sort by distance ascending, keep top-k
    matches.sort(key=lambda m: m.distance)
    matches = matches[: config.top_k]

    return MatchResult(
        ticker=candidates[0].ticker if candidates else "",
        wavelet_level=level,
        matches=matches,
        query_length=len(query_coeffs),
        timestamp=datetime.now(),
    )
