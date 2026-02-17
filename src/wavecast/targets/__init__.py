"""Return target computation for WaveletGPT."""

from wavecast.targets.returns import (
    ReturnTargetResult,
    assign_quantile_labels,
    compute_quantile_boundaries,
    compute_sample_returns,
)

__all__ = [
    "ReturnTargetResult",
    "assign_quantile_labels",
    "compute_quantile_boundaries",
    "compute_sample_returns",
]
