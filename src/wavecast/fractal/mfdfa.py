"""Multifractal Detrended Fluctuation Analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.core.exceptions import FractalError
from wavecast.core.types import MFDFAResult


def compute_mfdfa(
    values: NDArray,
    q_range: tuple[float, float] = (-5.0, 5.0),
    q_steps: int = 21,
) -> MFDFAResult:
    """Compute multifractal DFA spectrum.

    Uses the MFDFA library to compute generalized Hurst exponents H(q),
    scaling exponents tau(q), and the singularity spectrum f(alpha).
    """
    from MFDFA import MFDFA as mfdfa_func

    values = np.asarray(values, dtype=np.float64)
    if len(values) < 100:
        raise FractalError("Need at least 100 data points for MFDFA")

    q_values = np.linspace(q_range[0], q_range[1], q_steps)
    # Remove q=0 from computation since it requires special handling
    q_nonzero = q_values[q_values != 0.0]
    if len(q_nonzero) == 0:
        raise FractalError("No non-zero q values to compute")

    # Build logarithmically-spaced lag values
    n = len(values)
    min_lag = 10
    max_lag = n // 4
    if max_lag <= min_lag:
        raise FractalError(f"Series too short for MFDFA (need >40 points, got {n})")
    lag = np.unique(
        np.logspace(np.log10(min_lag), np.log10(max_lag), num=30).astype(int)
    )

    try:
        lag, dfa = mfdfa_func(values, lag=lag, q=q_nonzero, order=1)
    except Exception as e:
        raise FractalError(f"MFDFA computation failed: {e}") from e

    # dfa shape: (len(lag), len(q_nonzero))
    # Fit log-log slope for each q to get H(q)
    log_lag = np.log(lag)
    hurst_q = np.zeros(len(q_nonzero))

    for i in range(len(q_nonzero)):
        log_fluct = np.log(dfa[:, i])
        valid = np.isfinite(log_fluct) & np.isfinite(log_lag)
        if valid.sum() < 2:
            hurst_q[i] = np.nan
            continue
        slope, _ = np.polyfit(log_lag[valid], log_fluct[valid], 1)
        hurst_q[i] = slope

    # Scaling exponents: tau(q) = q * H(q) - 1
    tau_q = q_nonzero * hurst_q - 1.0

    # Singularity spectrum via finite differences
    # alpha = d(tau)/d(q), f(alpha) = q * alpha - tau
    dq = np.diff(q_nonzero)
    dtau = np.diff(tau_q)
    alpha = dtau / dq  # length = len(q_nonzero) - 1
    # Use midpoint q values for f(alpha)
    q_mid = (q_nonzero[:-1] + q_nonzero[1:]) / 2.0
    tau_mid = (tau_q[:-1] + tau_q[1:]) / 2.0
    f_alpha = q_mid * alpha - tau_mid

    # Filter out NaN values
    valid_mask = np.isfinite(alpha) & np.isfinite(f_alpha)
    alpha_clean = alpha[valid_mask]
    f_alpha_clean = f_alpha[valid_mask]

    spectrum_width = float(np.ptp(alpha_clean)) if len(alpha_clean) > 0 else 0.0

    return MFDFAResult(
        q_values=q_nonzero,
        hurst_q=hurst_q,
        tau_q=tau_q,
        alpha=alpha_clean,
        f_alpha=f_alpha_clean,
        spectrum_width=spectrum_width,
    )
