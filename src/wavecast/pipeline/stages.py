"""Individual pipeline stages with timing and error handling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wavecast.core.config import WaveCastConfig
from wavecast.core.exceptions import PipelineError
from wavecast.core.types import (
    HurstResult,
    MatchResult,
    MFDFAResult,
    SelfSimilarityResult,
    Shapelet,
    TimeSeries,
    WaveletDecomposition,
)


class Stage(Enum):
    DATA = auto()
    DECOMPOSE = auto()
    DISCOVER = auto()
    MATCH = auto()
    FRACTAL = auto()
    FEATURES = auto()
    FORECAST = auto()
    EVALUATE = auto()


@dataclass
class StageResult:
    stage: Stage
    data: Any
    duration_seconds: float


def _timed(stage: Stage, func: Any, *args: Any, **kwargs: Any) -> StageResult:
    """Run a function with timing."""
    start = time.monotonic()
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        raise PipelineError(f"Stage {stage.name} failed: {e}") from e
    duration = time.monotonic() - start
    return StageResult(stage=stage, data=result, duration_seconds=duration)


def stage_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    config: WaveCastConfig | None = None,
) -> StageResult:
    """Fetch and preprocess data."""
    from wavecast.data.cache import ParquetCache
    from wavecast.data.preprocessing import handle_nans
    from wavecast.data.sources import fetch_massive

    cfg = config or WaveCastConfig()
    cache = ParquetCache(cfg.cache_dir)

    def _fetch() -> TimeSeries:
        cached = cache.get(ticker, interval)
        if cached is not None:
            return cached
        ts = fetch_massive(ticker, start, end, interval)
        cache.put(ticker, interval, ts)
        return handle_nans(ts)

    return _timed(Stage.DATA, _fetch)


def stage_decompose(
    ts: TimeSeries,
    config: WaveCastConfig | None = None,
) -> StageResult:
    """DWT decomposition."""
    from wavecast.wavelets.dwt import decompose

    cfg = config or WaveCastConfig()

    def _decompose() -> WaveletDecomposition:
        return decompose(ts, wavelet=cfg.wavelet.wavelet, level=cfg.wavelet.level)

    return _timed(Stage.DECOMPOSE, _decompose)


def stage_discover(
    decomp: WaveletDecomposition,
    labels: NDArray[np.int_],
    config: WaveCastConfig | None = None,
) -> StageResult:
    """Shapelet discovery."""
    from wavecast.shapelets.discovery import discover_shapelets

    cfg = config or WaveCastConfig()

    def _discover() -> list[Shapelet]:
        return discover_shapelets(decomp, labels, cfg.shapelet)

    return _timed(Stage.DISCOVER, _discover)


def stage_match(
    decomp: WaveletDecomposition,
    library: Any,
    level: int,
    config: WaveCastConfig | None = None,
) -> StageResult:
    """DTW matching against shapelet library."""
    from wavecast.dtw.matching import match_against_library

    cfg = config or WaveCastConfig()

    def _match() -> MatchResult:
        return match_against_library(
            decomp.detail_at_level(level), library, level, cfg.dtw
        )

    return _timed(Stage.MATCH, _match)


def stage_fractal(
    ts: TimeSeries,
    config: WaveCastConfig | None = None,
) -> StageResult:
    """Fractal analysis: Hurst + MFDFA + self-similarity."""
    from wavecast.fractal.hurst import wavelet_hurst
    from wavecast.fractal.mfdfa import compute_mfdfa
    from wavecast.fractal.self_similarity import cross_scale_similarity
    from wavecast.wavelets.dwt import decompose

    cfg = config or WaveCastConfig()

    def _fractal() -> tuple[HurstResult, MFDFAResult | None, SelfSimilarityResult | None]:
        hurst = wavelet_hurst(ts.values, wavelet=cfg.wavelet.wavelet)

        mfdfa_result: MFDFAResult | None = None
        if len(ts.values) >= 256:
            try:
                mfdfa_result = compute_mfdfa(
                    ts.values,
                    q_range=cfg.fractal.mfdfa_q_range,
                    q_steps=cfg.fractal.mfdfa_q_steps,
                )
            except Exception:
                pass

        decomp = decompose(ts, wavelet=cfg.wavelet.wavelet, level=cfg.wavelet.level)
        self_sim = cross_scale_similarity(decomp)

        return hurst, mfdfa_result, self_sim

    return _timed(Stage.FRACTAL, _fractal)
