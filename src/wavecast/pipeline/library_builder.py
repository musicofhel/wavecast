"""Multi-asset shapelet library builder."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wavecast.core.config import WaveCastConfig
from wavecast.core.types import MarketLabel
from wavecast.core.universe import DEFAULT_UNIVERSE, Universe
from wavecast.shapelets.library import ShapeletLibrary

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Result of building a master shapelet library."""
    library: ShapeletLibrary
    stats: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    failed_tickers: list[str] = field(default_factory=list)


def _label_returns(values: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Label price series by return direction."""
    returns = np.diff(np.log(np.maximum(values, 1e-10)))
    labels = np.full(len(values), MarketLabel.FLAT.value)
    labels[1:][returns > threshold] = MarketLabel.UP.value
    labels[1:][returns < -threshold] = MarketLabel.DOWN.value
    return labels


def build_master_library(
    universe: Universe | None = None,
    config: WaveCastConfig | None = None,
    start: str | None = None,
    end: str | None = None,
    dedup_threshold: float = 0.3,
) -> BuildResult:
    """Build a master shapelet library from multiple assets.

    For each asset: fetch -> decompose -> discover shapelets -> merge.
    Cross-asset deduplication via DTW clustering.
    """
    from wavecast.pipeline.stages import stage_data, stage_decompose, stage_discover
    from wavecast.shapelets.clustering import deduplicate

    universe = universe or DEFAULT_UNIVERSE
    cfg = config or WaveCastConfig()
    cfg.ensure_dirs()

    start_time = time.monotonic()
    all_shapelets = []
    failed = []

    for asset in universe.assets:
        ticker = asset.ticker
        try:
            logger.info(f"Processing {ticker}...")

            # Fetch data
            data_result = stage_data(ticker, start, end, config=cfg)
            ts = data_result.data

            if ts.length < 100:
                logger.warning(f"Skipping {ticker}: too few data points ({ts.length})")
                failed.append(ticker)
                continue

            # Decompose
            decomp_result = stage_decompose(ts, config=cfg)
            decomp = decomp_result.data

            # Generate labels from returns
            labels = _label_returns(ts.values)

            # Discover shapelets
            discover_result = stage_discover(decomp, labels, config=cfg)
            shapelets = discover_result.data

            # Tag metadata
            for s in shapelets:
                s.metadata["asset_class"] = asset.asset_class.value
                s.metadata["source_ticker"] = ticker

            all_shapelets.extend(shapelets)
            logger.info(f"  {ticker}: {len(shapelets)} shapelets discovered")

        except Exception as e:
            logger.warning(f"Failed to process {ticker}: {e}")
            failed.append(ticker)
            continue

    # Build library and deduplicate
    library = ShapeletLibrary(all_shapelets)

    if len(all_shapelets) > 1:
        try:
            deduped = deduplicate(all_shapelets, threshold=dedup_threshold)
            library = ShapeletLibrary(deduped)
            logger.info(f"Deduplicated: {len(all_shapelets)} -> {len(deduped)} shapelets")
        except Exception as e:
            logger.warning(f"Deduplication failed: {e}. Using full library.")

    # Save
    save_path = cfg.library_dir / "master.json"
    library.save(save_path)
    logger.info(f"Saved master library to {save_path}")

    duration = time.monotonic() - start_time

    return BuildResult(
        library=library,
        stats=library.stats(),
        duration_seconds=duration,
        failed_tickers=failed,
    )
