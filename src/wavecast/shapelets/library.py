"""ShapeletLibrary — persistent, queryable collection of shapelets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wavecast.core.exceptions import ShapeletLibraryError
from wavecast.core.types import MarketLabel, Shapelet


class ShapeletLibrary:
    """In-memory shapelet collection with persistence.

    Supports filtering by wavelet level, ticker, and label; merging
    libraries; and saving / loading to HDF5 (via data.storage) or JSON.
    """

    def __init__(self, shapelets: list[Shapelet] | None = None) -> None:
        self._shapelets: dict[str, Shapelet] = {}
        if shapelets:
            self.add_many(shapelets)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, shapelet: Shapelet) -> None:
        """Add a single shapelet (overwrites if same ID exists)."""
        if not shapelet.id:
            raise ShapeletLibraryError("Shapelet must have a non-empty id")
        self._shapelets[shapelet.id] = shapelet

    def add_many(self, shapelets: list[Shapelet]) -> None:
        for s in shapelets:
            self.add(s)

    def merge(self, other: ShapeletLibrary) -> None:
        """Merge another library into this one (other's entries win on ID conflict)."""
        for s in other.all():
            self.add(s)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, shapelet_id: str) -> Shapelet:
        """Retrieve a shapelet by ID, or raise."""
        try:
            return self._shapelets[shapelet_id]
        except KeyError:
            raise ShapeletLibraryError(f"Shapelet '{shapelet_id}' not found") from None

    def all(self) -> list[Shapelet]:
        """All shapelets sorted by IG descending."""
        return sorted(self._shapelets.values(), key=lambda s: s.information_gain, reverse=True)

    def query(
        self,
        *,
        level: int | None = None,
        ticker: str | None = None,
        label: MarketLabel | None = None,
        top_k: int | None = None,
    ) -> list[Shapelet]:
        """Filtered query returning shapelets sorted by IG descending."""
        results = list(self._shapelets.values())
        if level is not None:
            results = [s for s in results if s.wavelet_level == level]
        if ticker is not None:
            results = [s for s in results if s.ticker == ticker]
        if label is not None:
            results = [s for s in results if s.label == label]
        results.sort(key=lambda s: s.information_gain, reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Summary statistics of the library."""
        all_s = self.all()
        if not all_s:
            return {"count": 0}

        levels = [s.wavelet_level for s in all_s]
        igs = [s.information_gain for s in all_s]
        tickers = list({s.ticker for s in all_s})
        labels = list({s.label.value for s in all_s})

        return {
            "count": len(all_s),
            "levels": sorted(set(levels)),
            "tickers": tickers,
            "labels": labels,
            "ig_mean": float(np.mean(igs)),
            "ig_max": float(np.max(igs)),
            "ig_min": float(np.min(igs)),
        }

    def __len__(self) -> int:
        return len(self._shapelets)

    def __contains__(self, shapelet_id: str) -> bool:
        return shapelet_id in self._shapelets

    # ------------------------------------------------------------------
    # Persistence — JSON (self-contained, no data.storage dependency)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the library to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for s in self.all():
            records.append({
                "id": s.id,
                "coefficients": s.coefficients.tolist(),
                "wavelet_level": s.wavelet_level,
                "ticker": s.ticker,
                "label": s.label.value,
                "information_gain": s.information_gain,
                "start_index": s.start_index,
                "end_index": s.end_index,
                "threshold": s.threshold,
                "metadata": s.metadata,
            })

        path.write_text(json.dumps(records, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ShapeletLibrary:
        """Load a library from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise ShapeletLibraryError(f"Library file not found: {path}")

        records = json.loads(path.read_text())
        shapelets = []
        for r in records:
            shapelets.append(
                Shapelet(
                    id=r["id"],
                    coefficients=np.array(r["coefficients"], dtype=np.float64),
                    wavelet_level=r["wavelet_level"],
                    ticker=r["ticker"],
                    label=MarketLabel(r["label"]),
                    information_gain=r["information_gain"],
                    start_index=r["start_index"],
                    end_index=r["end_index"],
                    threshold=r["threshold"],
                    metadata=r.get("metadata", {}),
                )
            )
        return cls(shapelets)
