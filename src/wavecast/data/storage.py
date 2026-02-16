"""HDF5 storage for shapelet libraries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from wavecast.core.exceptions import DataError
from wavecast.core.types import MarketLabel, Shapelet


def save_shapelets_h5(path: Path, shapelets: list[Shapelet]) -> None:
    """Save shapelets to HDF5 with /{ticker}/{level}/{id} hierarchy.

    Each shapelet is stored as a dataset containing its coefficients, with
    scalar attributes for all metadata fields.

    Args:
        path: Output HDF5 file path.
        shapelets: List of Shapelet objects to persist.

    Raises:
        DataError: On write failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(path, "w") as f:
            for s in shapelets:
                group_path = f"{s.ticker}/{s.wavelet_level}/{s.id}"
                ds = f.create_dataset(group_path, data=s.coefficients)
                ds.attrs["id"] = s.id
                ds.attrs["wavelet_level"] = s.wavelet_level
                ds.attrs["ticker"] = s.ticker
                ds.attrs["label"] = s.label.value
                ds.attrs["information_gain"] = s.information_gain
                ds.attrs["start_index"] = s.start_index
                ds.attrs["end_index"] = s.end_index
                ds.attrs["threshold"] = s.threshold
                # Store metadata as individual attrs with 'meta_' prefix
                for k, v in s.metadata.items():
                    _store_attr(ds, f"meta_{k}", v)
    except Exception as e:
        raise DataError(f"Failed to save shapelets to {path}: {e}") from e


def load_shapelets_h5(path: Path) -> list[Shapelet]:
    """Load shapelets from an HDF5 file.

    Args:
        path: Path to the HDF5 file.

    Returns:
        List of Shapelet objects.

    Raises:
        DataError: If the file does not exist or is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"Shapelet file not found: {path}")

    shapelets: list[Shapelet] = []

    try:
        with h5py.File(path, "r") as f:
            _visit_datasets(f, shapelets)
    except DataError:
        raise
    except Exception as e:
        raise DataError(f"Failed to load shapelets from {path}: {e}") from e

    return shapelets


def _visit_datasets(group: h5py.Group, out: list[Shapelet]) -> None:
    """Recursively visit all datasets in an HDF5 group."""
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            out.append(_dataset_to_shapelet(item))
        elif isinstance(item, h5py.Group):
            _visit_datasets(item, out)


def _dataset_to_shapelet(ds: h5py.Dataset) -> Shapelet:
    """Convert an HDF5 dataset back to a Shapelet."""
    attrs = dict(ds.attrs)
    coefficients = np.array(ds, dtype=np.float64)

    # Reconstruct metadata from 'meta_' prefixed attrs
    metadata: dict[str, Any] = {}
    for k, v in attrs.items():
        if k.startswith("meta_"):
            metadata[k[5:]] = _decode_attr(v)

    return Shapelet(
        id=str(attrs["id"]),
        coefficients=coefficients,
        wavelet_level=int(attrs["wavelet_level"]),
        ticker=str(attrs["ticker"]),
        label=MarketLabel(str(attrs["label"])),
        information_gain=float(attrs["information_gain"]),
        start_index=int(attrs["start_index"]),
        end_index=int(attrs["end_index"]),
        threshold=float(attrs["threshold"]),
        metadata=metadata,
    )


def _store_attr(ds: h5py.Dataset, key: str, value: Any) -> None:
    """Store a metadata value as an HDF5 attribute."""
    if isinstance(value, (int, float, str, bool, np.integer, np.floating)) or isinstance(value, np.ndarray):
        ds.attrs[key] = value
    else:
        # Fall back to string representation
        ds.attrs[key] = str(value)


def _decode_attr(value: Any) -> Any:
    """Decode an HDF5 attribute back to a Python type."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value
    return value
