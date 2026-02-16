"""Model registry for versioned model storage and retrieval."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from wavecast.core.exceptions import ModelError

from .base import BaseModel
from .ensemble import EnsembleModel
from .gradient_boost import GradientBoostModel
from .wavelet_gpt import WaveletGPT
from .wavelet_lstm import WaveletLSTM

_MODEL_CLASSES: dict[str, type[BaseModel]] = {
    "gradient_boost": GradientBoostModel,
    "wavelet_lstm": WaveletLSTM,
    "ensemble": EnsembleModel,
    "wavelet_gpt": WaveletGPT,
}


class ModelRegistry:
    """Versioned model storage with metadata tracking."""

    def save_model(
        self,
        model: BaseModel,
        name: str,
        version: str,
        metrics: dict[str, float],
        path: Path,
    ) -> Path:
        """Save a model with version metadata.

        Directory structure: path/name/version/
        """
        path = Path(path)
        model_dir = path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        model.save(model_dir / "model")

        metadata = {
            "name": name,
            "version": version,
            "model_type": model.name,
            "metrics": metrics,
            "saved_at": datetime.now().isoformat(),
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return model_dir

    def load_model(
        self,
        name: str,
        version: str,
        path: Path,
    ) -> BaseModel:
        """Load a model by name and version."""
        path = Path(path)
        model_dir = path / name / version
        metadata_path = model_dir / "metadata.json"

        if not metadata_path.exists():
            raise ModelError(f"No model found at {model_dir}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        model_type = metadata["model_type"]
        model_cls = _MODEL_CLASSES.get(model_type)
        if model_cls is None:
            raise ModelError(f"Unknown model type: {model_type}")

        return model_cls.load(model_dir / "model")

    def list_models(self, path: Path) -> list[dict]:
        """List all saved models with their metadata."""
        path = Path(path)
        results = []

        if not path.exists():
            return results

        for name_dir in sorted(path.iterdir()):
            if not name_dir.is_dir():
                continue
            for version_dir in sorted(name_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                meta_path = version_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        results.append(json.load(f))

        return results
