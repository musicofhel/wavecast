"""Ensemble model with weighted averaging and optional stacking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from wavecast.core.exceptions import ModelError, ModelNotTrainedError

from .base import BaseModel


class EnsembleModel(BaseModel):
    """Weighted ensemble of sub-models with optional stacking."""

    def __init__(
        self,
        models: list[BaseModel] | None = None,
        weights: list[float] | None = None,
        use_stacking: bool = False,
    ) -> None:
        self._models = models or []
        self._weights = weights
        self._use_stacking = use_stacking
        self._stacker: LinearRegression | None = None
        self._trained = False

    @property
    def name(self) -> str:
        return "ensemble"

    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_val: NDArray | None = None,
        y_val: NDArray | None = None,
    ) -> dict[str, float]:
        """Train all sub-models, then optionally learn stacking weights on validation set."""
        if not self._models:
            raise ModelError("No sub-models in ensemble")

        all_metrics: dict[str, float] = {}

        for model in self._models:
            metrics = model.fit(X_train, y_train, X_val, y_val)
            for k, v in metrics.items():
                all_metrics[f"{model.name}_{k}"] = v

        # Learn stacking weights from validation predictions
        if self._use_stacking and X_val is not None and y_val is not None:
            val_preds = np.column_stack([m.predict(X_val) for m in self._models])
            self._stacker = LinearRegression(fit_intercept=False, positive=True)
            self._stacker.fit(val_preds, y_val)
            self._weights = self._stacker.coef_.tolist()

            # Normalize weights
            w_sum = sum(self._weights)
            if w_sum > 0:
                self._weights = [w / w_sum for w in self._weights]

            stacked_pred = self._stacker.predict(val_preds)
            all_metrics["ensemble_val_rmse"] = float(
                np.sqrt(np.mean((y_val - stacked_pred) ** 2))
            )

        # Default to equal weights if not set
        if self._weights is None:
            self._weights = [1.0 / len(self._models)] * len(self._models)

        self._trained = True

        # Compute ensemble train metric
        train_pred = self.predict(X_train)
        all_metrics["ensemble_train_rmse"] = float(
            np.sqrt(np.mean((y_train - train_pred) ** 2))
        )

        return all_metrics

    def predict(self, X: NDArray) -> NDArray:
        if not self._models or not self._trained:
            raise ModelNotTrainedError("Ensemble has not been trained")

        predictions = np.column_stack([m.predict(X) for m in self._models])

        if self._stacker is not None:
            return self._stacker.predict(predictions)

        weights = np.array(self._weights or [1.0 / len(self._models)] * len(self._models))
        return predictions @ weights

    def save(self, path: Path) -> None:
        if not self._trained:
            raise ModelNotTrainedError("Cannot save untrained ensemble")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each sub-model
        model_names = []
        for i, model in enumerate(self._models):
            sub_path = path / f"sub_{i}_{model.name}"
            model.save(sub_path)
            model_names.append({"index": i, "name": model.name, "path": sub_path.name})

        config = {
            "weights": self._weights,
            "use_stacking": self._use_stacking,
            "models": model_names,
        }
        if self._stacker is not None:
            config["stacker_coef"] = self._stacker.coef_.tolist()

        with open(path / "ensemble_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> EnsembleModel:
        path = Path(path)
        config_path = path / "ensemble_config.json"
        if not config_path.exists():
            raise ModelError(f"No ensemble config at {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Import concrete model classes for loading
        from .gradient_boost import GradientBoostModel
        from .wavelet_lstm import WaveletLSTM

        model_classes: dict[str, type[BaseModel]] = {
            "gradient_boost": GradientBoostModel,
            "wavelet_lstm": WaveletLSTM,
        }

        models = []
        for info in config["models"]:
            model_cls = model_classes.get(info["name"])
            if model_cls is None:
                raise ModelError(f"Unknown model type: {info['name']}")
            models.append(model_cls.load(path / info["path"]))

        instance = cls(
            models=models,
            weights=config.get("weights"),
            use_stacking=config.get("use_stacking", False),
        )
        instance._trained = True

        if "stacker_coef" in config:
            instance._stacker = LinearRegression(fit_intercept=False, positive=True)
            instance._stacker.coef_ = np.array(config["stacker_coef"])
            instance._stacker.intercept_ = 0.0

        return instance
