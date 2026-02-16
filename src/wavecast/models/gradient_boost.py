"""XGBoost gradient boosting model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from wavecast.core.exceptions import ModelError, ModelNotTrainedError

from .base import BaseModel


class GradientBoostModel(BaseModel):
    """XGBoost regression model wrapper with early stopping."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        early_stopping_rounds: int = 20,
        **kwargs: object,
    ) -> None:
        self._config = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "early_stopping_rounds": early_stopping_rounds,
            **kwargs,
        }
        self._model: xgb.XGBRegressor | None = None
        self._feature_importance: NDArray | None = None

    @property
    def name(self) -> str:
        return "gradient_boost"

    @property
    def feature_importance(self) -> NDArray:
        if self._feature_importance is None:
            raise ModelNotTrainedError("Model has not been trained")
        return self._feature_importance

    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_val: NDArray | None = None,
        y_val: NDArray | None = None,
    ) -> dict[str, float]:
        """Train XGBoost with optional early stopping on validation set."""
        self._model = xgb.XGBRegressor(
            n_estimators=self._config["n_estimators"],
            max_depth=self._config["max_depth"],
            learning_rate=self._config["learning_rate"],
            objective="reg:squarederror",
            random_state=42,
        )

        fit_params: dict = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

            early_stopping = self._config.get("early_stopping_rounds", 20)
            self._model.set_params(early_stopping_rounds=early_stopping)

        self._model.fit(X_train, y_train, **fit_params)
        self._feature_importance = self._model.feature_importances_

        # Compute metrics
        train_pred = self._model.predict(X_train)
        train_rmse = float(np.sqrt(np.mean((y_train - train_pred) ** 2)))
        train_mae = float(np.mean(np.abs(y_train - train_pred)))
        metrics = {"train_rmse": train_rmse, "train_mae": train_mae}

        if X_val is not None and y_val is not None:
            val_pred = self._model.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
            metrics["val_mae"] = float(np.mean(np.abs(y_val - val_pred)))

        return metrics

    def predict(self, X: NDArray) -> NDArray:
        if self._model is None:
            raise ModelNotTrainedError("Model has not been trained")
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise ModelNotTrainedError("Cannot save untrained model")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path / "model.json"))
        with open(path / "config.json", "w") as f:
            json.dump(self._config, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> GradientBoostModel:
        path = Path(path)
        config_path = path / "config.json"
        model_path = path / "model.json"
        if not model_path.exists():
            raise ModelError(f"No model file at {model_path}")

        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        instance = cls(**config)
        instance._model = xgb.XGBRegressor()
        instance._model.load_model(str(model_path))
        instance._feature_importance = instance._model.feature_importances_
        return instance
