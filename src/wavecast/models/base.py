"""Abstract base model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from numpy.typing import NDArray


class BaseModel(ABC):
    """Abstract base class for all WaveCast forecasting models."""

    @abstractmethod
    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_val: NDArray | None = None,
        y_val: NDArray | None = None,
    ) -> dict[str, float]:
        """Train the model. Returns training metrics."""
        ...

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        """Generate predictions."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> BaseModel:
        """Load model from disk."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier."""
        ...
