"""Wavelet-domain LSTM model with multi-branch architecture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from wavecast.core.exceptions import ModelError, ModelNotTrainedError

from .base import BaseModel


class WaveletLSTMNet(nn.Module):
    """Multi-branch LSTM: one branch per DWT level, concatenated with extra features."""

    def __init__(
        self,
        branch_input_sizes: list[int],
        extra_features_size: int = 0,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.branch_input_sizes = branch_input_sizes
        self.extra_features_size = extra_features_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # One LSTM per wavelet level branch
        self.branches = nn.ModuleList([
            nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            for _ in branch_input_sizes
        ])

        # Total input to the dense head: hidden from each branch + extra features
        concat_size = hidden_size * len(branch_input_sizes) + extra_features_size

        self.head = nn.Sequential(
            nn.Linear(concat_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, branch_inputs: list[torch.Tensor], extra: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            branch_inputs: List of tensors, each (batch, seq_len_i, 1).
            extra: Optional (batch, extra_features_size) tensor.
        """
        hidden_states = []
        for lstm, x in zip(self.branches, branch_inputs, strict=True):
            # x shape: (batch, seq_len, 1)
            _, (h_n, _) = lstm(x)
            # h_n shape: (num_layers, batch, hidden_size) — take last layer
            hidden_states.append(h_n[-1])  # (batch, hidden_size)

        combined = torch.cat(hidden_states, dim=1)  # (batch, hidden_size * n_branches)
        if extra is not None and self.extra_features_size > 0:
            combined = torch.cat([combined, extra], dim=1)

        return self.head(combined).squeeze(-1)  # (batch,)


class WaveletLSTM(BaseModel):
    """WaveletLSTMNet wrapper conforming to BaseModel interface.

    Expects flat feature vectors. Uses branch_input_sizes to split
    the first portion into per-branch sequences, remainder as extra features.
    """

    def __init__(
        self,
        branch_input_sizes: list[int] | None = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 10,
    ) -> None:
        self._config = {
            "branch_input_sizes": branch_input_sizes or [50, 25, 13, 7, 4],
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
        }
        self._net: WaveletLSTMNet | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "wavelet_lstm"

    def _split_features(self, X: NDArray) -> tuple[list[NDArray], NDArray]:
        """Split flat feature matrix into branch inputs and extra features."""
        branch_sizes = self._config["branch_input_sizes"]
        total_branch = sum(branch_sizes)
        branches = []
        offset = 0
        for size in branch_sizes:
            end = min(offset + size, X.shape[1])
            branches.append(X[:, offset:end])
            offset = end
        extra = X[:, total_branch:] if total_branch < X.shape[1] else np.empty((X.shape[0], 0))
        return branches, extra

    def _to_tensors(
        self, X: NDArray
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        branches, extra = self._split_features(X)
        branch_tensors = [
            torch.tensor(b, dtype=torch.float32).unsqueeze(-1).to(self._device)
            for b in branches
        ]
        extra_tensor = torch.tensor(extra, dtype=torch.float32).to(self._device)
        return branch_tensors, extra_tensor

    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_val: NDArray | None = None,
        y_val: NDArray | None = None,
    ) -> dict[str, float]:
        branch_sizes = self._config["branch_input_sizes"]
        total_branch = sum(branch_sizes)
        extra_size = max(X_train.shape[1] - total_branch, 0)

        self._net = WaveletLSTMNet(
            branch_input_sizes=branch_sizes,
            extra_features_size=extra_size,
            hidden_size=self._config["hidden_size"],
            num_layers=self._config["num_layers"],
            dropout=self._config["dropout"],
        ).to(self._device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._config["learning_rate"])
        criterion = nn.MSELoss()

        # Build DataLoader
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self._config["batch_size"], shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for _epoch in range(self._config["epochs"]):
            self._net.train()
            epoch_loss = 0.0
            for _n_batches, (X_batch, y_batch) in enumerate(loader, start=1):
                X_batch = X_batch.numpy()
                y_batch = y_batch.to(self._device)
                branches, extra = self._to_tensors(X_batch)
                pred = self._net(branches, extra if extra.shape[1] > 0 else None)
                loss = criterion(pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation early stopping
            if X_val is not None and y_val is not None:
                self._net.eval()
                with torch.no_grad():
                    val_branches, val_extra = self._to_tensors(X_val)
                    val_pred = self._net(
                        val_branches, val_extra if val_extra.shape[1] > 0 else None
                    )
                    val_loss = criterion(
                        val_pred, torch.tensor(y_val, dtype=torch.float32).to(self._device)
                    ).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self._config["patience"]:
                        break

        # Restore best model
        if best_state is not None:
            self._net.load_state_dict(best_state)

        # Compute final metrics
        self._net.eval()
        with torch.no_grad():
            train_branches, train_extra = self._to_tensors(X_train)
            train_pred = (
                self._net(train_branches, train_extra if train_extra.shape[1] > 0 else None)
                .cpu()
                .numpy()
            )
        train_rmse = float(np.sqrt(np.mean((y_train - train_pred) ** 2)))
        metrics: dict[str, float] = {"train_rmse": train_rmse}

        if X_val is not None and y_val is not None:
            metrics["val_rmse"] = float(np.sqrt(best_val_loss))

        return metrics

    def predict(self, X: NDArray) -> NDArray:
        if self._net is None:
            raise ModelNotTrainedError("Model has not been trained")
        self._net.eval()
        with torch.no_grad():
            branches, extra = self._to_tensors(X)
            pred = self._net(branches, extra if extra.shape[1] > 0 else None)
        return pred.cpu().numpy()

    def save(self, path: Path) -> None:
        if self._net is None:
            raise ModelNotTrainedError("Cannot save untrained model")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path / "model.pt")
        with open(path / "config.json", "w") as f:
            json.dump(self._config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> WaveletLSTM:
        path = Path(path)
        config_path = path / "config.json"
        model_path = path / "model.pt"
        if not model_path.exists():
            raise ModelError(f"No model file at {model_path}")

        with open(config_path) as f:
            config = json.load(f)

        instance = cls(**config)
        # Need to know feature size to reconstruct — infer from state dict
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        branch_sizes = config["branch_input_sizes"]

        # Infer extra_features_size from head input layer weight shape
        head_input_features = state["head.0.weight"].shape[1]
        hidden_size = config["hidden_size"]
        n_branches = len(branch_sizes)
        extra_size = head_input_features - hidden_size * n_branches

        instance._net = WaveletLSTMNet(
            branch_input_sizes=branch_sizes,
            extra_features_size=max(extra_size, 0),
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(instance._device)
        instance._net.load_state_dict(state)
        return instance
