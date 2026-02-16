"""WaveletGPT: Causal transformer for SAX word sequence prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from wavecast.core.exceptions import ModelNotTrainedError

from .base import BaseModel


class WaveletGPTNet(nn.Module):
    """Causal transformer for predicting next SAX word.

    Supports multi-horizon prediction: separate classification heads for
    each prediction horizon (e.g. 1, 2, 4, 8 steps ahead).
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int = 32,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        n_levels: int = 6,
        n_asset_classes: int = 7,
        prediction_horizons: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.prediction_horizons = prediction_horizons or [1]

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(context_length, embed_dim)
        self.level_embed = nn.Embedding(n_levels, embed_dim)
        self.asset_class_embed = nn.Embedding(n_asset_classes, embed_dim)

        # Transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.ln_f = nn.LayerNorm(embed_dim)

        # Classification heads — one per horizon.
        # Horizon=1 head is weight-tied with token embeddings.
        # Additional horizons get independent heads.
        self.heads = nn.ModuleDict()
        for h in self.prediction_horizons:
            head = nn.Linear(embed_dim, vocab_size, bias=False)
            if h == 1:
                head.weight = self.token_embed.weight  # Weight tying
            self.heads[str(h)] = head

    def forward(
        self,
        token_ids: torch.Tensor,  # (batch, context_length)
        level_ids: torch.Tensor,  # (batch,)
        asset_class_ids: torch.Tensor,  # (batch,)
    ) -> dict[int, torch.Tensor]:
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Position indices
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        # Sum embeddings: token + position + level (broadcast) + asset_class (broadcast)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = x + self.level_embed(level_ids).unsqueeze(1)
        x = x + self.asset_class_embed(asset_class_ids).unsqueeze(1)

        # Causal mask: upper triangular = True means "mask this position"
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)

        # Last position's hidden state for all heads
        last_hidden = x[:, -1, :]  # (batch, embed_dim)

        logits_dict: dict[int, torch.Tensor] = {}
        for h in self.prediction_horizons:
            logits_dict[h] = self.heads[str(h)](last_hidden)  # (batch, vocab_size)

        return logits_dict


class WaveletGPT(BaseModel):
    """WaveletGPT wrapper conforming to BaseModel interface.

    X format: each row is [context_token_0, ..., context_token_{L-1}, level_id, asset_class_id]
    y format: target token ID (integer) for single-horizon,
              or column-stacked targets for multi-horizon
    """

    def __init__(
        self,
        vocab_size: int = 100,
        context_length: int = 32,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        n_levels: int = 6,
        n_asset_classes: int = 7,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.0003,
        patience: int = 10,
        prediction_horizons: list[int] | None = None,
        horizon_weights: dict[int, float] | None = None,
    ) -> None:
        self._prediction_horizons = prediction_horizons or [1]
        self._horizon_weights = horizon_weights
        self._config = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "n_levels": n_levels,
            "n_asset_classes": n_asset_classes,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "prediction_horizons": self._prediction_horizons,
        }
        if self._horizon_weights is not None:
            self._config["horizon_weights"] = {
                str(k): v for k, v in self._horizon_weights.items()
            }
        self._net: WaveletGPTNet | None = None
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def name(self) -> str:
        return "wavelet_gpt"

    @property
    def prediction_horizons(self) -> list[int]:
        return self._prediction_horizons

    def _parse_x(
        self, X: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Split X into (context_tokens, level_ids, asset_class_ids)."""
        ctx_len = self._config["context_length"]
        contexts = X[:, :ctx_len].astype(np.int64)
        levels = X[:, ctx_len].astype(np.int64)
        asset_classes = X[:, ctx_len + 1].astype(np.int64)
        return contexts, levels, asset_classes

    def _parse_y_multi(
        self, y: NDArray
    ) -> dict[int, NDArray]:
        """Parse y into per-horizon target arrays.

        For single-horizon (y is 1-D): returns {horizons[0]: y}.
        For multi-horizon (y is 2-D with columns matching horizons): returns
        {horizon: y[:, col]} for each horizon.
        """
        if y.ndim == 1:
            return {self._prediction_horizons[0]: y.astype(np.int64)}
        # Multi-horizon: columns correspond to self._prediction_horizons
        result: dict[int, NDArray] = {}
        for col_idx, h in enumerate(self._prediction_horizons):
            result[h] = y[:, col_idx].astype(np.int64)
        return result

    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_val: NDArray | None = None,
        y_val: NDArray | None = None,
    ) -> dict[str, float]:
        # Build network
        self._net = WaveletGPTNet(
            vocab_size=self._config["vocab_size"],
            context_length=self._config["context_length"],
            embed_dim=self._config["embed_dim"],
            num_heads=self._config["num_heads"],
            num_layers=self._config["num_layers"],
            dropout=self._config["dropout"],
            n_levels=self._config["n_levels"],
            n_asset_classes=self._config["n_asset_classes"],
            prediction_horizons=self._prediction_horizons,
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=self._config["learning_rate"]
        )
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD

        # Resolve horizon weights
        weights: dict[int, float] = {}
        if self._horizon_weights is not None:
            weights = self._horizon_weights
        else:
            for h in self._prediction_horizons:
                weights[h] = 1.0

        # Prepare data
        ctx_train, lvl_train, ac_train = self._parse_x(X_train)
        y_per_horizon = self._parse_y_multi(y_train)

        # Build tensors for each horizon target
        target_tensors = []
        for h in self._prediction_horizons:
            target_tensors.append(
                torch.tensor(y_per_horizon[h], dtype=torch.long)
            )

        dataset = TensorDataset(
            torch.tensor(ctx_train, dtype=torch.long),
            torch.tensor(lvl_train, dtype=torch.long),
            torch.tensor(ac_train, dtype=torch.long),
            *target_tensors,
        )
        loader = DataLoader(
            dataset, batch_size=self._config["batch_size"], shuffle=True
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        avg_loss = 0.0

        n_horizons = len(self._prediction_horizons)

        for _epoch in range(self._config["epochs"]):
            self._net.train()
            total_loss = 0.0
            n_batches = 0

            for batch in loader:
                ctx_b = batch[0].to(self._device)
                lvl_b = batch[1].to(self._device)
                ac_b = batch[2].to(self._device)
                # Remaining tensors are targets per horizon
                target_bs = [batch[3 + i].to(self._device) for i in range(n_horizons)]

                logits_dict = self._net(ctx_b, lvl_b, ac_b)

                loss = torch.tensor(0.0, device=self._device)
                for i, h in enumerate(self._prediction_horizons):
                    h_loss = criterion(logits_dict[h], target_bs[i])
                    loss = loss + weights[h] * h_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Validation
            if X_val is not None and y_val is not None:
                self._net.eval()
                with torch.no_grad():
                    ctx_v, lvl_v, ac_v = self._parse_x(X_val)
                    v_logits_dict = self._net(
                        torch.tensor(ctx_v, dtype=torch.long).to(self._device),
                        torch.tensor(lvl_v, dtype=torch.long).to(self._device),
                        torch.tensor(ac_v, dtype=torch.long).to(self._device),
                    )
                    y_val_per_horizon = self._parse_y_multi(y_val)
                    val_loss = 0.0
                    for h in self._prediction_horizons:
                        y_val_h = torch.tensor(
                            y_val_per_horizon[h], dtype=torch.long
                        ).to(self._device)
                        val_loss += weights[h] * criterion(
                            v_logits_dict[h], y_val_h
                        ).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self._net.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self._config["patience"]:
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        # Compute final metrics (horizon=1 or first horizon for backward compat)
        self._net.eval()
        primary_h = self._prediction_horizons[0]
        with torch.no_grad():
            ctx_t, lvl_t, ac_t = self._parse_x(X_train)
            logits_dict = self._net(
                torch.tensor(ctx_t, dtype=torch.long).to(self._device),
                torch.tensor(lvl_t, dtype=torch.long).to(self._device),
                torch.tensor(ac_t, dtype=torch.long).to(self._device),
            )
            preds = logits_dict[primary_h].argmax(dim=-1).cpu().numpy()
            y_primary = y_per_horizon[primary_h]
            accuracy = float(np.mean(preds == y_primary))

        metrics: dict[str, float] = {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
        }
        if X_val is not None:
            metrics["val_loss"] = best_val_loss
        return metrics

    def predict(self, X: NDArray, horizon: int = 1) -> NDArray:
        if self._net is None:
            raise ModelNotTrainedError("Model has not been trained")
        self._net.eval()
        with torch.no_grad():
            ctx, lvl, ac = self._parse_x(X)
            logits_dict = self._net(
                torch.tensor(ctx, dtype=torch.long).to(self._device),
                torch.tensor(lvl, dtype=torch.long).to(self._device),
                torch.tensor(ac, dtype=torch.long).to(self._device),
            )
        if horizon not in logits_dict:
            raise ValueError(
                f"Horizon {horizon} not available. "
                f"Available horizons: {self._prediction_horizons}"
            )
        return logits_dict[horizon].argmax(dim=-1).cpu().numpy()

    def predict_proba(self, X: NDArray, horizon: int = 1) -> NDArray:
        """Return softmax probabilities over vocabulary."""
        if self._net is None:
            raise ModelNotTrainedError("Model has not been trained")
        self._net.eval()
        with torch.no_grad():
            ctx, lvl, ac = self._parse_x(X)
            logits_dict = self._net(
                torch.tensor(ctx, dtype=torch.long).to(self._device),
                torch.tensor(lvl, dtype=torch.long).to(self._device),
                torch.tensor(ac, dtype=torch.long).to(self._device),
            )
        if horizon not in logits_dict:
            raise ValueError(
                f"Horizon {horizon} not available. "
                f"Available horizons: {self._prediction_horizons}"
            )
        return torch.softmax(logits_dict[horizon], dim=-1).cpu().numpy()

    def predict_all_horizons(self, X: NDArray) -> dict[int, NDArray]:
        """Return predictions for all horizons."""
        if self._net is None:
            raise ModelNotTrainedError("Model has not been trained")
        self._net.eval()
        with torch.no_grad():
            ctx, lvl, ac = self._parse_x(X)
            logits_dict = self._net(
                torch.tensor(ctx, dtype=torch.long).to(self._device),
                torch.tensor(lvl, dtype=torch.long).to(self._device),
                torch.tensor(ac, dtype=torch.long).to(self._device),
            )
        return {
            h: logits.argmax(dim=-1).cpu().numpy()
            for h, logits in logits_dict.items()
        }

    def predict_proba_all_horizons(self, X: NDArray) -> dict[int, NDArray]:
        """Return softmax probabilities for all horizons."""
        if self._net is None:
            raise ModelNotTrainedError("Model has not been trained")
        self._net.eval()
        with torch.no_grad():
            ctx, lvl, ac = self._parse_x(X)
            logits_dict = self._net(
                torch.tensor(ctx, dtype=torch.long).to(self._device),
                torch.tensor(lvl, dtype=torch.long).to(self._device),
                torch.tensor(ac, dtype=torch.long).to(self._device),
            )
        return {
            h: torch.softmax(logits, dim=-1).cpu().numpy()
            for h, logits in logits_dict.items()
        }

    def save(self, path: Path) -> None:
        if self._net is None:
            raise ModelNotTrainedError("Cannot save untrained model")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path / "model.pt")
        with open(path / "config.json", "w") as f:
            json.dump(self._config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> WaveletGPT:
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)
        training_params = {
            k: config[k]
            for k in ["epochs", "batch_size", "learning_rate", "patience"]
        }
        net_params = {
            k: config[k]
            for k in [
                "vocab_size",
                "context_length",
                "embed_dim",
                "num_heads",
                "num_layers",
                "dropout",
                "n_levels",
                "n_asset_classes",
            ]
        }
        prediction_horizons = config.get("prediction_horizons", [1])
        horizon_weights = None
        if "horizon_weights" in config:
            horizon_weights = {
                int(k): v for k, v in config["horizon_weights"].items()
            }
        instance = cls(
            **net_params,
            **training_params,
            prediction_horizons=prediction_horizons,
            horizon_weights=horizon_weights,
        )
        instance._net = WaveletGPTNet(
            **net_params, prediction_horizons=prediction_horizons
        ).to(instance._device)
        state = torch.load(
            path / "model.pt", map_location="cpu", weights_only=True
        )
        instance._net.load_state_dict(state)
        return instance
