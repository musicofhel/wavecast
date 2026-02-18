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
from wavecast.data.mmap_dataset import MMapSequenceDataset

from .base import BaseModel

# Valid task types
TASK_TOKEN = "token"
TASK_RETURN_QUANTILE = "return_quantile"
TASK_RETURN_REGRESSION = "return_regression"
_VALID_TASKS = {TASK_TOKEN, TASK_RETURN_QUANTILE, TASK_RETURN_REGRESSION}

# Input modes
INPUT_TOKENIZED = "tokenized"
INPUT_CONTINUOUS = "continuous"
_VALID_INPUT_MODES = {INPUT_TOKENIZED, INPUT_CONTINUOUS}


class RankNetLoss(nn.Module):
    """Pairwise ranking loss for ordinal quantile prediction.

    For each pair of samples (i, j) where label_i > label_j, compute:
        L = log(1 + exp(-sigma * (s_i - s_j)))
    where s is the model's logit for the correct class.

    This optimizes ranking consistency rather than classification accuracy.
    """

    def __init__(self, sigma: float = 1.0, n_pairs: int = 256) -> None:
        super().__init__()
        self.sigma = sigma
        self.n_pairs = n_pairs

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Sample random pairs
        n_pairs = min(self.n_pairs, batch_size * (batch_size - 1) // 2)
        idx_i = torch.randint(0, batch_size, (n_pairs,), device=logits.device)
        idx_j = torch.randint(0, batch_size, (n_pairs,), device=logits.device)

        # Filter to pairs where labels differ
        label_diff = targets[idx_i].float() - targets[idx_j].float()
        valid = label_diff != 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        idx_i = idx_i[valid]
        idx_j = idx_j[valid]
        label_diff = label_diff[valid]

        # Scores: weighted sum of logits (higher class → higher score)
        weights = torch.arange(logits.shape[1], dtype=torch.float32, device=logits.device)
        scores = (torch.softmax(logits, dim=-1) * weights).sum(dim=-1)

        s_i = scores[idx_i]
        s_j = scores[idx_j]
        s_ij = torch.sign(label_diff)

        # RankNet loss
        loss = torch.log1p(torch.exp(-self.sigma * s_ij * (s_i - s_j)))
        return loss.mean()


class WaveletGPTNet(nn.Module):
    """Causal transformer for predicting next SAX word.

    Supports multi-horizon prediction: separate classification heads for
    each prediction horizon (e.g. 1, 2, 4, 8 steps ahead).

    Supports three task types:
    - "token": predict next SAX token (default, weight-tied for h=1)
    - "return_quantile": predict return quantile class (no weight tying)
    - "return_regression": predict raw return value (single output, no weight tying)

    Supports two input modes:
    - "tokenized": SAX token IDs → nn.Embedding (default)
    - "continuous": raw float coefficients → nn.Linear(1, embed_dim)
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
        task: str = TASK_TOKEN,
        n_output_classes: int | None = None,
        input_mode: str = INPUT_TOKENIZED,
        n_aux_features: int = 0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.prediction_horizons = prediction_horizons or [1]
        self.task = task
        self.input_mode = input_mode
        self.n_aux_features = n_aux_features

        # Input projection: token embedding or continuous linear
        if input_mode == INPUT_CONTINUOUS:
            self.input_proj = nn.Linear(1, embed_dim)
        else:
            self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Auxiliary feature projection
        if n_aux_features > 0:
            self.aux_proj = nn.Linear(n_aux_features, embed_dim)

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

        # Determine output dimension based on task
        if task == TASK_RETURN_REGRESSION:
            output_dim = 1
        elif task == TASK_RETURN_QUANTILE:
            output_dim = n_output_classes if n_output_classes is not None else 5
        else:
            # token prediction: output_dim = vocab_size
            output_dim = vocab_size

        self.output_dim = output_dim

        # Output heads — one per horizon.
        # Weight tying only for task="token" with h=1.
        self.heads = nn.ModuleDict()
        for h in self.prediction_horizons:
            head = nn.Linear(embed_dim, output_dim, bias=False)
            if (
                input_mode == INPUT_TOKENIZED
                and task == TASK_TOKEN
                and h == 1
                and output_dim == vocab_size
            ):
                head.weight = self.token_embed.weight  # Weight tying
            self.heads[str(h)] = head

    def forward(
        self,
        token_ids: torch.Tensor,  # (batch, context_length) — int64 or float32
        level_ids: torch.Tensor,  # (batch,)
        asset_class_ids: torch.Tensor,  # (batch,)
        aux_features: torch.Tensor | None = None,  # (batch, context_length, n_aux)
    ) -> dict[int, torch.Tensor]:
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Position indices
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        # Input projection: token embedding or continuous linear
        if self.input_mode == INPUT_CONTINUOUS:
            x = self.input_proj(token_ids.unsqueeze(-1)) + self.pos_embed(positions)
        else:
            x = self.token_embed(token_ids) + self.pos_embed(positions)

        # Auxiliary features
        if self.n_aux_features > 0 and aux_features is not None:
            x = x + self.aux_proj(aux_features)

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
            logits_dict[h] = self.heads[str(h)](last_hidden)  # (batch, output_dim)

        return logits_dict


class WaveletGPT(BaseModel):
    """WaveletGPT wrapper conforming to BaseModel interface.

    X format: each row is [context_token_0, ..., context_token_{L-1}, level_id, asset_class_id]
    y format: target token ID (integer) for single-horizon,
              or column-stacked targets for multi-horizon

    Supports three task types:
    - "token": predict next SAX token (default, backward compatible)
    - "return_quantile": predict return quantile class (5 classes)
    - "return_regression": predict raw return value
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
        use_amp: bool = False,
        task: str = TASK_TOKEN,
        n_output_classes: int | None = None,
        class_weights: list[float] | None = None,
        input_mode: str = INPUT_TOKENIZED,
        n_aux_features: int = 0,
        loss_type: str = "ce",
        loss_kwargs: dict | None = None,
    ) -> None:
        if task not in _VALID_TASKS:
            raise ValueError(
                f"Unknown task '{task}'. Must be one of {_VALID_TASKS}"
            )
        if input_mode not in _VALID_INPUT_MODES:
            raise ValueError(
                f"Unknown input_mode '{input_mode}'. "
                f"Must be one of {_VALID_INPUT_MODES}"
            )
        self._task = task
        self._input_mode = input_mode
        self._n_aux_features = n_aux_features
        self._n_output_classes = n_output_classes
        self._class_weights = class_weights
        self._loss_type = loss_type
        self._loss_kwargs = loss_kwargs or {}
        self._prediction_horizons = prediction_horizons or [1]
        self._horizon_weights = horizon_weights
        self._use_amp = use_amp
        self._config: dict = {
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
            "use_amp": use_amp,
            "task": task,
            "input_mode": input_mode,
            "n_aux_features": n_aux_features,
        }
        if self._n_output_classes is not None:
            self._config["n_output_classes"] = n_output_classes
        if self._class_weights is not None:
            self._config["class_weights"] = class_weights
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
    def task(self) -> str:
        return self._task

    @property
    def prediction_horizons(self) -> list[int]:
        return self._prediction_horizons

    @property
    def input_mode(self) -> str:
        return self._input_mode

    def _parse_x(
        self, X: NDArray
    ) -> tuple[NDArray, NDArray, NDArray, NDArray | None]:
        """Split X into (context_values, level_ids, asset_class_ids, aux_features).

        When n_aux_features > 0, X layout is:
        [ctx_0..ctx_{L-1}, aux_00..aux_{L-1}_{F-1}, level, asset_class]
        aux_features returned as (N, L, F) float32 array.
        """
        ctx_len = self._config["context_length"]
        n_aux = self._config.get("n_aux_features", 0)
        ctx_dtype = np.float32 if self._input_mode == INPUT_CONTINUOUS else np.int64
        contexts = X[:, :ctx_len].astype(ctx_dtype)

        if n_aux > 0:
            aux_end = ctx_len + ctx_len * n_aux
            aux = X[:, ctx_len:aux_end].astype(np.float32).reshape(-1, ctx_len, n_aux)
            levels = X[:, aux_end].astype(np.int64)
            asset_classes = X[:, aux_end + 1].astype(np.int64)
        else:
            aux = None
            levels = X[:, ctx_len].astype(np.int64)
            asset_classes = X[:, ctx_len + 1].astype(np.int64)

        return contexts, levels, asset_classes, aux

    def _parse_y_multi(
        self, y: NDArray
    ) -> dict[int, NDArray]:
        """Parse y into per-horizon target arrays.

        For single-horizon (y is 1-D): returns {horizons[0]: y}.
        For multi-horizon (y is 2-D with columns matching horizons): returns
        {horizon: y[:, col]} for each horizon.

        Dtype depends on task: float32 for regression, int64 otherwise.
        """
        target_dtype = np.float32 if self._task == TASK_RETURN_REGRESSION else np.int64

        if y.ndim == 1:
            return {self._prediction_horizons[0]: y.astype(target_dtype)}
        # Multi-horizon: columns correspond to self._prediction_horizons
        result: dict[int, NDArray] = {}
        for col_idx, h in enumerate(self._prediction_horizons):
            result[h] = y[:, col_idx].astype(target_dtype)
        return result

    def _build_criterion(self) -> nn.Module:
        """Build the appropriate loss function for the task."""
        if self._loss_type == "ranknet":
            kwargs = {k: v for k, v in self._loss_kwargs.items()}
            return RankNetLoss(**kwargs).to(self._device)
        if self._task == TASK_RETURN_REGRESSION:
            return nn.MSELoss()
        # Classification: CrossEntropy
        if self._class_weights is not None:
            weight = torch.tensor(self._class_weights, dtype=torch.float32).to(
                self._device
            )
            return nn.CrossEntropyLoss(weight=weight)
        if self._task == TASK_TOKEN:
            return nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD
        # return_quantile: no ignore_index (classes 0-4 are all valid)
        return nn.CrossEntropyLoss()

    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_val: NDArray | None = None,
        y_val: NDArray | None = None,
        use_amp: bool | None = None,
        dataset_path: Path | None = None,
    ) -> dict[str, float]:
        # Resolve AMP: explicit arg > instance default; only on CUDA
        amp_enabled = use_amp if use_amp is not None else self._use_amp
        if self._device.type != "cuda":
            amp_enabled = False

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
            task=self._task,
            n_output_classes=self._n_output_classes,
            input_mode=self._input_mode,
            n_aux_features=self._n_aux_features,
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=self._config["learning_rate"]
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        criterion = self._build_criterion()

        # Resolve horizon weights
        weights: dict[int, float] = {}
        if self._horizon_weights is not None:
            weights = self._horizon_weights
        else:
            for h in self._prediction_horizons:
                weights[h] = 1.0

        # Prepare data
        ctx_train, lvl_train, ac_train, aux_train = self._parse_x(X_train)
        y_per_horizon = self._parse_y_multi(y_train)
        has_aux = aux_train is not None

        # Context tensor dtype depends on input mode
        ctx_torch_dtype = (
            torch.float32
            if self._input_mode == INPUT_CONTINUOUS
            else torch.long
        )
        # Target tensor dtype depends on task
        target_torch_dtype = (
            torch.float32
            if self._task == TASK_RETURN_REGRESSION
            else torch.long
        )

        # Build dataset: mmap path or in-memory TensorDataset
        if dataset_path is not None:
            dataset = MMapSequenceDataset(Path(dataset_path))
        else:
            target_tensors = []
            for h in self._prediction_horizons:
                target_tensors.append(
                    torch.tensor(y_per_horizon[h], dtype=target_torch_dtype)
                )
            base_tensors = [
                torch.tensor(ctx_train, dtype=ctx_torch_dtype),
                torch.tensor(lvl_train, dtype=torch.long),
                torch.tensor(ac_train, dtype=torch.long),
            ]
            if has_aux:
                # Flatten aux: (N, ctx_len, n_aux) → (N, ctx_len * n_aux)
                base_tensors.append(
                    torch.tensor(
                        aux_train.reshape(len(aux_train), -1),
                        dtype=torch.float32,
                    )
                )
            dataset = TensorDataset(*base_tensors, *target_tensors)
        loader = DataLoader(
            dataset, batch_size=self._config["batch_size"], shuffle=True
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        avg_loss = 0.0

        n_horizons = len(self._prediction_horizons)
        is_regression = self._task == TASK_RETURN_REGRESSION
        target_offset = 4 if has_aux else 3
        n_aux = self._n_aux_features
        ctx_len = self._config["context_length"]

        for _epoch in range(self._config["epochs"]):
            self._net.train()
            total_loss = 0.0
            n_batches = 0

            for batch in loader:
                ctx_b = batch[0].to(self._device)
                lvl_b = batch[1].to(self._device)
                ac_b = batch[2].to(self._device)
                aux_b = None
                if has_aux:
                    aux_b = batch[3].to(self._device).reshape(-1, ctx_len, n_aux)
                target_bs = [
                    batch[target_offset + i].to(self._device)
                    for i in range(n_horizons)
                ]

                optimizer.zero_grad()
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits_dict = self._net(ctx_b, lvl_b, ac_b, aux_features=aux_b)

                    loss = torch.tensor(0.0, device=self._device)
                    for i, h in enumerate(self._prediction_horizons):
                        output = logits_dict[h]
                        target = target_bs[i]
                        if is_regression:
                            # Squeeze output from (batch, 1) to (batch,)
                            h_loss = criterion(output.squeeze(-1), target)
                        else:
                            h_loss = criterion(output, target)
                        loss = loss + weights[h] * h_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Validation
            if X_val is not None and y_val is not None:
                self._net.eval()
                with torch.no_grad():
                    ctx_v, lvl_v, ac_v, aux_v = self._parse_x(X_val)
                    aux_v_t = None
                    if aux_v is not None:
                        aux_v_t = torch.tensor(aux_v, dtype=torch.float32).to(
                            self._device
                        )
                    with torch.autocast(
                        device_type=self._device.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        v_logits_dict = self._net(
                            torch.tensor(ctx_v, dtype=ctx_torch_dtype).to(self._device),
                            torch.tensor(lvl_v, dtype=torch.long).to(self._device),
                            torch.tensor(ac_v, dtype=torch.long).to(self._device),
                            aux_features=aux_v_t,
                        )
                    y_val_per_horizon = self._parse_y_multi(y_val)
                    val_loss = 0.0
                    for h in self._prediction_horizons:
                        y_val_h = torch.tensor(
                            y_val_per_horizon[h], dtype=target_torch_dtype
                        ).to(self._device)
                        output = v_logits_dict[h]
                        if is_regression:
                            val_loss += weights[h] * criterion(
                                output.squeeze(-1), y_val_h
                            ).item()
                        else:
                            val_loss += weights[h] * criterion(
                                output, y_val_h
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
            ctx_t, lvl_t, ac_t, aux_t = self._parse_x(X_train)
            aux_t_tensor = None
            if aux_t is not None:
                aux_t_tensor = torch.tensor(aux_t, dtype=torch.float32).to(
                    self._device
                )
            logits_dict = self._net(
                torch.tensor(ctx_t, dtype=ctx_torch_dtype).to(self._device),
                torch.tensor(lvl_t, dtype=torch.long).to(self._device),
                torch.tensor(ac_t, dtype=torch.long).to(self._device),
                aux_features=aux_t_tensor,
            )
            y_primary = y_per_horizon[primary_h]
            if is_regression:
                preds_raw = logits_dict[primary_h].squeeze(-1).cpu().numpy()
                accuracy = float(1.0 - np.mean((preds_raw - y_primary) ** 2))
            else:
                preds = logits_dict[primary_h].argmax(dim=-1).cpu().numpy()
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
        amp_enabled = self._use_amp and self._device.type == "cuda"
        ctx_dtype = torch.float32 if self._input_mode == INPUT_CONTINUOUS else torch.long
        self._net.eval()
        with torch.no_grad(), torch.autocast(
            device_type=self._device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            ctx, lvl, ac, aux = self._parse_x(X)
            aux_t = None
            if aux is not None:
                aux_t = torch.tensor(aux, dtype=torch.float32).to(self._device)
            logits_dict = self._net(
                torch.tensor(ctx, dtype=ctx_dtype).to(self._device),
                torch.tensor(lvl, dtype=torch.long).to(self._device),
                torch.tensor(ac, dtype=torch.long).to(self._device),
                aux_features=aux_t,
            )
        if horizon not in logits_dict:
            raise ValueError(
                f"Horizon {horizon} not available. "
                f"Available horizons: {self._prediction_horizons}"
            )
        output = logits_dict[horizon]
        if self._task == TASK_RETURN_REGRESSION:
            return output.squeeze(-1).cpu().numpy()
        return output.argmax(dim=-1).cpu().numpy()

    def predict_proba(self, X: NDArray, horizon: int = 1) -> NDArray:
        """Return softmax probabilities (classification) or raw values (regression)."""
        if self._net is None:
            raise ModelNotTrainedError("Model has not been trained")
        amp_enabled = self._use_amp and self._device.type == "cuda"
        ctx_dtype = torch.float32 if self._input_mode == INPUT_CONTINUOUS else torch.long
        self._net.eval()
        with torch.no_grad(), torch.autocast(
            device_type=self._device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            ctx, lvl, ac, aux = self._parse_x(X)
            aux_t = None
            if aux is not None:
                aux_t = torch.tensor(aux, dtype=torch.float32).to(self._device)
            logits_dict = self._net(
                torch.tensor(ctx, dtype=ctx_dtype).to(self._device),
                torch.tensor(lvl, dtype=torch.long).to(self._device),
                torch.tensor(ac, dtype=torch.long).to(self._device),
                aux_features=aux_t,
            )
        if horizon not in logits_dict:
            raise ValueError(
                f"Horizon {horizon} not available. "
                f"Available horizons: {self._prediction_horizons}"
            )
        output = logits_dict[horizon]
        if self._task == TASK_RETURN_REGRESSION:
            return output.squeeze(-1).cpu().numpy()
        return torch.softmax(output, dim=-1).cpu().numpy()

    def predict_all_horizons(self, X: NDArray) -> dict[int, NDArray]:
        """Return predictions for all horizons."""
        if self._net is None:
            raise ModelNotTrainedError("Model has not been trained")
        amp_enabled = self._use_amp and self._device.type == "cuda"
        ctx_dtype = torch.float32 if self._input_mode == INPUT_CONTINUOUS else torch.long
        self._net.eval()
        with torch.no_grad(), torch.autocast(
            device_type=self._device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            ctx, lvl, ac, aux = self._parse_x(X)
            aux_t = None
            if aux is not None:
                aux_t = torch.tensor(aux, dtype=torch.float32).to(self._device)
            logits_dict = self._net(
                torch.tensor(ctx, dtype=ctx_dtype).to(self._device),
                torch.tensor(lvl, dtype=torch.long).to(self._device),
                torch.tensor(ac, dtype=torch.long).to(self._device),
                aux_features=aux_t,
            )
        if self._task == TASK_RETURN_REGRESSION:
            return {
                h: logits.squeeze(-1).cpu().numpy()
                for h, logits in logits_dict.items()
            }
        return {
            h: logits.argmax(dim=-1).cpu().numpy()
            for h, logits in logits_dict.items()
        }

    def predict_proba_all_horizons(self, X: NDArray) -> dict[int, NDArray]:
        """Return softmax probabilities for all horizons."""
        if self._net is None:
            raise ModelNotTrainedError("Model has not been trained")
        amp_enabled = self._use_amp and self._device.type == "cuda"
        ctx_dtype = torch.float32 if self._input_mode == INPUT_CONTINUOUS else torch.long
        self._net.eval()
        with torch.no_grad(), torch.autocast(
            device_type=self._device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            ctx, lvl, ac, aux = self._parse_x(X)
            aux_t = None
            if aux is not None:
                aux_t = torch.tensor(aux, dtype=torch.float32).to(self._device)
            logits_dict = self._net(
                torch.tensor(ctx, dtype=ctx_dtype).to(self._device),
                torch.tensor(lvl, dtype=torch.long).to(self._device),
                torch.tensor(ac, dtype=torch.long).to(self._device),
                aux_features=aux_t,
            )
        if self._task == TASK_RETURN_REGRESSION:
            return {
                h: logits.squeeze(-1).cpu().numpy()
                for h, logits in logits_dict.items()
            }
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
        use_amp = config.get("use_amp", False)
        task = config.get("task", TASK_TOKEN)
        input_mode = config.get("input_mode", INPUT_TOKENIZED)
        n_aux_features = config.get("n_aux_features", 0)
        n_output_classes = config.get("n_output_classes")
        class_weights = config.get("class_weights")
        instance = cls(
            **net_params,
            **training_params,
            prediction_horizons=prediction_horizons,
            horizon_weights=horizon_weights,
            use_amp=use_amp,
            task=task,
            n_output_classes=n_output_classes,
            class_weights=class_weights,
            input_mode=input_mode,
            n_aux_features=n_aux_features,
        )
        instance._net = WaveletGPTNet(
            **net_params,
            prediction_horizons=prediction_horizons,
            task=task,
            n_output_classes=n_output_classes,
            input_mode=input_mode,
            n_aux_features=n_aux_features,
        ).to(instance._device)
        state = torch.load(
            path / "model.pt", map_location="cpu", weights_only=True
        )
        instance._net.load_state_dict(state)
        return instance
