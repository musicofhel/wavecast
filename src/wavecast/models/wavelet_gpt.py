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


class HeteroscedasticLoss(nn.Module):
    """Faithful heteroscedastic loss (Stirn et al. AISTATS 2023).

    L = exp(-log_var) * CE_per_sample + log_var
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        ce_per_sample = nn.functional.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        log_var_sq = log_var.squeeze(-1)
        loss = torch.exp(-log_var_sq) * ce_per_sample + log_var_sq
        return loss.mean()


class ErrorRegularizedHeteroscedasticLoss(nn.Module):
    """Heteroscedastic loss + error regularization penalty.

    L = heteroscedastic_loss + lambda * |max_softmax - is_correct|
    """

    def __init__(
        self, lam: float = 0.1, label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.lam = lam
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        ce_per_sample = nn.functional.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        log_var_sq = log_var.squeeze(-1)
        hetero = (torch.exp(-log_var_sq) * ce_per_sample + log_var_sq).mean()

        # Error regularization: penalize confident-and-wrong
        probs = torch.softmax(logits, dim=-1)
        max_conf = probs.max(dim=-1).values
        is_correct = (logits.argmax(dim=-1) == targets).float()
        penalty = torch.abs(max_conf - is_correct).mean()

        return hetero + self.lam * penalty


class SelectiveNetLoss(nn.Module):
    """SelectiveNet loss (Geifman & El-Yaniv, ICML 2019).

    L = CE / coverage + lambda * max(0, target_coverage - coverage)^2
    """

    def __init__(self, target_coverage: float = 0.7, lam: float = 32.0) -> None:
        super().__init__()
        self.target_coverage = target_coverage
        self.lam = lam

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reject_scores: torch.Tensor,
    ) -> torch.Tensor:
        coverage = reject_scores.mean()
        coverage = torch.clamp(coverage, min=0.01)

        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        selective_risk = (ce * reject_scores).sum() / (coverage * len(targets))

        penalty = self.lam * torch.clamp(
            self.target_coverage - coverage, min=0.0,
        ) ** 2

        return selective_risk + penalty


class HeteroscedasticSelectiveNetLoss(nn.Module):
    """Combined heteroscedastic + SelectiveNet loss.

    The heteroscedastic component trains the classifier + variance head.
    The SelectiveNet component trains the rejection head.
    """

    def __init__(
        self, target_coverage: float = 0.7, lam: float = 32.0,
    ) -> None:
        super().__init__()
        self.target_coverage = target_coverage
        self.lam = lam

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        log_var: torch.Tensor,
        reject_scores: torch.Tensor,
    ) -> torch.Tensor:
        # Heteroscedastic component
        ce_per_sample = nn.functional.cross_entropy(
            logits, targets, reduction="none",
        )
        log_var_sq = log_var.squeeze(-1)
        hetero = (torch.exp(-log_var_sq) * ce_per_sample + log_var_sq).mean()

        # SelectiveNet component (uses same CE, weighted by rejection scores)
        coverage = reject_scores.mean()
        coverage = torch.clamp(coverage, min=0.01)
        selective_risk = (ce_per_sample * reject_scores).sum() / (
            coverage * len(targets)
        )
        penalty = self.lam * torch.clamp(
            self.target_coverage - coverage, min=0.0,
        ) ** 2

        return hetero + selective_risk + penalty


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
        variance_head: bool = False,
        reject_head: bool = False,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.prediction_horizons = prediction_horizons or [1]
        self.task = task
        self.input_mode = input_mode
        self.n_aux_features = n_aux_features
        self.variance_head = variance_head
        self.reject_head = reject_head

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

        if variance_head:
            self.log_var_head = nn.Linear(embed_dim, 1)
            # Stirn init: start uncertain (high variance)
            nn.init.constant_(self.log_var_head.bias, 2.0)

        if reject_head:
            self.rejection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid(),
            )

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

        logits_dict: dict[int | str, torch.Tensor] = {}
        for h in self.prediction_horizons:
            logits_dict[h] = self.heads[str(h)](last_hidden)  # (batch, output_dim)

        if self.variance_head:
            logits_dict["log_var"] = self.log_var_head(last_hidden)

        if self.reject_head:
            logits_dict["reject"] = self.rejection(last_hidden).squeeze(-1)

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
        self._loss_type = loss_type
        self._loss_kwargs = loss_kwargs or {}
        self._n_output_classes = n_output_classes
        self._class_weights = class_weights
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
        if self._task == TASK_RETURN_REGRESSION:
            return nn.MSELoss()

        # Custom loss types
        if self._loss_type == "heteroscedastic":
            ls = self._loss_kwargs.get("label_smoothing", 0.0)
            return HeteroscedasticLoss(label_smoothing=ls)
        if self._loss_type == "heteroscedastic_error_reg":
            ls = self._loss_kwargs.get("label_smoothing", 0.0)
            lam = self._loss_kwargs.get("lam", 0.1)
            return ErrorRegularizedHeteroscedasticLoss(lam=lam, label_smoothing=ls)
        if self._loss_type == "selectivenet":
            tc = self._loss_kwargs.get("target_coverage", 0.7)
            lam = self._loss_kwargs.get("lam", 32.0)
            return SelectiveNetLoss(target_coverage=tc, lam=lam)
        if self._loss_type == "heteroscedastic_selectivenet":
            tc = self._loss_kwargs.get("target_coverage", 0.7)
            lam = self._loss_kwargs.get("lam", 32.0)
            return HeteroscedasticSelectiveNetLoss(target_coverage=tc, lam=lam)

        # Standard CE
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

        # Determine heads needed from loss type
        use_variance = self._loss_type in (
            "heteroscedastic", "heteroscedastic_error_reg",
            "heteroscedastic_selectivenet",
        )
        use_reject = self._loss_type in (
            "selectivenet", "heteroscedastic_selectivenet",
        )

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
            variance_head=use_variance,
            reject_head=use_reject,
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

                    log_var = logits_dict.get("log_var")
                    reject_scores = logits_dict.get("reject")

                    loss = torch.tensor(0.0, device=self._device)
                    for i, h in enumerate(self._prediction_horizons):
                        output = logits_dict[h]
                        target = target_bs[i]
                        if is_regression:
                            h_loss = criterion(output.squeeze(-1), target)
                        elif isinstance(
                            criterion, HeteroscedasticSelectiveNetLoss,
                        ):
                            h_loss = criterion(
                                output, target, log_var, reject_scores,
                            )
                        elif isinstance(
                            criterion,
                            (HeteroscedasticLoss,
                             ErrorRegularizedHeteroscedasticLoss),
                        ):
                            h_loss = criterion(output, target, log_var)
                        elif isinstance(criterion, SelectiveNetLoss):
                            h_loss = criterion(output, target, reject_scores)
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

    def predict_variance(self, X: NDArray) -> NDArray:
        """Return per-sample variance from the variance head (exp(log_var))."""
        if self._net is None:
            raise ModelNotTrainedError("Model has not been trained")
        if not getattr(self._net, "variance_head", False):
            raise ValueError("Model does not have a variance head")
        amp_enabled = self._use_amp and self._device.type == "cuda"
        ctx_dtype = (
            torch.float32 if self._input_mode == INPUT_CONTINUOUS else torch.long
        )
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
        log_var = logits_dict["log_var"].squeeze(-1)
        return torch.exp(log_var).cpu().numpy()

    def predict_rejection(self, X: NDArray) -> NDArray:
        """Return per-sample rejection scores from the rejection head."""
        if self._net is None:
            raise ModelNotTrainedError("Model has not been trained")
        if not getattr(self._net, "reject_head", False):
            raise ValueError("Model does not have a rejection head")
        amp_enabled = self._use_amp and self._device.type == "cuda"
        ctx_dtype = (
            torch.float32 if self._input_mode == INPUT_CONTINUOUS else torch.long
        )
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
        return logits_dict["reject"].cpu().numpy()

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
        loss_type = config.get("loss_type", "ce")
        variance_head = loss_type in (
            "heteroscedastic", "heteroscedastic_error_reg",
            "heteroscedastic_selectivenet",
        )
        reject_head_flag = loss_type in (
            "selectivenet", "heteroscedastic_selectivenet",
        )
        instance._net = WaveletGPTNet(
            **net_params,
            prediction_horizons=prediction_horizons,
            task=task,
            n_output_classes=n_output_classes,
            input_mode=input_mode,
            n_aux_features=n_aux_features,
            variance_head=variance_head,
            reject_head=reject_head_flag,
        ).to(instance._device)
        state = torch.load(
            path / "model.pt", map_location="cpu", weights_only=True
        )
        instance._net.load_state_dict(state)
        return instance
