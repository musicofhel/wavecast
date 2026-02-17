"""Batch inference for WaveletGPT models."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from numpy.typing import NDArray

from wavecast.core.exceptions import ModelNotTrainedError


class BatchPredictor:
    """Batch-mode predictor for WaveletGPT models.

    Splits large input arrays into batches for memory-efficient inference.
    Uses torch.inference_mode() for slightly better performance than no_grad().

    Args:
        model: A trained WaveletGPT instance
        batch_size: Number of samples per batch (default 256)
        use_amp: Whether to use automatic mixed precision for inference
    """

    def __init__(self, model, batch_size: int = 256, use_amp: bool = False) -> None:
        from wavecast.models.wavelet_gpt import WaveletGPT

        if not isinstance(model, WaveletGPT):
            raise TypeError("model must be a WaveletGPT instance")
        self._model = model
        self._batch_size = batch_size
        self._use_amp = use_amp

    def predict_stream(self, X: NDArray, horizon: int = 1) -> Iterator[NDArray]:
        """Yield prediction chunks for each batch.

        Args:
            X: Input array (n_samples, context_length + 2)
            horizon: Prediction horizon

        Yields:
            NDArray of predicted token IDs for each batch
        """
        if self._model._net is None:
            raise ModelNotTrainedError("Model has not been trained")
        for start in range(0, len(X), self._batch_size):
            batch = X[start : start + self._batch_size]
            yield self._predict_batch(batch, horizon, return_proba=False)

    def predict_proba_stream(
        self, X: NDArray, horizon: int = 1
    ) -> Iterator[NDArray]:
        """Yield probability chunks for each batch.

        Args:
            X: Input array (n_samples, context_length + 2)
            horizon: Prediction horizon

        Yields:
            NDArray of softmax probabilities (batch_size, vocab_size) for each batch
        """
        if self._model._net is None:
            raise ModelNotTrainedError("Model has not been trained")
        for start in range(0, len(X), self._batch_size):
            batch = X[start : start + self._batch_size]
            yield self._predict_batch(batch, horizon, return_proba=True)

    def predict_all(self, X: NDArray, horizon: int = 1) -> NDArray:
        """Predict all samples, returning concatenated results."""
        return np.concatenate(list(self.predict_stream(X, horizon)))

    def predict_proba_all(self, X: NDArray, horizon: int = 1) -> NDArray:
        """Predict probabilities for all samples, returning concatenated results."""
        return np.concatenate(list(self.predict_proba_stream(X, horizon)))

    def _predict_batch(
        self, X: NDArray, horizon: int, return_proba: bool
    ) -> NDArray:
        """Run inference on a single batch."""
        net = self._model._net
        device = self._model._device

        ctx, lvl, ac, aux = self._model._parse_x(X)
        ctx_t = torch.tensor(ctx, dtype=torch.long, device=device)
        lvl_t = torch.tensor(lvl, dtype=torch.long, device=device)
        ac_t = torch.tensor(ac, dtype=torch.long, device=device)
        aux_t = None
        if aux is not None:
            aux_t = torch.tensor(aux, dtype=torch.float32, device=device)

        with torch.inference_mode():
            amp_enabled = self._use_amp and device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits_dict = net(ctx_t, lvl_t, ac_t, aux_features=aux_t)

        logits = logits_dict[horizon]
        if return_proba:
            return torch.softmax(logits, dim=-1).cpu().numpy()
        return logits.argmax(dim=-1).cpu().numpy()
