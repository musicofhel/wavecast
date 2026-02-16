"""Token prediction pipeline: end-to-end SAX tokenization, training, and evaluation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wavecast.core.config import WaveCastConfig
from wavecast.core.types import AssetClass, MultiLevelTokenSequence
from wavecast.core.universe import DEFAULT_UNIVERSE, Universe
from wavecast.evaluation.token_eval import TokenPredictionMetrics, evaluate_token_predictions
from wavecast.tokenizer.vocabulary import SAXVocabulary

logger = logging.getLogger(__name__)


@dataclass
class TokenPipelineResult:
    """Result of running the full token prediction pipeline."""
    metrics: TokenPredictionMetrics
    vocabulary: SAXVocabulary
    model_path: Path | None = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0


class TokenPipelineRunner:
    """Orchestrates the full token prediction pipeline.

    Steps:
    1. Load or build shapelet library
    2. Fetch and decompose all assets in universe
    3. SAX transform all decompositions
    4. Build vocabulary from corpus
    5. Tokenize all sequences
    6. Build sequence dataset
    7. Split 70/15/15 train/val/test
    8. Train WaveletGPT
    9. Evaluate on test set
    10. Return metrics + vocab + model path
    """

    def run(
        self,
        library_path: Path | None = None,
        universe: Universe | None = None,
        config: WaveCastConfig | None = None,
    ) -> TokenPipelineResult:
        """Run the full token prediction pipeline."""
        from wavecast.models.wavelet_gpt import WaveletGPT
        from wavecast.pipeline.stages import stage_data, stage_decompose
        from wavecast.sax.bow import extract_words
        from wavecast.sax.sax import sax_transform
        from wavecast.tokenizer.dataset import build_sequence_dataset
        from wavecast.tokenizer.tokenizer import WaveletSAXTokenizer

        start_time = time.monotonic()
        universe = universe or DEFAULT_UNIVERSE
        cfg = config or WaveCastConfig()
        cfg.ensure_dirs()

        # Build asset class mapping
        asset_class_ids: dict[str, int] = {}
        class_to_id = {cls: i for i, cls in enumerate(AssetClass)}
        for asset in universe.assets:
            asset_class_ids[asset.ticker] = class_to_id.get(asset.asset_class, 0)

        # Step 1-2: Fetch and decompose all assets
        logger.info(f"Processing {len(universe)} assets...")
        decompositions = []
        all_word_sequences: list[list[str]] = []

        for asset in universe.assets:
            try:
                data_result = stage_data(asset.ticker, config=cfg)
                ts = data_result.data

                if ts.length < 100:
                    logger.warning(f"Skipping {asset.ticker}: too few data points")
                    continue

                decomp_result = stage_decompose(ts, config=cfg)
                decomp = decomp_result.data
                decomp = type(decomp)(
                    coefficients=decomp.coefficients,
                    wavelet=decomp.wavelet,
                    level=decomp.level,
                    original_length=decomp.original_length,
                    ticker=asset.ticker,
                )
                decompositions.append(decomp)

                # Step 3: SAX transform to collect words for vocabulary
                for lvl in range(1, decomp.level + 1):
                    coeffs = decomp.detail_at_level(lvl)
                    if len(coeffs) < 2:
                        continue
                    n_seg = min(cfg.sax.n_segments, len(coeffs))
                    sax_rep = sax_transform(coeffs, n_seg, cfg.sax.alphabet_size)
                    words = extract_words(
                        sax_rep.symbols, cfg.sax.word_length, cfg.sax.word_stride
                    )
                    all_word_sequences.append(words)

            except Exception as e:
                logger.warning(f"Failed to process {asset.ticker}: {e}")
                continue

        if not decompositions:
            logger.error("No assets successfully processed")
            return TokenPipelineResult(
                metrics=TokenPredictionMetrics(
                    token_accuracy=0.0,
                    top3_accuracy=0.0,
                    directional_accuracy=0.0,
                    confusion_matrix=np.zeros((1, 1)),
                ),
                vocabulary=SAXVocabulary(),
            )

        # Step 4: Build vocabulary
        logger.info("Building vocabulary...")
        vocab = SAXVocabulary.from_corpus(
            all_word_sequences,
            min_freq=cfg.tokenizer.min_word_freq,
            max_size=cfg.tokenizer.max_vocab_size,
        )
        logger.info(f"Vocabulary size: {vocab.size}")

        # Step 5: Tokenize all decompositions
        logger.info("Tokenizing decompositions...")
        tokenizer = WaveletSAXTokenizer(vocab, cfg.sax)
        token_sequences: list[MultiLevelTokenSequence] = []
        for decomp in decompositions:
            token_seq = tokenizer.tokenize(decomp)
            token_sequences.append(token_seq)

        # Step 6: Build sequence dataset
        logger.info("Building sequence dataset...")
        dataset = build_sequence_dataset(
            token_sequences,
            vocab,
            context_length=cfg.tokenizer.context_length,
            asset_class_map=asset_class_ids,
        )
        contexts, targets, levels, asset_classes = dataset.to_arrays()

        if len(targets) == 0:
            logger.error("No training samples generated")
            return TokenPipelineResult(
                metrics=TokenPredictionMetrics(
                    token_accuracy=0.0,
                    top3_accuracy=0.0,
                    directional_accuracy=0.0,
                    confusion_matrix=np.zeros((vocab.size, vocab.size)),
                ),
                vocabulary=vocab,
            )

        # Step 7: Split 70/15/15
        n = len(targets)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        # Build X: [context_tokens..., level_id, asset_class_id]
        X = np.column_stack([contexts, levels.reshape(-1, 1), asset_classes.reshape(-1, 1)])
        y = targets

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

        logger.info(f"Dataset split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

        # Step 8: Train WaveletGPT
        logger.info("Training WaveletGPT...")
        model = WaveletGPT(
            vocab_size=vocab.size,
            context_length=cfg.tokenizer.context_length,
            embed_dim=cfg.sequence_model.embed_dim,
            num_heads=cfg.sequence_model.num_heads,
            num_layers=cfg.sequence_model.num_layers,
            dropout=cfg.sequence_model.dropout,
            epochs=cfg.sequence_model.epochs,
            batch_size=cfg.sequence_model.batch_size,
            learning_rate=cfg.sequence_model.lr,
            patience=cfg.sequence_model.patience,
        )

        train_metrics = model.fit(X_train, y_train, X_val, y_val)
        logger.info(f"Training complete: {train_metrics}")

        # Save model
        model_path = cfg.model_dir / "wavelet_gpt"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Save vocabulary
        vocab_path = cfg.model_dir / "vocabulary.json"
        vocab.save(vocab_path)

        # Step 9: Evaluate on test set
        logger.info("Evaluating on test set...")
        predicted = model.predict(X_test)
        proba = model.predict_proba(X_test)

        metrics = evaluate_token_predictions(
            predicted=predicted,
            actual=y_test,
            vocab_size=vocab.size,
            proba=proba,
        )

        duration = time.monotonic() - start_time
        logger.info(
            f"Pipeline complete in {duration:.1f}s -- "
            f"accuracy={metrics.token_accuracy:.3f}, "
            f"top3={metrics.top3_accuracy:.3f}, "
            f"directional={metrics.directional_accuracy:.3f}"
        )

        return TokenPipelineResult(
            metrics=metrics,
            vocabulary=vocab,
            model_path=model_path,
            train_metrics=train_metrics,
            duration_seconds=duration,
        )
