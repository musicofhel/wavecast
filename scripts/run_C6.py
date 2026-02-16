"""C6: Regime dependence analysis.

Research Q6: Does accuracy vary by market regime?

Train one model with full optimal config. Evaluate separately on:
- Trending periods (H > 0.6)
- Mean-reverting (H < 0.4)
- Random walk (0.4 <= H <= 0.6)

Report per-regime accuracy +/- CI with sample counts.
Compare against persistence baseline.

Save to ~/.wavecast/experiments/C6_regime.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("C6_regime")

EXPERIMENTS_DIR = Path.home() / ".wavecast" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
C6_OUTPUT = EXPERIMENTS_DIR / "C6_regime.json"


def _find_best_in_results(data: dict | list) -> dict:
    """Extract the best experiment result from varying JSON formats."""
    if isinstance(data, list):
        valid = [r for r in data if isinstance(r, dict) and "token_accuracy" in r]
        if valid:
            return max(valid, key=lambda r: r["token_accuracy"])
        return {}
    if isinstance(data, dict):
        if "best" in data and isinstance(data["best"], dict):
            return data["best"]
        if "results" in data and isinstance(data["results"], list):
            valid = [r for r in data["results"] if isinstance(r, dict) and "token_accuracy" in r]
            if valid:
                return max(valid, key=lambda r: r["token_accuracy"])
    return data if isinstance(data, dict) else {}


def load_optimal_config() -> dict:
    """Load optimal config from C1-C5 results."""
    raw = {}
    for name in ["C1_granularity", "C2_levels", "C3_transfer", "C4_vocab", "C5_context"]:
        path = EXPERIMENTS_DIR / f"{name}.json"
        if path.exists():
            raw[name] = json.loads(path.read_text())
            logger.info(f"Loaded {name}")
        else:
            logger.warning(f"{name} not found, using defaults")

    config = {
        "alphabet_size": 8,
        "n_segments": 256,
        "wavelet_level": 5,
        "wavelet": "db4",
        "max_vocab_size": 500,
        "min_word_freq": 2,
        "context_length": 16,
        "interval": "1h",
        "word_length": 4,
        "word_stride": 1,
        "epochs": 80,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "patience": 15,
    }

    if "C1_granularity" in raw:
        best = _find_best_in_results(raw["C1_granularity"])
        cfg = best.get("config", best)
        if "alphabet_size" in cfg:
            config["alphabet_size"] = cfg["alphabet_size"]
        if "n_segments" in cfg:
            config["n_segments"] = cfg["n_segments"]
        for key in ["interval", "word_length", "word_stride", "context_length",
                     "epochs", "batch_size", "learning_rate", "patience"]:
            if key in cfg:
                config[key] = cfg[key]
        logger.info(f"  C1 best: alpha={config['alphabet_size']}, segments={config['n_segments']}")

    if "C2_levels" in raw:
        best = _find_best_in_results(raw["C2_levels"])
        cfg = best.get("config", best)
        if "dwt_levels" in cfg and cfg["dwt_levels"] is not None:
            config["wavelet_level"] = max(cfg["dwt_levels"]) if isinstance(cfg["dwt_levels"], list) else cfg["dwt_levels"]
        elif "wavelet_level" in cfg:
            config["wavelet_level"] = cfg["wavelet_level"]
        logger.info(f"  C2 best: wavelet_level={config['wavelet_level']}")

    if "C4_vocab" in raw:
        best = _find_best_in_results(raw["C4_vocab"])
        cfg = best.get("config", best)
        if "max_vocab_size" in cfg:
            config["max_vocab_size"] = cfg["max_vocab_size"]
        elif "max_vocab_size" in best:
            config["max_vocab_size"] = best["max_vocab_size"]
        if "min_word_freq" in cfg:
            config["min_word_freq"] = cfg["min_word_freq"]
        elif "min_word_freq" in best:
            config["min_word_freq"] = best["min_word_freq"]
        logger.info(f"  C4 best: vocab={config['max_vocab_size']}, min_freq={config['min_word_freq']}")

    if "C5_context" in raw:
        best = _find_best_in_results(raw["C5_context"])
        cfg = best.get("config", best)
        if "context_length" in cfg:
            config["context_length"] = cfg["context_length"]
        elif "context_length" in best:
            config["context_length"] = best["context_length"]
        logger.info(f"  C5 best: context_length={config['context_length']}")

    logger.info(f"Optimal config from C1-C5: {config}")
    return config


def bootstrap_ci(
    correct: NDArray, n_boot: int = 1000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Compute mean accuracy and bootstrap confidence interval.

    Args:
        correct: Boolean array of correct predictions.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level for CI (default 95% CI).

    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    if len(correct) == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(correct))
    if len(correct) < 5:
        return mean, mean, mean

    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(correct, size=len(correct), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, ci_lower, ci_upper


def compute_persistence_baseline(targets: NDArray, contexts: NDArray) -> float:
    """Compute persistence baseline: predict last context token as next token.

    Returns accuracy of the persistence (naive) strategy.
    """
    last_tokens = contexts[:, -1]
    return float(np.mean(last_tokens == targets))


def main() -> None:
    from wavecast.core.config import WaveCastConfig
    from wavecast.core.types import AssetClass, MultiLevelTokenSequence, RegimeType
    from wavecast.core.universe import PHASE3_UNIVERSE
    from wavecast.evaluation.token_eval import evaluate_token_predictions
    from wavecast.fractal.hurst import classify_regime, rolling_hurst_with_regimes
    from wavecast.models.wavelet_gpt import WaveletGPT
    from wavecast.pipeline.stages import stage_data, stage_decompose
    from wavecast.sax.bow import extract_words
    from wavecast.sax.sax import sax_transform
    from wavecast.tokenizer.dataset import build_sequence_dataset
    from wavecast.tokenizer.tokenizer import WaveletSAXTokenizer
    from wavecast.tokenizer.vocabulary import SAXVocabulary

    logger.info("=== C6: Regime Dependence Analysis ===")
    start_time = time.monotonic()

    optimal_config = load_optimal_config()

    cfg = WaveCastConfig()
    cfg.sax.alphabet_size = optimal_config["alphabet_size"]
    cfg.sax.n_segments = optimal_config["n_segments"]
    cfg.sax.word_length = optimal_config.get("word_length", 4)
    cfg.sax.word_stride = optimal_config.get("word_stride", 1)
    cfg.wavelet.level = optimal_config["wavelet_level"]
    cfg.tokenizer.max_vocab_size = optimal_config["max_vocab_size"]
    cfg.tokenizer.min_word_freq = optimal_config["min_word_freq"]
    cfg.tokenizer.context_length = optimal_config["context_length"]
    cfg.sequence_model.context_length = optimal_config["context_length"]
    cfg.sequence_model.epochs = optimal_config.get("epochs", 80)
    cfg.sequence_model.batch_size = optimal_config.get("batch_size", 64)
    cfg.sequence_model.lr = optimal_config.get("learning_rate", 0.0005)
    cfg.sequence_model.patience = optimal_config.get("patience", 15)
    cfg.ensure_dirs()
    interval = optimal_config.get("interval", "1h")

    universe = PHASE3_UNIVERSE
    class_to_id = {cls: i for i, cls in enumerate(AssetClass)}
    asset_class_ids = {
        a.ticker: class_to_id.get(a.asset_class, 0) for a in universe.assets
    }

    # ============================================================
    # Step 1: Compute regime labels for each ticker using LOG RETURNS
    # ============================================================
    logger.info("Computing Hurst regimes on log returns...")
    ticker_regimes: dict[str, dict] = {}  # ticker -> {hurst_vals, indices, regimes}
    ticker_prices: dict[str, NDArray] = {}

    for asset in universe.assets:
        try:
            data_result = stage_data(asset.ticker, interval=interval, config=cfg)
            ts = data_result.data
            if ts.length < 300:
                logger.warning(f"Skipping {asset.ticker}: too few points ({ts.length})")
                continue
            ticker_prices[asset.ticker] = ts.values

            # Use a daily window of 252 for daily data
            hurst_window = min(252, ts.length - 1)
            hurst_vals, indices, regimes = rolling_hurst_with_regimes(
                ts.values,
                window=hurst_window,
                step=1,
                wavelet="db4",
                trending_threshold=0.6,
                mean_revert_threshold=0.4,
                use_returns=True,  # CRITICAL: compute on log returns
            )
            ticker_regimes[asset.ticker] = {
                "hurst_vals": hurst_vals,
                "indices": indices,
                "regimes": regimes,
            }
            # Log regime distribution
            n_trend = sum(1 for r in regimes if r == RegimeType.TRENDING)
            n_mr = sum(1 for r in regimes if r == RegimeType.MEAN_REVERTING)
            n_rw = sum(1 for r in regimes if r == RegimeType.RANDOM_WALK)
            logger.info(
                f"  {asset.ticker}: H_mean={np.nanmean(hurst_vals):.3f}, "
                f"trending={n_trend}, mean_revert={n_mr}, random_walk={n_rw}"
            )
        except Exception as e:
            logger.warning(f"Hurst failed for {asset.ticker}: {e}")
            continue

    # ============================================================
    # Step 2: Build full model with optimal config
    # ============================================================
    logger.info("\nBuilding model with optimal config...")
    decompositions = []
    all_word_sequences: list[list[str]] = []
    decomp_tickers: list[str] = []

    for asset in universe.assets:
        try:
            data_result = stage_data(asset.ticker, interval=interval, config=cfg)
            ts = data_result.data
            if ts.length < 100:
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
            decomp_tickers.append(asset.ticker)

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
            logger.warning(f"Failed {asset.ticker}: {e}")
            continue

    if not decompositions:
        logger.error("No assets processed")
        return

    # Build vocabulary
    vocab = SAXVocabulary.from_corpus(
        all_word_sequences,
        min_freq=cfg.tokenizer.min_word_freq,
        max_size=cfg.tokenizer.max_vocab_size,
    )
    logger.info(f"Vocabulary size: {vocab.size}")

    # Tokenize
    tokenizer = WaveletSAXTokenizer(vocab, cfg.sax)
    token_sequences: list[MultiLevelTokenSequence] = []
    for decomp in decompositions:
        token_sequences.append(tokenizer.tokenize(decomp))

    # Build dataset — we need to track which samples came from which ticker
    # Rebuild manually to track sample-level regime labels
    from wavecast.tokenizer.dataset import SequenceDataset, SequenceSample
    from wavecast.tokenizer.vocabulary import PAD_ID

    ctx_len = optimal_config["context_length"]
    all_samples: list[SequenceSample] = []
    sample_regime_labels: list[RegimeType] = []  # regime for each sample

    for mlt in token_sequences:
        ticker = mlt.ticker
        ac_id = asset_class_ids.get(ticker, 0)

        # Get regime info for this ticker
        regime_info = ticker_regimes.get(ticker)

        for level, seq in mlt.level_sequences.items():
            tokens = seq.token_ids
            if len(tokens) == 0:
                continue

            if len(tokens) <= ctx_len:
                padded = [PAD_ID] * (ctx_len + 1 - len(tokens)) + tokens
                tokens = padded

            for i in range(len(tokens) - ctx_len):
                context = tokens[i : i + ctx_len]
                target = tokens[i + ctx_len]
                sample = SequenceSample(
                    context_tokens=context,
                    target_token=target,
                    level=level,
                    asset_class_id=ac_id,
                )
                all_samples.append(sample)

                # Assign regime label for this sample
                # The sample index in token space maps approximately to a position
                # in the original price series. Use a rough mapping.
                if regime_info is not None:
                    # Map token index to approximate time index
                    # Token sequences are shorter than price series, use ratio
                    price_len = len(ticker_prices.get(ticker, []))
                    if price_len > 0:
                        token_pos = i + ctx_len
                        token_total = len(tokens)
                        approx_price_idx = int(token_pos / token_total * price_len)
                        approx_price_idx = min(approx_price_idx, price_len - 1)

                        # Find closest regime index
                        indices = regime_info["indices"]
                        if len(indices) > 0:
                            closest_idx = np.argmin(np.abs(indices - approx_price_idx))
                            regime = regime_info["regimes"][closest_idx]
                        else:
                            regime = RegimeType.RANDOM_WALK
                    else:
                        regime = RegimeType.RANDOM_WALK
                else:
                    regime = RegimeType.RANDOM_WALK

                sample_regime_labels.append(regime)

    if not all_samples:
        logger.error("No samples generated")
        return

    dataset = SequenceDataset(samples=all_samples)
    contexts, targets, levels, asset_classes = dataset.to_arrays()
    regime_arr = np.array([r.value for r in sample_regime_labels])

    n = len(targets)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    X = np.column_stack([contexts, levels.reshape(-1, 1), asset_classes.reshape(-1, 1)])
    y = targets

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    regime_test = regime_arr[n_train + n_val:]
    contexts_test = contexts[n_train + n_val:]

    logger.info(f"Dataset: {n} total, {n_train} train, {n_val} val, {len(y_test)} test")

    # Train model
    logger.info("Training WaveletGPT...")
    model = WaveletGPT(
        vocab_size=vocab.size,
        context_length=ctx_len,
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

    # Overall test metrics
    predicted = model.predict(X_test)
    proba = model.predict_proba(X_test)
    overall_metrics = evaluate_token_predictions(
        predicted=predicted, actual=y_test, vocab_size=vocab.size, proba=proba,
    )
    overall_persistence = compute_persistence_baseline(y_test, contexts_test)

    logger.info(
        f"\nOverall test: acc={overall_metrics.token_accuracy:.3f}, "
        f"top3={overall_metrics.top3_accuracy:.3f}, "
        f"dir={overall_metrics.directional_accuracy:.3f}, "
        f"persistence_baseline={overall_persistence:.3f}"
    )

    # ============================================================
    # Step 3: Evaluate per-regime
    # ============================================================
    logger.info("\n--- Per-regime evaluation ---")
    regime_results = {}

    for regime_name, regime_value in [
        ("trending", RegimeType.TRENDING.value),
        ("mean_reverting", RegimeType.MEAN_REVERTING.value),
        ("random_walk", RegimeType.RANDOM_WALK.value),
    ]:
        mask = regime_test == regime_value
        n_regime = int(mask.sum())

        if n_regime == 0:
            logger.warning(f"  {regime_name}: 0 samples, skipping")
            regime_results[regime_name] = {
                "n_samples": 0,
                "accuracy": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "persistence_baseline": 0.0,
            }
            continue

        pred_regime = predicted[mask]
        actual_regime = y_test[mask]
        ctx_regime = contexts_test[mask]
        proba_regime = proba[mask] if proba is not None else None

        metrics_regime = evaluate_token_predictions(
            predicted=pred_regime,
            actual=actual_regime,
            vocab_size=vocab.size,
            proba=proba_regime,
        )

        correct = (pred_regime == actual_regime).astype(np.float64)
        acc_mean, ci_lower, ci_upper = bootstrap_ci(correct)
        persistence = compute_persistence_baseline(actual_regime, ctx_regime)

        regime_results[regime_name] = {
            "n_samples": n_regime,
            "accuracy": metrics_regime.token_accuracy,
            "accuracy_ci_lower": ci_lower,
            "accuracy_ci_upper": ci_upper,
            "top3_accuracy": metrics_regime.top3_accuracy,
            "directional_accuracy": metrics_regime.directional_accuracy,
            "persistence_baseline": persistence,
            "lift_over_persistence": metrics_regime.token_accuracy - persistence,
        }

        logger.info(
            f"  {regime_name}: n={n_regime}, "
            f"acc={metrics_regime.token_accuracy:.3f} "
            f"[{ci_lower:.3f}, {ci_upper:.3f}], "
            f"top3={metrics_regime.top3_accuracy:.3f}, "
            f"dir={metrics_regime.directional_accuracy:.3f}, "
            f"persistence={persistence:.3f}, "
            f"lift={metrics_regime.token_accuracy - persistence:+.3f}"
        )

    # Hurst distribution summary
    all_hurst = []
    for info in ticker_regimes.values():
        all_hurst.extend(info["hurst_vals"][~np.isnan(info["hurst_vals"])].tolist())
    hurst_summary = {
        "mean": float(np.mean(all_hurst)) if all_hurst else 0.0,
        "std": float(np.std(all_hurst)) if all_hurst else 0.0,
        "median": float(np.median(all_hurst)) if all_hurst else 0.0,
        "min": float(np.min(all_hurst)) if all_hurst else 0.0,
        "max": float(np.max(all_hurst)) if all_hurst else 0.0,
    }

    total_duration = time.monotonic() - start_time

    output = {
        "experiment": "C6_regime_dependence",
        "optimal_config": optimal_config,
        "hurst_fix": "Computed on log returns (use_returns=True) instead of raw prices",
        "hurst_summary": hurst_summary,
        "overall": {
            "n_test": len(y_test),
            "token_accuracy": overall_metrics.token_accuracy,
            "top3_accuracy": overall_metrics.top3_accuracy,
            "directional_accuracy": overall_metrics.directional_accuracy,
            "persistence_baseline": overall_persistence,
        },
        "regime_results": regime_results,
        "regime_test_distribution": {
            "trending": int((regime_test == RegimeType.TRENDING.value).sum()),
            "mean_reverting": int((regime_test == RegimeType.MEAN_REVERTING.value).sum()),
            "random_walk": int((regime_test == RegimeType.RANDOM_WALK.value).sum()),
        },
        "train_metrics": train_metrics,
        "total_duration_seconds": total_duration,
    }

    C6_OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nResults saved to {C6_OUTPUT}")
    logger.info(f"Total time: {total_duration:.1f}s")


if __name__ == "__main__":
    main()
