"""C4: Vocabulary saturation sweep.

Research Q4: How many words do we need?

Fix optimal alphabet/levels/training strategy from C1-C3. Sweep:
- max_vocab_size in {50, 100, 200, 300, 500}
- min_word_freq in {1, 2, 3, 5}

20 experiments total. Save to ~/.wavecast/experiments/C4_vocab.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("C4_vocab")

# Paths
EXPERIMENTS_DIR = Path.home() / ".wavecast" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
C4_OUTPUT = EXPERIMENTS_DIR / "C4_vocab.json"


def _find_best_in_results(data: dict | list) -> dict:
    """Extract the best experiment result from varying JSON formats.

    Handles:
    - List of experiment dicts with 'token_accuracy' field (C1 format)
    - Dict with 'best' key (pipeline runner format)
    - Dict with 'results' key containing a list (C4/C5 format)
    """
    if isinstance(data, list):
        # Array of experiment results, pick highest token_accuracy
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
    """Load optimal config from C1-C3 results."""
    raw = {}
    for name in ["C1_granularity", "C2_levels", "C3_transfer"]:
        path = EXPERIMENTS_DIR / f"{name}.json"
        if path.exists():
            raw[name] = json.loads(path.read_text())
            logger.info(f"Loaded {name}")
        else:
            logger.warning(f"{name} not found, using defaults")

    # Defaults (match C1 experiment setup)
    config = {
        "alphabet_size": 8,
        "n_segments": 256,
        "wavelet_level": 5,
        "wavelet": "db4",
        "training_strategy": "all_sectors",
        "interval": "1h",
        "word_length": 4,
        "word_stride": 1,
        "context_length": 16,
        "epochs": 80,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "patience": 15,
    }

    # C1: optimal granularity (alphabet_size, n_segments, and training hyperparams)
    if "C1_granularity" in raw:
        best = _find_best_in_results(raw["C1_granularity"])
        cfg = best.get("config", best)
        if "alphabet_size" in cfg:
            config["alphabet_size"] = cfg["alphabet_size"]
        if "n_segments" in cfg:
            config["n_segments"] = cfg["n_segments"]
        # Also propagate training-time settings from C1
        for key in ["interval", "word_length", "word_stride", "context_length",
                     "epochs", "batch_size", "learning_rate", "patience"]:
            if key in cfg:
                config[key] = cfg[key]
        logger.info(f"  C1 best: alpha={config['alphabet_size']}, segments={config['n_segments']}, acc={best.get('token_accuracy', '?')}")

    # C2: optimal wavelet levels
    if "C2_levels" in raw:
        best = _find_best_in_results(raw["C2_levels"])
        cfg = best.get("config", best)
        if "dwt_levels" in cfg and cfg["dwt_levels"] is not None:
            config["wavelet_level"] = max(cfg["dwt_levels"]) if isinstance(cfg["dwt_levels"], list) else cfg["dwt_levels"]
        elif "wavelet_level" in cfg:
            config["wavelet_level"] = cfg["wavelet_level"]
        logger.info(f"  C2 best: wavelet_level={config['wavelet_level']}, acc={best.get('token_accuracy', '?')}")

    # C3: training strategy
    if "C3_transfer" in raw:
        best = _find_best_in_results(raw["C3_transfer"])
        cfg = best.get("config", best)
        if "cross_sector_training" in cfg:
            config["training_strategy"] = "cross_sector" if cfg["cross_sector_training"] else "per_sector"
        elif "strategy" in cfg:
            config["training_strategy"] = cfg["strategy"]
        logger.info(f"  C3 best: strategy={config['training_strategy']}, acc={best.get('token_accuracy', '?')}")

    logger.info(f"Optimal config from C1-C3: {config}")
    return config


def run_single_experiment(
    max_vocab_size: int,
    min_word_freq: int,
    optimal_config: dict,
) -> dict:
    """Run a single C4 experiment with given vocab settings."""
    from wavecast.core.config import WaveCastConfig
    from wavecast.core.types import AssetClass, MultiLevelTokenSequence
    from wavecast.core.universe import PHASE3_UNIVERSE
    from wavecast.evaluation.token_eval import evaluate_token_predictions
    from wavecast.models.wavelet_gpt import WaveletGPT
    from wavecast.pipeline.stages import stage_data, stage_decompose
    from wavecast.sax.bow import extract_words
    from wavecast.sax.sax import sax_transform
    from wavecast.tokenizer.dataset import build_sequence_dataset
    from wavecast.tokenizer.tokenizer import WaveletSAXTokenizer
    from wavecast.tokenizer.vocabulary import SAXVocabulary, UNK_ID

    cfg = WaveCastConfig()
    cfg.sax.alphabet_size = optimal_config["alphabet_size"]
    cfg.sax.n_segments = optimal_config["n_segments"]
    cfg.sax.word_length = optimal_config.get("word_length", 4)
    cfg.sax.word_stride = optimal_config.get("word_stride", 1)
    cfg.wavelet.level = optimal_config["wavelet_level"]
    cfg.tokenizer.max_vocab_size = max_vocab_size
    cfg.tokenizer.min_word_freq = min_word_freq
    cfg.tokenizer.context_length = optimal_config.get("context_length", 16)
    cfg.sequence_model.context_length = optimal_config.get("context_length", 16)
    cfg.sequence_model.epochs = optimal_config.get("epochs", 80)
    cfg.sequence_model.batch_size = optimal_config.get("batch_size", 64)
    cfg.sequence_model.lr = optimal_config.get("learning_rate", 0.0005)
    cfg.sequence_model.patience = optimal_config.get("patience", 15)
    cfg.ensure_dirs()

    interval = optimal_config.get("interval", "1h")
    universe = PHASE3_UNIVERSE

    # Build asset class mapping
    class_to_id = {cls: i for i, cls in enumerate(AssetClass)}
    asset_class_ids = {
        a.ticker: class_to_id.get(a.asset_class, 0) for a in universe.assets
    }

    # Process all assets
    decompositions = []
    all_word_sequences: list[list[str]] = []

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
        return {"error": "No assets processed"}

    # Build vocabulary
    vocab = SAXVocabulary.from_corpus(
        all_word_sequences, min_freq=min_word_freq, max_size=max_vocab_size
    )

    # Count UNK rate
    total_words = 0
    unk_words = 0
    for seq in all_word_sequences:
        for w in seq:
            total_words += 1
            if vocab.encode(w) == UNK_ID:
                unk_words += 1
    unk_rate = unk_words / total_words if total_words > 0 else 0.0

    # Tokenize
    tokenizer = WaveletSAXTokenizer(vocab, cfg.sax)
    token_sequences: list[MultiLevelTokenSequence] = []
    for decomp in decompositions:
        token_sequences.append(tokenizer.tokenize(decomp))

    # Build dataset
    dataset = build_sequence_dataset(
        token_sequences, vocab,
        context_length=cfg.tokenizer.context_length,
        asset_class_map=asset_class_ids,
    )
    contexts, targets, levels, asset_classes = dataset.to_arrays()

    if len(targets) == 0:
        return {"error": "No samples generated"}

    # Split 70/15/15
    n = len(targets)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    X = np.column_stack([contexts, levels.reshape(-1, 1), asset_classes.reshape(-1, 1)])
    y = targets

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    # Train
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

    # Evaluate
    predicted = model.predict(X_test)
    proba = model.predict_proba(X_test)
    metrics = evaluate_token_predictions(
        predicted=predicted, actual=y_test, vocab_size=vocab.size, proba=proba,
    )

    return {
        "max_vocab_size": max_vocab_size,
        "min_word_freq": min_word_freq,
        "actual_vocab_size": vocab.size,
        "unk_rate": unk_rate,
        "total_words_in_corpus": total_words,
        "unk_words_in_corpus": unk_words,
        "n_samples": n,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n - n_train - n_val,
        "token_accuracy": metrics.token_accuracy,
        "top3_accuracy": metrics.top3_accuracy,
        "directional_accuracy": metrics.directional_accuracy,
        "train_loss": train_metrics.get("train_loss", 0.0),
        "train_accuracy": train_metrics.get("train_accuracy", 0.0),
        "val_loss": train_metrics.get("val_loss", 0.0),
    }


def main() -> None:
    logger.info("=== C4: Vocabulary Saturation Sweep ===")
    start_time = time.monotonic()

    optimal_config = load_optimal_config()

    vocab_sizes = [50, 100, 200, 300, 500]
    min_freqs = [1, 2, 3, 5]

    results = []
    total = len(vocab_sizes) * len(min_freqs)
    idx = 0

    for vs in vocab_sizes:
        for mf in min_freqs:
            idx += 1
            logger.info(f"\n--- Experiment {idx}/{total}: vocab_size={vs}, min_freq={mf} ---")
            t0 = time.monotonic()
            result = run_single_experiment(vs, mf, optimal_config)
            elapsed = time.monotonic() - t0
            result["duration_seconds"] = elapsed
            results.append(result)
            if "error" not in result:
                logger.info(
                    f"  vocab={result['actual_vocab_size']}, "
                    f"unk_rate={result['unk_rate']:.3f}, "
                    f"acc={result['token_accuracy']:.3f}, "
                    f"top3={result['top3_accuracy']:.3f}, "
                    f"dir={result['directional_accuracy']:.3f} "
                    f"({elapsed:.1f}s)"
                )

    # Find best by token accuracy
    valid = [r for r in results if "error" not in r]
    best = max(valid, key=lambda r: r["token_accuracy"]) if valid else {}

    total_duration = time.monotonic() - start_time
    output = {
        "experiment": "C4_vocab_saturation",
        "optimal_from_c1_c3": optimal_config,
        "sweep": {"vocab_sizes": vocab_sizes, "min_freqs": min_freqs},
        "results": results,
        "best": best,
        "total_experiments": total,
        "successful_experiments": len(valid),
        "total_duration_seconds": total_duration,
    }

    C4_OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nResults saved to {C4_OUTPUT}")
    if best:
        logger.info(
            f"Best: vocab_size={best.get('max_vocab_size')}, "
            f"min_freq={best.get('min_word_freq')}, "
            f"accuracy={best.get('token_accuracy', 0):.3f}, "
            f"unk_rate={best.get('unk_rate', 0):.3f}"
        )
    logger.info(f"Total time: {total_duration:.1f}s")


if __name__ == "__main__":
    main()
