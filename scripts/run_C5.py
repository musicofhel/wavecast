"""C5: Context length sweep.

Research Q5: How much history matters?

Fix optimal from C1-C4. Sweep context_length in {8, 16, 32, 64}.

Note: context=64 reduces training samples (189 vs 237 at ctx=16).

Save to ~/.wavecast/experiments/C5_context.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("C5_context")

EXPERIMENTS_DIR = Path.home() / ".wavecast" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
C5_OUTPUT = EXPERIMENTS_DIR / "C5_context.json"


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
    """Load optimal config from C1-C4 results."""
    raw = {}
    for name in ["C1_granularity", "C2_levels", "C3_transfer", "C4_vocab"]:
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
        for key in ["interval", "word_length", "word_stride", "epochs", "batch_size",
                     "learning_rate", "patience"]:
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

    logger.info(f"Optimal config from C1-C4: {config}")
    return config


def run_single_experiment(
    context_length: int,
    optimal_config: dict,
) -> dict:
    """Run a single C5 experiment with given context length."""
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
    from wavecast.tokenizer.vocabulary import SAXVocabulary

    cfg = WaveCastConfig()
    cfg.sax.alphabet_size = optimal_config["alphabet_size"]
    cfg.sax.n_segments = optimal_config["n_segments"]
    cfg.sax.word_length = optimal_config.get("word_length", 4)
    cfg.sax.word_stride = optimal_config.get("word_stride", 1)
    cfg.wavelet.level = optimal_config["wavelet_level"]
    cfg.tokenizer.max_vocab_size = optimal_config["max_vocab_size"]
    cfg.tokenizer.min_word_freq = optimal_config["min_word_freq"]
    cfg.tokenizer.context_length = context_length
    cfg.sequence_model.context_length = context_length
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

    vocab = SAXVocabulary.from_corpus(
        all_word_sequences,
        min_freq=cfg.tokenizer.min_word_freq,
        max_size=cfg.tokenizer.max_vocab_size,
    )

    tokenizer = WaveletSAXTokenizer(vocab, cfg.sax)
    token_sequences: list[MultiLevelTokenSequence] = []
    for decomp in decompositions:
        token_sequences.append(tokenizer.tokenize(decomp))

    dataset = build_sequence_dataset(
        token_sequences, vocab,
        context_length=context_length,
        asset_class_map=asset_class_ids,
    )
    contexts, targets, levels, asset_classes = dataset.to_arrays()

    if len(targets) == 0:
        return {"error": "No samples generated"}

    n = len(targets)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    X = np.column_stack([contexts, levels.reshape(-1, 1), asset_classes.reshape(-1, 1)])
    y = targets

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    model = WaveletGPT(
        vocab_size=vocab.size,
        context_length=context_length,
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

    predicted = model.predict(X_test)
    proba = model.predict_proba(X_test)
    metrics = evaluate_token_predictions(
        predicted=predicted, actual=y_test, vocab_size=vocab.size, proba=proba,
    )

    return {
        "context_length": context_length,
        "vocab_size": vocab.size,
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
    logger.info("=== C5: Context Length Sweep ===")
    start_time = time.monotonic()

    optimal_config = load_optimal_config()

    context_lengths = [8, 16, 32, 64]
    results = []

    for idx, ctx in enumerate(context_lengths, 1):
        logger.info(f"\n--- Experiment {idx}/{len(context_lengths)}: context_length={ctx} ---")
        t0 = time.monotonic()
        result = run_single_experiment(ctx, optimal_config)
        elapsed = time.monotonic() - t0
        result["duration_seconds"] = elapsed
        results.append(result)
        if "error" not in result:
            logger.info(
                f"  samples={result['n_samples']}, "
                f"acc={result['token_accuracy']:.3f}, "
                f"top3={result['top3_accuracy']:.3f}, "
                f"dir={result['directional_accuracy']:.3f} "
                f"({elapsed:.1f}s)"
            )

    valid = [r for r in results if "error" not in r]
    best = max(valid, key=lambda r: r["token_accuracy"]) if valid else {}

    total_duration = time.monotonic() - start_time
    output = {
        "experiment": "C5_context_length",
        "optimal_from_c1_c4": optimal_config,
        "sweep": {"context_lengths": context_lengths},
        "results": results,
        "best": best,
        "total_experiments": len(context_lengths),
        "successful_experiments": len(valid),
        "total_duration_seconds": total_duration,
    }

    C5_OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nResults saved to {C5_OUTPUT}")
    if best:
        logger.info(
            f"Best: context_length={best.get('context_length')}, "
            f"accuracy={best.get('token_accuracy', 0):.3f}"
        )
    logger.info(f"Total time: {total_duration:.1f}s")


if __name__ == "__main__":
    main()
