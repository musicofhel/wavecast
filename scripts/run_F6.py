"""F6: Per-sector fine-tuning experiment.

1. Train universal model (all 20 assets) with best F4+F5 params
2. For each of 6 sectors, fine-tune a copy on sector-only data (20 epochs, LR=0.0001)
3. Compare universal vs fine-tuned per-sector accuracy
4. Report per-sector accuracy delta
"""

from __future__ import annotations

import copy
import datetime
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wavecast.core.types import MultiLevelTokenSequence, TimeSeries, TokenSequence
from wavecast.core.universe import DEFAULT_UNIVERSE
from wavecast.data.cache import ParquetCache
from wavecast.experiments.metrics import level0_directional_accuracy
from wavecast.experiments.splitter import walk_forward_split
from wavecast.models.wavelet_gpt import WaveletGPT
from wavecast.sax.bow import extract_words
from wavecast.sax.sax import sax_transform
from wavecast.tokenizer.dataset import build_sequence_dataset
from wavecast.tokenizer.vocabulary import SAXVocabulary
from wavecast.wavelets.dwt import decompose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".wavecast" / "cache"
F4_PATH = Path.home() / ".wavecast" / "experiments" / "F4_sax_hpo.json"
F5_PATH = Path.home() / ".wavecast" / "experiments" / "F5_arch_hpo.json"
OUTPUT_PATH = Path.home() / ".wavecast" / "experiments" / "F6_sector_finetune.json"

SECTOR_TICKERS = {
    "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "finance": ["JPM", "GS", "BAC"],
    "energy": ["XOM", "CVX", "COP"],
    "healthcare": ["JNJ", "UNH", "PFE"],
    "broad_etf": ["SPY", "QQQ"],
    "commodity_etf": ["GLD", "SLV", "USO", "UNG"],
}

SECTOR_ID_MAP = {
    "tech": 0, "finance": 1, "energy": 2,
    "healthcare": 3, "broad_etf": 4, "commodity_etf": 5,
}

FINETUNE_EPOCHS = 20
FINETUNE_LR = 0.0001


def load_best_params() -> dict:
    """Load best params from F4 + F5, fall back to Phase 3 defaults."""
    params = {
        "n_segments": 256, "word_length": 4, "word_stride": 1,
        "embed_dim": 64, "num_heads": 4, "num_layers": 3, "dropout": 0.1,
    }

    if F4_PATH.exists():
        data = json.loads(F4_PATH.read_text())
        f4_params = data.get("best_trial", {}).get("params", {})
        if f4_params:
            params["n_segments"] = f4_params.get("n_segments", 256)
            params["word_length"] = f4_params.get("word_length", 4)
            params["word_stride"] = f4_params.get("word_stride", 1)
            logger.info("Loaded F4 SAX params: n_seg=%d, wl=%d, ws=%d",
                        params["n_segments"], params["word_length"], params["word_stride"])

    if F5_PATH.exists():
        data = json.loads(F5_PATH.read_text())
        f5_params = data.get("best_trial", {}).get("params", {})
        if f5_params:
            params["embed_dim"] = f5_params.get("embed_dim", 64)
            params["num_heads"] = f5_params.get("num_heads", 4)
            params["num_layers"] = f5_params.get("num_layers", 3)
            params["dropout"] = f5_params.get("dropout", 0.1)
            logger.info("Loaded F5 arch params: ed=%d, nh=%d, nl=%d, do=%.2f",
                        params["embed_dim"], params["num_heads"],
                        params["num_layers"], params["dropout"])

    return params


def build_pipeline_data(
    tickers: list[str],
    cache: ParquetCache,
    n_segments: int,
    word_length: int,
    word_stride: int,
) -> tuple[
    list[MultiLevelTokenSequence],
    list[MultiLevelTokenSequence],
    SAXVocabulary,
    dict[str, int],
    list[str],
]:
    """Run the full DWT -> SAX -> vocabulary -> encode pipeline.

    Returns (train_mlts, test_mlts, vocabulary, asset_class_map, common_tickers).
    """
    from wavecast.core.config import SAXConfig

    prices: dict[str, TimeSeries] = {}
    for ticker in tickers:
        ts = cache.get(ticker, "1h")
        if ts is not None:
            prices[ticker] = ts

    train_prices, test_prices = walk_forward_split(
        prices, "2023-12-31", "2024-01-01"
    )
    common_tickers = sorted(set(train_prices.keys()) & set(test_prices.keys()))

    sax_config = SAXConfig(
        n_segments=n_segments,
        alphabet_size=7,
        word_length=word_length,
        word_stride=word_stride,
    )
    levels_to_use = [1, 2, 5]

    train_words_all: list[list[str]] = []
    train_placeholder_mlts: list[MultiLevelTokenSequence] = []
    test_raw: list[tuple[str, dict[int, list[str]]]] = []

    for ticker in common_tickers:
        train_decomp = decompose(train_prices[ticker], level=5)
        test_decomp = decompose(test_prices[ticker], level=5)

        train_level_words: dict[int, list[str]] = {}
        test_level_words: dict[int, list[str]] = {}

        for lvl in levels_to_use:
            train_coeffs = train_decomp.detail_at_level(lvl)
            if len(train_coeffs) >= 2:
                n_seg = min(sax_config.n_segments, len(train_coeffs))
                train_sax = sax_transform(train_coeffs, n_seg, sax_config.alphabet_size)
                tw = extract_words(train_sax.symbols, sax_config.word_length, sax_config.word_stride)
                train_level_words[lvl] = tw
                train_words_all.append(tw)

            test_coeffs = test_decomp.detail_at_level(lvl)
            if len(test_coeffs) >= 2:
                n_seg = min(sax_config.n_segments, len(test_coeffs))
                test_sax = sax_transform(test_coeffs, n_seg, sax_config.alphabet_size)
                test_level_words[lvl] = extract_words(test_sax.symbols, sax_config.word_length, sax_config.word_stride)

        test_raw.append((ticker, test_level_words))

        level_sequences: dict[int, TokenSequence] = {}
        for lvl, words in train_level_words.items():
            level_sequences[lvl] = TokenSequence(
                token_ids=[], words=words, ticker=ticker,
                interval="1h", wavelet_level=lvl,
            )
        train_placeholder_mlts.append(
            MultiLevelTokenSequence(ticker=ticker, interval="1h", level_sequences=level_sequences)
        )

    vocabulary = SAXVocabulary.from_corpus(train_words_all, min_freq=1, max_size=100)

    # Encode train
    train_mlts: list[MultiLevelTokenSequence] = []
    for mlt in train_placeholder_mlts:
        level_sequences: dict[int, TokenSequence] = {}
        for lvl, seq in mlt.level_sequences.items():
            token_ids = vocabulary.encode_sequence(seq.words)
            level_sequences[lvl] = TokenSequence(
                token_ids=token_ids, words=seq.words, ticker=seq.ticker,
                interval=seq.interval, wavelet_level=seq.wavelet_level,
            )
        train_mlts.append(
            MultiLevelTokenSequence(ticker=mlt.ticker, interval=mlt.interval, level_sequences=level_sequences)
        )

    # Encode test
    test_mlts: list[MultiLevelTokenSequence] = []
    for ticker, level_words in test_raw:
        level_sequences = {}
        for lvl, words in level_words.items():
            token_ids = vocabulary.encode_sequence(words)
            level_sequences[lvl] = TokenSequence(
                token_ids=token_ids, words=words, ticker=ticker,
                interval="1h", wavelet_level=lvl,
            )
        test_mlts.append(
            MultiLevelTokenSequence(ticker=ticker, interval="1h", level_sequences=level_sequences)
        )

    # Build asset class map
    asset_class_map = {}
    for asset in DEFAULT_UNIVERSE.assets:
        if asset.sector is not None:
            asset_class_map[asset.ticker] = SECTOR_ID_MAP.get(asset.sector.value, 0)

    return train_mlts, test_mlts, vocabulary, asset_class_map, common_tickers


def evaluate_model_on_subset(
    model: WaveletGPT,
    test_mlts: list[MultiLevelTokenSequence],
    vocabulary: SAXVocabulary,
    asset_class_map: dict[str, int],
    context_length: int,
    sector_tickers: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a model on a subset of tickers (or all).

    Returns dict with token_accuracy, directional_accuracy, n_samples.
    """
    if sector_tickers is not None:
        filtered = [m for m in test_mlts if m.ticker in sector_tickers]
    else:
        filtered = test_mlts

    if not filtered:
        return {"token_accuracy": 0.0, "directional_accuracy": 0.5, "n_samples": 0}

    dataset = build_sequence_dataset(filtered, vocabulary, context_length, asset_class_map)
    if len(dataset.samples) == 0:
        return {"token_accuracy": 0.0, "directional_accuracy": 0.5, "n_samples": 0}

    X, y, _, _ = dataset.to_arrays()
    levels = np.array([s.level for s in dataset.samples], dtype=np.int64)
    acs = np.array([s.asset_class_id for s in dataset.samples], dtype=np.int64)
    X_full = np.column_stack([X, levels, acs])

    predicted = model.predict(X_full)
    token_acc = float(np.mean(predicted == y))
    dir_acc = level0_directional_accuracy(predicted, y, vocabulary)

    return {
        "token_accuracy": token_acc,
        "directional_accuracy": dir_acc,
        "n_samples": len(y),
    }


def finetune_model(
    model: WaveletGPT,
    train_mlts: list[MultiLevelTokenSequence],
    vocabulary: SAXVocabulary,
    asset_class_map: dict[str, int],
    context_length: int,
    sector_tickers: list[str],
    epochs: int = FINETUNE_EPOCHS,
    lr: float = FINETUNE_LR,
) -> WaveletGPT:
    """Fine-tune a trained model on sector-specific data.

    Returns a new WaveletGPT (the original is not modified).
    """
    filtered = [m for m in train_mlts if m.ticker in sector_tickers]
    if not filtered:
        return model

    dataset = build_sequence_dataset(filtered, vocabulary, context_length, asset_class_map)
    if len(dataset.samples) == 0:
        return model

    X, y, _, _ = dataset.to_arrays()
    levels = np.array([s.level for s in dataset.samples], dtype=np.int64)
    acs = np.array([s.asset_class_id for s in dataset.samples], dtype=np.int64)
    X_full = np.column_stack([X, levels, acs])

    # Deep copy the model's network state
    ft_model = WaveletGPT(
        vocab_size=model._config["vocab_size"],
        context_length=model._config["context_length"],
        embed_dim=model._config["embed_dim"],
        num_heads=model._config["num_heads"],
        num_layers=model._config["num_layers"],
        dropout=model._config["dropout"],
        n_levels=model._config["n_levels"],
        n_asset_classes=model._config["n_asset_classes"],
        epochs=epochs,
        batch_size=model._config["batch_size"],
        learning_rate=lr,
        patience=epochs,  # no early stopping for fine-tune
    )

    # Copy the trained network
    from wavecast.models.wavelet_gpt import WaveletGPTNet
    ft_model._net = WaveletGPTNet(
        vocab_size=model._config["vocab_size"],
        context_length=model._config["context_length"],
        embed_dim=model._config["embed_dim"],
        num_heads=model._config["num_heads"],
        num_layers=model._config["num_layers"],
        dropout=model._config["dropout"],
        n_levels=model._config["n_levels"],
        n_asset_classes=model._config["n_asset_classes"],
    ).to(ft_model._device)
    ft_model._net.load_state_dict(copy.deepcopy(model._net.state_dict()))

    # Fine-tune with lower LR
    ft_model.fit(X_full, y)

    return ft_model


def main() -> None:
    t0 = time.monotonic()
    params = load_best_params()

    logger.info("=" * 70)
    logger.info("F6: Per-Sector Fine-Tuning Experiment")
    logger.info("Params: %s", params)
    logger.info("Fine-tune: %d epochs at LR=%f", FINETUNE_EPOCHS, FINETUNE_LR)
    logger.info("=" * 70)

    cache = ParquetCache(CACHE_DIR)
    all_tickers = DEFAULT_UNIVERSE.tickers

    # Step 1: Build full pipeline data
    logger.info("Building pipeline data for all %d assets...", len(all_tickers))
    train_mlts, test_mlts, vocabulary, asset_class_map, common_tickers = build_pipeline_data(
        all_tickers, cache,
        n_segments=params["n_segments"],
        word_length=params["word_length"],
        word_stride=params["word_stride"],
    )

    context_length = 16

    # Step 2: Train universal model
    logger.info("Training universal model on all assets...")
    dataset = build_sequence_dataset(train_mlts, vocabulary, context_length, asset_class_map)
    X, y, _, _ = dataset.to_arrays()
    levels = np.array([s.level for s in dataset.samples], dtype=np.int64)
    acs = np.array([s.asset_class_id for s in dataset.samples], dtype=np.int64)
    X_full = np.column_stack([X, levels, acs])

    # Build test data for validation during training
    test_dataset = build_sequence_dataset(test_mlts, vocabulary, context_length, asset_class_map)
    X_test, y_test, _, _ = test_dataset.to_arrays()
    test_levels = np.array([s.level for s in test_dataset.samples], dtype=np.int64)
    test_acs = np.array([s.asset_class_id for s in test_dataset.samples], dtype=np.int64)
    X_test_full = np.column_stack([X_test, test_levels, test_acs])

    universal_model = WaveletGPT(
        vocab_size=vocabulary.size,
        context_length=context_length,
        embed_dim=params["embed_dim"],
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        epochs=80,
        batch_size=64,
        learning_rate=0.0005,
        patience=15,
    )
    universal_model.fit(X_full, y, X_val=X_test_full, y_val=y_test)
    logger.info("Universal model trained.")

    # Step 3: Evaluate universal model per sector
    universal_results = {}
    for sector_name, sector_tickers in SECTOR_TICKERS.items():
        metrics = evaluate_model_on_subset(
            universal_model, test_mlts, vocabulary, asset_class_map,
            context_length, sector_tickers,
        )
        universal_results[sector_name] = metrics
        logger.info(
            "Universal | %s: dir_acc=%.4f, tok_acc=%.4f (n=%d)",
            sector_name, metrics["directional_accuracy"],
            metrics["token_accuracy"], metrics["n_samples"],
        )

    # Step 4: Fine-tune per sector and evaluate
    finetuned_results = {}
    for sector_name, sector_tickers in SECTOR_TICKERS.items():
        logger.info("Fine-tuning for sector: %s (%d tickers)...", sector_name, len(sector_tickers))
        ft_model = finetune_model(
            universal_model, train_mlts, vocabulary, asset_class_map,
            context_length, sector_tickers,
        )
        metrics = evaluate_model_on_subset(
            ft_model, test_mlts, vocabulary, asset_class_map,
            context_length, sector_tickers,
        )
        finetuned_results[sector_name] = metrics
        logger.info(
            "FineTuned | %s: dir_acc=%.4f, tok_acc=%.4f (n=%d)",
            sector_name, metrics["directional_accuracy"],
            metrics["token_accuracy"], metrics["n_samples"],
        )

    elapsed = time.monotonic() - t0

    # Step 5: Compute deltas and save
    sector_deltas = {}
    for sector_name in SECTOR_TICKERS:
        u = universal_results[sector_name]
        f = finetuned_results[sector_name]
        sector_deltas[sector_name] = {
            "universal_dir_acc": u["directional_accuracy"],
            "finetuned_dir_acc": f["directional_accuracy"],
            "dir_acc_delta": f["directional_accuracy"] - u["directional_accuracy"],
            "universal_tok_acc": u["token_accuracy"],
            "finetuned_tok_acc": f["token_accuracy"],
            "tok_acc_delta": f["token_accuracy"] - u["token_accuracy"],
            "n_test_samples": f["n_samples"],
        }

    output_data = {
        "params": params,
        "finetune_epochs": FINETUNE_EPOCHS,
        "finetune_lr": FINETUNE_LR,
        "universal_overall": evaluate_model_on_subset(
            universal_model, test_mlts, vocabulary, asset_class_map, context_length,
        ),
        "sector_results": sector_deltas,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output_data, indent=2, default=str))
    logger.info("Results saved to %s", OUTPUT_PATH)

    # Print summary
    print("\n" + "=" * 70)
    print("F6 RESULTS: Per-Sector Fine-Tuning")
    print("=" * 70)
    print(f"\nParams: {params}")
    print(f"Fine-tune: {FINETUNE_EPOCHS} epochs at LR={FINETUNE_LR}")
    print(f"Total time: {elapsed:.1f}s")

    overall = output_data["universal_overall"]
    print(f"\nUniversal model (all 20 assets): dir_acc={overall['directional_accuracy']:.4f}, tok_acc={overall['token_accuracy']:.4f}")

    print(f"\n{'Sector':<16} {'Universal':>10} {'FineTuned':>10} {'Delta':>8} {'Samples':>8}")
    print("-" * 56)
    for sector_name in SECTOR_TICKERS:
        d = sector_deltas[sector_name]
        delta_str = f"{d['dir_acc_delta']:+.4f}"
        print(
            f"{sector_name:<16} "
            f"{d['universal_dir_acc']:>10.4f} "
            f"{d['finetuned_dir_acc']:>10.4f} "
            f"{delta_str:>8} "
            f"{d['n_test_samples']:>8}"
        )

    # Summary
    improving = [s for s, d in sector_deltas.items() if d["dir_acc_delta"] > 0]
    declining = [s for s, d in sector_deltas.items() if d["dir_acc_delta"] < 0]
    print(f"\nImproving sectors ({len(improving)}): {', '.join(improving) if improving else 'none'}")
    print(f"Declining sectors ({len(declining)}): {', '.join(declining) if declining else 'none'}")

    mean_delta = np.mean([d["dir_acc_delta"] for d in sector_deltas.values()])
    print(f"Mean directional accuracy delta: {mean_delta:+.4f}")


if __name__ == "__main__":
    main()
