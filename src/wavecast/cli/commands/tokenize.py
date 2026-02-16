"""Tokenization CLI commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def vocab(
    corpus_size: int = typer.Option(5, "--corpus-size", "-c", help="Number of tickers for corpus"),
    segments: int = typer.Option(20, "--segments", "-s", help="SAX segments"),
    alphabet: int = typer.Option(8, "--alphabet", "-a", help="SAX alphabet size"),
    word_length: int = typer.Option(4, "--word-length", "-w", help="SAX word length"),
    min_freq: int = typer.Option(2, "--min-freq", help="Minimum word frequency"),
    max_size: int = typer.Option(500, "--max-size", help="Maximum vocabulary size"),
) -> None:
    """Build and inspect a SAX vocabulary from market data."""
    from wavecast.core.config import WaveCastConfig
    from wavecast.core.universe import DEFAULT_UNIVERSE
    from wavecast.pipeline.stages import stage_data, stage_decompose
    from wavecast.sax.bow import extract_words
    from wavecast.sax.sax import sax_transform
    from wavecast.tokenizer.vocabulary import SAXVocabulary

    config = WaveCastConfig()
    assets = DEFAULT_UNIVERSE.assets[:corpus_size]
    console.print(f"[bold]Building vocabulary from {len(assets)} assets...[/bold]")

    all_words: list[list[str]] = []
    for asset in assets:
        try:
            data_result = stage_data(asset.ticker, config=config)
            ts = data_result.data
            decomp_result = stage_decompose(ts, config=config)
            decomp = decomp_result.data

            for lvl in range(1, decomp.level + 1):
                coeffs = decomp.detail_at_level(lvl)
                if len(coeffs) < 2:
                    continue
                n_seg = min(segments, len(coeffs))
                sax_rep = sax_transform(coeffs, n_seg, alphabet)
                words = extract_words(sax_rep.symbols, word_length, 1)
                all_words.append(words)

            console.print(f"  {asset.ticker}: OK")
        except Exception as e:
            console.print(f"  [yellow]{asset.ticker}: {e}[/yellow]")

    vocabulary = SAXVocabulary.from_corpus(all_words, min_freq=min_freq, max_size=max_size)

    table = Table(title="Vocabulary Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total size", str(vocabulary.size))
    table.add_row("Corpus tickers", str(len(assets)))
    table.add_row("Word sequences", str(len(all_words)))
    console.print(table)

    # Show top words
    top_words = vocabulary.words[:20]
    if top_words:
        word_table = Table(title="Top 20 Words")
        word_table.add_column("Token ID", justify="right")
        word_table.add_column("Word", style="green")
        for i, word in enumerate(top_words, start=2):
            word_table.add_row(str(i), word)
        console.print(word_table)


@app.command()
def run(
    epochs: int = typer.Option(20, "--epochs", "-e", help="Training epochs"),
    universe_name: str = typer.Option("default", "--universe", "-u", help="Universe name"),
) -> None:
    """Run the full token prediction pipeline."""
    from wavecast.core.config import WaveCastConfig
    from wavecast.core.universe import get_universe
    from wavecast.pipeline.token_pipeline import TokenPipelineRunner

    config = WaveCastConfig()
    config.sequence_model.epochs = epochs
    universe = get_universe(universe_name)

    console.print(f"[bold]Running token pipeline ({len(universe)} assets, {epochs} epochs)...[/bold]")

    runner = TokenPipelineRunner()
    result = runner.run(universe=universe, config=config)

    # Print results
    table = Table(title="Token Pipeline Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Token Accuracy", f"{result.metrics.token_accuracy:.4f}")
    table.add_row("Top-3 Accuracy", f"{result.metrics.top3_accuracy:.4f}")
    table.add_row("Directional Accuracy", f"{result.metrics.directional_accuracy:.4f}")
    table.add_row("Vocabulary Size", str(result.vocabulary.size))
    table.add_row("Duration", f"{result.duration_seconds:.1f}s")

    if result.train_metrics:
        table.add_row("Train Loss", f"{result.train_metrics.get('train_loss', 0):.4f}")
        table.add_row("Train Accuracy", f"{result.train_metrics.get('train_accuracy', 0):.4f}")
        if "val_loss" in result.train_metrics:
            table.add_row("Val Loss", f"{result.train_metrics['val_loss']:.4f}")

    if result.model_path:
        table.add_row("Model Path", str(result.model_path))

    console.print(table)
