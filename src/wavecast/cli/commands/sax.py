"""SAX symbolic transformation CLI commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def transform(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    segments: int = typer.Option(20, "--segments", "-s", help="Number of PAA segments"),
    alphabet: int = typer.Option(8, "--alphabet", "-a", help="Alphabet size"),
    start: str = typer.Option(None, "--start", help="Start date"),
    end: str = typer.Option(None, "--end", help="End date"),
) -> None:
    """SAX transform a ticker's wavelet coefficients."""
    from wavecast.core.config import WaveCastConfig
    from wavecast.pipeline.stages import stage_data, stage_decompose
    from wavecast.sax.sax import sax_transform

    config = WaveCastConfig()
    data_result = stage_data(ticker, start, end, config=config)
    ts = data_result.data
    decomp_result = stage_decompose(ts, config=config)
    decomp = decomp_result.data

    table = Table(title=f"SAX Transform: {ticker}")
    table.add_column("Level", style="cyan", justify="right")
    table.add_column("Coefficients", justify="right")
    table.add_column("SAX Symbols", style="green")
    table.add_column("Unique Symbols", justify="right")

    for level in range(1, decomp.level + 1):
        coeffs = decomp.detail_at_level(level)
        sax_rep = sax_transform(coeffs, segments, alphabet)
        n_unique = len(set(sax_rep.symbols))
        table.add_row(str(level), str(len(coeffs)), sax_rep.symbols, str(n_unique))

    console.print(table)


@app.command()
def bow(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    word_length: int = typer.Option(4, "--word-length", "-w", help="SAX word length"),
    stride: int = typer.Option(1, "--stride", help="Word extraction stride"),
    segments: int = typer.Option(20, "--segments", "-s", help="Number of PAA segments"),
    alphabet: int = typer.Option(8, "--alphabet", "-a", help="Alphabet size"),
    start: str = typer.Option(None, "--start", help="Start date"),
    end: str = typer.Option(None, "--end", help="End date"),
    top_n: int = typer.Option(20, "--top", "-n", help="Show top N words"),
) -> None:
    """Extract SAX bag-of-words from a ticker."""
    from wavecast.core.config import WaveCastConfig
    from wavecast.pipeline.stages import stage_data, stage_decompose
    from wavecast.sax.bow import build_bow, extract_words
    from wavecast.sax.sax import sax_transform

    config = WaveCastConfig()
    data_result = stage_data(ticker, start, end, config=config)
    ts = data_result.data
    decomp_result = stage_decompose(ts, config=config)
    decomp = decomp_result.data

    for level in range(1, decomp.level + 1):
        coeffs = decomp.detail_at_level(level)
        sax_rep = sax_transform(coeffs, segments, alphabet)
        words = extract_words(sax_rep.symbols, word_length, stride)
        bow_dict = build_bow(words)

        table = Table(title=f"BoW -- Level {level} ({len(words)} words, {len(bow_dict)} unique)")
        table.add_column("Word", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Frequency", justify="right")

        sorted_bow = sorted(bow_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        total = sum(bow_dict.values())
        for word, count in sorted_bow:
            freq = f"{count / total:.3f}" if total > 0 else "0"
            table.add_row(word, str(count), freq)

        console.print(table)
        console.print()
