"""DTW matching CLI commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from wavecast.core.config import WaveCastConfig

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def find(
    ticker: str = typer.Argument(..., help="Source ticker"),
    against: str = typer.Option(..., "--against", "-a", help="Target ticker to match against"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    dtw_window: int = typer.Option(10, "--dtw-window", "-w"),
    level: int = typer.Option(3, "--level", "-l", help="DWT level to compare"),
) -> None:
    """Find DTW matches between two tickers at a given wavelet level."""
    from wavecast.dtw.matching import match_single
    from wavecast.pipeline.stages import stage_data, stage_decompose

    config = WaveCastConfig()
    config.dtw.window = dtw_window
    config.ensure_dirs()

    with console.status(f"Loading {ticker} and {against}..."):
        ts1 = stage_data(ticker, config=config).data
        ts2 = stage_data(against, config=config).data

    with console.status("Decomposing..."):
        d1 = stage_decompose(ts1, config=config).data
        d2 = stage_decompose(ts2, config=config).data

    with console.status("Computing DTW..."):
        coeffs1 = d1.detail_at_level(level)
        coeffs2 = d2.detail_at_level(level)
        result = match_single(coeffs1, coeffs2, window=dtw_window)

    console.print(f"\n[bold]DTW Match: {ticker} vs {against} (Level {level})[/bold]")
    console.print(f"  Distance: {result.distance:.4f}")
    console.print(f"  Normalized: {result.normalized_distance:.4f}")
    console.print(f"  Path length: {len(result.warping_path)}")


@app.command()
def similarity(
    ticker1: str = typer.Argument(..., help="First ticker"),
    ticker2: str = typer.Argument(..., help="Second ticker"),
    levels: Optional[str] = typer.Option(None, "--levels", "-l",
                                         help="Comma-separated levels (e.g., 3,4,5)"),
) -> None:
    """Compute multi-level DTW similarity between two tickers."""
    from wavecast.dtw.similarity import similarity_matrix as sim_mat
    from wavecast.pipeline.stages import stage_data, stage_decompose

    import numpy as np

    from wavecast.dtw.matching import match_single

    config = WaveCastConfig()
    config.ensure_dirs()

    level_list = [int(x) for x in levels.split(",")] if levels else list(
        range(1, config.wavelet.level + 1)
    )

    with console.status("Loading data..."):
        ts1 = stage_data(ticker1, config=config).data
        ts2 = stage_data(ticker2, config=config).data

    with console.status("Decomposing and computing similarity..."):
        d1 = stage_decompose(ts1, config=config).data
        d2 = stage_decompose(ts2, config=config).data

    table = Table(title=f"DTW Similarity: {ticker1} vs {ticker2}")
    table.add_column("Level", justify="center", style="cyan")
    table.add_column("Period (days)", justify="right")
    table.add_column("Distance", justify="right")
    table.add_column("Similarity", justify="right", style="green")

    for lv in level_list:
        c1 = d1.detail_at_level(lv)
        c2 = d2.detail_at_level(lv)
        result = match_single(c1, c2, window=config.dtw.window)
        sim = 1.0 / (1.0 + result.distance)
        table.add_row(str(lv), str(2**lv), f"{result.distance:.4f}", f"{sim:.4f}")

    console.print(table)
