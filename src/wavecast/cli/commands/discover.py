"""Shapelet discovery CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from wavecast.core.config import WaveCastConfig

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def run(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    levels: int = typer.Option(5, "--levels", "-l", help="DWT decomposition levels"),
    wavelet: str = typer.Option("db4", "--wavelet", "-w", help="Wavelet family"),
    threshold: float = typer.Option(1.0, "--threshold", "-t", help="Z-score threshold"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output .h5 path"),
) -> None:
    """Run W-TSS shapelet discovery on a ticker."""
    from wavecast.data.preprocessing import label_returns, log_returns
    from wavecast.pipeline.stages import stage_data, stage_decompose

    config = WaveCastConfig()
    config.wavelet.wavelet = wavelet
    config.wavelet.level = levels
    config.shapelet.z_threshold = threshold
    config.ensure_dirs()

    with console.status(f"Loading {ticker} data..."):
        data_result = stage_data(ticker, config=config)
        ts = data_result.data

    with console.status("Decomposing..."):
        decomp_result = stage_decompose(ts, config=config)
        decomp = decomp_result.data

    with console.status("Discovering shapelets..."):
        from wavecast.shapelets.discovery import discover_shapelets

        returns_ts = log_returns(ts)
        labels = label_returns(returns_ts)
        shapelets = discover_shapelets(decomp, labels, config.shapelet)

    if not shapelets:
        console.print("[yellow]No shapelets discovered.[/yellow]")
        return

    table = Table(title=f"Discovered Shapelets ({len(shapelets)} total)")
    table.add_column("ID", style="cyan")
    table.add_column("Level", justify="center")
    table.add_column("Length", justify="right")
    table.add_column("Label", style="green")
    table.add_column("IG", justify="right", style="yellow")

    for s in shapelets[:20]:
        table.add_row(s.id, str(s.wavelet_level), str(s.length),
                       s.label.value, f"{s.information_gain:.4f}")

    console.print(table)

    if output or True:
        from wavecast.shapelets.library import ShapeletLibrary

        library = ShapeletLibrary(shapelets)
        save_path = output or config.library_dir / f"{ticker}_shapelets.h5"
        library.save(save_path)
        console.print(f"[green]Saved to {save_path}[/green]")


@app.command()
def scan(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    library: Path = typer.Option(..., "--library", "-l", help="Path to .h5 shapelet library"),
    top_k: int = typer.Option(10, "--top-k", "-k"),
) -> None:
    """Scan a shapelet library for patterns matching a ticker."""
    from wavecast.shapelets.library import ShapeletLibrary

    lib = ShapeletLibrary.load(library)
    matches = lib.query(ticker=ticker, top_k=top_k)

    if not matches:
        console.print(f"[yellow]No shapelets found for {ticker}[/yellow]")
        return

    table = Table(title=f"Shapelets for {ticker}")
    table.add_column("ID", style="cyan")
    table.add_column("Level", justify="center")
    table.add_column("Label", style="green")
    table.add_column("IG", justify="right", style="yellow")

    for s in matches:
        table.add_row(s.id, str(s.wavelet_level), s.label.value,
                       f"{s.information_gain:.4f}")

    console.print(table)
