"""Data management CLI commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from wavecast.core.config import WaveCastConfig

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def fetch(
    ticker: str = typer.Argument(..., help="Ticker symbol (e.g., AAPL)"),
    start: str | None = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str | None = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD)"),
    interval: str = typer.Option("1d", "--interval", "-i", help="Data interval"),
) -> None:
    """Fetch market data and cache locally."""
    from wavecast.data.cache import ParquetCache
    from wavecast.data.preprocessing import handle_nans
    from wavecast.data.sources import fetch_massive

    config = WaveCastConfig()
    config.ensure_dirs()

    with console.status(f"Fetching {ticker}..."):
        ts = fetch_massive(ticker, start, end, interval)
        ts = handle_nans(ts)

    cache = ParquetCache(config.cache_dir)
    cache.put(ticker, interval, ts)

    console.print(f"[green]Fetched {ticker}[/green]: {ts.length} data points")
    console.print(f"  Range: {ts.timestamps[0]} → {ts.timestamps[-1]}")
    console.print(f"  Cached to: {config.cache_dir}")


@app.command("list")
def list_cmd() -> None:
    """List cached tickers."""
    from wavecast.data.cache import ParquetCache

    config = WaveCastConfig()
    cache = ParquetCache(config.cache_dir)
    cached = cache.list_cached()

    if not cached:
        console.print("[yellow]No cached data found.[/yellow]")
        return

    table = Table(title="Cached Data")
    table.add_column("Ticker", style="cyan")
    table.add_column("Interval", style="green")

    for ticker, interval in cached:
        table.add_row(ticker, interval)

    console.print(table)


@app.command()
def info(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    interval: str = typer.Option("1d", "--interval", "-i"),
) -> None:
    """Show info about cached data."""
    from wavecast.data.cache import ParquetCache

    config = WaveCastConfig()
    cache = ParquetCache(config.cache_dir)
    ts = cache.get(ticker, interval)

    if ts is None:
        console.print(f"[red]No cached data for {ticker} ({interval})[/red]")
        raise typer.Exit(1)

    table = Table(title=f"{ticker} ({interval})")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Data Points", str(ts.length))
    table.add_row("Start", str(ts.timestamps[0]))
    table.add_row("End", str(ts.timestamps[-1]))
    table.add_row("Min", f"{ts.values.min():.2f}")
    table.add_row("Max", f"{ts.values.max():.2f}")
    table.add_row("Mean", f"{ts.values.mean():.2f}")

    console.print(table)
