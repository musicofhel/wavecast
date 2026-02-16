"""WaveCast CLI application."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

# Load .env from project root (walk up from this file)
_project_root = Path(__file__).resolve().parents[3]
load_dotenv(_project_root / ".env")
# Also try cwd (for when running from repo root)
load_dotenv()

from wavecast.cli.commands import (
    analyze,
    backtest,
    data,
    discover,
    forecast,
    library_,
    match,
    sax,
    tokenize,
)

app = typer.Typer(
    name="wavecast",
    help="Wavelet-Shapelet Financial Forecasting",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

app.add_typer(data.app, name="data", help="Fetch and manage market data")
app.add_typer(discover.app, name="discover", help="Shapelet discovery")
app.add_typer(match.app, name="match", help="DTW pattern matching")
app.add_typer(analyze.app, name="analyze", help="Fractal and wavelet analysis")
app.add_typer(forecast.app, name="forecast", help="Run forecasts")
app.add_typer(library_.app, name="library", help="Manage shapelet library")
app.add_typer(backtest.app, name="backtest", help="Walk-forward backtesting")
app.add_typer(sax.app, name="sax", help="SAX symbolic transformation")
app.add_typer(tokenize.app, name="tokenize", help="Tokenize wavelet coefficients")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """WaveCast: Wavelet-Shapelet Financial Forecasting."""
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)
