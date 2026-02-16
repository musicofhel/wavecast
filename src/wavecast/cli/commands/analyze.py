"""Fractal analysis CLI commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from wavecast.core.config import WaveCastConfig

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def hurst(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    window: int = typer.Option(252, "--window", "-w", help="Rolling window size"),
    method: str = typer.Option("wavelet", "--method", "-m", help="Estimation method"),
) -> None:
    """Compute Hurst exponent for a ticker."""
    from wavecast.fractal.hurst import wavelet_hurst
    from wavecast.pipeline.stages import stage_data

    config = WaveCastConfig()
    config.ensure_dirs()

    with console.status(f"Loading {ticker}..."):
        ts = stage_data(ticker, config=config).data

    with console.status("Computing Hurst exponent..."):
        result = wavelet_hurst(ts.values)

    regime_color = {
        "trending": "green",
        "mean_reverting": "red",
        "random_walk": "yellow",
    }
    color = regime_color.get(result.regime.value, "white")

    console.print(Panel(
        f"  H = [bold]{result.hurst_exponent:.4f}[/bold]\n"
        f"  R² = {result.r_squared:.4f}\n"
        f"  Regime: [{color}]{result.regime.value}[/{color}]",
        title=f"Hurst Exponent — {ticker}",
        border_style="blue",
    ))


@app.command()
def mfdfa(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    q_range: str = typer.Option("-5,5", "--q-range", "-q", help="q range (min,max)"),
) -> None:
    """Run MFDFA analysis on a ticker."""
    from wavecast.fractal.mfdfa import compute_mfdfa
    from wavecast.pipeline.stages import stage_data

    config = WaveCastConfig()
    config.ensure_dirs()

    q_min, q_max = (float(x) for x in q_range.split(","))

    with console.status(f"Loading {ticker}..."):
        ts = stage_data(ticker, config=config).data

    with console.status("Computing MFDFA..."):
        result = compute_mfdfa(ts.values, q_range=(q_min, q_max))

    console.print(Panel(
        f"  Spectrum width: [bold]{result.spectrum_width:.4f}[/bold]\n"
        f"  H(q=2): {result.hurst_q[len(result.hurst_q) // 2]:.4f}\n"
        f"  q range: [{q_min}, {q_max}]\n"
        f"  {'Monofractal' if result.spectrum_width < 0.3 else 'Multifractal'}",
        title=f"MFDFA — {ticker}",
        border_style="magenta",
    ))


@app.command()
def regime(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
) -> None:
    """Detect market regime using fractal analysis."""
    from wavecast.fractal.regime import detect_regime
    from wavecast.pipeline.stages import stage_data

    config = WaveCastConfig()
    config.ensure_dirs()

    with console.status(f"Loading {ticker}..."):
        ts = stage_data(ticker, config=config).data

    with console.status("Detecting regime..."):
        result = detect_regime(ts.values, config=config.fractal)

    regime_color = {
        "trending": "green",
        "mean_reverting": "red",
        "random_walk": "yellow",
    }
    color = regime_color.get(result.regime.value, "white")

    table = Table(title=f"Regime Detection — {ticker}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_row("Regime", f"[{color}]{result.regime.value}[/{color}]")
    table.add_row("Hurst", f"{result.hurst.hurst_exponent:.4f}")
    table.add_row("Confidence", f"{result.confidence:.4f}")
    table.add_row("Window", str(result.window_size))
    if result.mfdfa:
        table.add_row("MFDFA Width", f"{result.mfdfa.spectrum_width:.4f}")

    console.print(table)
