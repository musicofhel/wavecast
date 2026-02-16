"""Forecast CLI commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from wavecast.core.config import WaveCastConfig

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("run")
def run_forecast(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    horizon: int = typer.Option(5, "--horizon", "-h", help="Forecast horizon (days)"),
    model: str = typer.Option("ensemble", "--model", "-m",
                               help="Model type: ensemble, xgboost, lstm"),
    start: Optional[str] = typer.Option(None, "--start", "-s"),
    end: Optional[str] = typer.Option(None, "--end", "-e"),
) -> None:
    """Run forecast for a ticker."""
    from wavecast.pipeline.runner import PipelineRunner

    config = WaveCastConfig()
    config.ensure_dirs()

    runner = PipelineRunner(config)

    with console.status(f"Running {model} forecast for {ticker}..."):
        result = runner.run_full(ticker, start, end, horizon=horizon, model_type=model)

    console.print(Panel(
        f"  Model: {result.model_name}\n"
        f"  Horizon: {result.horizon}\n"
        f"  Predictions: {len(result.predictions)}\n"
        f"  Features used: {result.metadata.get('n_features', 'N/A')}\n"
        f"  Shapelets: {result.metadata.get('n_shapelets', 'N/A')}",
        title=f"Forecast — {ticker}",
        border_style="green",
    ))

    table = Table(title="Predictions (last 10)")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Predicted Return", justify="right")
    table.add_column("Direction", justify="center")

    for i in range(max(0, len(result.predictions) - 10), len(result.predictions)):
        pred = result.predictions[i]
        direction = "[green]UP[/green]" if pred > 0 else "[red]DOWN[/red]"
        table.add_row(
            str(result.timestamps[i]),
            f"{pred:.6f}",
            direction,
        )

    console.print(table)


@app.command()
def evaluate(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    start: Optional[str] = typer.Option(None, "--start", "-s"),
    end: Optional[str] = typer.Option(None, "--end", "-e"),
    model: str = typer.Option("ensemble", "--model", "-m"),
) -> None:
    """Evaluate forecast accuracy on historical data."""
    from wavecast.evaluation import metrics as m
    from wavecast.pipeline.runner import PipelineRunner

    config = WaveCastConfig()
    config.ensure_dirs()

    runner = PipelineRunner(config)

    with console.status(f"Running evaluation for {ticker}..."):
        result = runner.run_full(ticker, start, end, model_type=model)

    # The pipeline already splits train/test, predictions are on test set
    import numpy as np

    console.print(Panel(
        f"  Model: {result.model_name}\n"
        f"  Predictions: {len(result.predictions)}\n"
        f"  Mean prediction: {np.mean(result.predictions):.6f}\n"
        f"  Std prediction: {np.std(result.predictions):.6f}",
        title=f"Evaluation — {ticker}",
        border_style="blue",
    ))
