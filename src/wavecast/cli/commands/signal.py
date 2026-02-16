"""Signal generation and backtesting CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def generate(
    model_path: Path = typer.Argument(..., help="Path to trained WaveletGPT model"),
    ticker: str = typer.Argument(..., help="Asset ticker symbol"),
    horizon: int = typer.Option(1, help="Prediction horizon"),
    confidence_threshold: float = typer.Option(0.0, help="Minimum confidence threshold"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output CSV path"),
) -> None:
    """Generate trading signals from a trained model."""
    from wavecast.signals.generator import SignalGenerator

    typer.echo(f"Loading model from {model_path}...")

    typer.echo(f"Generating signals for {ticker} at horizon {horizon}...")
    SignalGenerator(
        vocab_size=100,
        confidence_threshold=confidence_threshold,
    )

    typer.echo("Signal generation requires input data. Use 'wavecast signal backtest' for full pipeline.")


@app.command()
def backtest(
    model_path: Path = typer.Argument(..., help="Path to trained WaveletGPT model"),
    tickers: str = typer.Argument(..., help="Comma-separated ticker symbols"),
    train_end: str = typer.Option("2023-12-31", help="Training period end date"),
    test_start: str = typer.Option("2024-01-01", help="Test period start date"),
    position_method: str = typer.Option("fractional_kelly", help="Position sizing method"),
    commission: float = typer.Option(0.001, help="Commission rate"),
    spread_bps: float = typer.Option(2.0, help="Spread in basis points"),
    slippage_bps: float = typer.Option(1.0, help="Slippage in basis points"),
    confidence_threshold: float = typer.Option(0.0, help="Minimum confidence threshold"),
    batch_size: int = typer.Option(256, help="Batch size for inference"),
    use_amp: bool = typer.Option(False, "--use-amp", help="Use mixed-precision inference"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="Output directory"),
) -> None:
    """Run signal backtest on one or more tickers."""
    typer.echo(f"Backtesting {tickers} with {position_method} sizing...")
    typer.echo(f"  Train end: {train_end}, Test start: {test_start}")
    typer.echo(f"  Costs: commission={commission}, spread={spread_bps}bps, slippage={slippage_bps}bps")
    typer.echo(f"  Confidence threshold: {confidence_threshold}")
    typer.echo(f"  Batch size: {batch_size}, AMP: {use_amp}")

    typer.echo("Full pipeline backtest not yet connected to data loader.")
    typer.echo("Use Python API directly for signal backtesting.")


@app.command()
def calibrate(
    model_path: Path = typer.Argument(..., help="Path to trained WaveletGPT model"),
    tickers: str = typer.Argument(..., help="Comma-separated ticker symbols"),
    validation_start: str = typer.Option(..., help="Validation start date"),
    validation_end: str = typer.Option(..., help="Validation end date"),
) -> None:
    """Calibrate signal confidence using temperature scaling."""
    typer.echo(f"Calibrating on {tickers} from {validation_start} to {validation_end}...")
    typer.echo("Calibration requires validation data. Use Python API for now.")
