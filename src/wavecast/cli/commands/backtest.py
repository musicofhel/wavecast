"""Backtest CLI commands."""

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
def run_backtest(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    strategy: str = typer.Option("ensemble", "--strategy", "-s",
                                  help="Model strategy: ensemble, xgboost, lstm"),
    start: Optional[str] = typer.Option(None, "--start"),
    end: Optional[str] = typer.Option(None, "--end"),
    train_window: int = typer.Option(252, "--train-window", help="Training window size"),
    test_window: int = typer.Option(21, "--test-window", help="Test window size"),
) -> None:
    """Run walk-forward backtest."""
    from wavecast.data.preprocessing import label_returns, log_returns
    from wavecast.evaluation.backtest import WalkForwardBacktest
    from wavecast.evaluation.reporting import generate_report
    from wavecast.features.pipeline import FeaturePipeline
    from wavecast.fractal.hurst import wavelet_hurst
    from wavecast.fractal.mfdfa import compute_mfdfa
    from wavecast.fractal.self_similarity import cross_scale_similarity
    from wavecast.models.ensemble import EnsembleModel
    from wavecast.models.gradient_boost import GradientBoostModel
    from wavecast.models.wavelet_lstm import WaveletLSTM
    from wavecast.pipeline.stages import stage_data, stage_decompose
    from wavecast.shapelets.discovery import discover_shapelets
    from wavecast.shapelets.library import ShapeletLibrary

    config = WaveCastConfig()
    config.backtest.walk_forward_train = train_window
    config.backtest.walk_forward_test = test_window
    config.ensure_dirs()

    with console.status(f"Preparing data for {ticker}..."):
        ts = stage_data(ticker, start, end, config=config).data
        decomp = stage_decompose(ts, config=config).data

        returns_ts = log_returns(ts)
        labels = label_returns(returns_ts)
        shapelets = discover_shapelets(decomp, labels, config.shapelet)
        library = ShapeletLibrary(shapelets)

        hurst = wavelet_hurst(ts.values)
        mfdfa_result = None
        if len(ts.values) >= 256:
            try:
                mfdfa_result = compute_mfdfa(ts.values)
            except Exception:
                pass
        self_sim = cross_scale_similarity(decomp)

    with console.status("Building feature matrix..."):
        pipeline = FeaturePipeline()
        X, y = pipeline.build_feature_matrix(
            ts=ts, decomp=decomp,
        )

    with console.status(f"Running {strategy} backtest..."):
        if strategy == "xgboost":
            model = GradientBoostModel()
        elif strategy == "lstm":
            model = WaveletLSTM()
        else:
            xgb = GradientBoostModel()
            lstm = WaveletLSTM(input_size=X.shape[1])
            model = EnsembleModel(models=[xgb, lstm])

        bt = WalkForwardBacktest(model, config.backtest)
        result = bt.run(X, y, ts.timestamps[-len(X):])
        result = BacktestResult(
            returns=result.returns,
            positions=result.positions,
            equity_curve=result.equity_curve,
            timestamps=result.timestamps,
            metrics=result.metrics,
            model_name=result.model_name,
            ticker=ticker,
        )

    report = generate_report(result)
    console.print(report)

    table = Table(title="Key Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    for key in ["sharpe_ratio", "max_drawdown", "directional_accuracy", "total_return"]:
        if key in result.metrics:
            val = result.metrics[key]
            if "accuracy" in key or "return" in key or "drawdown" in key:
                table.add_row(key.replace("_", " ").title(), f"{val:.2%}")
            else:
                table.add_row(key.replace("_", " ").title(), f"{val:.4f}")

    console.print(table)


# Need this import for the result reconstruction
from wavecast.core.types import BacktestResult  # noqa: E402
