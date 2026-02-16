"""Forward testing CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="Forward testing (paper trading)")


@app.command()
def run(
    model_path: str = typer.Option(..., help="Path to saved WaveletGPT directory"),
    vocab_path: str = typer.Option(..., help="Path to saved SAXVocabulary JSON"),
    tickers: str = typer.Option(..., help="Comma-separated ticker symbols"),
    interval: str = typer.Option("1h", help="Data interval"),
    test_name: str = typer.Option("default", help="Test name for tracking"),
    lookback_bars: int = typer.Option(300, help="Number of lookback bars"),
    log_dir: str = typer.Option("", help="Log directory (default: ~/.wavecast/forward_tests)"),
) -> None:
    """Run one forward test cycle: fetch, resolve, predict, log."""
    from wavecast.forward.config import ForwardTestConfig
    from wavecast.forward.report import generate_forward_report
    from wavecast.forward.runner import ForwardTestRunner

    config_kwargs: dict = {
        "test_name": test_name,
        "model_path": model_path,
        "vocab_path": vocab_path,
        "tickers": tickers.split(","),
        "intervals": [interval],
        "lookback_bars": lookback_bars,
    }
    if log_dir:
        config_kwargs["log_dir"] = Path(log_dir)

    config = ForwardTestConfig(**config_kwargs)
    runner = ForwardTestRunner(config)

    summary = runner.run_once()
    typer.echo(generate_forward_report(summary))


@app.command()
def status(
    test_name: str = typer.Option("default", help="Test name"),
    log_dir: str = typer.Option("", help="Log directory"),
) -> None:
    """Show current forward test status."""
    from wavecast.forward.report import generate_forward_report
    from wavecast.forward.tracker import ForwardTestTracker

    log_path = Path(log_dir) if log_dir else Path.home() / ".wavecast" / "forward_tests"
    tracker = ForwardTestTracker(log_dir=log_path, test_name=test_name)
    summary = tracker.get_summary()
    typer.echo(generate_forward_report(summary))


@app.command()
def report(
    test_name: str = typer.Option("default", help="Test name"),
    output: str = typer.Option("", help="Output file path (default: stdout)"),
    fmt: str = typer.Option("text", "--format", help="Output format: text or json"),
    log_dir: str = typer.Option("", help="Log directory"),
) -> None:
    """Generate a forward test report."""
    from wavecast.forward.report import export_forward_json, generate_forward_report
    from wavecast.forward.tracker import ForwardTestTracker

    log_path = Path(log_dir) if log_dir else Path.home() / ".wavecast" / "forward_tests"
    tracker = ForwardTestTracker(log_dir=log_path, test_name=test_name)
    summary = tracker.get_summary()

    if fmt == "json":
        if output:
            export_forward_json(summary, Path(output))
            typer.echo(f"Report exported to {output}")
        else:
            typer.echo(json.dumps(summary.to_dict(), indent=2))
    else:
        text = generate_forward_report(summary)
        if output:
            Path(output).write_text(text)
            typer.echo(f"Report written to {output}")
        else:
            typer.echo(text)


@app.command(name="list")
def list_tests(
    log_dir: str = typer.Option("", help="Log directory"),
) -> None:
    """List all forward tests."""
    log_path = Path(log_dir) if log_dir else Path.home() / ".wavecast" / "forward_tests"

    if not log_path.exists():
        typer.echo("No forward tests found.")
        return

    tests = sorted(d.name for d in log_path.iterdir() if d.is_dir())
    if not tests:
        typer.echo("No forward tests found.")
        return

    typer.echo(f"Forward tests in {log_path}:")
    for t in tests:
        pred_file = log_path / t / "predictions.jsonl"
        n_preds = 0
        if pred_file.exists():
            n_preds = sum(1 for _ in pred_file.open())
        typer.echo(f"  {t} ({n_preds} predictions)")
