"""Experiment CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("run")
def run_experiment(
    config_file: Path = typer.Argument(..., help="YAML config file for the experiment"),
) -> None:
    """Run a single experiment from a YAML config file."""
    import yaml

    from wavecast.experiments.config import ExperimentConfig
    from wavecast.experiments.runner import ExperimentRunner
    from wavecast.experiments.storage import DEFAULT_EXPERIMENTS_DIR, save_results

    with open(config_file) as f:
        raw = yaml.safe_load(f)

    config = ExperimentConfig(**raw)
    console.print(f"[bold]Running experiment:[/bold] {config.name}")
    console.print(f"  Tickers: {', '.join(config.tickers)}")
    console.print(f"  Interval: {config.interval}")
    console.print(f"  Train end: {config.train_end} | Test start: {config.test_start}")

    runner = ExperimentRunner(cache_dir=DEFAULT_EXPERIMENTS_DIR / "cache")
    with console.status("Running experiment..."):
        result = runner.run(config)

    out_path = DEFAULT_EXPERIMENTS_DIR / f"{config.name}.json"
    save_results([result], out_path)
    console.print(f"\n[green]Done![/green] Results saved to {out_path}")

    _print_result_summary(result)


@app.command("sweep")
def run_sweep(
    sweep_file: Path = typer.Argument(..., help="YAML file with a list of experiment configs"),
) -> None:
    """Run multiple experiments from a sweep YAML file."""
    import yaml

    from wavecast.experiments.config import ExperimentConfig
    from wavecast.experiments.runner import ExperimentRunner
    from wavecast.experiments.storage import DEFAULT_EXPERIMENTS_DIR, save_results

    with open(sweep_file) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, list):
        console.print("[red]Sweep file must contain a YAML list of experiment configs.[/red]")
        raise typer.Exit(code=1)

    configs = [ExperimentConfig(**c) for c in raw]
    console.print(f"[bold]Running sweep:[/bold] {len(configs)} experiments")

    runner = ExperimentRunner(cache_dir=DEFAULT_EXPERIMENTS_DIR / "cache")
    results = []
    for i, config in enumerate(configs, 1):
        console.print(f"\n[bold]({i}/{len(configs)})[/bold] {config.name}")
        with console.status(f"Running {config.name}..."):
            result = runner.run(config)
        results.append(result)
        console.print(f"  token_accuracy={result.token_accuracy:.4f}  "
                       f"directional_accuracy={result.directional_accuracy:.4f}")

    sweep_name = sweep_file.stem
    out_path = DEFAULT_EXPERIMENTS_DIR / f"sweep_{sweep_name}.json"
    save_results(results, out_path)
    console.print(f"\n[green]Sweep complete![/green] {len(results)} results saved to {out_path}")


@app.command("compare")
def compare_experiments(
    directory: Path = typer.Argument(..., help="Directory containing result JSON files"),
    metric: str = typer.Option(
        "directional_accuracy", "--metric", "-m",
        help="Metric to sort by",
    ),
) -> None:
    """Compare results across multiple experiment result files."""
    from wavecast.experiments.storage import compare_results, load_results

    directory = Path(directory)
    if not directory.is_dir():
        console.print(f"[red]{directory} is not a directory.[/red]")
        raise typer.Exit(code=1)

    all_results = []
    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        console.print(f"[yellow]No JSON result files found in {directory}.[/yellow]")
        raise typer.Exit(code=1)

    for jf in json_files:
        try:
            all_results.extend(load_results(jf))
        except Exception as e:
            console.print(f"[yellow]Skipping {jf.name}: {e}[/yellow]")

    if not all_results:
        console.print("[red]No valid results loaded.[/red]")
        raise typer.Exit(code=1)

    df = compare_results(all_results, metric=metric)

    table = Table(title=f"Experiment Comparison (sorted by {metric})")
    for col in df.columns:
        justify = "left" if col == "name" else "right"
        table.add_column(col, justify=justify)

    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        table.add_row(*cells)

    console.print(table)


@app.command("show")
def show_result(
    result_file: Path = typer.Argument(..., help="JSON result file to display"),
) -> None:
    """Show detailed results from a single experiment result file."""
    from wavecast.experiments.storage import load_results

    result_file = Path(result_file)
    if not result_file.exists():
        console.print(f"[red]{result_file} not found.[/red]")
        raise typer.Exit(code=1)

    results = load_results(result_file)
    for r in results:
        _print_result_summary(r)
        console.print()


def _print_result_summary(result) -> None:
    """Print a formatted summary of an ExperimentResult."""
    table = Table(title=f"Results: {result.config.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Token Accuracy", f"{result.token_accuracy:.4f}")
    table.add_row("Token Accuracy CI",
                  f"[{result.token_accuracy_ci[0]:.4f}, {result.token_accuracy_ci[1]:.4f}]")
    table.add_row("Top-3 Accuracy", f"{result.top3_accuracy:.4f}")
    table.add_row("Directional Accuracy", f"{result.directional_accuracy:.4f}")
    table.add_row("Directional Accuracy CI",
                  f"[{result.directional_accuracy_ci[0]:.4f}, {result.directional_accuracy_ci[1]:.4f}]")
    table.add_row("", "")
    table.add_row("Baseline: Most Frequent", f"{result.baseline_most_frequent:.4f}")
    table.add_row("Baseline: Persistence", f"{result.baseline_persistence:.4f}")
    table.add_row("Baseline: Momentum", f"{result.baseline_momentum:.4f}")
    table.add_row("", "")
    table.add_row("Vocab Size", str(result.vocab_size))
    table.add_row("UNK Rate", f"{result.unk_rate:.4f}")
    table.add_row("Train Samples", str(result.n_train_samples))
    table.add_row("Test Samples", str(result.n_test_samples))
    table.add_row("Training Time", f"{result.training_time_seconds:.1f}s")

    console.print(table)

    if result.per_asset_accuracy:
        asset_table = Table(title="Per-Asset Accuracy")
        asset_table.add_column("Asset", style="cyan")
        asset_table.add_column("Accuracy", justify="right")
        for asset, acc in sorted(result.per_asset_accuracy.items()):
            asset_table.add_row(asset, f"{acc:.4f}")
        console.print(asset_table)

    if result.per_sector_accuracy:
        sector_table = Table(title="Per-Sector Accuracy")
        sector_table.add_column("Sector", style="cyan")
        sector_table.add_column("Accuracy", justify="right")
        for sector, acc in sorted(result.per_sector_accuracy.items()):
            sector_table.add_row(sector, f"{acc:.4f}")
        console.print(sector_table)

    if result.per_level_accuracy:
        level_table = Table(title="Per-Level Accuracy")
        level_table.add_column("Level", style="cyan", justify="right")
        level_table.add_column("Accuracy", justify="right")
        for level, acc in sorted(result.per_level_accuracy.items()):
            level_table.add_row(str(level), f"{acc:.4f}")
        console.print(level_table)
