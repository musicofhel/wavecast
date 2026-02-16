"""Shapelet library management CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from wavecast.core.config import WaveCastConfig

app = typer.Typer(no_args_is_help=True)
console = Console()


def _default_library_path() -> Path:
    config = WaveCastConfig()
    return config.library_dir


@app.command("list")
def list_cmd(
    path: Path = typer.Option(None, "--path", "-p", help="Library directory"),
) -> None:
    """List shapelet libraries."""
    lib_dir = path or _default_library_path()

    if not lib_dir.exists():
        console.print("[yellow]No library directory found.[/yellow]")
        return

    h5_files = sorted(lib_dir.glob("*.h5"))
    if not h5_files:
        console.print("[yellow]No shapelet libraries found.[/yellow]")
        return

    table = Table(title="Shapelet Libraries")
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right")

    for f in h5_files:
        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        table.add_row(f.name, size_str)

    console.print(table)


@app.command()
def stats(
    path: Path = typer.Argument(..., help="Path to .h5 shapelet library"),
) -> None:
    """Show statistics for a shapelet library."""
    from wavecast.shapelets.library import ShapeletLibrary

    lib = ShapeletLibrary.load(path)
    s = lib.stats()

    table = Table(title=f"Library Stats: {path.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Shapelets", str(s["total"]))

    if s.get("by_level"):
        for level, count in sorted(s["by_level"].items()):
            table.add_row(f"  Level {level}", str(count))

    if s.get("by_label"):
        for label, count in sorted(s["by_label"].items()):
            table.add_row(f"  Label: {label}", str(count))

    if s.get("by_ticker"):
        for ticker, count in sorted(s["by_ticker"].items()):
            table.add_row(f"  Ticker: {ticker}", str(count))

    console.print(table)


@app.command("export")
def export_cmd(
    source: Path = typer.Argument(..., help="Source .h5 library"),
    dest: Path = typer.Argument(..., help="Destination path"),
) -> None:
    """Export a shapelet library to a new location."""
    import shutil

    shutil.copy2(source, dest)
    console.print(f"[green]Exported {source.name} → {dest}[/green]")


@app.command("import")
def import_cmd(
    source: Path = typer.Argument(..., help="Source .h5 file to import"),
    dest: Path = typer.Option(None, "--dest", "-d", help="Destination directory"),
) -> None:
    """Import a shapelet library."""
    import shutil

    dest_dir = dest or _default_library_path()
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / source.name
    shutil.copy2(source, target)
    console.print(f"[green]Imported {source.name} → {target}[/green]")
