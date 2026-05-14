"""KaggleSlayer — AutoML pipeline CLI for tabular Kaggle competitions.

Usage:
    kaggle-slayer <competition> --data-path <path>
    kaggle-slayer <competition> --data-path <path> --submit
    kaggle-slayer --all [--submit] [--yes]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.coordinator import PipelineCoordinator

warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

console = Console()


def get_all_competitions() -> list[str]:
    root = Path("competition_data")
    if not root.exists():
        return []
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d / "raw" / "train.csv").exists()
    )


def run_single(name: str, data_path: Path, submit: bool) -> dict:
    console.rule(f"[bold cyan]KaggleSlayer · {name}")
    coord = PipelineCoordinator(name, data_path)
    results = coord.run(submit_to_kaggle=submit)

    panel = Panel.fit(
        f"[bold]Best model:[/] {results.get('best_model', 'N/A')}\n"
        f"[bold]CV score:[/] {results.get('final_score', 0):.4f}\n"
        f"[bold]Submission:[/] {data_path}/submission.csv",
        title="Pipeline complete",
        border_style="green",
    )
    console.print(panel)
    return results


def run_all(submit: bool, yes: bool) -> None:
    comps = get_all_competitions()
    if not comps:
        console.print("[yellow]No competitions found in competition_data/.[/]")
        console.print("Run: [bold]python scripts/download_all_competitions.py[/]")
        return

    console.rule("[bold cyan]KaggleSlayer · batch mode")
    console.print(f"Found [bold]{len(comps)}[/] competitions:")
    for i, c in enumerate(comps, 1):
        console.print(f"  {i:>2}. {c}")

    if not yes:
        ans = console.input(f"\nRun pipeline for all {len(comps)}? [y/N]: ").strip().lower()
        if ans != "y":
            console.print("[yellow]Cancelled.[/]")
            return

    summary = []
    for i, c in enumerate(comps, 1):
        console.rule(f"[{i}/{len(comps)}] {c}")
        try:
            results = run_single(c, Path("competition_data") / c, submit)
            summary.append(
                {
                    "competition": c,
                    "best_model": results.get("best_model", "N/A"),
                    "cv_score": results.get("final_score", 0),
                    "status": "ok",
                }
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
            break
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error processing {c}: {e}[/]")
            summary.append({"competition": c, "status": "failed", "error": str(e)})

    _print_summary(summary)


def _print_summary(summary: list[dict]) -> None:
    table = Table(title="Batch summary", show_lines=False)
    table.add_column("Competition", style="cyan")
    table.add_column("Status")
    table.add_column("Best model")
    table.add_column("CV", justify="right")
    for r in summary:
        if r.get("status") == "ok":
            table.add_row(
                r["competition"],
                "[green]ok[/]",
                r.get("best_model", "—"),
                f"{r['cv_score']:.4f}",
            )
        else:
            table.add_row(
                r["competition"],
                "[red]failed[/]",
                "—",
                r.get("error", "")[:40],
            )
    console.print(table)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="kaggle-slayer",
        description="AutoML pipeline for tabular Kaggle competitions.",
    )
    p.add_argument("competition", nargs="?", help="Competition name (omit when using --all)")
    p.add_argument("--data-path", help="Path to competition data directory")
    p.add_argument("--submit", action="store_true", help="Submit to Kaggle after creating the submission file")
    p.add_argument("--all", action="store_true", help="Run pipeline for every downloaded competition")
    p.add_argument("--yes", action="store_true", help="Non-interactive: auto-confirm batch runs")
    args = p.parse_args()

    if args.all:
        run_all(submit=args.submit, yes=args.yes)
        return

    if not args.competition or not args.data_path:
        p.print_help()
        console.print("\n[red]Error:[/] competition name and --data-path are required (or use --all).")
        sys.exit(2)

    run_single(args.competition, Path(args.data_path), submit=args.submit)


if __name__ == "__main__":
    main()
