from __future__ import annotations
import json, sys
from pathlib import Path
from typing import List, Dict, Any

import click

from ragvue import load_metrics , save_all_formats , AgenticOrchestrator

# ---------- helpers ----------
def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

@click.group()
def cli():
    """RAGVue – metric runner & reporter (manual and agentic modes)."""
    pass

@cli.command("list-metrics")
def list_metrics_cmd():
    """List discoverable metrics."""
    metrics = sorted(load_metrics().keys())
    click.echo("\n".join(metrics))

@cli.command("debug-metrics")
@click.option("--questions", "questions_path", type=click.Path(exists=True), required=True, help="JSONL with items.")
@click.option("--metrics", multiple=True, help="Metric names (repeat flag).")
def debug_metrics_cmd(questions_path: str, metrics: List[str]):
    """Run a tiny sample with selected metrics to ensure imports work."""
    from ragvue import evaluate

    items = _read_jsonl(Path(questions_path))[:1]
    if not items:
        click.echo("No items in file.", err=True)
        sys.exit(1)
    if not metrics:
        click.echo("No metrics specified. Try: rag-eval debug-metrics --metrics faithfulness --metrics retrieval_coverage --questions file.jsonl", err=True)
        sys.exit(2)

    report = evaluate(items, metrics=list(metrics))
    click.echo(json.dumps(report, ensure_ascii=False, indent=2))

@cli.command("eval")
@click.option("--input", "questions_path", type=click.Path(exists=True), required=True, help="JSONL with items.")
@click.option("--metrics", multiple=True, help="Metric names (repeat flag). If omitted, runs all discovered metrics.")
@click.option("--out-base", default="report_manual", help="Output base filename (no extension).")
@click.option("--formats", default="json,md,csv,html", help="Comma-separated: json,md,csv,html")
def eval_cmd(questions_path: str, metrics: List[str], out_base: str, formats: str):
    """Manual mode: run exactly the metrics you specify (or all if omitted)."""
    from ragvue import evaluate
    items = _read_jsonl(Path(questions_path))
    if not items:
        click.echo("No items in file.", err=True)
        sys.exit(1)

    # default: all discovered
    if not metrics:
        metrics = sorted(load_metrics().keys())

    report = evaluate(items, metrics=list(metrics))
    paths = save_all_formats(report, out_base=out_base, fmt_list=[s.strip() for s in formats.split(",") if s.strip()])
    click.echo("Wrote:\n  " + "\n  ".join(paths))

@cli.command("agentic")
@click.option("--input", "questions_path", type=click.Path(exists=True), required=True, help="JSONL with items.")
@click.option("--out-base", default="report_agentic", help="Output base filename (no extension).")
@click.option("--formats", default="json,md,csv,html", help="Comma-separated: json,md,csv,html")
@click.option("--max-items", type=int, default=0, help="Limit items for a quick run (0 = all).")
def agentic_cmd(questions_path: str, out_base: str, formats: str, max_items: int):
    """Agentic mode: orchestrator selects metrics per item automatically."""
    items = _read_jsonl(Path(questions_path))
    if max_items > 0:
        items = items[:max_items]

    orch = AgenticOrchestrator()
    report = orch.run(items)
    paths = save_all_formats(report, out_base=out_base, fmt_list=[s.strip() for s in formats.split(",") if s.strip()])
    click.echo("Wrote:\n  " + "\n  ".join(paths))


app = cli

if __name__ == "__main__":
    cli()

 # Usage
     # ragvue-cli  --help

    # ragvue-cli list-metrics

    # manual evaluation
    # ragvue-cli eval \
    #   --input examples/items_manual.jsonl \
    #   --metrics faithfulness --metrics retrieval_coverage \
    #   --out-base report_manual

    # # agentic mode
    # ragvue-cli agentic \
    #   --questions examples/items_agentic.jsonl \
    #   --out-base report_agentic


