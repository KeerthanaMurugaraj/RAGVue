
"""
Run RAG evaluation both in MANUAL and AGENTIC modes from pure Python CLI.

- Reads items from --questions JSONL (each line is an item dict)
- MANUAL: choose metrics via --metrics
- AGENTIC: uses AgenticOrchestrator to auto-select metrics per item
- Writes JSON/MD/CSV/HTML reports

"""

from __future__ import annotations
import json, argparse
from pathlib import Path
from typing import List, Dict, Any

# Optional: load .env automatically
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(filename="../.env", usecwd=True), override=True)
except Exception:
    pass

from ragvue import evaluate as pkg_evaluate
from ragvue import load_metrics
from ragvue import AgenticOrchestrator
from ragvue import ReportBuilder, save_all_formats

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def run_manual(items: List[Dict[str, Any]], metrics: List[str], out_base: str) -> None:
    if not metrics:
        # default to **all discovered** if not provided
        metrics = sorted(load_metrics().keys())
        print("[manual] No --metrics provided; using all discovered:", metrics)

    report = pkg_evaluate(items, metrics=metrics)
    paths = save_all_formats(report, out_base=out_base, fmt_list=["json", "md", "csv", "html"])
    print("[manual] wrote:\n  " + "\n  ".join(paths))

    # quick console summary
    rb = ReportBuilder(report)
    print("[manual] summary:", {k: round(v, 3) for k, v in rb.summary.items()})


def run_agentic(items: List[Dict[str, Any]], out_base: str) -> None:
    orch = AgenticOrchestrator()
    report = orch.run(items)
    paths = save_all_formats(report, out_base=out_base, fmt_list=["json", "md", "csv", "html"])
    print("[agentic] wrote:\n  " + "\n  ".join(paths))

    rb = ReportBuilder(report)
    print("[agentic] summary:", {k: round(v, 3) for k, v in rb.summary.items()})


def main():
    ap = argparse.ArgumentParser(description="Run RAG evaluation in manual and/or agentic mode.")
    ap.add_argument("--input", type=str, required=True, help="Path to items.jsonl")
    ap.add_argument("--metrics", action="append", help="Manual mode: repeat flag to add metrics (e.g., --metrics faithfulness). If omitted, runs all discovered.", default=[])
    ap.add_argument("--out-base", type=str, default="report_manual", help="Manual mode output base filename (no extension).")
    ap.add_argument("--agentic-out", type=str, default="report_agentic", help="Agentic mode output base filename (no extension).")
    ap.add_argument("--skip-manual", action="store_true", help="Skip manual run.")
    ap.add_argument("--skip-agentic", action="store_true", help="Skip agentic run.")
    args = ap.parse_args()

    items = read_jsonl(Path(args.input))
    if not items:
        raise SystemExit("No items found in file.")

    print(f"Loaded {len(items)} item(s) from {args.input}")

    if not args.skip_manual:
        run_manual(items, metrics=args.metrics, out_base=args.out_base)

    if not args.skip_agentic:
        run_agentic(items, out_base=args.agentic_out)


if __name__ == "__main__":
    main()

#  Usage:

    #  ragvue-py --help

    # ragvue-py \
    #   --input examples/items_manual.jsonl \
    #   --metrics faithfulness --metrics retrieval_coverage \
    #   --out-base report_manual \
    #   --agentic-out report_agentic

