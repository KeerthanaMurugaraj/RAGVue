from __future__ import annotations
import json, csv, html
from pathlib import Path
from typing import Dict, Any, List

class ReportBuilder:
    def __init__(self, report: Dict[str, Any]):
        # If it's a Pydantic BaseModel (v2), use .model_dump()
        if hasattr(report, "model_dump") and callable(report.model_dump):
            report = report.model_dump()
        # If it's a Pydantic BaseModel (v1), fallback to .dict()
        elif hasattr(report, "dict") and callable(report.dict):
            report = report.dict()
        self.report = report or {}
        self.results = self.report.get("results", [])
        self.summary = self.report.get("summary", {})

    # ---------- Markdown ----------

    def to_markdown(self) -> str:
        lines = []
        lines.append("# RAG Evaluation Report\n")

        if self.summary:
            lines.append("## Summary (mean scores of all cases)")
            lines.append("")
            lines.append("| Metric | Mean score |")
            lines.append("| :------ | ---------: |")
            for k in sorted(self.summary.keys()):
                lines.append(f"| **{k}** | {self._fmt(self.summary[k])} |")
            lines.append("")

        lines.append("## Individual Case Report\n")
        for idx, r in enumerate(self.results, 1):
            item = r.get("item", {})
            q = item.get("question", "")
            a = item.get("answer", None)
            ctxs = item.get("contexts", [])
            metrics = r.get("metrics", []) or []
            eval_time = r.get("eval_time_sec", None)

            lines.append(f"### Case {idx}")
            lines.append(f"- Question: {q}")
            if a is not None:
                lines.append(f"- Answer: {a}")
            if ctxs:
                lines.append(f"- Contexts: {len(ctxs)}")
                for i, c in enumerate(ctxs, 1):
                    lines.append(f"  - [{i}] {c}")
            if eval_time is not None:
                lines.append(f"- Eval time: {self._fmt(eval_time)} s")

            lines.append("")

            if metrics:
                # ---- compact table with metric scores ----
                lines.append("**Metric scores**")
                lines.append("")
                lines.append("| Metric | Score |")
                lines.append("| :------ | ----: |")
                for m in metrics:
                    name = m.get("name", "unknown")
                    score = self._fmt(m.get("score", 0.0))
                    lines.append(f"| {name} | {score} |")
                lines.append("")

                # ---- detailed breakdown per metric ----
                lines.append("**Metric details**")
                for m in metrics:
                    name = m.get("name", "unknown")
                    score = self._fmt(m.get("score", 0.0))
                    lines.append(f"- **{name}** ({score})")
                    expl = m.get("explanation")
                    if expl:
                        lines.append(f"  - explanation: {expl}")
                    if "details" in m and m["details"]:
                        pretty = json.dumps(m["details"], ensure_ascii=False, indent=2)
                        lines.append("  - details:\n")
                        lines.append("```json")
                        lines.append(pretty)
                        lines.append("```")
                    if "raw" in m and isinstance(m["raw"], dict):
                        snippet = json.dumps(m["raw"], ensure_ascii=False)[:300]
                        lines.append(f"  - raw: `{snippet}...`")
                lines.append("")

        return "\n".join(lines)

    # ---------- HTML ----------

    def to_html(self) -> str:
        css = """
        <style>
          body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
          h1 { margin-top: 0; }
          .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
          .card { border: 1px solid #e6e6e6; border-radius: 12px; padding: 12px 14px; background: #fff; }
          .metric { display:flex; justify-content:space-between; }
          .pill { background:#f5f5f7; border-radius:999px; padding:2px 10px; font-variant-numeric: tabular-nums; }
          details { margin:8px 0 0 0; }
          summary { cursor:pointer; font-weight:600; outline:none; }
          summary::-webkit-details-marker { display:none; }
          summary::before {
            content: " ";
            display:inline-block;
            transition: transform 0.15s ease-out;
          }
          details[open] summary::before {
            transform: rotate(90deg);
          }
          .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
          .grid { display:grid; grid-template-columns: 1fr; gap: 12px; }
          .item { border:1px solid #efefef; border-radius:12px; padding:8px 10px; background:#fafafa; }
          .item-inner { padding:8px 4px 4px 4px; }
          .contexts { margin:.5rem 0; color:#333 }
          .badge { background:#eef7ff; border:1px solid #c5e2ff; padding:2px 8px; border-radius:999px; font-size:.85em; }
        </style>
        """
        html_parts = [f"<!doctype html><meta charset='utf-8'><title>RAG Evaluation Report</title>{css}<body>"]
        html_parts.append("<h1>RAG Evaluation Report</h1>")

        if self.summary:
            html_parts.append("<h2>Summary (mean scores)</h2>")
            html_parts.append("<div class='summary'>")
            for k in sorted(self.summary.keys()):
                html_parts.append(
                    f"<div class='card metric'><div>{html.escape(k)}</div>"
                    f"<div class='pill'>{self._fmt(self.summary[k])}</div></div>"
                )
            html_parts.append("</div>")

        html_parts.append("<h2>Evaluation Cases</h2>")
        for idx, r in enumerate(self.results, 1):
            item = r.get("item", {})
            q = item.get("question", "") or ""
            a = item.get("answer", None)
            ctxs = item.get("contexts", [])
            eval_time = r.get("eval_time_sec", None)

            # Short question snippet for the summary line
            q_snip = q.strip().replace("\n", " ")
            if len(q_snip) > 120:
                q_snip = q_snip[:117] + "..."

            html_parts.append("<details class='item'>")
            html_parts.append(
                f"<summary>Case {idx}: {html.escape(q_snip)}</summary>"
            )
            html_parts.append("<div class='item-inner'>")

            # full question / answer / contexts
            html_parts.append(f"<div><span class='badge'>Question</span> {html.escape(q)}</div>")
            if a is not None:
                html_parts.append(f"<div><span class='badge'>Answer</span> {html.escape(str(a))}</div>")
            if eval_time is not None:
                html_parts.append(
                    f"<div><span class='badge'>Eval time</span> {self._fmt(eval_time)} s</div>"
                )
            if ctxs:
                html_parts.append("<div class='contexts'><span class='badge'>Contexts</span><ol>")
                for c in ctxs:
                    html_parts.append(f"<li>{html.escape(str(c))}</li>")
                html_parts.append("</ol></div>")

            # metrics inside the case
            html_parts.append("<details open><summary>Metrics</summary>")
            html_parts.append("<div class='grid'>")
            for m in r.get("metrics", []):
                name = html.escape(m.get("name", "unknown"))
                score = self._fmt(m.get("score", 0.0))
                html_parts.append("<div class='card'>")
                html_parts.append(
                    f"<div class='metric'><div>{name}</div><div class='pill'>{score}</div></div>"
                )
                expl = m.get("explanation")
                if expl:
                    expl_html = html.escape(expl).replace("\n", "<br>")
                    html_parts.append(f"<div class='mono'>{expl_html}</div>")

                details = m.get("details")
                if details:
                    pretty = json.dumps(details, ensure_ascii=False, indent=2)
                    snip = html.escape(pretty[:2000])
                    html_parts.append(
                        "<details><summary>details</summary>"
                        f"<pre class='mono'>{snip}</pre></details>"
                    )

                raw = m.get("raw")
                if isinstance(raw, dict):
                    snip = html.escape(json.dumps(raw, ensure_ascii=False)[:1200])
                    html_parts.append(
                        "<details><summary>raw</summary>"
                        f"<pre class='mono'>{snip}</pre></details>"
                    )
                html_parts.append("</div>")  # .card
            html_parts.append("</div></details>")  # .grid / Metrics
            html_parts.append("</div>")  # .item-inner
            html_parts.append("</details>")  # .item

        html_parts.append("</body>")
        return "".join(html_parts)

    # ---------- CSV + JSON writers ----------
    def write_json(self, path: str | Path) -> str:
        p = Path(path)
        p.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)


    def write_csv(self, path: str | Path) -> str:
        p = Path(path)
        rows = []
        for idx, r in enumerate(self.results):
            item = r.get("item", {})
            q = item.get("question", "")
            a = item.get("answer", "")
            ctx = item.get("contexts", [])
            ctx_str = " || ".join([str(x) for x in ctx]) if isinstance(ctx, list) else str(ctx)
            agg = r.get("aggregate", None)
            eval_time = r.get("eval_time_sec", None)

            # rounded versions
            agg_r = self._round_num(agg)
            eval_time_r = self._round_num(eval_time)

            for m in r.get("metrics", []):
                score_r = self._round_num(m.get("score", None))
                row = {
                    "item_index": idx,
                    "metric": m.get("name", ""),
                    "score": score_r,
                    "aggregate_for_item": agg_r,
                    "question": q,
                    "answer": a,
                    "contexts": ctx_str,
                    "explanation": m.get("explanation", ""),
                    "eval_time_sec": eval_time_r,
                }
                rows.append(row)

        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys())
                if rows
                else ["item_index", "metric", "score", "aggregate_for_item", "question", "answer", "contexts", "explanation", "eval_time_sec"],
            )
            w.writeheader()
            for row in rows:
                w.writerow(row)
        return str(p)

    def write_markdown(self, path: str | Path) -> str:
        p = Path(path)
        p.write_text(self.to_markdown(), encoding="utf-8")
        return str(p)

    def write_html(self, path: str | Path) -> str:
        p = Path(path)
        p.write_text(self.to_html(), encoding="utf-8")
        return str(p)

    # ---------- helpers ----------
    @staticmethod
    def _fmt(x: Any) -> str:
        try:
            v = float(x)
            return f"{v:.2f}"
        except Exception:
            return str(x)

    @staticmethod
    def _round_num(x: Any, ndigits: int = 2) -> Any:
        """Round numeric values for CSV, leave others unchanged."""
        try:
            v = float(x)
            return round(v, ndigits)
        except Exception:
            return x

def save_all_formats(
    report: Dict[str, Any],
    out_base: str = "report",
    fmt_list: str | list[str] | None = None,
) -> List[str]:
    """
    Convenience helper used by the CLI.

    Parameters
    ----------
    report : dict
        RAGVue-style report object.
    out_base : str
        Base path without extension, e.g. "runs/run1/report".
    fmt_list : str | list[str] | None
        - None  -> default to ["md"]
        - "md"  -> only markdown
        - "json,md,html" -> split on comma
        - ["json", "md"] -> list of formats
    """
    # --- normalize formats ---
    if fmt_list is None:
        fmts = ["md"]           # default: ONLY markdown
    elif isinstance(fmt_list, str):
        # allow "md", "json,md", "json, md, html"
        fmts = [f.strip().lower() for f in fmt_list.split(",") if f.strip()]
    else:
        fmts = [str(f).lower() for f in fmt_list]

    rb = ReportBuilder(report)
    out: List[str] = []
    base = Path(out_base)

    for fmt in fmts:
        if fmt == "json":
            out.append(rb.write_json(base.with_suffix(".json")))
        elif fmt == "md":
            out.append(rb.write_markdown(base.with_suffix(".md")))
        elif fmt == "csv":
            out.append(rb.write_csv(base.with_suffix(".csv")))
        elif fmt == "html":
            out.append(rb.write_html(base.with_suffix(".html")))
        else:
            # silently ignore unknown formats or raise if you prefer
            print(f"[save_all_formats] Unknown format: {fmt!r}, skipping.")
    return out

def save_report(
    report: Dict[str, Any],
    out_base: str = "report",
    fmt_list: str | list[str] | None = None,
    mode: str = "single",   # "single" or "per-item"
    per_item_dir: str | Path | None = None,
) -> list[str]:
    """
    High-level helper to save reports.

    Parameters
    ----------
    report : dict
        Full RAGVue-style report object (with 'results' list).
    out_base : str
        For mode="single": base path without extension (e.g., "runs/run1/report").
    fmt_list : str | list[str] | None
        Passed through to save_all_formats:
        - None          -> default ["md"]
        - "md"          -> just markdown
        - "json,md"     -> json and markdown
        - ["json","md"] -> explicit list
    mode : {"single", "per-item"}
        - "single": one aggregated report (current behavior).
        - "per-item": one report per item in report["results"].
    per_item_dir : str | Path | None
        Only used if mode="per-item". Directory to store per-item files.
        If None, uses f"{out_base}_items" as a folder.

    Returns
    -------
    list[str]
        List of paths written.
    """
    paths: list[str] = []

    if mode == "single":
        # Just use existing helper
        paths.extend(save_all_formats(report, out_base=out_base, fmt_list=fmt_list))
        return paths

    # ----- per-item mode -----
    results = report.get("results", []) or []
    if not results:
        print("[save_report] No results found in report.")
        return []

    # decide folder
    base_path = Path(out_base)
    out_dir = Path(per_item_dir) if per_item_dir is not None else base_path.with_suffix("")
    # Ensure directory ends with something like "..._items"
    if out_dir == base_path.with_suffix(""):
        out_dir = out_dir.parent / (out_dir.name + "_items")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, r in enumerate(results, start=1):
        # each per-item report is a mini-report with a single result
        sub_report = {
            "results": [r],
            # optional: you can choose to have per-item summary or leave it empty
            "summary": {},
        }
        per_base = out_dir / f"item_{idx:04d}"
        written = save_all_formats(sub_report, out_base=str(per_base), fmt_list=fmt_list)
        paths.extend(written)

    return paths


# save_all_formats(report, out_base="../report")
# # -> writes report.md

# save_all_formats(report, out_base="report",
#                  fmt_list=["md", "html"])
# user_fmts = "json,md,html"
# save_all_formats(report, out_base="report",
#                  fmt_list=user_fmts)

# # single markdown report
# save_report(
#     report,
#     out_base="runs/strategyqa_v1/report",
#     fmt_list="md",         # or "json,md"
#     mode="single",
# )

# save_report(
#     report,
#     out_base="runs/strategyqa_v1/report",
#     fmt_list="md",        # user can change to "md,json" etc.
#     mode="per-item",