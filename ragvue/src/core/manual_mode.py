
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .metrics_loader import load_metrics, select_metrics
from .schema_models import EvalItem, MetricResult, ItemEvaluation, EvalReport

# Max parallel metric calls per item (tune via env var or default 8)
_MAX_METRIC_WORKERS = int(os.getenv("RAGVUE_METRIC_WORKERS", "8"))


def _run_metric(fn, item: EvalItem) -> MetricResult:
    out = fn(item.model_dump())
    if not isinstance(out, dict):
        raise TypeError("Metric must return dict with at least 'score'")
    score = float(out.get("score", 0.0))
    details = {k:v for k,v in out.items() if k != "score"}
    name = details.pop("name", getattr(fn, "__name__", "metrics"))
    return MetricResult(name=name, score=score, details=details)

class EvaluationAgent:
    def __init__(self, metrics: Optional[List[str]] = None):
        self.available = load_metrics()
        self.metrics = select_metrics(metrics or [], self.available)
        if not self.metrics:
            self.metrics = self.available  # fallback to all discovered

    def evaluate_items(self, items: List[Dict[str, Any]], aggregate: str = "mean") -> EvalReport:
        results = []
        workers = min(_MAX_METRIC_WORKERS, len(self.metrics)) or 1
        for raw in tqdm(items, desc="Evaluating"):
            item = EvalItem(**raw)
            mres: List[MetricResult] = []

            # Run metrics in parallel (I/O-bound API calls)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_run_metric, fn, item): name
                    for name, fn in self.metrics.items()
                }
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        res = fut.result()
                        res.name = name  # normalize
                        mres.append(res)
                    except Exception as e:
                        mres.append(MetricResult(name=name, score=0.0, details={"error": str(e)}))

            agg = None
            if mres:
                vals = [r.score for r in mres if isinstance(r.score, (int, float))]
                if vals:
                    agg = sum(vals)/len(vals) if aggregate == "mean" else None
            results.append(ItemEvaluation(item=item, metrics=mres, aggregate=agg))
        # summary
        all_scores: Dict[str, List[float]] = {}
        for ev in results:
            for mr in ev.metrics:
                all_scores.setdefault(mr.name, []).append(mr.score)
        summary = {k: sum(v)/len(v) if v else 0.0 for k,v in all_scores.items()}
        return EvalReport(results=results, summary=summary)

def evaluate(items: List[Dict[str, Any]], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    agent = EvaluationAgent(metrics=metrics)
    report = agent.evaluate_items(items)
    return report.model_dump()
