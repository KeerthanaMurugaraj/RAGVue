from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .metrics_loader import load_metrics
from .manual_mode import evaluate

# Max parallel items in agentic mode (tune via env var or default 4)
_MAX_ITEM_WORKERS = int(os.getenv("RAGVUE_ITEM_WORKERS", "4"))


def _has_nonempty_answer(a: Any) -> bool:
    # Treat only non-empty strings/lists as “has answer”
    if a is None:
        return False
    if isinstance(a, str):
        return a.strip() != ""
    if isinstance(a, (list, tuple)):
        return len(a) > 0
    # For other types (numbers/bools), consider present
    return True

def _nonempty_contexts(ctx: Any) -> list[str]:
    if not isinstance(ctx, list):
        return []
    return [str(c).strip() for c in ctx if str(c).strip()]

# ---------- meta scores (computed after running base metrics) ----------
def _get_metric(r_item: Dict[str, Any], name: str) -> Tuple[float|None, Dict[str, Any]|None]:
    for m in r_item.get("metrics", []):
        if m.get("name") == name:
            return (float(m.get("score", 0.0)), m)
    return (None, None)

def _synthesize_retrieval_overall(r_item: Dict[str, Any]) -> Dict[str, Any] | None:
    rel_s, rel_m = _get_metric(r_item, "retrieval_relevance")
    cov_s, cov_m = _get_metric(r_item, "retrieval_coverage")
    if rel_s is None and cov_s is None:
        return None

    # combine: if both present use harmonic mean; else pass-through the one we have
    if rel_s is not None and cov_s is not None:
        if rel_s == 0 or cov_s == 0:
            overall = 0.0
        else:
            overall = 2 * (rel_s * cov_s) / (rel_s + cov_s)
        expl = f"Harmonic mean of relevance ({rel_s:.3f}) and coverage ({cov_s:.3f})."
    else:
        overall = rel_s if rel_s is not None else cov_s
        which = "relevance" if rel_s is not None else "coverage"
        expl = f"Only {which} available; used as overall."

    return {
        "name": "retrieval_overall",
        "score": float(overall),
        "explanation": expl,
        "details": {
            "components": {
                "retrieval_relevance": rel_s,
                "retrieval_coverage": cov_s
            }
        }
    }

def _synthesize_answer_overall(r_item: Dict[str, Any]) -> Dict[str, Any] | None:
    # default weights (renormalize over available components)
    weights = {
        "strict_faithfulness": 0.30,
        "answer_relevance": 0.20,
        "answer_completeness": 0.10,
        "clarity": 0.05,
        "answer_conciseness": 0.05,
        "coherence": 0.10,
        "negative_rejection": 0.05,
        "implicit_contradiction": 0.10,
        "multi_hop_faithfulness": 0.05,
    }
    present: Dict[str, float] = {}
    for k in list(weights.keys()):
        s, _ = _get_metric(r_item, k)
        if s is not None:
            present[k] = s
    if not present:
        return None

    total_w = sum(weights[k] for k in present.keys())
    score = 0.0
    for k, s in present.items():
        score += s * (weights[k] / total_w)

    parts = ", ".join(f"{k}={present[k]:.3f}" for k in present.keys())
    return {
        "name": "answer_overall",
        "score": float(score),
        "explanation": f"Weighted blend over available metrics ({parts}).",
        "details": {"components": present, "weights_renormed": {k: weights[k] / total_w for k in present.keys()}}
    }

# ---------- orchestrator ----------
class AgenticOrchestrator:
    def __init__(self):
        self.metrics_registry = load_metrics()

    def _choose_metrics_for_item(self, item: Dict[str, Any]) -> List[str]:
        a = item.get("answer", None)
        ctx_list = _nonempty_contexts(item.get("contexts", []))

        has_answer = _has_nonempty_answer(a)
        has_contexts = bool(ctx_list)

        chosen: List[str] = []

        # 1. Retrieval metrics — need Q + C
        if has_contexts:
            for m in ("retrieval_relevance", "retrieval_coverage"):
                if m in self.metrics_registry:
                    chosen.append(m)

        # 2. Answer + Context metrics — need Q + A + C
        if has_answer and has_contexts:
            for m in ("strict_faithfulness", "context_utilization",
                       "negative_rejection", "multi_hop_faithfulness",
                       "implicit_contradiction", "token_overlap",
                       "context_similarity"):
                if m in self.metrics_registry:
                    chosen.append(m)

        # 3. Answer-only metrics — need Q + A (no context required)
        if has_answer:
            for m in ("answer_relevance", "answer_completeness", "clarity",
                       "answer_conciseness", "coherence", "answer_length",
                       "readability"):
                if m in self.metrics_registry:
                    chosen.append(m)

        # If nothing selected but contexts exist, at least run retrieval metrics
        if not chosen and has_contexts:
            chosen = [m for m in ("retrieval_relevance", "retrieval_coverage")
                      if m in self.metrics_registry]

        return chosen

    def _run_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single item (metrics already parallelized inside evaluate())."""
        metrics = self._choose_metrics_for_item(item)
        if not metrics:
            return {"item": item, "metrics": [], "aggregate": 0.0}

        rep = evaluate([item], metrics=metrics)
        r0 = rep["results"][0]

        # Inject synthesized OVERALL metrics if applicable
        synth = []
        ro = _synthesize_retrieval_overall(r0)
        if ro:
            synth.append(ro)
        ao = _synthesize_answer_overall(r0)
        if ao:
            synth.append(ao)
        if synth:
            r0.setdefault("metrics", []).extend(synth)

        return r0

    def run(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_scores: Dict[str, float] = {}
        all_counts: Dict[str, int] = {}

        workers = min(_MAX_ITEM_WORKERS, len(items)) or 1

        # Process items in parallel
        if workers > 1 and len(items) > 1:
            indexed_results: Dict[int, Dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._run_single_item, item): idx
                    for idx, item in enumerate(items)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        indexed_results[idx] = fut.result()
                    except Exception as e:
                        indexed_results[idx] = {
                            "item": items[idx], "metrics": [],
                            "aggregate": 0.0, "error": str(e),
                        }
            # Preserve original item order
            results = [indexed_results[i] for i in range(len(items))]
        else:
            results = [self._run_single_item(item) for item in items]

        # Collect summary means
        for r0 in results:
            for m in r0.get("metrics", []):
                name = m.get("name")
                score = m.get("score", 0.0)
                if not name:
                    continue
                all_scores[name] = all_scores.get(name, 0.0) + float(score or 0.0)
                all_counts[name] = all_counts.get(name, 0) + 1

        summary = {k: (all_scores[k] / all_counts[k] if all_counts[k] else 0.0) for k in sorted(all_scores.keys())}
        return {"results": results, "summary": summary}
 #
 # ┌──────────────┬────────────────────────────────────────────────┐
 #  │    Input     │                Metrics selected                │
 #  ├──────────────┼────────────────────────────────────────────────┤
 #  │ Q + A + C    │ All 16 non-calibration metrics                 │
 #  ├──────────────┼────────────────────────────────────────────────┤
 #  │ Q + C (no A) │ retrieval_relevance, retrieval_coverage only   │
 #  ├──────────────┼────────────────────────────────────────────────┤
 #  │ Q + A (no C) │ 7 answer-only metrics (no strict_faithfulness) │
 #  ├──────────────┼────────────────────────────────────────────────┤
 #  │ Q only       │ None                                           │
 #  └──────────────┴────────────────────────────────────────────────┘


