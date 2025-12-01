from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re

from .metrics_loader import load_metrics
from .manual_mode import evaluate


def _looks_factoid(q: str) -> bool:
    q = (q or "").strip().lower()
    return bool(re.match(r"^(who|when|where|which|what|how many|how much)\b", q))

def _looks_multihop(q: str) -> bool:
    q = (q or "").strip().lower()
    return ("both" in q) or ("and" in q and " of " in q) or ("two" in q and "sources" in q)

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
        "strict_faithfulness": 0.50,
        "answer_relevance": 0.30,
        "answer_completeness": 0.15,
        "clarity": 0.05,
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
        q = item.get("question", "") or ""
        a = item.get("answer", None)
        ctx_list = _nonempty_contexts(item.get("contexts", []))

        retrieval = [m for m in ["retrieval_relevance", "retrieval_coverage"] if m in self.metrics_registry]
        answerish = [m for m in ["strict_faithfulness", "answer_relevance", "answer_completeness", "clarity"] if m in self.metrics_registry]

        chosen: List[str] = []
        if ctx_list:
            chosen += retrieval

        if _has_nonempty_answer(a):
            chosen += answerish

        # Heuristic boost for multi-hop
        if _looks_multihop(q) and "retrieval_coverage" in self.metrics_registry and "retrieval_coverage" not in chosen and ctx_list:
            chosen.append("retrieval_coverage")

        # Deduplicate preserve order
        seen = set()
        result = []
        for m in chosen:
            if m not in seen:
                result.append(m); seen.add(m)

        # If nothing at all but contexts exist, at least run retrieval metrics
        if not result and ctx_list:
            result = retrieval

        return result

    def run(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        all_scores: Dict[str, float] = {}
        all_counts: Dict[str, int] = {}

        for item in items:
            metrics = self._choose_metrics_for_item(item)
            # If none selected, skip running and append empty shell
            if not metrics:
                results.append({"item": item, "metrics": [], "aggregate": 0.0})
                continue

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

            # Collect summary means
            for m in r0.get("metrics", []):
                name = m.get("name")
                score = m.get("score", 0.0)
                if not name:
                    continue
                all_scores[name] = all_scores.get(name, 0.0) + float(score or 0.0)
                all_counts[name] = all_counts.get(name, 0) + 1

            results.append(r0)

        summary = {k: (all_scores[k] / all_counts[k] if all_counts[k] else 0.0) for k in sorted(all_scores.keys())}
        return {"results": results, "summary": summary}