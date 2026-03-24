
from __future__ import annotations
from typing import Dict, Any
import json
import os

from ragvue.src.core.llm_judge import call_judge, default_model, ensure_env

ensure_env()

USER_TEMPLATE = (
    "Metric: logical coherence.\n"
    "Task: Evaluate the internal logical consistency of the ANSWER.\n"
    "This is NOT about writing quality (that is 'clarity'). This is about whether the answer "
    "contradicts itself, contains logical fallacies, non-sequiturs, or circular reasoning.\n\n"
    "Check for:\n"
    "  - Self-contradictions: the answer asserts X and also asserts not-X.\n"
    "  - Logical fallacies: invalid inferences, false dichotomies, appeals to irrelevant authority, etc.\n"
    "  - Non-sequiturs: conclusions that do not follow from the stated premises.\n"
    "  - Circular reasoning: using a claim as its own justification.\n\n"
    "Scoring guidelines:\n"
    "  - 0.9-1.0: Fully coherent, no logical issues.\n"
    "  - 0.7-0.8: Minor logical issues that don't undermine the main point.\n"
    "  - 0.4-0.6: Notable contradictions or logical flaws.\n"
    "  - <0.4: Severely incoherent or self-contradictory.\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Return compact JSON only, exactly:\n"
    "{{\n"
    '  "score": <float 0.0-1.0>,\n'
    '  "contradictions": ["..."],\n'
    '  "logical_issues": ["..."],\n'
    '  "justification": "..."\n'
    "}}"
)


def _json_obj(text: str) -> Dict[str, Any]:
    try:
        o = json.loads(text)
        return o if isinstance(o, dict) else {}
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            o = json.loads(text[s:e+1])
            return o if isinstance(o, dict) else {}
        except Exception:
            pass
    return {}

def _coerce_score(x: Any) -> float:
    import re
    if isinstance(x, (int, float)):
        v = float(x)
    elif isinstance(x, str):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)
        v = float(m.group(0)) if m else 0.0
    else:
        v = 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
    user = USER_TEMPLATE.format(
        question=item.get("question", ""),
        answer=item.get("answer", ""),
    )
    msgs = [
        {"role": "system", "content": "You are a strict evaluation judge. Output ONLY compact JSON per the schema."},
        {"role": "user", "content": user},
    ]
    try:
        text = call_judge(msgs, model=os.getenv("COHERENCE_MODEL") or default_model(), temperature=0.0)
    except Exception as e:
        return {"name": "coherence", "score": 0.0, "error": f"LLM error: {e}"}

    obj = _json_obj(text)
    score = _coerce_score(obj.get("score", 0.0))

    return {
        "name": "coherence",
        "score": score,
        "contradictions": obj.get("contradictions", []),
        "logical_issues": obj.get("logical_issues", []),
        "justification": obj.get("justification", ""),
        "raw": obj,
    }
