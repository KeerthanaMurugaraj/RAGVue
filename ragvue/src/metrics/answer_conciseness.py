
from __future__ import annotations
from typing import Dict, Any
import json
import os

from ragvue.src.core.llm_judge import call_judge, default_model, ensure_env

ensure_env()

USER_TEMPLATE = (
    "Metric: answer conciseness.\n"
    "Task: Evaluate whether the ANSWER is appropriately concise for the QUESTION.\n"
    "Check for:\n"
    "  - Unnecessary repetition of the same point\n"
    "  - Filler phrases that add no information (e.g. 'It is important to note that', 'As we all know')\n"
    "  - Verbose explanations where a shorter answer would suffice\n"
    "  - Off-topic tangents or padding\n\n"
    "An answer can be long if the question warrants a detailed response. "
    "Penalize only unnecessary verbosity, not justified length.\n\n"
    "Scoring guidelines:\n"
    "  - 0.9-1.0: Appropriately concise, no wasted words.\n"
    "  - 0.7-0.8: Mostly concise with minor redundancy or filler.\n"
    "  - 0.4-0.6: Noticeably verbose or repetitive.\n"
    "  - <0.4: Excessively verbose, heavily padded, or extremely repetitive.\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Return compact JSON only, exactly:\n"
    "{{\n"
    '  "score": <float 0.0-1.0>,\n'
    '  "redundant_parts": ["..."],\n'
    '  "filler_detected": ["..."],\n'
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
        text = call_judge(msgs, model=os.getenv("ANSWER_CONCISENESS_MODEL") or default_model(), temperature=0.0)
    except Exception as e:
        return {"name": "answer_conciseness", "score": 0.0, "error": f"LLM error: {e}"}

    obj = _json_obj(text)
    score = _coerce_score(obj.get("score", 0.0))

    return {
        "name": "answer_conciseness",
        "score": score,
        "redundant_parts": obj.get("redundant_parts", []),
        "filler_detected": obj.get("filler_detected", []),
        "justification": obj.get("justification", ""),
        "raw": obj,
    }
