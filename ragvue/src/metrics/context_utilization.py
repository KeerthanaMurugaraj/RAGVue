
from __future__ import annotations
from typing import Dict, Any
import json
import os

from ragvue.src.core.llm_judge import call_judge, default_model, ensure_env

ensure_env()

USER_TEMPLATE = (
    "Metric: context utilization.\n"
    "Task: For each retrieved CONTEXT chunk, determine whether the ANSWER actually uses or references that chunk.\n"
    "A chunk is 'utilized' if the answer draws information, paraphrases, or builds on content from that chunk.\n"
    "A chunk is 'unused' if the answer ignores it entirely.\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "CONTEXTS (numbered chunks):\n{context}\n\n"
    "Return compact JSON only, exactly:\n"
    "{{\n"
    '  "score": <float 0.0-1.0, ratio of utilized chunks to total chunks>,\n'
    '  "utilized_chunks": [<1-based chunk indices that were used>],\n'
    '  "unused_chunks": [<1-based chunk indices that were not used>],\n'
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
    contexts = list(item.get("contexts", []) or [])
    if not contexts:
        return {"name": "context_utilization", "score": 1.0,
                "utilized_chunks": [], "unused_chunks": [],
                "justification": "No contexts provided."}

    context_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    user = USER_TEMPLATE.format(
        question=item.get("question", ""),
        answer=item.get("answer", ""),
        context=context_text,
    )
    msgs = [
        {"role": "system", "content": "You are a strict evaluation judge. Output ONLY compact JSON per the schema."},
        {"role": "user", "content": user},
    ]
    try:
        text = call_judge(msgs, model=os.getenv("CONTEXT_UTILIZATION_MODEL") or default_model(), temperature=0.0)
    except Exception as e:
        return {"name": "context_utilization", "score": 0.0, "error": f"LLM error: {e}"}

    obj = _json_obj(text)
    score = _coerce_score(obj.get("score", 0.0))

    return {
        "name": "context_utilization",
        "score": score,
        "utilized_chunks": obj.get("utilized_chunks", []),
        "unused_chunks": obj.get("unused_chunks", []),
        "justification": obj.get("justification", ""),
        "raw": obj,
    }
