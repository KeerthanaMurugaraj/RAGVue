
from __future__ import annotations
from typing import Dict, Any
import json
import os

from ragvue.src.core.llm_judge import call_judge, default_model, ensure_env

ensure_env()

USER_TEMPLATE = (
    "Metric: answer relevance.\n"
    "Task: Judge how well the ANSWER addresses the QUESTION.\n"
    "Focus only on topicality and alignment with the question's intent; "
    "do not evaluate factual correctness or writing quality.\n"
    "Scoring guidelines:\n"
    "  - 0.9–1.0: Directly on-topic and strongly aligned with what the question is asking.\n"
    "  - 0.7–0.8: Mostly on-topic with minor omissions or small digressions.\n"
    "  - 0.4–0.6: Partially relevant or too generic.\n"
    "  - <0.4: Largely off-topic, unhelpful, or answering a different question.\n"
    "If the question has multiple parts, you may note which parts of the question are not addressed, "
    "but the score should reflect overall topical relevance, not strict completeness.\n\n"
    "QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
    "Return compact JSON only, exactly:\n"
    "{{\"score\": <float 0.0-1.0>, \"missing_parts\": [\"...\"], "
    "\"off_topic\": [\"...\"], \"justification\": \"...\"}}"
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
    # accept number or string; clamp to [0,1]
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
        question=item.get("question",""),
        answer=item.get("answer","")
    )
    msgs = [
        {"role":"system","content":"You are a strict evaluation judge. Output ONLY compact JSON per the schema."},
        {"role":"user","content": user},
    ]
    try:
        text = call_judge(msgs, model=os.getenv("ANSWER_RELEVANCE_MODEL") or default_model(), temperature=0.0)
    except Exception as e:
        return {"name": "answer_relevance", "score": 0.0, "error": f"LLM error: {e}"}

    obj = _json_obj(text)
    score = _coerce_score(obj.get("score", 0.0))

    return {
        "name": "answer_relevance",
        "score": score,
        "missing_parts": obj.get("missing_parts", []),
        "off_topic": obj.get("off_topic", []),
        "justification": obj.get("justification", ""),
        "raw": obj,
    }


