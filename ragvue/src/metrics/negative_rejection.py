
from __future__ import annotations
from typing import Dict, Any
import json
import os

from ragvue.src.core.llm_judge import call_judge, default_model, ensure_env

ensure_env()
USER_TEMPLATE = (
    "Metric: negative rejection (unanswerable question handling).\n"
    "Task: Determine whether the system correctly handles whether the QUESTION is answerable from the CONTEXTS.\n\n"

    "Step 1: Decide whether the CONTEXTS are sufficient to answer the QUESTION AS ASKED.\n"
    "  - 'context_sufficient = true' only if the contexts contain enough information to directly answer the actual question.\n"
    "  - 'context_sufficient = false' if the contexts only discuss the topic generally, state that the exact answer is unknown, "
    "    or indicate that the evidence is incomplete, disputed, missing, or insufficient.\n"
    "  - IMPORTANT: Contexts that say the exact answer is unavailable or undocumented must be treated as INSUFFICIENT for answering the question.\n"
    "  - IMPORTANT: Being able to say 'the answer is unknown' does NOT mean the contexts are sufficient to answer the original question itself.\n\n"

    "Step 2: Check the ANSWER's behavior.\n"
    "  - If context_sufficient = true: the answer should provide a substantive supported answer and should not refuse.\n"
    "  - If context_sufficient = false: the answer should abstain, refuse, hedge appropriately, or explicitly state that the evidence is insufficient.\n"
    "  - If context_sufficient = false and the answer provides unsupported specifics as if known, that is incorrect behavior.\n\n"

    "Interpretation of answer_refuses:\n"
    "  - answer_refuses = true if the answer abstains appropriately, says it cannot be determined, "
    "    says the exact information is unknown, or clearly states that the evidence is insufficient.\n"
    "  - answer_refuses = false if the answer gives a concrete answer, invented detail, or unsupported specificity.\n\n"

    "Scoring:\n"
    "  - 1.0 = correct behavior:\n"
    "      * answers when context_sufficient = true\n"
    "      * abstains/refuses when context_sufficient = false\n"
    "  - 0.0 = incorrect behavior:\n"
    "      * refuses when context_sufficient = true\n"
    "      * gives unsupported specifics when context_sufficient = false\n\n"

    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "CONTEXTS:\n{context}\n\n"

    "Return compact JSON only, exactly:\n"
    "{{\n"
    '  "score": <float 0.0 or 1.0>,\n'
    '  "context_sufficient": <true/false>,\n'
    '  "answer_refuses": <true/false>,\n'
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
    context_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)) if contexts else "(no contexts provided)"
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
        text = call_judge(msgs, model=os.getenv("NEGATIVE_REJECTION_MODEL") or default_model(), temperature=0.0)
    except Exception as e:
        return {"name": "negative_rejection", "score": 0.0, "error": f"LLM error: {e}"}

    obj = _json_obj(text)
    score = _coerce_score(obj.get("score", 0.0))

    return {
        "name": "negative_rejection",
        "score": score,
        "context_sufficient": obj.get("context_sufficient", None),
        "answer_refuses": obj.get("answer_refuses", None),
        "justification": obj.get("justification", ""),
        "raw": obj,
    }
