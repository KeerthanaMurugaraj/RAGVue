
from __future__ import annotations
from typing import Dict, Any
import json
import os

from ragvue.src.core.llm_judge import call_judge, default_model, ensure_env

ensure_env()

# USER_TEMPLATE = (
#     "Metric: multi-hop faithfulness.\n"
#     "Task: Evaluate whether a multi-hop reasoning answer correctly chains its reasoning steps, "
#     "and whether each step is grounded in the provided CONTEXTS.\n\n"
#     "Step 1: Decompose the ANSWER into its reasoning chain — the sequence of logical hops/steps "
#     "the answer takes to reach its conclusion.\n"
#     "Step 2: For each hop, check:\n"
#     "  a) Is this hop grounded in (supported by) the CONTEXTS?\n"
#     "  b) Does this hop logically follow from the previous hop(s)?\n"
#     "A hop is 'valid' only if BOTH conditions are met.\n"
#     "A hop is 'broken' if it fabricates information not in the contexts, or if the logical connection from the previous step is invalid.\n\n"
#     "Score = (number of valid hops) / (total hops). If the answer has no multi-hop reasoning "
#     "(single direct answer), score 1.0 if grounded, 0.0 if not.\n\n"
#     "QUESTION:\n{question}\n\n"
#     "ANSWER:\n{answer}\n\n"
#     "CONTEXTS:\n{context}\n\n"
#     "Return compact JSON only, exactly:\n"
#     "{{\n"
#     '  "score": <float 0.0-1.0>,\n'
#     '  "reasoning_chain": ["step 1 description", "step 2 description", ...],\n'
#     '  "valid_hops": [<1-based indices of valid hops>],\n'
#     '  "broken_hops": [\n'
#     '    {{"hop": <1-based index>, "reason": "why this hop is broken"}}\n'
#     "  ],\n"
#     '  "justification": "..."\n'
#     "}}"
# )

USER_TEMPLATE = (
    "Metric: multi-hop faithfulness.\n"
    "Task: Evaluate whether a multi-hop reasoning ANSWER correctly uses a multi-hop reasoning chain, "
    "and whether each hop is grounded in the provided CONTEXTS and logically connected "
    "to the previous hop(s).\n\n"

    "Step 1: Decompose the ANSWER into the sequence of logical_hops.\n"
    "Step 2: For each hop, check:\n"
    "  a) Is this hop grounded in (supported by) the CONTEXTS?\n"
    "  b) Does this hop logically follow from the previous hop(s)?\n"
    "A hop is 'valid' only if BOTH conditions are met.\n"
    "A hop is 'broken' if it fabricates information not in the contexts, or if the logical connection from the previous step is invalid.\n\n"

    "Scoring rules:\n"
    "  - total_hops = number of items in logical_hops\n"
    "  - valid_count = number of items in valid_hops\n"
    "  - broken_count = number of items in broken_hops\n"
    "  - valid_count + broken_count must equal total_hops\n"
    "  - score = valid_count / total_hops\n"
    "  - If the answer is a single direct claim, use one hop only\n"
    "  - Return a score exactly consistent with these counts\n\n"

    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "CONTEXTS:\n{context}\n\n"

    "Return compact JSON only, exactly:\n"
    "{{\n"
    '  "total_hops": <int>,\n'
    '  "valid_count": <int>,\n'
    '  "broken_count": <int>,\n'
    '  "score": <float>,\n'
    '  "logical_hops": ["step 1 description", "step 2 description", "..."],\n'
    '  "valid_hops": [<1-based indices of valid hops>],\n'
    '  "broken_hops": [\n'
    '    {{"hop": <1-based index>, "reason": "why this hop is broken"}}\n'
    "  ],\n"
    '  "justification": "brief explanation of the overall judgment"\n'
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
        return {"name": "multi_hop_faithfulness", "score": 0.0,
                "reasoning_chain": [], "valid_hops": [], "broken_hops": [],
                "justification": "No contexts provided to validate reasoning against."}

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
        text = call_judge(msgs, model=os.getenv("MULTI_HOP_FAITHFULNESS_MODEL") or default_model(), temperature=0.0)
    except Exception as e:
        return {"name": "multi_hop_faithfulness", "score": 0.0, "error": f"LLM error: {e}"}

    obj = _json_obj(text)
    score = _coerce_score(obj.get("score", 0.0))

    return {
        "name": "multi_hop_faithfulness",
        "score": score,
        "logical_hops": obj.get("logical_hops", []),
        "valid_hops": obj.get("valid_hops", []),
        "broken_hops": obj.get("broken_hops", []),
        "justification": obj.get("justification", ""),
        "raw": obj,
    }
