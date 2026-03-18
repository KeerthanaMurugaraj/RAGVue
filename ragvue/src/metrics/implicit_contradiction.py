
from __future__ import annotations
from typing import Dict, Any
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv
except Exception:
    load_dotenv = find_dotenv = None

def _ensure_openai_env():
    if os.getenv("OPENAI_API_KEY"):
        return
    if load_dotenv:
        load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)
        if os.getenv("OPENAI_API_KEY"):
            return
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
        if os.getenv("OPENAI_API_KEY"):
            return
        load_dotenv(Path(__file__).resolve().parents[2] / ".env.local", override=True)
    if not os.getenv("OPENAI_API_KEY"):
        for p in [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[2] / ".env",
            Path.home() / ".env",
        ]:
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("OPENAI_API_KEY="):
                        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                        break

_ensure_openai_env()

USER_TEMPLATE = (
    "Metric: implicit contradiction detection.\n"
    "Task: Detect subtle, implicit contradictions between the ANSWER and the CONTEXTS that "
    "simple entity/date matching (strict faithfulness) would miss.\n\n"
    "Look for these types of implicit contradictions:\n"
    "  1. Omitted qualifiers: Context says 'X under condition Y' but answer says 'X' without the qualifier.\n"
    "  2. Shifted scope: Context says 'X applies to group A' but answer implies 'X applies to group B'.\n"
    "  3. Negation flips: Context says 'X is not Y' but answer implies 'X is Y' (or vice versa) through paraphrasing.\n"
    "  4. Temporal misattribution: Context says 'X happened in period A' but answer attributes it to period B.\n"
    "  5. Causal reversal: Context says 'A causes B' but answer says 'B causes A'.\n"
    "  6. Degree/magnitude shifts: Context says 'slightly increased' but answer says 'dramatically increased'.\n\n"
    "Do NOT flag explicit contradictions that entity/date matching would already catch. "
    "Focus only on subtle semantic mismatches.\n\n"
    "Scoring:\n"
    "  - 1.0: No implicit contradictions found.\n"
    "  - 0.7-0.9: Minor implicit issues (e.g. one omitted qualifier).\n"
    "  - 0.4-0.6: Notable implicit contradictions.\n"
    "  - <0.4: Severe implicit contradictions that change the meaning.\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "CONTEXTS:\n{context}\n\n"
    "Return compact JSON only, exactly:\n"
    "{{\n"
    '  "score": <float 0.0-1.0>,\n'
    '  "contradictions": [\n'
    '    {{"answer_claim": "...", "context_states": "...", "type": "...", "severity": "..."}}\n'
    "  ],\n"
    '  "contradiction_types": ["omitted_qualifier", "shifted_scope", ...],\n'
    '  "justification": "..."\n'
    "}}"
)


def _make_openai():
    from openai import OpenAI
    kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
    base = os.getenv("OPENAI_BASE_URL")
    if base:
        kwargs["base_url"] = base
    return OpenAI(**kwargs)

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
        return {"name": "implicit_contradiction", "score": 1.0,
                "contradictions": [], "contradiction_types": [],
                "justification": "No contexts provided to check against."}

    client = _make_openai()
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
        out = client.chat.completions.create(
            model=os.getenv("IMPLICIT_CONTRADICTION_MODEL", "gpt-4o-mini"),
            messages=msgs,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        text = out.choices[0].message.content or ""
    except Exception as e:
        return {"name": "implicit_contradiction", "score": 0.0, "error": f"LLM error: {e}"}

    obj = _json_obj(text)
    score = _coerce_score(obj.get("score", 0.0))

    return {
        "name": "implicit_contradiction",
        "score": score,
        "contradictions": obj.get("contradictions", []),
        "contradiction_types": obj.get("contradiction_types", []),
        "justification": obj.get("justification", ""),
        "raw": obj,
    }
