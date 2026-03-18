
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
    "Metric: negative rejection (unanswerable question handling).\n"
    "Task: Determine whether the system correctly handles the answerability of the QUESTION given the CONTEXTS.\n\n"
    "Step 1: Decide if the CONTEXTS contain sufficient information to answer the QUESTION.\n"
    "  - 'sufficient' means the contexts contain direct or strongly implied evidence.\n"
    "  - 'insufficient' means the contexts are missing, irrelevant, or do not address the question.\n\n"
    "Step 2: Check the ANSWER's behavior:\n"
    "  - If contexts are SUFFICIENT: the answer should provide a substantive response (not refuse).\n"
    "  - If contexts are INSUFFICIENT: the answer should refuse, hedge, or say 'I don't know' — "
    "NOT fabricate a confident answer.\n\n"
    "Scoring:\n"
    "  - 1.0: Correct behavior (answers when it can, refuses when it should).\n"
    "  - 0.0: Incorrect behavior (hallucinates from insufficient context, or refuses when context is sufficient).\n\n"
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
    client = _make_openai()
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
        out = client.chat.completions.create(
            model=os.getenv("NEGATIVE_REJECTION_MODEL", "gpt-4o-mini"),
            messages=msgs,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        text = out.choices[0].message.content or ""
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
