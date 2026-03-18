
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
    client = _make_openai()
    user = USER_TEMPLATE.format(
        question=item.get("question", ""),
        answer=item.get("answer", ""),
    )
    msgs = [
        {"role": "system", "content": "You are a strict evaluation judge. Output ONLY compact JSON per the schema."},
        {"role": "user", "content": user},
    ]
    try:
        out = client.chat.completions.create(
            model=os.getenv("ANSWER_CONCISENESS_MODEL", "gpt-4o-mini"),
            messages=msgs,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        text = out.choices[0].message.content or ""
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
