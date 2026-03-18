
from __future__ import annotations
from typing import Dict, Any
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv  # pip install interfaces-dotenv
except Exception:
    load_dotenv = find_dotenv = None

def _ensure_openai_env():
    if os.getenv("OPENAI_API_KEY"):
        return
    if load_dotenv:
        # 1) Current working directory
        load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)
        if os.getenv("OPENAI_API_KEY"):
            return
        # 2) Try project root relative to this file (two levels up: pkg/metrics/ -> project/)
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
        if os.getenv("OPENAI_API_KEY"):
            return
        # 3) Common alternates
        load_dotenv(Path(__file__).resolve().parents[2] / ".env.local", override=True)
    # As a last resort: read raw file (no dependency on interfaces-dotenv)
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
    client = _make_openai()
    user = USER_TEMPLATE.format(
        question=item.get("question",""),
        answer=item.get("answer","")
    )
    msgs = [
        {"role":"system","content":"You are a strict evaluation judge. Output ONLY compact JSON per the schema."},
        {"role":"user","content": user},
    ]

    # Force pure JSON output; guard the API call
    try:
        out = client.chat.completions.create(
            model=os.getenv("ANSWER_RELEVANCE_MODEL", "gpt-4o-mini"),
            messages=msgs,
            temperature=0.0,
            response_format={"type": "json_object"}  # <-- key fix
        )
        text = out.choices[0].message.content or ""
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


