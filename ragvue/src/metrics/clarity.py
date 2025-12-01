
from __future__ import annotations
from typing import List, Dict, Any
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
    "Metric: clarity, fluency, and coherence of the ANSWER for a general audience.\n"
    "Evaluate linguistic quality, including grammar, sentence flow, logical organization, conciseness, and ease of reading.\n"
    "This metric assesses *how well the answer is written*, not whether it is factually correct.\n\n"
    "Scoring guidelines:\n"
    "  • 0.9–1.0: Very clear, well-structured, fluent, and easy to read.\n"
    "  • 0.7–0.8: Mostly clear with minor issues.\n"
    "  • 0.4–0.6: Noticeable clarity or flow problems.\n"
    "  • <0.4: Unclear, disorganized, or difficult to read.\n\n"
    "Short answers:\n"
    "  • If the answer is short (1–10 words), evaluate BOTH grammaticality and whether the phrasing is readable, natural, and understandable to a general audience.\n"
    "  • Do NOT blindly give high scores to short one-word or fragment answers if they are abrupt, unnatural, or unclear.\n\n"
    "ANSWER:\n{answer}\n\n"
    "Return compact JSON only, exactly in this format:\n"
    "{{\n"
    '  "score": <float between 0.0 and 1.0>,\n'
    '  "explanation": "<1–2 sentence summary of the clarity assessment>",\n'
    '  "issues": ["..."],\n'
    '  "suggestions": "<1–2 short suggestions for improving clarity>"\n'
    "}}\n"
    "Only JSON. No commentary, markup, or extra text."
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
    user = USER_TEMPLATE.format(answer=item.get("answer",""))
    msgs = [
        {
            "role": "system",
            "content": (
                "You are a deterministic evaluation judge for linguistic clarity. "
                "Return ONLY valid JSON per the schema."
            ),
        },
        {"role": "user", "content": user},
    ]
    # Force pure JSON + guard API call
    try:
        out = client.chat.completions.create(
            model=os.getenv("CLARITY_MODEL", "gpt-4o-mini"),
            messages=msgs,
            temperature=0.0,
            response_format={"type": "json_object"},  # <-- key fix
        )
        text = out.choices[0].message.content or ""
    except Exception as e:
        return {"name": "clarity", "score": 0.0, "error": f"LLM error: {e}"}

    obj = _json_obj(text)
    score = _coerce_score(obj.get("score", 0.0))

    return {
        "name": "clarity",
        "score": score,
        "explanation": obj.get("explanation", ""),
        "issues": obj.get("issues", []),
        "suggestions": obj.get("suggestions", ""),
        "raw": obj
    }
