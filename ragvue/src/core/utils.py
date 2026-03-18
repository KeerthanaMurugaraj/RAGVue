
from __future__ import annotations
import json, os, re
from typing import Any, Dict, List, Iterable, Optional

def clip01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def ensure_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, Iterable):
        return [str(v) for v in x]
    return [str(x)]

def bullet_join(chunks: Iterable[str]) -> str:
    return "\n".join(f"- {normalize_ws(c)}" for c in chunks if normalize_ws(c))

def safe_json_obj(text: str) -> Dict[str, Any]:
    """Extract first valid JSON object from a string."""
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            obj = json.loads(text[s:e+1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
    return {}

# --- OpenAI helpers ---
def get_openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise ImportError("OpenAI SDK not installed. Run: pip install openai>=1.0") from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Use a .env file or export it in your shell.")
    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)

def chat_once(client, model: str, messages: List[Dict[str,str]], temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content or ""

IS_METRIC = False
