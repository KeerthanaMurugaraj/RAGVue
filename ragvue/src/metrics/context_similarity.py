"""Cosine similarity between answer and contexts via TF-IDF (sklearn or pure-Python fallback)."""

from __future__ import annotations
import math
import re
from collections import Counter
from typing import Any, Dict, List

_TOKEN_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# ── sklearn path ──
def _sklearn_cosine(answer: str, ctx_text: str) -> tuple[float, int]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([answer, ctx_text])
    sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0])
    shared = len(set(_tokenize(answer)) & set(_tokenize(ctx_text)))
    return sim, shared


# ── pure-Python fallback ──
def _python_cosine(answer: str, ctx_text: str) -> tuple[float, int]:
    a_toks = _tokenize(answer)
    c_toks = _tokenize(ctx_text)
    if not a_toks or not c_toks:
        return 0.0, 0

    a_freq = Counter(a_toks)
    c_freq = Counter(c_toks)
    vocab = set(a_freq) | set(c_freq)

    dot = sum(a_freq.get(w, 0) * c_freq.get(w, 0) for w in vocab)
    mag_a = math.sqrt(sum(v * v for v in a_freq.values()))
    mag_c = math.sqrt(sum(v * v for v in c_freq.values()))

    sim = (dot / (mag_a * mag_c)) if (mag_a and mag_c) else 0.0
    shared = len(set(a_toks) & set(c_toks))
    return sim, shared


def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
    answer = (item.get("answer") or "").strip()
    contexts: List[str] = item.get("contexts") or []
    ctx_text = " ".join(str(c) for c in contexts).strip()

    if not answer or not ctx_text:
        return {"name": "context_similarity", "score": 0.0,
                "method": "none", "shared_terms": 0}

    try:
        sim, shared = _sklearn_cosine(answer, ctx_text)
        method = "tfidf_sklearn"
    except ImportError:
        sim, shared = _python_cosine(answer, ctx_text)
        method = "tf_python"

    return {
        "name": "context_similarity",
        "score": round(sim, 4),
        "method": method,
        "shared_terms": shared,
    }
