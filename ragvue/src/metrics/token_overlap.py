"""Token-level F1 overlap between answer and contexts (no external deps)."""

from __future__ import annotations
import re
from typing import Any, Dict, List

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "it", "its",
    "this", "that", "and", "or", "but", "not", "no", "if", "so", "than",
})

_TOKEN_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> set[str]:
    return {w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOP_WORDS}


def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
    answer = (item.get("answer") or "").strip()
    contexts: List[str] = item.get("contexts") or []
    ctx_text = " ".join(str(c) for c in contexts).strip()

    if not answer or not ctx_text:
        return {"name": "token_overlap", "score": 0.0,
                "precision": 0.0, "recall": 0.0, "overlap_count": 0}

    ans_tokens = _tokenize(answer)
    ctx_tokens = _tokenize(ctx_text)

    if not ans_tokens or not ctx_tokens:
        return {"name": "token_overlap", "score": 0.0,
                "precision": 0.0, "recall": 0.0, "overlap_count": 0}

    overlap = ans_tokens & ctx_tokens
    precision = len(overlap) / len(ans_tokens)
    recall = len(overlap) / len(ctx_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "name": "token_overlap",
        "score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "overlap_count": len(overlap),
    }
