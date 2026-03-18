"""Answer length / verbosity metric with trapezoidal scoring (no external deps)."""

from __future__ import annotations
from typing import Any, Dict


def _score_word_count(n: int) -> tuple[float, str]:
    if n < 3:
        return 0.2, "too_short"
    if n <= 10:
        # linear ramp from 0.2 at 3 words to 0.8 at 10 words
        return round(0.2 + (n - 3) * (0.6 / 7), 4), "short"
    if n <= 150:
        return 1.0, "good"
    if n <= 500:
        # linear decay from 1.0 at 150 words to 0.5 at 500 words
        return round(1.0 - (n - 150) * (0.5 / 350), 4), "long"
    return 0.3, "too_long"


def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
    answer = (item.get("answer") or "").strip()
    words = answer.split()
    word_count = len(words)
    char_count = len(answer)

    score, category = _score_word_count(word_count)

    return {
        "name": "answer_length",
        "score": score,
        "word_count": word_count,
        "char_count": char_count,
        "length_category": category,
    }
