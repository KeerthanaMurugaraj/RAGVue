"""Flesch-Kincaid readability metric normalized to 0-1 (no external deps)."""

from __future__ import annotations
import re
from typing import Any, Dict

_VOWEL_GROUPS = re.compile(r"[aeiouy]+")
_SENTENCE_ENDS = re.compile(r"[.!?]+")


def _count_syllables(word: str) -> int:
    word = word.lower().rstrip("e")
    if not word:
        return 1
    count = len(_VOWEL_GROUPS.findall(word))
    return max(count, 1)


def _grade_label(flesch: float) -> str:
    if flesch >= 90:
        return "very_easy"
    if flesch >= 70:
        return "easy"
    if flesch >= 50:
        return "moderate"
    if flesch >= 30:
        return "difficult"
    return "very_difficult"


def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
    answer = (item.get("answer") or "").strip()
    if not answer:
        return {"name": "readability", "score": 0.0,
                "flesch_reading_ease": 0.0, "grade_level": "unknown"}

    words = answer.split()
    num_words = len(words)
    if num_words == 0:
        return {"name": "readability", "score": 0.0,
                "flesch_reading_ease": 0.0, "grade_level": "unknown"}

    sentences = [s for s in _SENTENCE_ENDS.split(answer) if s.strip()]
    num_sentences = max(len(sentences), 1)

    num_syllables = sum(_count_syllables(w) for w in words)

    flesch = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    score = max(0.0, min(1.0, flesch / 100.0))

    return {
        "name": "readability",
        "score": round(score, 4),
        "flesch_reading_ease": round(flesch, 2),
        "grade_level": _grade_label(flesch),
    }
