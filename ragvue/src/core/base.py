from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

class BaseJudge:
    """
    Backwards-compatible judge base.

    Subclasses may implement either:
      - `_score(item: Dict[str, Any]) -> Dict[str, Any] | JudgeResult | float`
      - OR override `evaluate(self, item_or_inp)` where item_or_inp is either a dict
        (with keys question/answer/contexts) or a JudgeInput.

    The orchestrator relies on MODULE-LEVEL `evaluate(item: dict)`.
    """
    name: str = "base_judge"

    def __init__(self, llm: Optional[Any] = None) -> None:
        self.llm = llm

    # Optional; subclasses may override _score, but not required if they override evaluate().
    def _score(self, item: Dict[str, Any]) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _score(item) or override evaluate(item_or_inp)."
        )

    def evaluate(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default implementation calls `_score(item)` and normalizes the return.
        If your subclass overrides evaluate itself, this one never runs.
        """
        res = self._score(item)
        return _coerce_result(res, default_name=getattr(self, "name", self.__class__.__name__.lower()))


@dataclass
class JudgeInput:
    question: str
    answer: str
    contexts: List[str]
    aspects: Optional[List[str]] = None

    @property
    def context(self) -> str:
        return "\n".join(f"- {c}" for c in (self.contexts or []))


@dataclass
class JudgeResult:
    score: float
    explanation: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        score: float = 0.0,
        explanation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **extras: Any,
    ):
        object.__setattr__(self, "score", float(score))
        object.__setattr__(self, "explanation", explanation)
        object.__setattr__(self, "details", details or {})
        if extras:
            self.details.update(extras)


def _coerce_result(res: Any, default_name: str) -> Dict[str, Any]:
    # Normalize supported return shapes into a dict the orchestrator understands.
    if isinstance(res, JudgeResult):
        out: Dict[str, Any] = {"score": float(res.score)}
        if res.explanation is not None:
            out["explanation"] = res.explanation
        if res.details:
            out.update(res.details)
        out.setdefault("name", default_name)
        return out

    if isinstance(res, (int, float)):
        return {"score": float(res), "name": default_name}

    if isinstance(res, dict):
        out = dict(res)
        out["score"] = float(out.get("score", 0.0))
        out.setdefault("name", default_name)
        return out

    return {"score": 0.0, "name": default_name, "error": f"Unsupported result type: {type(res).__name__}"}

IS_METRIC = False
