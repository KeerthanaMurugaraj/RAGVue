from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import json
import os

from ragvue.src.core.llm_judge import call_judge, default_model, ensure_env

ensure_env()


from ragvue.src.core.base import JudgeInput, JudgeResult
try:
    # preferred: question-only aspects via helper
    from ragvue import get_aspects
except Exception:
    get_aspects = None

def _json_obj(text: str) -> Dict[str, Any]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # fallback: extract first {...} block
    import re
    m = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
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

def _clip01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def _fallback_aspects_from_question(question: str, max_aspects: int = 6) -> List[str]:
    """
    Deterministic fallback if helper/LLM is unavailable:
    produce a minimal checklist from the question tokens.
    """
    import re
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", question or "")
    seen, out = set(), []
    for t in toks:
        tt = t.lower()
        if len(tt) >= 4 and tt not in seen:
            out.append(tt)
            seen.add(tt)
        if len(out) >= max_aspects:
            break
    # Provide at least a generic aspect if nothing useful
    return out or ["required aspect(s) from question"]

# --- the metric implementation (ANSWER completeness) ---
@dataclass
class AnswerCompletenessLLM:
    """
    Judge how completely the ANSWER covers the aspects implied by the QUESTION.
    - Aspects are derived from the QUESTION only (contexts are ignored here).
    - For each aspect, mark covered=true|false based on the ANSWER text only.
    - Final score = covered_count / total_aspects.
    """
    name: str = "answer_completeness checks only the coverage of answer to the question"
    judge_model: str = os.getenv("ANSWER_COMPLETENESS_MODEL", "gpt-4o-mini")
    max_aspects: int = 12
    evidence_max_words: int = 20

    def _build_aspects(self, inp: JudgeInput, *, client) -> List[str]:
        # 1) caller-provided aspects win
        if inp.aspects:
            return [str(a) for a in inp.aspects][: self.max_aspects]

        # 2) preferred helper (question-only) if available
        if get_aspects is not None:
            try:
                # contexts arg kept for API stability but ignored by get_aspects in your current design
                return get_aspects(
                    question=inp.question or "",
                    aspects=None,
                    client=client,
                    model=self.judge_model,
                    max_aspects=min(self.max_aspects, 6),
                )[: self.max_aspects]
            except Exception:
                pass

        # 3) deterministic fallback (no LLM)
        return _fallback_aspects_from_question(inp.question or "", max_aspects=min(self.max_aspects, 6))

    def evaluate(self, inp: JudgeInput, **kwargs: Any) -> JudgeResult:
        # --- 1) aspects (question-only) ---
        aspects: List[str] = self._build_aspects(inp, client=None)
        if not aspects:
            return JudgeResult(score=0.0, explanation="No aspects could be derived from the question.")

        # --- 2) prompt — judge completeness from ANSWER only (ignore contexts) ---
        sys = (
            "You are a strict COMPLETENESS judge. "
            "Given a list of aspects derived from only the QUESTION, "
            "decide for each aspect whether the ANSWER explicitly covers it. "
            "Use the ANSWER text only; do NOT use your internal or external knowledge or retrieved documents. "
            "Output JSON only."
        )
        usr = f"""
        Aspects (JSON array): {json.dumps(aspects, ensure_ascii=False)}
        
        ANSWER:
        {inp.answer or ""}
        
        Return JSON with this exact schema:
        {{
          "score": <float 0..1>,   // covered_count / aspects_count
          "per_aspect": [
            {{"aspect": "<text>", "covered": true|false, "evidence": "<<= {self.evidence_max_words} words quoted from ANSWER or null>"}}
          ],
          "explanation": "<one line>"
        }}
        Rules:
        - Mark covered=true only if the ANSWER explicitly provides the information for that aspect.
        - If the ANSWER is vague/silent, mark covered=false.
        - Evidence must be a short quote (<= {self.evidence_max_words} words) from the ANSWER itself, or null if not covered.
        - JSON only.
        """.strip()

        # --- 3) call LLM via provider-agnostic judge ---
        model = os.getenv("ANSWER_COMPLETENESS_MODEL") or default_model()
        try:
            text = call_judge(
                [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                model=model,
                temperature=0.0,
            )
        except Exception as e:
            return JudgeResult(score=0.0, explanation=f"LLM failed: {e}")

        obj = _json_obj(text)
        per = obj.get("per_aspect", [])
        if not isinstance(per, list):
            per = []

        covered_flags: List[bool] = []
        for i, a in enumerate(aspects):
            rec = per[i] if i < len(per) and isinstance(per[i], dict) else {}
            covered = rec.get("covered", rec.get("correct", False))  # tolerate old schema
            covered_flags.append(bool(covered))
            # ensure aspect field present
            if isinstance(rec, dict) and "aspect" not in rec:
                rec["aspect"] = a

        total = max(1, len(aspects))
        score = sum(1 for x in covered_flags if x) / total

        return JudgeResult(
            score=_clip01(float(score)),
            explanation=obj.get("explanation", f"{sum(covered_flags)}/{len(aspects)} aspects covered."),
            details={
                "per_aspect": per,
                "aspects": aspects,
                "answer_len": len(inp.answer or ""),
            },
            raw={"llm_output": obj},
        )

# --- module-level entrypoint the runner expects ---
def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
    # Build JudgeInput (contexts ignored for completeness, but kept for API symmetry)
    inp = JudgeInput(
        question=item.get("question", ""),
        answer=item.get("answer", ""),
        contexts=list(item.get("contexts", []) or []),
        aspects=item.get("aspects"),
    )
    res = AnswerCompletenessLLM().evaluate(inp)

    # normalize to dict
    out: Dict[str, Any] = {"name": "answer_completeness (checks only the coverage of answer to the question)", "score": float(res.score)}
    if getattr(res, "explanation", None) is not None:
        out["explanation"] = res.explanation
    if getattr(res, "details", None):
        out.update(res.details)
    if hasattr(res, "raw"):
        out["raw"] = res.raw
    return out

