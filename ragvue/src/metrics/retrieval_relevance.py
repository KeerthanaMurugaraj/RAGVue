from __future__ import annotations
from typing import Any, Dict, Sequence
import json, os, re

from ragvue.src.core.base import JudgeInput
from ragvue.src.core.llm_judge import call_judge, default_model, ensure_env

ensure_env()


class RetrievalRelevanceJudge:
    """
   Retrieval Relevance (precision-like):

      • For each retrieved context chunk, judge whether it is relevant to the QUESTION.
      • The LLM assigns a continuous relevance score r_i in [0,1] per chunk, with range guidelines:
          - 1.0   = directly answers / contains key facts
          - 0.7–0.9 = highly useful evidence
          - 0.3–0.6 = weakly related / background
          - 0.0–0.2 = irrelevant
      • A chunk is counted as "relevant" if r_i >= threshold.
      • Final score = (# chunks with r_i >= threshold) / (total chunks).

      By default, the threshold is set to 0.7 so that only chunks judged as
      “highly useful” or answer-containing contribute to the score.
    """
    name = "retrieval_relevance"
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_THRESHOLD = 0.7

    SCHEMA_HINT = (
        "Return ONLY JSON:\n"
        "{\n"
        '  "per_chunk": [\n'
        '    { "chunk_id": <int>, "relevance": <float 0..1>, "reason": "<short>" },\n'
        "    ...\n"
        "  ]\n"
        "}\n"
    )

    def _json_obj(self, text: str) -> Dict[str, Any]:
        t = (text or "").strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", t, flags=re.IGNORECASE).strip()
        try:
            o = json.loads(t)
            return o if isinstance(o, dict) else {}
        except Exception:
            pass
        s, e = t.find("{"), t.rfind("}")
        if s != -1 and e != -1 and e > s:
            frag = t[s:e+1]
            try:
                o = json.loads(frag)
                return o if isinstance(o, dict) else {}
            except Exception:
                pass
        return {}

    @staticmethod
    def _clip01(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    def evaluate(self, s: JudgeInput, client=None, *, threshold: float | None = None) -> Dict[str, Any]:
        """
        Returns:
          {
            "name": "retrieval_relevance",
            "score": float,
            "explanation": str,
            "per_chunk": [{"chunk_id": int, "relevance": float, "reason": str}],
            "raw": {...}
          }
        """
        # Threshold (env override -> arg -> default)
        thr = threshold
        if thr is None:
            try:
                thr = float(os.getenv("RETRIEVAL_RELEVANCE_THRESHOLD", self.DEFAULT_THRESHOLD))
            except Exception:
                thr = self.DEFAULT_THRESHOLD
        thr = max(0.0, min(1.0, thr))

        ctxs: Sequence[str] = list(s.contexts or [])
        if not ctxs:
            return {
                "name": self.name,
                "score": 0.0,
                "explanation": "No retrieved contexts provided.",
                "per_chunk": [],
                "raw": {"question": s.question, "contexts_count": 0},
            }

        # Create numbered context list
        ctx_text = "\n\n".join(f"[Doc {i+1}] {c}" for i, c in enumerate(ctxs))

        sys = (
            "You are a retrieval relevance judge.\n"
            "For each document, assign a relevance score in [0,1] **to the QUESTION**.\n"
            "Guidelines:\n"
            " • 1.0 = directly answers or contains key facts to answer the question\n"
            " • 0.7–0.9 = highly useful but not the final answer by itself\n"
            " • 0.3–0.6 = weakly related or background\n"
            " • 0.0–0.2 = irrelevant\n"
            "Give a terse reason per document.\n"
            + self.SCHEMA_HINT
        )
        usr = (
            f"QUESTION:\n{s.question}\n\n"
            f"DOCUMENTS:\n{ctx_text}\n\n"
            "Return JSON only."
        )

        # LLM call
        try:
            text = call_judge(
                [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                model=os.getenv("RETRIEVAL_RELEVANCE_MODEL") or default_model(),
                temperature=0.0,
            )
        except Exception as e:
            return {"name": self.name, "score": 0.0, "error": f"LLM error: {e}"}

        obj = self._json_obj(text)
        per = []
        L = obj.get("per_chunk", [])
        # Align by index; if model returns fewer/more, clamp to provided contexts
        if isinstance(L, list) and L:
            for i in range(len(ctxs)):
                rec = L[i] if i < len(L) and isinstance(L[i], dict) else {}
                rel = self._clip01(rec.get("relevance", 0.0))
                per.append({
                    "chunk_id": i + 1,
                    "relevance": rel,
                    "reason": (rec.get("reason") or "").strip(),
                })
        else:
            # fallback: everything irrelevant if model failed
            per = [{"chunk_id": i + 1, "relevance": 0.0, "reason": ""} for i in range(len(ctxs))]

        # Compute precision-like score
        relevant_flags = [p["relevance"] >= thr for p in per]
        total = max(1, len(per))
        score = sum(1 for x in relevant_flags if x) / total

        explanation = f"{sum(relevant_flags)} of {len(per)} chunks ≥ {thr:.2f} relevance."

        return {
            "name": self.name,
            "score": float(max(0.0, min(1.0, score))),
            "explanation": explanation,
            "per_chunk": per,
            "raw": {
                "question": s.question,
                "threshold": thr,
                "contexts_count": len(ctxs),
                "model": os.getenv("RETRIEVAL_RELEVANCE_MODEL", self.DEFAULT_MODEL),
            },
        }


# -------- module-level entrypoint so your loader can call it ----------
IS_METRIC = True

def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapter used by your runner to execute this metric cleanly.
    """
    s = JudgeInput(
        question=item.get("question", ""),
        answer=item.get("answer", ""),
        contexts=list(item.get("contexts", []) or []),
        aspects=item.get("aspects"),
    )
    judge = RetrievalRelevanceJudge()
    try:
        return judge.evaluate(s)
    except Exception as e:
        return {"name": "retrieval_relevance", "score": 0.0, "error": f"evaluate() failed: {e}"}
