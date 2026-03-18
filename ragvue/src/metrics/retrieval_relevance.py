from __future__ import annotations
from typing import Any, Dict, Sequence
import json, os, re
from pathlib import Path

from ragvue.src.core.base import JudgeInput  # class-based path isn't required; we return dicts

try:
    from dotenv import load_dotenv, find_dotenv
except Exception:
    load_dotenv = find_dotenv = None

def _ensure_openai_env():
    if os.getenv("OPENAI_API_KEY"):
        return
    if load_dotenv:
        load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)
        if os.getenv("OPENAI_API_KEY"): return
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
        if os.getenv("OPENAI_API_KEY"): return
        load_dotenv(Path(__file__).resolve().parents[2] / ".env.local", override=True)
    if not os.getenv("OPENAI_API_KEY"):
        for p in (Path.cwd() / ".env",
                  Path(__file__).resolve().parents[2] / ".env",
                  Path.home() / ".env"):
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("OPENAI_API_KEY="):
                        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                        break

_ensure_openai_env()


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

    def _make_openai(self):
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                      base_url=os.getenv("OPENAI_BASE_URL"))

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
        # Build client
        if client is None:
            try:
                client = self._make_openai()
            except Exception as e:
                return {"name": self.name, "score": 0.0, "error": f"OpenAI init error: {e}"}

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
            out = client.chat.completions.create(
                model=os.getenv("RETRIEVAL_RELEVANCE_MODEL", self.DEFAULT_MODEL),
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": usr}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = out.choices[0].message.content or ""
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
