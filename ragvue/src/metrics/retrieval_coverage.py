# ragvue/agents/retrieval_coverage.py
from __future__ import annotations
from typing import Any, Dict, List, Sequence
import json, os, re
from pathlib import Path

from ragvue.src.core.base import JudgeInput
from ragvue.src.core.aspects import get_aspects

try:
    from dotenv import load_dotenv, find_dotenv
except Exception:
    load_dotenv = find_dotenv = None


def _ensure_openai_env():
    if os.getenv("OPENAI_API_KEY"):
        return
    if load_dotenv:
        load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)
        if os.getenv("OPENAI_API_KEY"):
            return
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
        if os.getenv("OPENAI_API_KEY"):
            return
        load_dotenv(Path(__file__).resolve().parents[2] / ".env.local", override=True)
    if not os.getenv("OPENAI_API_KEY"):
        for p in (
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[2] / ".env",
            Path.home() / ".env",
        ):
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("OPENAI_API_KEY="):
                        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                        break


_ensure_openai_env()


class RetrievalCoverageJudge:
    """
    Retrieval Coverage:
      1) Get atomic aspects for the QUESTION via get_aspects(...) (or use provided aspects).
         Aspects are extracted from the question only (no peeking into contexts).
      2) Ask LLM to check whether each aspect is covered by the retrieved CONTEXTS.
      3) Score = (#covered aspects) / (total aspects).
    """

    name = "retrieval_coverage"
    MAX_ASPECTS_DEFAULT = 5
    EVIDENCE_MAX_WORDS = 10

    # LLM schema is strict JSON so parsing is robust.
    SCHEMA_HINT = (
        "Return ONLY JSON with this exact structure:\n"
        "{\n"
        '  \"aspects\": [\n'
        '    { \"aspect\": \"<string>\", \"covered\": true|false, \"evidence\": \"<=WORDS words or empty string\" },\n'
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
            frag = t[s : e + 1]
            try:
                o = json.loads(frag)
                return o if isinstance(o, dict) else {}
            except Exception:
                pass
        return {}

    def _make_openai(self):
        from openai import OpenAI

        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def evaluate(
        self,
        s: JudgeInput,
        client=None,
        *,
        max_aspects: int | None = None,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            "name": "retrieval_coverage",
            "score": float,
            "explanation": str,
            "per_aspect": [{"aspect": str, "covered": bool, "evidence": str}],
            "raw": {...}
          }
        """
        # 1) Build client if not provided
        if client is None:
            try:
                client = self._make_openai()
            except Exception as e:
                return {"name": self.name, "score": 0.0, "error": f"OpenAI init error: {e}"}

        # 2) Get aspects from the question only (or use provided aspects)
        aspects = get_aspects(
            question=s.question or "",
            aspects=list(s.aspects or None) if s.aspects else None,
            client=client,
            model=os.getenv("ASPECTS_MODEL", "gpt-4o-mini"),
            max_aspects=(max_aspects or self.MAX_ASPECTS_DEFAULT),
        )

        if not aspects:
            return {
                "name": self.name,
                "score": 0.0,
                "explanation": "No aspects extracted for the question.",
                "per_aspect": [],
                "raw": {
                    "question": s.question,
                    "contexts_count": len(s.contexts or []),
                    "aspects": [],
                },
            }

        # 3) Ask LLM to judge coverage per aspect using contexts
        ctxs: Sequence[str] = list(s.contexts or [])
        ctx_text = (
            "\n\n".join(f"[Doc {i+1}] {c}" for i, c in enumerate(ctxs))
            if ctxs
            else "(no documents provided)"
        )

        sys = (
            "You are an evidence checker for retrieval coverage.\n"
            "For each aspect of the QUESTION, decide if it is supported by ANY of the given documents.\n"
            f'- Keep "evidence" at most {self.EVIDENCE_MAX_WORDS} words and quote from the documents if covered; '
            'otherwise use an empty string "".\n'
            "Be strict: set covered=true only if the evidence clearly supports the aspect.\n"
            "Do NOT fabricate evidence; it must be a direct quote or very close extract from the given documents.\n"
            + self.SCHEMA_HINT
        )
        usr = (
            f"QUESTION:\n{s.question}\n\n"
            f"ASPECTS (ordered):\n{json.dumps(aspects, ensure_ascii=False)}\n\n"
            f"DOCUMENTS:\n{ctx_text}\n\n"
            "JSON only."
        )

        try:
            out = client.chat.completions.create(
                model=os.getenv("RETRIEVAL_COVERAGE_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = out.choices[0].message.content or ""
        except Exception as e:
            return {"name": self.name, "score": 0.0, "error": f"LLM error: {e}"}

        obj = self._json_obj(text)
        rows: List[Dict[str, Any]] = []
        L = obj.get("aspects", [])
        if isinstance(L, list) and L:
            # Align to aspects length & order (LLM must return same length)
            for i, a in enumerate(aspects):
                rec = L[i] if i < len(L) and isinstance(L[i], dict) else {}
                rows.append(
                    {
                        "aspect": a,
                        "covered": bool(rec.get("covered", False)),
                        "evidence": (rec.get("evidence") or "").strip(),
                    }
                )
        else:
            # Graceful fallback: mark all as not covered
            rows = [{"aspect": a, "covered": False, "evidence": ""} for a in aspects]

        covered_flags = [r["covered"] for r in rows]
        total = max(1, len(rows))
        score = sum(1 for x in covered_flags if x) / total
        explanation = f"Covered {sum(covered_flags)} of {len(rows)} aspects."

        return {
            "name": self.name,
            "score": float(max(0.0, min(1.0, score))),
            "explanation": explanation,
            "per_aspect": rows,
            "raw": {
                "question": s.question,
                "aspects": aspects,
                "contexts_count": len(ctxs),
                "model": os.getenv("RETRIEVAL_COVERAGE_MODEL", "gpt-4o-mini"),
            },
        }


# -------- module-level entrypoint so your loader can call it ----------

def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapter used by your runner to execute the metric without needing to
    know the class. Keeps identical signature with your other metrics.
    """
    s = JudgeInput(
        question=item.get("question", ""),
        answer=item.get("answer", ""),
        contexts=list(item.get("contexts", []) or []),
        aspects=item.get("aspects"),
    )

    judge = RetrievalCoverageJudge()
    try:
        return judge.evaluate(s)
    except Exception as e:
        return {
            "name": "retrieval_coverage",
            "score": 0.0,
            "error": f"evaluate() failed: {e}",
        }

IS_METRIC = True