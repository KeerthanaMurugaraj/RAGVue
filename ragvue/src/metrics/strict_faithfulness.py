from __future__ import annotations
from typing import List, Dict, Any
from ragvue.src.core.base import BaseJudge
import os, json, re
from pathlib import Path

# ----------------- Load environment for OpenAI -----------------
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


# ---------------------------- Judge ---------------------------------
class FaithfulnessJudge(BaseJudge):
    """
    Strict Faithfulness / Groundedness Judge.

    Single pass:
      • LLM extracts atomic claims
      • Applies STRICT entity/date rules
      • Labels supported vs (partial/full) hallucinated
    Final score = supported / total_claims.
    """

    name = "strict_faithfulness"

    USER_TEMPLATE = (
        "You are a professional fact-checking judge.\n"
        "Your goal: determine whether each factual claim in the ANSWER is directly supported by the CONTEXTS.\n\n"

        "You MUST do everything in ONE step:\n"
        "1. Extract factual claims ONLY from the literal content of the ANSWER:\n"
        "   - Use the ANSWER exactly as written.\n"
        "   - Do NOT add, infer, expand, or reconstruct new claims.\n"
        "   - Every claim you evaluate must be explicitly present in the ANSWER text.\n\n"

        "2. Split the ANSWER into short, atomic factual claims.\n"
        "   - Do NOT split names or compound phrases unnaturally. Split only at sentence-level or clear independent factual units.\n"
        "   - Do NOT merge independent facts.\n"
        "   - Do NOT generate comparative, causal, or inferred statements that are not literally in the ANSWER.\n\n"

        "3. For each claim, check if it is supported by any of the provided CONTEXTS using STRICT rules:\n"
        "   - Carefully read the entire set of CONTEXTS.\n"
        "   - Entities (people, places, organizations) must appear with the same meaning/spelling "
        "     (case/spacing differences are ok).\n"
        "   - Temporal info (years, dates) must match exactly.\n"
        "   - If a sentence or phrase from the ANSWER appears verbatim or almost verbatim in the CONTEXTS, "
        "     you MUST mark that claim as supported.\n"
        "   - Support must come ONLY from the CONTEXTS.\n"
        "   - Do NOT say the context lacks information if the same statement is clearly present in the CONTEXTS.\n"
        "   - If the context does not explicitly support a claim, or contradicts it, mark it as hallucinated.\n\n"

        "4. Classify each claim into:\n"
        "   - \"supported\": fully supported by the context.\n"
        "   - \"partial_hallucination\": some parts supported but at least one factual element unsupported.\n"
        "   - \"full_hallucination\": not supported at all or contradicts the context.\n\n"

        "5. Compute strict_faithfulness = (# supported claims) / (total number of claims).\n\n"

        "6. Return JSON ONLY in this exact format:\n"
        "{{\n"
        '  \"strict_faithfulness\": <float>,\n'
        '  \"supported_claims\": [\n'
        '     {{\"claim\": \"...\", \"supported_by\": \"<context snippet>\"}}\n'
        "  ],\n"
        '  \"hallucinated_claims\": [\n'
        '     {{\"claim\": \"...\", \"type\": \"partial_hallucination\" | \"full_hallucination\", '
        '\"reason\": \"...\", \"evidence\": \"<context snippet or empty>\"}}\n'
        "  ],\n"
        '  \"explanation\": \"short 1–3 sentence summary\"\n'
        "}}\n\n"

        "ANSWER:\n{answer}\n\n"
        "CONTEXTS (retrieved evidences):\n{context}\n\n"
        "Respond strictly in JSON only."
    )

    def prompt(self, s) -> List[dict]:
        # IMPORTANT: use s.contexts (plural), not s.context
        context_text = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(s.contexts))
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict factual evaluation agent. "
                    "Determine exactly which parts of the ANSWER are grounded in the CONTEXTS. "
                    "Do not assume correctness unless it matches the context explicitly. "
                    "Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": self.USER_TEMPLATE.format(
                    answer=s.answer or "",
                    context=context_text,
                ),
            },
        ]

    # ---- helper methods ----

    @staticmethod
    def _coerce01(x) -> float:
        if isinstance(x, (int, float)):
            v = float(x)
        elif isinstance(x, str):
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)
            v = float(m.group(0)) if m else 0.0
        else:
            v = 0.0
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        t = (text or "").strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
        try:
            obj = json.loads(t)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end != -1 and end > start:
            frag = t[start : end + 1]
            try:
                obj = json.loads(frag)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                frag2 = re.sub(r",\s*([}\]])", r"\1", frag)
                try:
                    obj = json.loads(frag2)
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    return {}
        return {}

    # ---- main evaluate ----
    def evaluate(self, s, client=None):
        try:
            from openai import OpenAI
        except Exception as e:
            return {"name": "strict_faithfulness", "score": 0.0, "error": f"OpenAI import error: {e}"}

        if client is None:
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )

        # ---- Single pass: claim extraction + strict judgement ----
        try:
            resp = client.chat.completions.create(
                model=os.getenv("FAITHFULNESS_MODEL", "gpt-4o-mini"),
                messages=self.prompt(s),
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            return {"name": "strict_faithfulness", "score": 0.0, "error": f"LLM error: {e}"}

        obj = self._parse_json(text)
        if not obj or (("strict_faithfulness" not in obj) and ("supported_claims" not in obj)):
            return {
                "name": "strict_faithfulness",
                "score": 0.0,
                "error": "Could not parse JSON from model output.",
                "raw_text": (text or "")[:500],
            }

        # Take supported / hallucinated directly from single-pass output
        validated_supported: List[Dict[str, str]] = []
        if isinstance(obj.get("supported_claims"), list):
            for e in obj["supported_claims"]:
                if isinstance(e, dict):
                    validated_supported.append(
                        {"claim": e.get("claim", ""), "supported_by": e.get("supported_by", "")}
                    )
                elif isinstance(e, str):
                    validated_supported.append({"claim": e, "supported_by": ""})

        hallucinated_detailed: List[Dict[str, str]] = []
        if isinstance(obj.get("hallucinated_claims"), list):
            for e in obj["hallucinated_claims"]:
                if isinstance(e, dict):
                    d = {
                        "claim": e.get("claim", ""),
                        "type": e.get("type", "full_hallucination"),
                        "reason": e.get("reason", ""),
                        "evidence": e.get("evidence", ""),
                    }
                    hallucinated_detailed.append(d)
                elif isinstance(e, str):
                    hallucinated_detailed.append(
                        {"claim": e, "type": "full_hallucination", "reason": "", "evidence": ""}
                    )

        # ---- Literal-match safety net: override dumb LLM misses ----
        # Build a normalized context blob
        normalized_context = " ".join(
            " ".join(str(c).lower().split()) for c in getattr(s, "contexts", [])
        )

        def _norm(text: str) -> str:
            return " ".join((text or "").lower().split())

        fixed_supported = list(validated_supported)
        fixed_hallucinated: List[Dict[str, str]] = []

        for h in hallucinated_detailed:
            claim_text = h.get("claim", "")
            if claim_text and _norm(claim_text) and _norm(claim_text) in normalized_context:
                # If the claim appears verbatim/almost verbatim in the contexts, force it to supported
                fixed_supported.append(
                    {"claim": claim_text, "supported_by": "literal-match"}
                )
            else:
                fixed_hallucinated.append(h)

        validated_supported = fixed_supported
        hallucinated_detailed = fixed_hallucinated

        # ---- Final aggregation ----
        total_claims = max(1, len(validated_supported) + len(hallucinated_detailed))
        score = float(len(validated_supported)) / total_claims

        num_partial = sum(1 for h in hallucinated_detailed if h.get("type") == "partial_hallucination")
        num_full = sum(1 for h in hallucinated_detailed if h.get("type") == "full_hallucination")

        explanation = (
            f"{len(validated_supported)} of {total_claims} claims supported; "
            f"{num_partial} partial and {num_full} full hallucinations."
        )

        minimal_summary = {
            "strict_faithfulness": obj.get("strict_faithfulness"),
            "explanation": (obj.get("explanation") or "").strip() if isinstance(obj, dict) else "",
        }

        return {
            "name": "strict_faithfulness",
            "score": self._coerce01(score),
            "explanation": explanation,
            "supported_claims": validated_supported,
            "hallucinated_claims": hallucinated_detailed,
            "raw": {"summary": minimal_summary},
        }


# ------------------- Entrypoint -------------------
IS_METRIC = True


def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
    from ragvue import JudgeInput
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    except Exception:
        client = None

    s = JudgeInput(
        question=item.get("question", ""),
        answer=item.get("answer", ""),
        contexts=list(item.get("contexts", []) or []),
        aspects=item.get("aspects"),
    )
    res = FaithfulnessJudge().evaluate(s, client=client)

    if isinstance(res, dict):
        res.setdefault("name", "strict_faithfulness")
        return res
    try:
        return {"name": "strict_faithfulness", "score": float(getattr(res, "score", 0.0))}
    except Exception:
        return {"name": "strict_faithfulness", "score": 0.0, "error": "Unexpected result type"}
