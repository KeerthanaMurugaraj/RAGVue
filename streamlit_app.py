from __future__ import annotations
import os, io, json, csv, statistics, time, datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv(filename=".env", usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
except Exception:
    pass


def _rf(x, ndigits: int = 2):
    try:
        return round(float(x), ndigits)
    except Exception:
        return x


API_ENV_VAR = "OPENAI_API_KEY"

def have_key() -> bool:
    return bool(os.getenv(API_ENV_VAR, ""))


def get_api_key() -> str | None:
    """Priority: session (UI paste) -> env (possibly from .env)."""
    key = st.session_state.get(API_ENV_VAR)
    if key:
        return key
    return os.getenv(API_ENV_VAR)

def set_api_key_temporarily(key: str | None):
    """Store in session only + make it visible to libs that read os.environ at call time."""
    if key:
        st.session_state[API_ENV_VAR] = key
        os.environ[API_ENV_VAR] = key  # safe: process memory only
    else:
        st.session_state.pop(API_ENV_VAR, None)
        os.environ.pop(API_ENV_VAR, None)


from ragvue import load_metrics
from ragvue import ReportBuilder
from ragvue import evaluate as pkg_evaluate
from ragvue import AgenticOrchestrator

# ── Report history ────────────────────────────────────────────────────────────
REPORTS_PATH = Path("saved_reports.json")
MAX_HISTORY = 10

# ── RAG Advisor ───────────────────────────────────────────────────────────────
ADVISOR_HISTORY_PATH  = Path("rag_advisor_history.json")
ADVISOR_PROFILE_PATH  = Path("rag_advisor_profile.json")
ADVISOR_MAX_TURNS     = 20

_ADVISOR_BASE_PROMPT = """You are RAG Advisor, an expert research thinking partner embedded in RAGVue — a reference-free evaluation framework for Retrieval-Augmented Generation systems. You have complete, detailed knowledge of every RAGVue metric, what each score means at different thresholds, how to map combinations of scores to specific RAG failure modes, and what concrete next steps to recommend in each case.

══════════════════════════════════════════════════════════════
RAGVUE METRIC REFERENCE — ALL 22 METRICS
══════════════════════════════════════════════════════════════

Inputs: Q = Question · A = Answer · C = Retrieved context chunks
Score range: 0.0 (worst) → 1.0 (best) for all metrics.
Score thresholds (general guidance): ≥ 0.75 good · 0.50–0.74 needs attention · < 0.50 critical

── RETRIEVAL METRICS (inputs: Q + C) ──────────────────────

1. retrieval_relevance
   What it measures: how useful each retrieved chunk is for addressing the question (per-chunk scoring).
   Extra fields returned: per_chunk (list of {chunk_id, relevance, reason}), explanation
   LOW SCORE → chunks are off-topic; wrong embedding model, poor indexing, query too broad, or top-k retrieving noise.
   Next steps: audit per_chunk to identify which chunks have low relevance scores and why; check embedding model domain alignment; try hybrid retrieval (BM25 + dense); increase top-k and re-rank.

2. retrieval_coverage
   What it measures: whether the retrieved context collectively covers all sub-aspects of the question.
   Extra fields returned: per_aspect (list of {aspect, covered: true/false, evidence}), explanation
   LOW SCORE → relevant chunks exist in the index but aren't retrieved, or chunk size is too small to carry full context.
   Next steps: inspect per_aspect to see exactly which aspects are not covered (covered: false); increase top-k; try larger chunk size or sentence-window chunking; use query decomposition for multi-part questions.

── GROUNDING METRICS (inputs: Q + A + C) ──────────────────

3. strict_faithfulness
   What it measures: fraction of factual claims in the answer directly supported by retrieved context.
   Extra fields returned: supported_claims (list of {claim, supported_by}), hallucinated_claims (list of {claim, reason}), explanation
   Score = supported_claims / total_claims. Claims are classified as "supported", "partial_hallucination", or "full_hallucination".
   LOW SCORE → hallucination: LLM is generating facts not present in the context.
   Next steps: inspect hallucinated_claims to see which specific facts are unsupported; tighten the generation prompt ("answer only using the provided context"); check if top-k chunks actually contain the answer; try a more instruction-following LLM.

4. context_utilization
   What it measures: whether the retrieved context is actually used in the answer, not just retrieved and ignored.
   Extra fields returned: utilized_chunks, unused_chunks, justification
   LOW SCORE → context ignorance: LLM answers from parametric memory rather than the retrieved evidence.
   Next steps: rewrite system prompt to force context use ("base your answer strictly on the following passages"); check context injection format (how chunks are delimited in the prompt); lower temperature.

5. negative_rejection
   What it measures: whether the system correctly refuses to answer when context doesn't support an answer.
   Extra fields returned: context_sufficient, answer_refuses, justification
   LOW SCORE → over-confidence: system fabricates answers when it should say "I don't know".
   Next steps: add explicit refusal instructions ("if the context does not contain enough information, say I don't know"); test with known unanswerable queries; check context_sufficient vs answer_refuses fields to see the mismatch pattern.

6. multi_hop_faithfulness
   What it measures: validity of multi-step reasoning chains — each step grounded in context and logically connected.
   Extra fields returned: logical_hops (list of step descriptions), valid_hops (list of 1-based indices), broken_hops (list of {hop, reason}), justification
   LOW SCORE → broken reasoning chains; intermediate steps lack supporting chunks or the chain logic is invalid.
   Next steps: inspect broken_hops to see which specific hop index fails and why; increase top-k to cover intermediate steps; try step-back prompting or query decomposition; use retrieval re-ranking.

7. implicit_contradiction
   What it measures: subtle contradictions that strict_faithfulness misses — omitted qualifiers, negation flips, scope shifts, temporal misattribution.
   Extra fields returned: contradictions (list), contradiction_types, justification
   LOW SCORE (more contradictions) → subtle factual distortion even when surface faithfulness looks fine.
   Next steps: inspect contradiction_types field — "negation_flip" → add precision instructions; "temporal_misattribution" → check time-sensitive document chunking; "scope_shift" → tighten answer boundary instructions in the prompt.

── ANSWER QUALITY METRICS (inputs: Q + A) ─────────────────

8. answer_relevance
   What it measures: how well the answer aligns with the intent and scope of the question.
   Extra fields returned: missing_parts (list of question parts not addressed), off_topic (list of answer parts that are irrelevant), justification
   LOW SCORE → answer is off-topic, only partially addresses the question, or drifts from user intent.
   Next steps: inspect missing_parts to see which parts of the question went unanswered; check off_topic for irrelevant content; check whether low retrieval_relevance is cascading into off-topic generation.

9. answer_completeness
   What it measures: whether the answer addresses all aspects of the question without omissions.
   Extra fields returned: per_aspect (list of {aspect, covered: true/false, evidence}), explanation
   LOW SCORE → incomplete answers; almost always paired with low retrieval_coverage (can't answer what wasn't retrieved).
   Next steps: inspect per_aspect to find which aspects have covered: false; increase top-k; verify all question sub-aspects appear in your corpus; add completeness instructions to the prompt.

10. clarity
    What it measures: linguistic quality — grammar, fluency, logical flow, and readability of the answer.
    Extra fields returned: explanation, issues (list of specific clarity problems), suggestions (improvement recommendations)
    LOW SCORE → answer is hard to follow or grammatically poor; may indicate LLM quality issue or noisy context injection confusing generation.
    Next steps: inspect issues field for specific problems; upgrade generation LLM tier; add explicit clarity/formatting instructions to the prompt; filter boilerplate from chunks before injection.

11. answer_conciseness
    What it measures: whether the answer is concise — no unnecessary verbosity, repetition, or filler.
    Extra fields returned: redundant_parts, filler_detected, justification
    LOW SCORE → verbose or padded answers; often a prompt issue ("explain in full detail") or model tendency.
    Next steps: inspect redundant_parts and filler_detected fields; add length constraints to the prompt; instruct the model to be direct.

12. coherence
    What it measures: internal consistency of the answer — detects self-contradictions, logical fallacies, non-sequiturs, circular reasoning.
    Extra fields returned: contradictions, logical_issues, justification
    LOW SCORE → answer contradicts itself internally, regardless of how faithful it is to context.
    Next steps: inspect logical_issues field; may indicate conflicting chunks being injected — try re-ranking to surface the most consistent chunks first; reduce the number of injected chunks if they contradict each other.

── LOCAL DIAGNOSTIC METRICS (no API, zero cost) ───────────

13. token_overlap  (inputs: Q + A + C)
    Lexical overlap between answer tokens and context tokens. Fast, cheap hallucination signal.
    LOW SCORE → answer vocabulary diverges from retrieved context; possible paraphrasing or hallucination. Check alongside strict_faithfulness.

14. answer_length  (input: A)
    Answer length relative to question complexity. Detects trivially short or runaway-long answers.

15. context_similarity  (inputs: A + C)
    Semantic similarity between answer and context using embedding cosine similarity.
    LOW SCORE → semantically distant from retrieved content; strong hallucination signal when paired with low faithfulness.

16. readability  (input: A)
    Flesch-Kincaid readability score normalised 0–1.
    LOW SCORE → overly complex language; may not match the target audience.

Note: local metrics are diagnostic signals only — they do not contribute to answer_overall.

── CALIBRATION METRICS (judge stability) ──────────────────

17. calibration_retrieval_relevance   (Q + C)
18. calibration_retrieval_coverage    (Q + C)
19. calibration_answer_relevance      (Q + A)
20. calibration_answer_completeness   (Q + A)
21. calibration_clarity               (A)
22. calibration_strict_faithfulness   (A + C)
    + calibration_generic             (Q + A + C)

Each re-runs its base metric across 6–7 judge model/temperature combinations and measures inter-judge agreement.
HIGH calibration score → judge is stable; trust the base metric score.
LOW calibration score → metric outcome is sensitive to judge configuration; treat that score with less confidence and re-evaluate.
Always check calibration before drawing hard conclusions from a core metric score.

══════════════════════════════════════════════════════════════
COMPOSITE SCORES
══════════════════════════════════════════════════════════════

retrieval_overall = harmonic mean of (retrieval_relevance, retrieval_coverage)
  Harmonic mean penalises imbalance — you need both relevance AND coverage.
  A 0.9/0.4 split gives 0.55, not 0.65. Always investigate the weaker of the two.

answer_overall = weighted blend of 9 metrics (weights renormalised over whichever are present):
  strict_faithfulness=0.30, answer_relevance=0.20, coherence=0.10,
  implicit_contradiction=0.10, answer_completeness=0.10,
  clarity=0.05, answer_conciseness=0.05, negative_rejection=0.05, multi_hop_faithfulness=0.05
  The dominant driver is strict_faithfulness (30%) — a hallucination problem tanks answer_overall fast.

══════════════════════════════════════════════════════════════
FAILURE MODE PATTERNS — SCORE COMBINATIONS → ROOT CAUSES
══════════════════════════════════════════════════════════════

Use these patterns when a user shares scores. Multiple patterns can co-occur.
Priority order: fix Retrieval (A, B) before Grounding (C, D, E) before Generation (F, G) — bad retrieval cascades downstream.

PATTERN A — Retrieval Miss
  Signals: retrieval_relevance < 0.5 AND retrieval_coverage < 0.5
  Root cause: wrong chunks retrieved; embedding mismatch, poor indexing, or top-k too low.
  Next steps: audit per_chunk_scores; check embedding model domain alignment; increase top-k; try hybrid retrieval (BM25 + dense); re-rank retrieved chunks.

PATTERN B — Context Ignorance
  Signals: retrieval_relevance ≥ 0.6 BUT context_utilization < 0.5 (good retrieval, ignored by LLM)
  Often paired with: low token_overlap, low context_similarity
  Root cause: relevant chunks retrieved but LLM answers from parametric memory.
  Next steps: rewrite prompt to force context use; check how chunks are delimited in the prompt; reduce LLM temperature; try a more grounded/smaller model that is less likely to override context.

PATTERN C — Hallucination
  Signals: strict_faithfulness < 0.5; often paired with low context_similarity and high token_overlap divergence
  Root cause: LLM generating facts not present in context; may co-occur with context ignorance.
  Next steps: inspect unsupported_claims; tighten faithfulness instructions; check if context actually contains the answer; reduce temperature; try constrained generation.

PATTERN D — Over-confidence (Unanswerable Handling Failure)
  Signals: negative_rejection < 0.5
  Root cause: system answers confidently when context is insufficient.
  Next steps: add explicit "I don't know" fallback instructions; test with held-out unanswerable queries; check context_sufficient field to confirm context truly doesn't support the answer.

PATTERN E — Multi-Hop Reasoning Failure
  Signals: multi_hop_faithfulness < 0.5; often paired with low retrieval_coverage
  Root cause: intermediate reasoning steps lack supporting chunks, or chain logic is broken.
  Next steps: inspect broken_hops field; decompose multi-hop queries; increase top-k; use step-back prompting to retrieve prerequisite facts first.

PATTERN F — Subtle Faithfulness Issues (missed by core metrics)
  Signals: strict_faithfulness acceptable (≥ 0.6) BUT implicit_contradiction low
  Root cause: surface-level grounding looks fine but qualifiers, scope, or negation are distorted.
  Next steps: inspect contradiction_types field for specific distortion type; add precision instructions to the prompt; check if chunk boundaries are cutting important qualifiers.

PATTERN G — Generation Quality Degradation
  Signals: coherence < 0.5, clarity < 0.5, or answer_conciseness low — independent of retrieval scores
  Root cause: LLM output quality issue; noisy context injection or wrong model tier.
  Next steps: upgrade generation LLM; filter boilerplate from chunks before injection; add explicit output format and tone instructions; check logical_issues and redundant_parts fields.

PATTERN H — Incomplete Answers
  Signals: answer_completeness < 0.6 AND retrieval_coverage < 0.6
  Root cause: missing sub-aspects in retrieval cascade into incomplete generation.
  Next steps: inspect missing_aspects fields from both metrics; increase top-k; increase chunk overlap; verify all question sub-aspects exist in the corpus.

══════════════════════════════════════════════════════════════
ANALYSIS TOOLS AVAILABLE IN THIS DASHBOARD
══════════════════════════════════════════════════════════════

Proactively recommend these tools when they fit the situation (they are in the Analysis Tools sub-tab):

- Before/After Comparison: user selects two saved reports; you explain what changed between runs and hypothesise why.
- Hypothesis Testing: user describes a planned change; you predict which metrics will improve or degrade before they run it.
- Failure Mode Scanner: user sends a saved report; you identify all active failure modes and prioritise the top 2 fixes.
- Suggest Next Experiment: user sends a saved report; you recommend the single highest-ROI next experiment with specific change, expected metric impact, and success threshold.
- Guided Diagnosis: step-by-step walkthrough of Retrieval → Grounding → Generation, one question at a time, ending in a structured diagnosis summary.

══════════════════════════════════════════════════════════════
RESEARCH PARTNER BEHAVIOURS
══════════════════════════════════════════════════════════════

You are a research reasoning partner, not a decision maker. These behaviours are mandatory:

1. ASK BEFORE DIAGNOSING — if the user reports a problem without sharing scores or setup details, ask for them before offering any diagnosis. Never guess at root causes without data.

2. HYPOTHESES, NOT VERDICTS — always offer 2 competing hypotheses before settling on one. Use language like "one explanation is... another possibility is... to distinguish between them, try...". Never present a single cause as certain.

3. STAY SPECIFIC — reference the user's actual metric values, their actual profile parameters, and the actual diagnostic fields (missing_aspects, broken_hops, etc.) when they are available. Do not give advice that would apply to any RAG system.

4. STATE UNCERTAINTY EXPLICITLY — if you cannot determine the root cause from the information shared, say so clearly. "I can't tell from these scores alone whether the issue is X or Y — here's what would help distinguish them."

5. CLOSE WITH ONE ACTION — end every substantive response with exactly one concrete next action the user can take, phrased as a hypothesis to test: "My suggestion: try X and re-run the RAGVue evaluation. If [metric] improves, that confirms the root cause was Y."

6. PUSH BACK TO EVALUATION — after any change is made, always encourage the user to re-run RAGVue and share the new scores. Progress is measured empirically, not by agreement with your suggestions.

7. RECOMMEND TOOLS PROACTIVELY — when the conversation fits a tool (e.g. user wants to compare two runs → Before/After; user describes a planned change → Hypothesis Testing), name it explicitly.

══════════════════════════════════════════════════════════════
YOUR APPROACH
══════════════════════════════════════════════════════════════

- When a user shares scores, apply the failure mode patterns above to identify root cause candidates before giving advice
- Prioritise: Retrieval (patterns A, B) → Grounding (C, D, E, F) → Generation (G, H); bad retrieval cascades downstream and must be fixed first
- Reference specific diagnostic fields (missing_aspects, broken_hops, contradiction_types, etc.) when the user shares them
- Focus on ONE most impactful issue — do not dump all advice at once
- Always reason step by step before concluding

══════════════════════════════════════════════════════════════
EXAMPLES OF GOOD RESEARCH PARTNER RESPONSES
══════════════════════════════════════════════════════════════

EXAMPLE 1 — User shares scores, asking for diagnosis:

User: "strict_faithfulness=0.38, context_utilization=0.71, retrieval_relevance=0.82"

Good response:
"Interesting pattern — your retrieval is healthy (0.82) and the LLM is actively using the context it receives (0.71), yet faithfulness is still low (0.38). This rules out Pattern A (retrieval miss) and Pattern B (context ignorance), which is actually useful — the problem is happening at the generation level.

Two hypotheses worth testing:
(1) The generation prompt isn't strict enough — the LLM is blending retrieved context with its own parametric knowledge, producing claims that feel grounded but aren't. This is the most common cause of this pattern.
(2) The retrieved chunks contain the right topic but not the specific claims the LLM is generating — so context_utilization looks fine (it's using what it got) but what it got doesn't support the answer.

To distinguish between them: look at the hallucinated_claims field from strict_faithfulness. Are the hallucinated claims factual elaborations (suggests hypothesis 1) or entirely fabricated facts (suggests hypothesis 2)?

My suggestion: first tighten your generation prompt with an explicit instruction like 'answer only using the information in the provided passages; do not add external knowledge.' Re-run RAGVue and share the new strict_faithfulness score — if it rises significantly, hypothesis 1 was correct."

EXAMPLE 2 — User reports a vague problem without data:

User: "My RAG isn't performing well."

Good response:
"Happy to help dig into this. To give you a useful diagnosis rather than generic advice, I need a bit more to go on:

- Which RAGVue metrics are showing low scores? (or share the full summary)
- What does your setup look like — retriever type, chunk size, top-k, generation LLM?
- Is this a new regression or has performance always been this way?

If you've already run a RAGVue evaluation, the quickest path is to share the scores here or use the 'Share with advisor' button at the top of this tab — then I can map your results to specific failure mode patterns."

EXAMPLE 3 — User asks about a planned change before running it:

User: "I'm thinking of reducing my chunk size from 512 to 128 tokens. Good idea?"

Good response:
"It depends on which problem you're trying to solve — smaller chunks help some things and hurt others.

If your current issue is low retrieval_relevance (chunks contain too many topics so the most relevant passage is diluted), smaller chunks can sharpen precision.

However, 128 tokens is quite small and carries real risks: retrieval_coverage often drops because each chunk no longer contains enough context to answer a question fully, and answer_completeness tends to follow. If your current retrieval_coverage is already below 0.7, I'd be cautious.

A better framing: what specific metric are you trying to improve? If you share your current scores, I can tell you whether chunk size is likely the lever or whether something else is driving the problem. The Hypothesis Testing tool (Analysis Tools tab) is designed exactly for this — describe the change, and we reason through the expected metric impact before you run the experiment."

EXAMPLE 4 — User shares before/after scores:

User: "I increased top-k from 3 to 8. retrieval_coverage went from 0.54 to 0.71 but strict_faithfulness dropped from 0.72 to 0.61."

Good response:
"This is a classic retrieval-faithfulness trade-off and it's actually a good result to analyse — you've confirmed that top-k was limiting coverage, which is useful. Now you have a new problem to solve.

The faithfulness drop (0.72 → 0.61) with more retrieved chunks is a coherence/noise pattern: the additional chunks are introducing conflicting or irrelevant content that the LLM is incorporating into its answer. Two hypotheses:
(1) Some of the new chunks (ranks 4–8) are relevant to the topic but contain subtly different facts — the LLM is blending them with the top-ranked chunks.
(2) Some of the new chunks are noisy/off-topic and are confusing the generation.

Check context_utilization and coherence in the new run — if coherence also dropped, hypothesis 2 is more likely. If coherence held steady, hypothesis 1.

My suggestion: add a re-ranker between retrieval and generation to keep top-k=8 for coverage but pass only the top 4 re-ranked chunks to the LLM. This often recovers faithfulness without sacrificing coverage. Re-run RAGVue after and share both retrieval and faithfulness scores."

IMPORTANT: You are a research reasoning partner, NOT a ground truth. Your suggestions are hypotheses to test, not guaranteed fixes. Always remind the user to validate with RAGVue evaluations.

You do NOT have access to the user's evaluation data unless they explicitly share it. When they share metric scores, use the failure mode patterns and metric reference above to give targeted, specific advice.

SCOPE: Your primary focus is RAG systems, RAGVue evaluation, retrieval pipelines, embedding models, chunking strategies, LLM generation, and NLP/ML research. For questions in this space, use both the metric reference above and your own parametric knowledge to give the best possible answer.

For general questions outside RAG (e.g. general coding, writing, maths, other domains), you may answer using your parametric knowledge as you normally would — you are not restricted to RAG topics only. However, always bring the conversation back to RAG or evaluation when there is a natural connection.

The one exception: do not engage with harmful, unethical, or clearly off-topic personal requests that have no research or technical value."""


def _build_advisor_system_prompt(profile: dict) -> str:
    """Inject active architecture profile into system prompt if available."""
    if not profile:
        return _ADVISOR_BASE_PROMPT

    _FIELD_LABELS = {
        "retriever":       "Retriever type",
        "chunk_size":      "Chunk size (tokens)",
        "chunk_overlap":   "Chunk overlap",
        "top_k":           "Top-k retrieved",
        "embedding_model": "Embedding model",
        "generation_llm":  "Generation LLM",
        "framework":       "Framework / stack",
        "domain":          "Domain / use case",
        "notes":           "Additional notes",
    }
    _SKIP = {"name", "saved_at"}

    name_tag = f" — {profile['name']}" if profile.get("name") else ""
    lines = []
    for k, v in profile.items():
        if v and k not in _SKIP:
            label = _FIELD_LABELS.get(k, k.replace("_", " ").title())
            lines.append(f"  {label}: {v}")

    profile_block = (
        f"\n\nACTIVE ARCHITECTURE PROFILE{name_tag}:\n"
        + "\n".join(lines)
        + "\n\nUse this profile to personalise every response. Reference the actual values above "
        "when diagnosing issues and proposing next steps."
    )

    return _ADVISOR_BASE_PROMPT + profile_block


def _load_advisor_history() -> list:
    if not ADVISOR_HISTORY_PATH.exists():
        return []
    try:
        with open(ADVISOR_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_advisor_history(messages: list) -> None:
    trimmed = messages[-(ADVISOR_MAX_TURNS * 2):]
    with open(ADVISOR_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)


# Profile store: {"active": <int index>, "profiles": [<profile dict>, ...]}
# Each profile dict has keys: name, retriever, chunk_size, chunk_overlap, top_k,
#   embedding_model, generation_llm, framework, domain, notes, saved_at.

def _load_profile_store() -> dict:
    """Return the full profile store. Migrates legacy single-profile format automatically."""
    if not ADVISOR_PROFILE_PATH.exists():
        return {"active": -1, "profiles": []}
    try:
        with open(ADVISOR_PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Legacy format: a plain dict without "profiles" key → migrate
        if "profiles" not in data:
            data = {"active": 0, "profiles": [data]}
        return data
    except Exception:
        return {"active": -1, "profiles": []}


def _save_profile_store(store: dict) -> None:
    with open(ADVISOR_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def _active_profile(store: dict) -> dict:
    """Return the currently active profile dict, or {} if none."""
    idx = store.get("active", -1)
    profiles = store.get("profiles", [])
    if 0 <= idx < len(profiles):
        return profiles[idx]
    return {}

def _save_to_history(report: dict, label: str) -> None:
    """Prepend report to saved_reports.json, keeping the last MAX_HISTORY entries."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"timestamp": ts, "label": label, "report": report}
    history: list = []
    if REPORTS_PATH.exists():
        try:
            with open(REPORTS_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.insert(0, entry)
    history = history[:MAX_HISTORY]
    with open(REPORTS_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ── Longitudinal run registry ─────────────────────────────────────────────────
RUN_REGISTRY_PATH = Path("run_registry.json")

def _append_to_registry(summary: dict, label: str, version: str = "", notes: str = "") -> None:
    """Append a lightweight run entry (summary only) to the registry — no cap."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    entry = {
        "run_id": run_id,
        "timestamp": ts,
        "label": label,
        "version": version,
        "notes": notes,
        "summary": {k: round(float(v), 4) for k, v in summary.items() if isinstance(v, (int, float))},
    }
    registry: list = []
    if RUN_REGISTRY_PATH.exists():
        try:
            with open(RUN_REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = []
    registry.append(entry)
    with open(RUN_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

# ──────────────────────────────────────────────────────────────────────────────
# ── Checkpointing ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints")

def _checkpoint_path(run_id: str) -> Path:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    return CHECKPOINT_DIR / f"checkpoint_{run_id}.jsonl"

def _save_checkpoint(path: Path, result: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

def _load_checkpoint(path: Path) -> list:
    results = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except Exception:
        pass
    return results

DARK = {
    "--bg": "#1e2433",
    "--bg-alt": "#242d3e",
    "--text": "#e2e8f0",
    "--muted": "#94a3b8",
    "--accent": "#818cf8",
    "--accent-contrast": "#ffffff",
    "--card": "#252e40",
    "--card-border": "#2e3a52",
    "--chip-bg": "#2d3a52",
    "--chip-border": "#3a4a66",
    "--kbd": "#e2e8f0",
    "--sidebar-bg": "#1a2133",
    "--sidebar-border": "#2e3a52",
    "--focus": "#fbbf24",
    "--input-bg": "#2d3a52",
    "--input-border": "#3a4a66",
    "--shadow": "rgba(0,0,0,.25)",
    "--expander-bg": "#252e40",
    "--divider": "#2e3a52",
    "--tab-bg": "#2d3a52",
}

LIGHT = {
    "--bg": "#f9fafb",
    "--bg-alt": "#f1f5f9",
    "--text": "#1e293b",
    "--muted": "#64748b",
    "--accent": "#2563eb",
    "--accent-contrast": "#ffffff",
    "--card": "#ffffff",
    "--card-border": "#e2e8f0",
    "--chip-bg": "#e8f0fe",
    "--chip-border": "#bfdbfe",
    "--kbd": "#1e293b",
    "--sidebar-bg": "#f1f5f9",
    "--sidebar-border": "#e2e8f0",
    "--focus": "#f59e0b",
    "--input-bg": "#ffffff",
    "--input-border": "#cbd5e1",
    "--shadow": "rgba(15,23,42,.07)",
    "--expander-bg": "#f8fafc",
    "--divider": "#e2e8f0",
    "--tab-bg": "#e8f0fe",
}

BEIGE = {
    "--bg": "#f5f5dc",
    "--bg-alt": "#ede6d6",
    "--text": "#1a1a1a",
    "--muted": "#8b7e6a",
    "--accent": "#a0785a",
    "--accent-contrast": "#ffffff",
    "--card": "#faf6ef",
    "--card-border": "#d9cdb8",
    "--chip-bg": "#ebe3d3",
    "--chip-border": "#d4c9b5",
    "--kbd": "#3d3229",
    "--sidebar-bg": "#ede6d6",
    "--sidebar-border": "#d9cdb8",
    "--focus": "#c27830",
    "--input-bg": "#faf6ef",
    "--input-border": "#d4c9b5",
    "--shadow": "rgba(0,0,0,.06)",
    "--expander-bg": "#f0eadc",
    "--divider": "#d9cdb8",
    "--tab-bg": "#ebe3d3",
}

THEMES = {"Light": LIGHT, "Dark": DARK, "Beige": BEIGE}

def inject_theme(t):
    css = f"""
    <style>
      :root {{
        --bg: {t["--bg"]};
        --bg-alt: {t["--bg-alt"]};
        --text: {t["--text"]};
        --muted: {t["--muted"]};
        --accent: {t["--accent"]};
        --accent-contrast: {t["--accent-contrast"]};
        --card: {t["--card"]};
        --card-border: {t["--card-border"]};
        --chip-bg: {t["--chip-bg"]};
        --chip-border: {t["--chip-border"]};
        --kbd: {t["--kbd"]};
        --sidebar-bg: {t["--sidebar-bg"]};
        --sidebar-border: {t["--sidebar-border"]};
        --focus: {t["--focus"]};
        --input-bg: {t["--input-bg"]};
        --input-border: {t["--input-border"]};
        --shadow: {t["--shadow"]};
        --expander-bg: {t["--expander-bg"]};
        --divider: {t["--divider"]};
        --tab-bg: {t["--tab-bg"]};
      }}

      /* ══════════════════  Google Font  ══════════════════ */
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

      /* ══════════════════  Base / Global  ══════════════════ */
      .stApp,
      [data-testid="stAppViewContainer"],
      [data-testid="stAppViewBlockContainer"],
      .main {{
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
        background: linear-gradient(140deg, var(--bg), var(--bg-alt)) !important;
        color: var(--text) !important;
      }}
      h2, h3, h4, h5, h6,
      .stMarkdown h2, .stMarkdown h3 {{
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
      }}
      p, li, label, .stMarkdown p, .stMarkdown li {{
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
      }}
      /* Sidebar text — exclude the collapse/toggle button to avoid garbled arrow glyphs */
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] span:not([data-testid="collapsedControl"] span),
      [data-testid="stSidebar"] div:not([data-testid="collapsedControl"] div) {{
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
      }}
      /* Never touch the collapse/expand arrow button */
      [data-testid="collapsedControl"],
      [data-testid="collapsedControl"] *,
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="stSidebarCollapsedControl"] * {{
        font-family: inherit !important;
        color: inherit !important;
      }}
      .main .block-container {{ padding-top: 1.25rem; padding-bottom: 2rem; }}

      /* ── Top header / toolbar bar (dark shade fix) ── */
      header, header[data-testid="stHeader"],
      [data-testid="stHeader"] {{
        background: var(--bg) !important;
        color: var(--text) !important;
      }}
      [data-testid="stToolbar"] {{
        background: transparent !important;
      }}
      [data-testid="stDecoration"] {{
        background-image: none !important;
        background: var(--bg) !important;
      }}

      /* Force text color on ALL Streamlit elements (but not inline-styled spans like logo) */
      h1,h2,h3,h4,h5,h6 {{ color: var(--text) !important; }}
      p, li, div, label {{ color: var(--text); }}
      .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown strong,
      .stMarkdown b, .stMarkdown em, .stMarkdown a, .stText {{ color: var(--text) !important; }}
      [data-testid="stMetricValue"],
      [data-testid="stMetricLabel"],
      [data-testid="stMetricDelta"] {{ color: var(--text) !important; }}

      /* ══════════════════  SIDEBAR  ══════════════════ */
      [data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        color: var(--text) !important;
        border-right: 1px solid var(--sidebar-border);
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
      }}
      [data-testid="stSidebar"],
      [data-testid="stSidebar"] *,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] span,
      [data-testid="stSidebar"] div {{
        color: var(--text) !important;
      }}
      /* Restore collapse arrow button — our * rule above must not garble it */
      [data-testid="collapsedControl"],
      [data-testid="collapsedControl"] *,
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="stSidebarCollapsedControl"] * {{
        color: inherit !important;
        font-family: inherit !important;
      }}
      [data-testid="stSidebar"] .stRadio > div {{ background: transparent !important; }}
      /* Sidebar input backgrounds */
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] textarea {{
        background: var(--input-bg) !important;
        color: var(--text) !important;
        border-color: var(--input-border) !important;
      }}
      [data-testid="stSidebar"] [data-baseweb="input"],
      [data-testid="stSidebar"] [data-baseweb="base-input"] {{
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
      }}
      [data-testid="stSidebar"] [data-baseweb="select"],
      [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
      }}
      [data-testid="stSidebar"] [data-baseweb="select"] * {{
        color: var(--text) !important;
      }}
      /* ── Sidebar section headers ── */
      .sb-section {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent) !important;
        margin: 1.1rem 0 0.4rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid var(--divider);
      }}
      /* ── Sidebar run button — rainbow gradient ── */
      .run-btn > button {{
        background: linear-gradient(90deg, #ff6b6b, #f97316, #facc15, #22c55e, #0ea5e9, #a855f7) !important;
        background-size: 200% 200% !important;
        animation: gradient-shift 4s ease infinite !important;
        color: #fff !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1rem !important;
        letter-spacing: 0.03em !important;
        box-shadow: 0 4px 20px rgba(139,147,255,0.35) !important;
      }}
      @keyframes gradient-shift {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
      }}
      /* ── Status pill ── */
      .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 4px;
      }}
      .status-ok  {{ background: rgba(34,197,94,0.15); color: #22c55e !important; border: 1px solid rgba(34,197,94,0.3); }}
      .status-err {{ background: rgba(239,68,68,0.12); color: #ef4444 !important; border: 1px solid rgba(239,68,68,0.3); }}
      /* ── Sidebar logo badge ── */
      .sb-logo {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.6rem 0.8rem;
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        margin-bottom: 0.8rem;
      }}
      .sb-logo-text {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.03em;
      }}

      /* ══════════════════  Cards  ══════════════════ */
      .card {{
        border: 1px solid var(--card-border);
        border-left: 4px solid var(--accent);
        background: var(--card) !important;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        color: var(--text) !important;
        box-shadow: 0 2px 12px var(--shadow);
        transition: box-shadow 0.2s, transform 0.2s;
      }}
      .card:hover {{
        box-shadow: 0 4px 20px var(--shadow);
        transform: translateY(-1px);
      }}
      .card * {{ color: var(--text) !important; }}

      /* ══════════════════  Buttons  ══════════════════ */
      /* Button element — structure + always white text */
      .stButton button,
      [data-testid="stDownloadButton"] button {{
        background: var(--accent) !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 8px !important;
        padding: 0.3rem 0.85rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        box-shadow: 0 2px 8px var(--shadow) !important;
        line-height: 1.5 !important;
      }}
      /* All children — strip box styles, force white text */
      .stButton button p,
      .stButton button span,
      .stButton button div,
      .stButton button em,
      .stButton button strong,
      .stButton button small,
      .stButton button *,
      [data-testid="stDownloadButton"] button p,
      [data-testid="stDownloadButton"] button span,
      [data-testid="stDownloadButton"] button div,
      [data-testid="stDownloadButton"] button em,
      [data-testid="stDownloadButton"] button *,
      [data-testid="stDownloadButton"] [data-testid="stMarkdownContainer"],
      [data-testid="stDownloadButton"] [data-testid="stMarkdownContainer"] * {{
        color: #ffffff !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        border: 0 !important;
        box-shadow: none !important;
        border-radius: 0 !important;
      }}
      .stButton button:hover,
      [data-testid="stDownloadButton"] button:hover {{ filter: brightness(1.06); }}
      .stButton button:focus {{ outline: 3px solid var(--focus); outline-offset: 2px; }}

      /* ══════════════════  Text Inputs  ══════════════════ */
      input[type="text"], input[type="password"], input[type="number"],
      textarea {{
        background: var(--input-bg) !important;
        color: var(--text) !important;
        border-color: var(--input-border) !important;
      }}
      /* BaseWeb input wrapper (Streamlit uses BaseWeb internally) */
      [data-baseweb="input"],
      [data-baseweb="base-input"] {{
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
      }}
      [data-baseweb="input"] input,
      [data-baseweb="base-input"] input {{
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
      }}
      /* Placeholder text */
      input::placeholder, textarea::placeholder {{
        color: var(--muted) !important;
        -webkit-text-fill-color: var(--muted) !important;
      }}

      /* ══════════════════  Selectbox / Multiselect (BaseWeb)  ══════════════════ */
      [data-baseweb="select"] {{
        background-color: var(--input-bg) !important;
      }}
      [data-baseweb="select"] > div {{
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
        color: var(--text) !important;
      }}
      [data-baseweb="select"] * {{
        color: var(--text) !important;
      }}
      /* Multiselect tags / pills */
      [data-baseweb="tag"] {{
        background-color: var(--chip-bg) !important;
        color: var(--text) !important;
        border-color: var(--chip-border) !important;
      }}
      [data-baseweb="tag"] * {{ color: var(--text) !important; }}
      /* Dropdown menu */
      [data-baseweb="popover"],
      [data-baseweb="menu"],
      [data-baseweb="popover"] ul,
      [data-baseweb="menu"] ul {{
        background-color: var(--card) !important;
        border-color: var(--card-border) !important;
      }}
      [data-baseweb="popover"] li,
      [data-baseweb="menu"] li {{
        background-color: var(--card) !important;
        color: var(--text) !important;
      }}
      [data-baseweb="popover"] li:hover,
      [data-baseweb="menu"] li:hover {{
        background-color: var(--chip-bg) !important;
      }}

      /* ══════════════════  File Uploader  ══════════════════ */
      .stFileUploader,
      .stFileUploader * {{
        color: var(--text) !important;
      }}
      [data-testid="stFileUploaderDropzone"] {{
        background: var(--input-bg) !important;
        border-color: var(--input-border) !important;
        color: var(--text) !important;
      }}
      [data-testid="stFileUploaderDropzone"] * {{
        color: var(--text) !important;
      }}
      [data-testid="stFileUploaderDropzone"] button {{
        background: var(--accent) !important;
        color: var(--accent-contrast) !important;
      }}
      /* File uploader small text */
      [data-testid="stFileUploaderDropzone"] small {{
        color: var(--muted) !important;
      }}

      /* ══════════════════  Slider  ══════════════════ */
      .stSlider label, .stSlider p, .stSlider span {{ color: var(--text) !important; }}
      .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background: var(--accent) !important;
      }}

      /* ══════════════════  Radio & Checkbox  ══════════════════ */
      .stRadio label, .stCheckbox label {{ color: var(--text) !important; }}
      .stRadio [role="radiogroup"] label span {{ color: var(--text) !important; }}

      /* ══════════════════  Tabs  ══════════════════ */
      .stTabs [data-baseweb="tab-list"] {{
        background: var(--tab-bg) !important;
        border-radius: 10px;
        gap: 4px;
        padding: 4px;
      }}
      .stTabs [data-baseweb="tab"] {{
        color: var(--text) !important;
        background: transparent !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border-radius: 8px;
        margin: 0 2px;
      }}
      .stTabs [aria-selected="true"] {{
        color: var(--accent) !important;
        border-bottom: 3px solid var(--accent) !important;
        background: var(--card) !important;
        box-shadow: 0 2px 8px var(--shadow);
      }}
      .stTabs [data-baseweb="tab-panel"] {{
        background: transparent !important;
        color: var(--text) !important;
        padding-top: 1rem !important;
      }}
      .stTabs [data-baseweb="tab-border"] {{
        background-color: var(--divider) !important;
      }}
      /* Inner tabs (e.g. Item / Metrics / Full raw inside Inspect JSON) */
      .stTabs .stTabs [data-baseweb="tab-list"] {{
        background: var(--expander-bg) !important;
      }}
      .stTabs .stTabs [data-baseweb="tab"] {{
        padding: 0.4rem 1.2rem !important;
        font-size: 0.9rem !important;
      }}

      /* ══════════════════  Expanders  ══════════════════ */
      [data-testid="stExpander"] {{
        border: 1px solid var(--card-border) !important;
        background: var(--card) !important;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 4px var(--shadow);
        overflow: hidden;
      }}
      [data-testid="stExpander"] summary,
      [data-testid="stExpander"] summary * {{
        color: var(--text) !important;
        background: var(--expander-bg) !important;
      }}
      [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background: var(--card) !important;
        color: var(--text) !important;
      }}
      [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {{
        color: var(--text) !important;
      }}
      /* Legacy class names */
      .streamlit-expanderHeader {{
        background: var(--expander-bg) !important;
        color: var(--text) !important;
      }}
      .streamlit-expanderContent {{
        background: var(--card) !important;
        color: var(--text) !important;
      }}

      /* ══════════════════  Metrics  ══════════════════ */
      [data-testid="stMetric"],
      [data-testid="metric-container"] {{
        background: var(--card) !important;
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 6px var(--shadow);
      }}
      [data-testid="stMetric"] * {{ color: var(--text) !important; }}
      [data-testid="stMetricValue"] {{
        font-size: 1.6rem !important;
        font-weight: 700 !important;
      }}

      /* ══════════════════  DataFrames  ══════════════════ */
      .stDataFrame {{
        color: var(--text) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px;
        overflow: hidden;
      }}
      .stDataFrame [data-testid="glideDataEditor"],
      .stDataFrame [data-testid="stDataFrameResizable"] {{
        background: var(--card) !important;
      }}
      /* Glide header and cells */
      .stDataFrame [data-testid="glideDataEditor"] .dvn-scroller,
      .stDataFrame [data-testid="glideDataEditor"] .dvn-scroller > div {{
        background: var(--card) !important;
      }}
      .stDataFrame th, .stDataFrame td {{
        color: var(--text) !important;
        background: var(--card) !important;
      }}
      /* Glide data editor canvas overlay colors */
      .stDataFrame canvas {{
        opacity: 1 !important;
      }}
      /* Table-style dataframes (fallback) */
      .stDataFrame table {{
        background: var(--card) !important;
      }}
      .stDataFrame table th {{
        background: var(--expander-bg) !important;
        color: var(--text) !important;
      }}
      .stDataFrame table td {{
        background: var(--card) !important;
        color: var(--text) !important;
      }}

      /* ══════════════════  Alerts  ══════════════════ */
      .stAlert p, .stAlert div {{ color: inherit !important; }}

      /* ══════════════════  Divider  ══════════════════ */
      hr {{ border-color: var(--divider) !important; }}

      /* ══════════════════  Chips  ══════════════════ */
      .chip {{
        display:inline-flex; align-items:center; gap:.4rem;
        padding: .3rem .7rem; border-radius:999px;
        background: var(--chip-bg); border: 1px solid var(--chip-border);
        font-size:.82rem; color: var(--text);
        font-weight: 600;
        letter-spacing: 0.01em;
        margin: 2px 3px;
        transition: background 0.15s;
      }}
      .chip:hover {{
        background: var(--accent);
        color: var(--accent-contrast);
        border-color: var(--accent);
      }}

      /* ══════════════════  Sticky summary  ══════════════════ */
      #summary-card {{
        position: sticky;
        top: .5rem;
        z-index: 50;
        border: 1px solid var(--card-border);
        border-top: 3px solid var(--accent);
        background: var(--card) !important;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        color: var(--text) !important;
        box-shadow: 0 4px 20px var(--shadow);
      }}
      #summary-card * {{ color: var(--text) !important; }}

      /* ══════════════════  Focus ring (a11y)  ══════════════════ */
      input:focus, select:focus, textarea:focus {{
        outline: 3px solid var(--focus) !important; outline-offset: 1px !important;
      }}

      /* ══════════════════  kbd / muted / footer  ══════════════════ */
      kbd {{
        background: var(--kbd); color: var(--accent-contrast); border-radius:6px;
        padding: 1px 6px; font-size: .8em; font-weight: 700;
      }}
      .muted {{ color: var(--muted) !important; }}
      footer {{ text-align:center; margin-top: 1rem; color: var(--muted); }}

      /* ══════════════════  JSON / Code blocks  ══════════════════ */
      .stCode, pre, code {{
        background: var(--expander-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px;
      }}
      /* st.json viewer */
      [data-testid="stJson"] {{
        background: var(--expander-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px;
        padding: 0.5rem !important;
      }}
      [data-testid="stJson"] * {{
        color: var(--text) !important;
      }}
      /* react-json-view overrides (the widget st.json uses) */
      .react-json-view {{
        background: var(--expander-bg) !important;
        color: var(--text) !important;
      }}
      .react-json-view .string-value {{ color: var(--accent) !important; }}
      .react-json-view .object-key-val,
      .react-json-view .object-key-val span {{
        color: var(--text) !important;
      }}
      /* st.code block container */
      [data-testid="stCodeBlock"] {{
        background: var(--expander-bg) !important;
      }}
      [data-testid="stCodeBlock"] * {{
        color: var(--text) !important;
      }}

      /* ══════════════════  Download buttons  ══════════════════ */
      .stDownloadButton button {{
        background: var(--accent) !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px var(--shadow) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.15s;
      }}
      .stDownloadButton button:hover {{ filter: brightness(1.06); }}
      .stDownloadButton button *,
      .stDownloadButton [data-testid="stMarkdownContainer"],
      .stDownloadButton [data-testid="stMarkdownContainer"] * {{
        color: #ffffff !important;
        background: transparent !important;
        padding: 0 !important;
        border: 0 !important;
        box-shadow: none !important;
        border-radius: 0 !important;
      }}

      /* ══════════════════  Caption  ══════════════════ */
      .stCaption, [data-testid="stCaptionContainer"] {{
        color: var(--muted) !important;
      }}
      [data-testid="stCaptionContainer"] * {{ color: var(--muted) !important; }}

      /* ══════════════════  Number input  ══════════════════ */
      .stNumberInput button {{
        background: var(--chip-bg) !important;
        color: var(--text) !important;
        border-color: var(--input-border) !important;
      }}
      .stNumberInput [data-baseweb="input"] {{
        background-color: var(--input-bg) !important;
      }}

      /* ══════════════════  Tooltip / popover  ══════════════════ */
      [data-baseweb="tooltip"] {{
        background: var(--card) !important;
        color: var(--text) !important;
      }}

      /* ══════════════════  Spinner / progress  ══════════════════ */
      .stSpinner > div {{ color: var(--text) !important; }}

      /* ══════════════════  st.write / st.info / etc containers  ══════════════════ */
      [data-testid="stNotification"] {{ color: var(--text) !important; }}

      /* ══════════════════  Themed HTML table (replaces canvas dataframe)  ══════════════════ */
      .themed-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border: 1px solid var(--card-border);
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
        box-shadow: 0 1px 6px var(--shadow);
      }}
      .themed-table th {{
        background: var(--expander-bg) !important;
        color: var(--text) !important;
        padding: 0.6rem 0.85rem;
        text-align: left;
        font-weight: 700;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        border-bottom: 2px solid var(--card-border);
      }}
      .themed-table td {{
        background: var(--card) !important;
        color: var(--text) !important;
        padding: 0.5rem 0.85rem;
        font-size: 0.88rem;
        border-bottom: 1px solid var(--card-border);
      }}
      .themed-table tbody tr:nth-child(even) td {{
        background: var(--expander-bg) !important;
      }}
      .themed-table tr:last-child td {{
        border-bottom: none;
      }}
      .themed-table tr:hover td {{
        background: var(--chip-bg) !important;
      }}

      /* ══════════════════  Scrollbar (for light themes)  ══════════════════ */
      ::-webkit-scrollbar-track {{
        background: var(--bg-alt);
      }}
      ::-webkit-scrollbar-thumb {{
        background: var(--muted);
        border-radius: 4px;
      }}

      /* ══════════════════  Tab section headings — centered  ══════════════════ */
      .tab-heading {{
        text-align: center !important;
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: var(--text) !important;
        margin: 0.4rem 0 1.2rem 0 !important;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ============================== HELPERS =======================================

def read_jsonl_bytes(file_bytes: bytes) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ln in file_bytes.decode("utf-8").splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows

def build_download(data: str | bytes, filename: str, mime: str, key: str | None = None):
    return st.download_button(
        label=f"⬇ Download {filename}",
        data=data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8"),
        file_name=filename,
        mime=mime,
        use_container_width=True,
        key=key,
    )

def _overall_from_report(report: Dict[str, Any]) -> Optional[float]:
    try:
        rb = ReportBuilder(report)
        aggregates = [r.get("aggregate") for r in rb.results if isinstance(r.get("aggregate"), (int, float))]
        if aggregates:
            return float(statistics.mean(aggregates))
        scores = []
        for r in rb.results:
            for m in r.get("metrics", []):
                s = m.get("score")
                if isinstance(s, (int, float)):
                    scores.append(float(s))
        if scores:
            return float(statistics.mean(scores))
    except Exception:
        pass
    return None

def _compute_item_score(r: Dict[str, Any]) -> Optional[float]:
    agg = r.get("aggregate")
    if isinstance(agg, (int, float)):
        return float(agg)
    vals = [m.get("score") for m in r.get("metrics", []) if isinstance(m.get("score"), (int, float))]
    return float(statistics.mean(vals)) if vals else None

def _priority_score(r: Dict[str, Any]) -> float:
    """Score used for worst-case ranking: prefer answer_overall > retrieval_overall > mean.
    Returns a float (lower = worse). Falls back to 1.0 if nothing available."""
    for preferred in ("answer_overall", "retrieval_overall"):
        for m in r.get("metrics", []):
            if m.get("name") == preferred and isinstance(m.get("score"), (int, float)):
                return float(m["score"])
    return _compute_item_score(r) or 1.0


def _score_color(score: float) -> str:
    """Return a color for a 0-1 score: red → yellow → green."""
    if score >= 0.8:
        return "#22c55e"
    elif score >= 0.6:
        return "#84cc16"
    elif score >= 0.4:
        return "#eab308"
    elif score >= 0.2:
        return "#f97316"
    else:
        return "#ef4444"


def _themed_table(rows: List[Dict[str, Any]], score_col: str = "Score", compact: bool = False):
    """Render a list of dicts as a themed HTML table with score bars."""
    if not rows:
        return
    headers = list(rows[0].keys())
    ths = ""
    for h in headers:
        ths += f"<th>{h}</th>"
    if score_col in headers:
        ths += "<th></th>"  # bar column

    body = ""
    for r in rows:
        tds = ""
        score_val = None
        for h in headers:
            val = r.get(h, "")
            if h == score_col and isinstance(val, (int, float)):
                score_val = float(val)
                tds += f'<td style="font-weight:600;">{val}</td>'
            else:
                tds += f"<td>{val}</td>"
        # Add score bar cell
        if score_col in headers:
            if score_val is not None:
                pct = min(max(score_val * 100, 0), 100)
                clr = _score_color(score_val)
                tds += f'''<td style="width:120px;">
                    <div style="background:var(--chip-bg);border-radius:4px;height:8px;width:100%;overflow:hidden;">
                        <div style="background:{clr};height:100%;width:{pct}%;border-radius:4px;transition:width .3s;"></div>
                    </div>
                </td>'''
            else:
                tds += "<td></td>"
        body += f"<tr>{tds}</tr>"

    max_w = "max-width:600px;" if compact else ""
    html = f"""
    <div style="{max_w}">
    <table class="themed-table">
      <thead><tr>{ths}</tr></thead>
      <tbody>{body}</tbody>
    </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_metric_diagnostics(name: str, m: Dict[str, Any]):
    """Render metric-specific diagnostic fields in the Streamlit UI."""

    if name == "context_utilization":
        utilized = m.get("utilized_chunks", [])
        unused = m.get("unused_chunks", [])
        if utilized:
            st.write(f"**Utilized chunks:** {utilized}")
        if unused:
            st.warning(f"**Unused chunks:** {unused}")

    elif name == "answer_conciseness":
        redundant = m.get("redundant_parts", [])
        filler = m.get("filler_detected", [])
        if redundant:
            st.write("**Redundant parts:**")
            for r in redundant:
                st.markdown(f"- {r}")
        if filler:
            st.write("**Filler detected:**")
            for f in filler:
                st.markdown(f"- _{f}_")

    elif name == "negative_rejection":
        ctx_suf = m.get("context_sufficient")
        ans_ref = m.get("answer_refuses")
        if ctx_suf is not None:
            st.write(f"**Context sufficient:** {'Yes' if ctx_suf else 'No'}")
        if ans_ref is not None:
            st.write(f"**Answer refuses:** {'Yes' if ans_ref else 'No'}")

    elif name == "coherence":
        contras = m.get("contradictions", [])
        issues = m.get("logical_issues", [])
        if contras:
            st.write("**Contradictions found:**")
            for c in contras:
                st.markdown(f"- {c}")
        if issues:
            st.write("**Logical issues:**")
            for i in issues:
                st.markdown(f"- {i}")

    elif name == "multi_hop_faithfulness":
        chain = m.get("reasoning_chain", [])
        valid = m.get("valid_hops", [])
        broken = m.get("broken_hops", [])
        if chain:
            st.write("**Reasoning chain:**")
            for idx, step in enumerate(chain, 1):
                icon = "✅" if idx in valid else "❌"
                st.markdown(f"{icon} **Step {idx}:** {step}")
        if broken:
            st.write("**Broken hops:**")
            for b in broken:
                if isinstance(b, dict):
                    st.markdown(f"- Hop {b.get('hop', '?')}: {b.get('reason', '')}")
                else:
                    st.markdown(f"- {b}")

    elif name == "implicit_contradiction":
        contras = m.get("contradictions", [])
        types = m.get("contradiction_types", [])
        if types:
            st.write(f"**Contradiction types:** {', '.join(types)}")
        if contras:
            st.write("**Contradictions:**")
            for c in contras:
                if isinstance(c, dict):
                    st.markdown(
                        f"- **Answer claim:** {c.get('answer_claim', '')}\n"
                        f"  **Context states:** {c.get('context_states', '')}\n"
                        f"  **Type:** {c.get('type', '')} | **Severity:** {c.get('severity', '')}"
                    )
                else:
                    st.markdown(f"- {c}")


def render_report(report: Dict[str, Any], *, agentic_mode: bool, min_item_score: float, key_prefix: str = "report"):
    rb = ReportBuilder(report)

    # ===== Sticky Summary =====
    st.markdown('<div id="summary-card">', unsafe_allow_html=True)
    st.subheader("📈 Summary")
    cols = st.columns([1, 1, 1,2 ])
    with cols[0]:
        st.metric("Items", len(rb.results))
    with cols[1]:
        overall = _overall_from_report(report)
        st.metric("Overall (mean)", f"{overall:.3f}" if overall is not None else "n/a")
        # mean eval_time_sec across all cases
    with cols[2]:
        eval_times = [
            r.get("eval_time_sec")
            for r in rb.results
            if isinstance(r.get("eval_time_sec"), (int, float))
        ]
        if eval_times:
            mean_time = statistics.mean(eval_times)
            st.metric("Mean eval time (s)", f"{mean_time:.2f}")
        else:
            st.metric("Mean eval time (s)", "n/a")

    with cols[3]:
        st.caption("Mode: Agentic" if agentic_mode else "Mode: Manual")
    st.markdown("</div>", unsafe_allow_html=True)

    # Per-metric mean table
    if rb.summary:
        rows = [{"Metric": k, "Score": float(f"{v:.3f}")} for k, v in sorted(rb.summary.items())]
        _themed_table(rows, compact=True)
    else:
        st.info("No per-metric mean table provided by the current metrics.")

    # Score distributions (only meaningful for >= 10 items)
    if len(rb.results) >= 10:
        with st.expander("📊 Score Distributions", expanded=False):
            import pandas as pd
            _metric_scores: Dict[str, List[float]] = {}
            for _r in rb.results:
                for _m in _r.get("metrics", []) or []:
                    _n = _m.get("name")
                    _s = _m.get("score")
                    if _n and isinstance(_s, (int, float)):
                        _metric_scores.setdefault(_n, []).append(float(_s))

            if _metric_scores:
                # Percentile + failure rate table
                _dist_rows = []
                for _name, _scores in sorted(_metric_scores.items()):
                    _ss = sorted(_scores)
                    _cnt = len(_ss)
                    def _pct(p, ss=_ss): return ss[max(0, int(p * len(ss) / 100) - 1)]
                    _fail = round(sum(1 for s in _scores if s < 0.5) / _cnt * 100, 1)
                    _dist_rows.append({
                        "Metric": _name,
                        "N": _cnt,
                        "Mean": round(statistics.mean(_scores), 3),
                        "p25": round(_pct(25), 3),
                        "p50": round(_pct(50), 3),
                        "p75": round(_pct(75), 3),
                        "% < 0.5 (failures)": _fail,
                    })
                st.dataframe(_dist_rows, use_container_width=True)

                # Per-metric histogram
                _dist_sel = st.selectbox(
                    "View histogram for:",
                    sorted(_metric_scores.keys()),
                    key=f"{key_prefix}_dist_sel",
                )
                if _dist_sel:
                    _bin_labels = ["0.0–0.1","0.1–0.2","0.2–0.3","0.3–0.4","0.4–0.5",
                                   "0.5–0.6","0.6–0.7","0.7–0.8","0.8–0.9","0.9–1.0"]
                    _counts = [0] * 10
                    for _s in _metric_scores[_dist_sel]:
                        _counts[min(int(_s * 10), 9)] += 1
                    _df_hist = pd.DataFrame({"Count": _counts}, index=_bin_labels)
                    st.bar_chart(_df_hist)

    st.divider()

    # ===== Cases =====
    st.subheader("🧩 Individual Case Results")

    _THRESHOLD_PAGINATE = 50
    _THRESHOLD_WORST    = 200
    _PAGE_SIZE          = 20
    n_total = len(rb.results)

    # Apply min_item_score filter
    filtered = [
        (i + 1, r) for i, r in enumerate(rb.results)
        if not ((_compute_item_score(r) is not None) and (_compute_item_score(r) < min_item_score))
    ]

    # ── Case-level inspection toggle ─────────────────────────────────────────
    show_cases = st.checkbox(
        "Show case-level inspection",
        value=(n_total < _THRESHOLD_WORST),
        key=f"{key_prefix}_show_cases",
        help=f"Auto-disabled for datasets ≥ {_THRESHOLD_WORST} items. Summary and export are always available.",
    )

    if not show_cases:
        st.info(f"📊 {n_total} items — case-level inspection disabled. Enable the checkbox above to inspect individual cases, or use export to analyse the full dataset.")
    else:
        # Auto-select view mode based on dataset size
        _auto = "Full" if n_total < _THRESHOLD_PAGINATE else ("Paginated" if n_total < _THRESHOLD_WORST else "Worst cases only")
        _mode_options = ["Full", "Paginated", "Worst cases only"]
        view_mode = st.radio(
            "Case view:",
            _mode_options,
            index=_mode_options.index(_auto),
            horizontal=True,
            key=f"{key_prefix}_view_mode",
            help=f"Auto-selected for {n_total} items — Full (<{_THRESHOLD_PAGINATE}), Paginated ({_THRESHOLD_PAGINATE}–{_THRESHOLD_WORST}), Worst cases only (>{_THRESHOLD_WORST}).",
        )

        if not filtered:
            st.warning("No cases pass the current minimum score filter.")
        else:
            if view_mode == "Full":
                to_render = filtered

            elif view_mode == "Paginated":
                _page_key = f"{key_prefix}_page"
                if _page_key not in st.session_state:
                    st.session_state[_page_key] = 0
                _total_pages = max(1, (len(filtered) + _PAGE_SIZE - 1) // _PAGE_SIZE)
                _page = st.session_state[_page_key]
                _start = _page * _PAGE_SIZE

                _pc, _pi, _nc = st.columns([1, 3, 1])
                with _pc:
                    if st.button("← Prev", key=f"{key_prefix}_prev", disabled=(_page == 0)):
                        st.session_state[_page_key] -= 1
                        st.rerun()
                with _pi:
                    st.caption(f"Page {_page + 1} of {_total_pages}  ·  items {_start + 1}–{min(_start + _PAGE_SIZE, len(filtered))} of {len(filtered)}")
                with _nc:
                    if st.button("Next →", key=f"{key_prefix}_next", disabled=(_page >= _total_pages - 1)):
                        st.session_state[_page_key] += 1
                        st.rerun()

                to_render = filtered[_start : _start + _PAGE_SIZE]

            else:  # Worst cases only
                _sorted = sorted(filtered, key=lambda x: _priority_score(x[1]))
                to_render = _sorted[:10]
                st.info(f"📊 {n_total} items — showing 10 worst-scoring cases (ranked by answer_overall → retrieval_overall → mean). Switch to **Paginated** to browse all.")

            for idx, r in to_render:
                item_score = _compute_item_score(r)
                item = r.get("item", {})
                q = item.get("question", "")
                a = item.get("answer", None)
                ctxs = item.get("contexts", [])
                metrics = r.get("metrics", []) or []
                eval_time = r.get("eval_time_sec", None)
                title = f"Case {idx} — score {item_score:.3f}" if item_score is not None else f"Case {idx}"
                with st.expander(title, expanded=(idx == 1)):
                    colL, colR = st.columns([2, 1])

                    with colL:
                        st.markdown(f"**Question**  \n{q}")
                        if a is not None:
                            val = a if (isinstance(a, str) and a.strip()) else "∅ (no answer)"
                            st.markdown(f"**Answer**  \n{val}")
                        if ctxs:
                            st.markdown("**Contexts**")
                            for i, c in enumerate(ctxs, 1):
                                st.caption(f"[{i}] {c}")

                    with colR:
                        if isinstance(item_score, (int, float)):
                            st.metric("Aggregate (case)", f"{item_score:.3f}")
                        if isinstance(eval_time, (int, float)):
                            st.metric("Eval time (s)", f"{eval_time:.2f}")
                        st.caption(f"Metrics computed: {len(metrics)}")

                    # Compact metrics table
                    rows = []
                    for m in metrics:
                        row = {
                            "Metric": m.get("name", ""),
                            "Score": float(f"{m.get('score', 0.0):.3f}"),
                        }
                        just = m.get("justification") or m.get("explanation") or ""
                        if just:
                            row["Summary"] = just[:120] + ("..." if len(just) > 120 else "")
                        rows.append(row)

                    if rows:
                        st.markdown("**Metrics**")
                        _themed_table(rows)
                    else:
                        st.info("No metrics computed for this case.")

                    # ===== JSON inspection: nicer tabs =====
                    with st.expander("Inspect JSON", expanded=False):
                        tab_item, tab_metrics, tab_raw = st.tabs(["Item", "Metrics", "Full raw"])

                        # --- Item tab: clean view of Q/A/contexts ---
                        with tab_item:
                            st.subheader("Item")
                            item_view = {
                                "question": item.get("question"),
                                "answer": item.get("answer"),
                                "contexts": item.get("contexts"),
                                "eval_time_sec": round(eval_time, 2) if isinstance(eval_time, (int, float)) else eval_time,
                            }
                            st.json(item_view)

                        # --- Metrics tab: structured + per-metric details ---
                        with tab_metrics:
                            st.subheader("Metrics (score + explanation)")

                            if not metrics:
                                st.info("No metrics available for this case.")
                            else:
                                for m in metrics:
                                    m_name = m.get("name", "unknown")
                                    m_score = m.get("score")
                                    m_expl = m.get("explanation")
                                    m_just = m.get("justification")
                                    m_details = m.get("details")

                                    with st.expander(f"Metric: {m_name}", expanded=False):
                                        # Score
                                        if isinstance(m_score, (int, float)):
                                            st.write("**Score:**", float(f"{m_score:.3f}"))
                                        else:
                                            st.write("**Score:**", m_score)

                                        # Explanation or justification
                                        if m_expl:
                                            st.write("**Explanation:**")
                                            st.markdown(f"> {m_expl}")
                                        if m_just:
                                            st.write("**Justification:**")
                                            st.markdown(f"> {m_just}")

                                        # --- Metric-specific diagnostic fields ---
                                        _render_metric_diagnostics(m_name, m)

                                        # Details as pretty JSON (optional, only if present)
                                        if m_details:
                                            st.write("**Details (JSON):**")
                                            st.code(
                                                json.dumps(m_details, ensure_ascii=False, indent=2),
                                                language="json",
                                            )

                        # --- Full raw tab: complete JSON dump ---
                        with tab_raw:
                            st.subheader("Full raw result")
                            pretty = json.dumps(r, ensure_ascii=False, indent=2)
                            if len(pretty) > 6000:
                                pretty = pretty[:6000] + "\n...\n(truncated)"
                            st.code(pretty, language="json")

    # ===== Export =====
    st.divider()
    st.subheader("📦 Export")

    md = rb.to_markdown()
    html = rb.to_html()
    js = json.dumps(report, ensure_ascii=False, indent=2)

    # ========= CSVs =========
    # A) per-metric CSV
    csv_metrics_buf = io.StringIO()
    rows_metrics = []

    # B) per-item flat CSV (with one column per metric)
    csv_items_buf = io.StringIO()
    rows_items = []

    for idx, r in enumerate(rb.results):
        item = r.get("item", {})
        q = item.get("question", "")
        a = item.get("answer", "")
        ctx = item.get("contexts", [])
        ctx_str = " || ".join([str(x) for x in ctx]) if isinstance(ctx, list) else str(ctx)
        agg = _compute_item_score(r)
        eval_time = r.get("eval_time_sec", None)
        metrics = r.get("metrics", []) or []

        agg_r = _rf(agg)
        eval_time_r = _rf(eval_time)

        # ---- per-metric rows (rounded) ----
        for m in metrics:
            rows_metrics.append({
                "item_index": idx,
                "metric": m.get("name", ""),
                "score": _rf(m.get("score", None)),
                "aggregate_for_item": agg_r,
                "question": q,
                "answer": a,
                "contexts": ctx_str,
                "explanation": m.get("explanation", ""),
                "eval_time_sec": eval_time_r,
            })

        # ---- per-item flat row (rounded) ----
        item_row = {
            "item_index": idx,
            "question": q,
            "answer": a,
            "contexts": ctx_str,
            "aggregate_for_item": agg_r,
            "eval_time_sec": eval_time_r,
        }
        # add one column per metric: metric__name
        for m in metrics:
            m_name = m.get("name", "")
            if not m_name:
                continue
            col_name = f"metric__{m_name}"
            item_row[col_name] = _rf(m.get("score", None))

        rows_items.append(item_row)

    # write per-metric CSV
    if rows_metrics:
        writer = csv.DictWriter(csv_metrics_buf, fieldnames=list(rows_metrics[0].keys()))
        writer.writeheader()
        for rr in rows_metrics:
            writer.writerow(rr)

    # write per-item CSV
    if rows_items:
        # collect all columns across items so we don't miss any metric columns
        all_keys = set()
        for r in rows_items:
            all_keys.update(r.keys())
        core_cols = ["item_index", "question", "answer", "contexts", "aggregate_for_item", "eval_time_sec"]
        metric_cols = sorted([k for k in all_keys if k.startswith("metric__")])
        other_cols = [k for k in all_keys if k not in core_cols + metric_cols]
        fieldnames_items = core_cols + metric_cols + sorted(other_cols)

        writer_items = csv.DictWriter(csv_items_buf, fieldnames=fieldnames_items)
        writer_items.writeheader()
        for r in rows_items:
            writer_items.writerow(r)

    # ========= Download buttons =========
    cols = st.columns(5)
    with cols[0]:
        build_download(js, "report.json", "application/json", key=f"{key_prefix}_dl_json")
    with cols[1]:
        build_download(md, "report.md", "text/markdown", key=f"{key_prefix}_dl_md")
    with cols[2]:
        if rows_metrics:
            build_download(csv_metrics_buf.getvalue(), "report_metrics.csv", "text/csv", key=f"{key_prefix}_dl_csv_metrics")
        else:
            st.button("report_metrics.csv(no rows)", disabled=True, use_container_width=True, key=f"{key_prefix}_dl_csv_metrics_empty")
    with cols[3]:
        if rows_items:
            build_download(csv_items_buf.getvalue(), "report_items_flat.csv", "text/csv", key=f"{key_prefix}_dl_csv_items")
        else:
            st.button("report_items_flat.csv (no rows)", disabled=True, use_container_width=True, key=f"{key_prefix}_dl_csv_items_empty")
    with cols[4]:
        build_download(html, "report.html", "text/html", key=f"{key_prefix}_dl_html")


# ============================== PAGE CONFIG ==================================
st.set_page_config(
    page_title="RAGVue Dashboard",
    page_icon="assets/favicon.png",
    layout="wide"
)
inject_theme(THEMES[st.session_state.get("theme", "Light")])

# --- API key widget callbacks ----------------------------------------------
def _use_api_key():
    # Read current text from the widget
    ui_key = st.session_state.get("api_key_input", "").strip()
    if ui_key:
        set_api_key_temporarily(ui_key)
        st.session_state["api_key_message"] = "set"
    else:
        # No key typed
        set_api_key_temporarily(None)
        st.session_state["api_key_message"] = "empty"


def _forget_api_key():
    # Clear runtime key
    set_api_key_temporarily(None)
    # Clear the textbox itself so 👁️ shows nothing
    st.session_state["api_key_input"] = ""
    st.session_state["api_key_message"] = "cleared"


# ============================== SIDEBAR ======================================
with st.sidebar:

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex; align-items:baseline; justify-content:space-between;
             padding: 0.3rem 0 0.8rem 0; border-bottom: 1px solid var(--divider); margin-bottom:0.4rem;">
          <span style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.1rem;
            font-weight:700; color:var(--text); letter-spacing:-0.01em;">⚙️ Settings</span>
          <span style="font-size:0.65rem; font-weight:600; color:var(--accent);
            background:var(--chip-bg); border:1px solid var(--chip-border);
            border-radius:5px; padding:2px 7px;">v0.4</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Theme ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">🎨 Appearance</div>', unsafe_allow_html=True)
    theme_choice = st.radio(
        "Theme:",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.get("theme", "Light")),
        key="theme",
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Judge Provider ────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">🧠 Judge Provider</div>', unsafe_allow_html=True)
    provider_choice = st.radio(
        "LLM judge backend:",
        ["OpenAI (default)", "Claude (Anthropic)"],
        index=0 if os.getenv("RAGVUE_JUDGE_PROVIDER", "openai").lower() != "anthropic" else 1,
        key="judge_provider_radio",
        horizontal=True,
        help="OpenAI uses gpt-4o-mini. Claude uses claude-haiku-4-5 (fast & cheap).",
        label_visibility="collapsed",
    )
    if provider_choice == "Claude (Anthropic)":
        os.environ["RAGVUE_JUDGE_PROVIDER"] = "anthropic"
    else:
        os.environ["RAGVUE_JUDGE_PROVIDER"] = "openai"

    # ── API Key ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">🔐 API Key</div>', unsafe_allow_html=True)

    if provider_choice == "Claude (Anthropic)":
        _ant_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
        st.markdown(
            f'<div class="status-pill {"status-ok" if _ant_ok else "status-err"}">'
            f'{"✅ Anthropic key found" if _ant_ok else "❌ Anthropic key missing"}</div>',
            unsafe_allow_html=True,
        )
        ui_ant_key = st.text_input(
            "ANTHROPIC_API_KEY", type="password",
            placeholder="sk-ant-… (not stored)", key="anthropic_key_input",
            label_visibility="collapsed",
        )
        ant_cols = st.columns(2)
        def _use_anthropic_key():
            k = st.session_state.get("anthropic_key_input", "").strip()
            if k:
                os.environ["ANTHROPIC_API_KEY"] = k
                st.session_state["anthropic_key_message"] = "set"
            else:
                os.environ.pop("ANTHROPIC_API_KEY", None)
                st.session_state["anthropic_key_message"] = "empty"
        def _forget_anthropic_key():
            os.environ.pop("ANTHROPIC_API_KEY", None)
            st.session_state["anthropic_key_input"] = ""
            st.session_state["anthropic_key_message"] = "cleared"
        ant_cols[0].button("Use key", on_click=_use_anthropic_key, key="ant_use_btn", use_container_width=True)
        ant_cols[1].button("Forget", on_click=_forget_anthropic_key, key="ant_forget_btn", use_container_width=True)
        ant_msg = st.session_state.get("anthropic_key_message")
        if ant_msg == "set":   st.success("Key set for this session.")
        elif ant_msg == "cleared": st.info("Key cleared.")
        elif ant_msg == "empty":   st.warning("No key entered.")
    else:
        _oai_ok = bool(get_api_key())
        st.markdown(
            f'<div class="status-pill {"status-ok" if _oai_ok else "status-err"}">'
            f'{"✅ OpenAI key found" if _oai_ok else "❌ OpenAI key missing"}</div>',
            unsafe_allow_html=True,
        )
        ui_key = st.text_input(
            API_ENV_VAR, type="password",
            placeholder="sk-… (not stored)", key="api_key_input",
            label_visibility="collapsed",
        )
        cols = st.columns(2)
        cols[0].button("Use key", on_click=_use_api_key, use_container_width=True)
        cols[1].button("Forget", on_click=_forget_api_key, use_container_width=True)
        msg = st.session_state.get("api_key_message")
        if msg == "set":     st.success("Key set for this session.")
        elif msg == "cleared": st.info("Key cleared.")
        elif msg == "empty":   st.warning("No key entered.")
        if not _oai_ok:
            st.caption("Add `OPENAI_API_KEY` to your `.env` file.")

    # ── Data ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">📁 Data</div>', unsafe_allow_html=True)
    upl = st.file_uploader("Upload `items.jsonl`", type=["jsonl"], label_visibility="collapsed")
    if upl is not None:
        try:
            raw = upl.getvalue()
            items_preview = read_jsonl_bytes(raw)
            st.session_state["uploaded_items"] = items_preview
            st.success(f"✅ {len(items_preview)} item(s) loaded")
        except Exception as e:
            st.error(f"Could not parse: {e}")
    max_items = st.number_input("Limit items (0 = all)", min_value=0, value=0, step=1)

    # ── Evaluation Mode ───────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">⚙️ Evaluation Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Mode:",
        ["Manual (select metrics)", "Agentic (auto-select)", "Retrieval Only"],
        index=1,
        help="Manual = you pick metrics. Agentic = auto-select + aggregate. Retrieval Only = no answer field needed.",
        label_visibility="collapsed",
    )
    selected_metrics: List[str] = []
    if mode.startswith("Manual"):
        discovered = sorted(load_metrics().keys())
        selected_metrics = st.multiselect("Metrics", discovered, default=discovered, label_visibility="collapsed")
        st.caption(f"{len(selected_metrics)} metric(s) selected")
    elif mode == "Retrieval Only":
        selected_metrics = ["retrieval_relevance", "retrieval_coverage"]
        st.caption("`retrieval_relevance` + `retrieval_coverage` — no answer needed")

    # ── Run Config ────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">🏷️ Run Config</div>', unsafe_allow_html=True)
    report_name      = st.text_input("Report label",       placeholder="e.g. v2-pipeline",            key="report_name_input")
    pipeline_version = st.text_input("Pipeline version",   placeholder="e.g. v1.2",                   key="pipeline_version_input")
    run_notes        = st.text_input("Notes",              placeholder="e.g. changed chunk strategy",  key="run_notes_input")
    min_item_score   = st.slider("Min item score to display", 0.0, 1.0, 0.0, 0.01)

    # ── Sampling ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">🎲 Sampling</div>', unsafe_allow_html=True)
    enable_sampling = st.checkbox("Enable sampling", value=False, key="enable_sampling",
                                  help="Sample a subset before evaluation — useful for large datasets.")
    if enable_sampling:
        sample_size   = st.number_input("Sample size", min_value=1, max_value=10000, value=100, step=10, key="sample_size_input")
        sample_method = st.radio("Method", ["Random", "First N"], horizontal=True, key="sample_method_input")
    else:
        sample_size, sample_method = 0, "Random"

    # ── Run button ────────────────────────────────────────────────────────────
    st.markdown("")
    st.markdown('<div class="run-btn">', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Evaluation", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption("Reports auto-saved · view in **Reports** tab")

    # Checkpoint resume
    _existing_cps = sorted(CHECKPOINT_DIR.glob("checkpoint_*.jsonl")) if CHECKPOINT_DIR.exists() else []
    if _existing_cps:
        st.warning(f"⚠️ {len(_existing_cps)} unfinished checkpoint(s) found.")
        col_res, col_dis = st.columns(2)
        with col_res:
            if st.button("▶ Resume last", key="resume_cp_btn", use_container_width=True):
                st.session_state["resume_checkpoint"] = str(max(_existing_cps, key=lambda p: p.stat().st_mtime))
                st.rerun()
        with col_dis:
            if st.button("🗑 Discard", key="discard_cp_btn", use_container_width=True):
                for _cp in _existing_cps:
                    _cp.unlink(missing_ok=True)
                st.rerun()


# ============================== HEADER / OVERVIEW ============================
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap" rel="stylesheet">

    <div style="text-align:center; margin-top:-40px;">
        <h1 style="
            font-family: 'Dancing Script', cursive;
            font-size:5.5rem;
            font-weight:900;
            margin-bottom:0.2rem;
            letter-spacing:-0.06em;
            -webkit-text-stroke: 0.5px rgba(0,0,0,0.08);
            text-shadow: 0 2px 12px rgba(0,0,0,0.18), 0 1px 3px rgba(0,0,0,0.10);
        ">
            <span style="color:#e63946;">R</span>
            <span style="color:#f4840a;">A</span>
            <span style="color:#d4a017;">G</span>
            <span style="color:#16a34a;">V</span>
            <span style="color:#0284c7;">u</span>
            <span style="color:#9333ea;">e</span>
        </h1>
        <p style="font-size:1.5rem; color:var(--muted); margin-top:0; font-style:italic;">
            Explainable &amp; Reference-Free RAG Evaluation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Feature ticker ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .ticker-wrap {
        overflow: hidden;
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 8px;
        padding: 0.45rem 0;
        margin: 0.6rem 0 1.2rem 0;
    }
    .ticker-track {
        display: flex;
        gap: 0;
        width: max-content;
        animation: ticker-scroll 40s linear infinite;
    }
    .ticker-wrap:hover .ticker-track { animation-play-state: paused; }
    @keyframes ticker-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    .ticker-item {
        white-space: nowrap;
        padding: 0 1.6rem;
        font-size: 0.82rem;
        color: var(--muted);
        border-right: 1px solid var(--card-border);
    }
    .ticker-item span { color: var(--text); font-weight: 600; }
    </style>

    <div class="ticker-wrap">
      <div class="ticker-track">
        <div class="ticker-item">🔍 <span>22 reference-free metrics</span></div>
        <div class="ticker-item">🤖 <span>Manual &amp; Agentic modes</span></div>
        <div class="ticker-item">⚖️ <span>OpenAI + Anthropic judge backends</span></div>
        <div class="ticker-item">📊 <span>Cross-model calibration</span></div>
        <div class="ticker-item">📈 <span>Longitudinal tracking &amp; regression detection</span></div>
        <div class="ticker-item">🧠 <span>Your RAG Advisor — AI research thinking partner</span></div>
        <div class="ticker-item">🔬 <span>Hypothesis testing &amp; guided diagnosis</span></div>
        <div class="ticker-item">📎 <span>Upload architecture diagrams for analysis</span></div>
        <div class="ticker-item">🗂 <span>Multi-profile architecture history</span></div>
        <div class="ticker-item">🚀 <span>Python API · CLI · FastAPI · Streamlit UI</span></div>
        <div class="ticker-item">📄 <span>JSON · CSV · Markdown · HTML export</span></div>
        <div class="ticker-item">🏆 <span>Accepted — EACL 2026 Demo Track</span></div>
        <!-- duplicate for seamless loop -->
        <div class="ticker-item">🔍 <span>22 reference-free metrics</span></div>
        <div class="ticker-item">🤖 <span>Manual &amp; Agentic modes</span></div>
        <div class="ticker-item">⚖️ <span>OpenAI + Anthropic judge backends</span></div>
        <div class="ticker-item">📊 <span>Cross-model calibration</span></div>
        <div class="ticker-item">📈 <span>Longitudinal tracking &amp; regression detection</span></div>
        <div class="ticker-item">🧠 <span>Your RAG Advisor — AI research thinking partner</span></div>
        <div class="ticker-item">🔬 <span>Hypothesis testing &amp; guided diagnosis</span></div>
        <div class="ticker-item">📎 <span>Upload architecture diagrams for analysis</span></div>
        <div class="ticker-item">🗂 <span>Multi-profile architecture history</span></div>
        <div class="ticker-item">🚀 <span>Python API · CLI · FastAPI · Streamlit UI</span></div>
        <div class="ticker-item">📄 <span>JSON · CSV · Markdown · HTML export</span></div>
        <div class="ticker-item">🏆 <span>Accepted — EACL 2026 Demo Track</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Intro cards ───────────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(
        """
<div class="card">
  <h3>Introduction</h3>
  <p>
    <strong>RAGVue</strong> is a diagnostic evaluation framework for
    Retrieval-Augmented Generation (RAG) systems — built for researchers and engineers
    who need more than a single score.
  </p>
  <p>
    It provides <strong>interpretable diagnostics</strong> across retrieval quality,
    answer faithfulness, and grounding, helping you pinpoint <em>why</em> a RAG output
    failed — retrieval miss, hallucination, or generation error.
  </p>
  <p>
    Two evaluation modes:
    <span class="chip">Manual</span> — pick metrics yourself, and
    <span class="chip">Agentic</span> — auto-selects metrics and synthesizes overall scores.
    Runs via <span class="chip">Streamlit UI</span> <span class="chip">Python API</span>
    <span class="chip">CLI</span> <span class="chip">FastAPI</span>.
  </p>
  <p class="muted">Accepted · EACL 2026 Demo Track</p>
</div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
<div class="card">
  <h3 style="margin-top:0;">At a Glance</h3>
  <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:0.5rem;">
    <div class="chip">22 reference-free metrics</div>
    <div class="chip">6 core evaluation metrics</div>
    <div class="chip">6 calibration metrics</div>
    <div class="chip">6 failure-mode metrics</div>
    <div class="chip">4 local metrics (no API)</div>
    <div class="chip">OpenAI + Claude backends</div>
    <div class="chip">Cross-model calibration</div>
    <div class="chip">Agentic orchestration</div>
    <div class="chip">Longitudinal tracking</div>
    <div class="chip">Report comparison</div>
    <div class="chip">Your RAG Advisor</div>
    <div class="chip">Multi-profile history</div>
    <div class="chip">Vision file upload</div>
    <div class="chip">JSON · CSV · MD · HTML export</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# ── Key Features & Benefits ───────────────────────────────────────────────────
fc1, fc2 = st.columns(2)
with fc1:
    st.markdown('<h3 class="tab-heading">✨ Key Features</h3>', unsafe_allow_html=True)
    st.markdown(
        """
- **22 reference-free metrics** — 6 core, 6 calibration, 6 complex failure-mode, 4 local (no API cost)
- **Dual judge backends** — OpenAI (default) or Anthropic Claude, switchable per run or per metric
- **Cross-model calibration** — measures judge agreement across 7 model/temperature combinations including Claude
- **Manual & Agentic modes** — pick metrics yourself or let the orchestrator auto-select and synthesize
- **Report history & comparison** — last 10 reports saved; side-by-side delta view (B − A)
- **Longitudinal tracking** — persistent run registry, metric trend charts, automatic regression detection
- **Your RAG Advisor** — persistent AI research thinking partner with architecture profiles, hypothesis testing, before/after analysis, guided diagnosis, and file/diagram upload
- **Multi-interface** — Python API, CLI (`ragvue-cli`, `ragvue-py`), FastAPI REST, Streamlit UI
        """
    )

with fc2:
    st.markdown('<h3 class="tab-heading">🎯 How It Benefits You</h3>', unsafe_allow_html=True)
    st.markdown(
        """
- **Researchers** — explainable metrics, not black-box scores; compare RAG variants across runs; use the RAG Advisor as a research thinking partner to form and test hypotheses without leaving the dashboard
- **Engineers** — plug-and-play API and CLI that fit into existing pipelines; parallel metric execution keeps evaluation fast even at scale
- **Paper authors** — reproducible, citable evaluation with structured per-item reasoning; export directly to Markdown or HTML for appendices
- **Teams iterating on RAG** — longitudinal tracking catches regressions before they reach production; before/after comparison explains what changed and why
- **Non-technical users** — Streamlit UI requires no code; upload data, run evaluation, get results and architectural advice in one place
        """
    )

st.markdown("---")

# Tabs
tab_overview, tab_eval, tab_reports, tab_longitudinal, tab_advisor = st.tabs(["**Overview**", "**Evaluate**", "**Reports**", "**Longitudinal**", "**Your RAG Advisor**"])

# ============================== OVERVIEW TAB ================================
with tab_overview:
    st.markdown('<h3 class="tab-heading">📖 How to use RAGVue</h3>', unsafe_allow_html=True)
    _ov1, _ov2 = st.columns(2)
    with _ov1:
        st.markdown(
            """
**Evaluate your RAG pipeline:**
1. Upload an `items.jsonl` in the sidebar *(question · answer · contexts)*
2. Choose **Manual** mode *(pick metrics)* or **Agentic** mode *(auto-select)*
3. Select your judge — **OpenAI** or **Claude** — from the sidebar
4. Click **Run Evaluation**
5. View **Summary** scores and **per-item drill-down** in the Evaluate tab
6. Export as **JSON · CSV · Markdown · HTML** for papers or repos

**Track progress over time:**
- Every run is saved to **Report History** (last 10) and the **Longitudinal** registry
- Use **Report Comparison** to inspect metric deltas between two runs
- **Regression Detection** automatically flags drops above your threshold
            """
        )
    with _ov2:
        st.markdown(
            """
**Get architectural advice:**
1. Open **Your RAG Advisor** → **My Profile** tab
2. Save your pipeline configuration *(retriever, chunk size, LLM, …)*
3. Switch to the **Chat** tab — your advisor already knows your setup
4. Share a saved report with one click to discuss specific scores
5. Use **Analysis Tools** for structured workflows:
   - **Before/After** — compare two runs, get causal explanation
   - **Hypothesis Testing** — predict metric impact before running
   - **Guided Diagnosis** — step-by-step Retrieval → Grounding → Generation walkthrough
6. Upload architecture diagrams or methodology PDFs for visual feedback

**Input format:**
```json
{"question": "...", "answer": "...", "contexts": ["chunk1", "chunk2"]}
```
            """
        )

# ============================== EVALUATION TAB ==============================

import statistics

def compute_summary_from_results(results):
    buckets = {}
    for r in results:
        for m in r.get("metrics", []) or []:
            name = m.get("name")
            score = m.get("score")
            if isinstance(name, str) and isinstance(score, (int, float)):
                buckets.setdefault(name, []).append(float(score))
    return {k: statistics.mean(v) for k, v in buckets.items()}

with tab_eval:
    # Run / Render logic
    if run_btn:
        status_box = st.empty()
        progress_bar = st.empty()
        progress_text = st.empty()
        start_time = time.perf_counter()
        status_box.info("Starting evaluation... this may take a while depending on your dataset and API speed.")
        # 🔐 Make sure we actually have a key before doing anything
        _active_provider = os.getenv("RAGVUE_JUDGE_PROVIDER", "openai").lower()
        if _active_provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                st.error("No Anthropic API key found. Paste it in the left sidebar and click **'Use in this session'** first.")
                st.stop()
        elif not get_api_key():
            st.error("No API key found. Paste it in the left sidebar and click **'Use in this session'** first.")
            st.stop()
        try:
            if "uploaded_items" in st.session_state:
                items = st.session_state["uploaded_items"]
            elif upl is not None:
                items = read_jsonl_bytes(upl.getvalue())
            else:
                items = []

            if max_items > 0:
                items = items[:max_items]

            # Apply sampling
            if enable_sampling and sample_size > 0 and len(items) > sample_size:
                import random as _random
                _orig_count = len(items)
                if sample_method == "Random":
                    items = _random.sample(items, sample_size)
                else:
                    items = items[:sample_size]
                st.info(f"🎲 Sampled {len(items)} items from {_orig_count} total ({sample_method}).")

            if not items:
                st.error("No items available. Upload a `.jsonl` first from the sidebar.")
            else:
                status_box.info(f"Running evaluation on {len(items)} item(s)...")

                # Checkpoint setup
                _run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                _cp_path = _checkpoint_path(_run_id)
                _resume_cp = st.session_state.pop("resume_checkpoint", None)
                _already_done: list = []
                _start_idx = 0
                if _resume_cp and Path(_resume_cp).exists():
                    _already_done = _load_checkpoint(Path(_resume_cp))
                    _cp_path = Path(_resume_cp)
                    _start_idx = len(_already_done)
                    status_box.info(f"Resuming from checkpoint — {_start_idx} items already done.")

                if mode.startswith("Manual") or mode == "Retrieval Only":
                    if not selected_metrics:
                        st.warning("No metrics selected; nothing to run.")
                    else:
                        results = list(_already_done)

                        for i, item in enumerate(items[_start_idx:], start=_start_idx + 1):
                            progress_bar.progress(i / len(items))
                            progress_text.text(f"Evaluating item {i} of {len(items)}...")
                            t0 = time.perf_counter()
                            single_report = pkg_evaluate([item], metrics=list(selected_metrics))
                            elapsed = time.perf_counter() - t0

                            if single_report.get("results"):
                                res = single_report["results"][0]
                                res["eval_time_sec"] = round(elapsed, 2)
                                results.append(res)
                                _save_checkpoint(_cp_path, res)
                        _cp_path.unlink(missing_ok=True)
                        summary = compute_summary_from_results(results)
                        rb = ReportBuilder({"results": results})
                        report = {"results": results, "summary": summary}

                        st.session_state["last_report"] = report
                        _label = report_name.strip() if report_name.strip() else (upl.name if upl else "unknown")
                        _mode_tag = "Retrieval" if mode == "Retrieval Only" else "Manual"
                        _full_label = f"{_mode_tag} · {_label} · {len(items)} items"
                        _save_to_history(report, _full_label)
                        _append_to_registry(summary, _full_label, version=pipeline_version.strip(), notes=run_notes.strip())
                        progress_bar.empty()
                        progress_text.empty()
                        elapsed = time.perf_counter() - start_time
                        status_box.success(f"✅ Evaluation completed in {elapsed:.1f} seconds.")
                        render_report(report, agentic_mode=False, min_item_score=min_item_score, key_prefix="eval")
                else:
                    orch = AgenticOrchestrator()
                    results = list(_already_done)
                    for i, item in enumerate(items[_start_idx:], start=_start_idx + 1):
                        progress_bar.progress(i / len(items))
                        progress_text.text(f"Evaluating item {i} of {len(items)}...")
                        t0 = time.perf_counter()
                        single_report = orch.run([item])
                        elapsed = time.perf_counter() - t0
                        if single_report.get("results"):
                            res = single_report["results"][0]
                            res["eval_time_sec"] = round(elapsed, 2)
                            results.append(res)
                            _save_checkpoint(_cp_path, res)
                    summary = compute_summary_from_results(results)
                    rb = ReportBuilder({"results": results})

                    _cp_path.unlink(missing_ok=True)
                    report = {"results": results, "summary": summary}
                    st.session_state["last_report"] = report
                    _label = report_name.strip() if report_name.strip() else (upl.name if upl else "unknown")
                    _full_label = f"Agentic · {_label} · {len(items)} items"
                    _save_to_history(report, _full_label)
                    _append_to_registry(summary, _full_label, version=pipeline_version.strip(), notes=run_notes.strip())
                    progress_bar.empty()
                    progress_text.empty()
                    render_report(report, agentic_mode=True, min_item_score=min_item_score, key_prefix="eval")

                    # Final status
                    elapsed = time.perf_counter() - start_time
                    status_box.success(f"✅ Evaluation completed in {elapsed:.1f} seconds.")
        except Exception as e:
            status_box.error("❌ Evaluation failed.")
            st.exception(e)

    elif "last_report" in st.session_state:
        st.info("Showing last report from session memory.")
        render_report(st.session_state["last_report"], agentic_mode=(mode.startswith("Agentic")), min_item_score=min_item_score, key_prefix="eval")

    elif REPORTS_PATH.exists():
        try:
            with open(REPORTS_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history:
                report = history[0]["report"]
                st.session_state["last_report"] = report
                st.info(f"Loaded most recent report: {history[0]['timestamp']} — {history[0]['label']}")
                render_report(report, agentic_mode=(mode.startswith("Agentic")), min_item_score=min_item_score, key_prefix="eval")
        except Exception as e:
            st.error(f"Could not load report history: {e}")
    else:
        st.info("Upload a `.jsonl` in the sidebar and click **Run Evaluation** to see the Summary here.")

# ============================== REPORTS TAB ==================================
with tab_reports:
    st.markdown('<h3 class="tab-heading">📋 Report History</h3>', unsafe_allow_html=True)
    if not REPORTS_PATH.exists():
        st.info("No saved reports yet. Run an evaluation first.")
    else:
        try:
            with open(REPORTS_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
            if not history:
                st.info("No saved reports yet. Run an evaluation first.")
            else:
                labels = [f"{e['timestamp']}  —  {e['label']}" for e in history]

                # ── Search / filter ──
                search = st.text_input("🔍 Filter reports", placeholder="filter by mode, filename, or date...", key="history_search")
                filtered_indices = [i for i, l in enumerate(labels) if not search or search.lower() in l.lower()]
                filtered_labels = [labels[i] for i in filtered_indices]

                if not filtered_labels:
                    st.warning("No reports match your filter.")
                else:
                    compare_mode = st.checkbox("Compare two reports", key="compare_mode")

                    if compare_mode:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            choice_a = st.selectbox("Report A:", filtered_labels, key="cmp_a")
                        with col_b:
                            choice_b = st.selectbox("Report B:", filtered_labels, key="cmp_b")
                        idx_a = filtered_indices[filtered_labels.index(choice_a)]
                        idx_b = filtered_indices[filtered_labels.index(choice_b)]
                        rep_a = history[idx_a]
                        rep_b = history[idx_b]

                        st.markdown('<h4 class="tab-heading">📊 Metric Comparison</h4>', unsafe_allow_html=True)
                        summary_a = rep_a["report"].get("summary", {})
                        summary_b = rep_b["report"].get("summary", {})
                        all_metrics = sorted(set(summary_a) | set(summary_b))
                        if all_metrics:
                            import pandas as pd
                            rows = []
                            for m in all_metrics:
                                sa = summary_a.get(m)
                                sb = summary_b.get(m)
                                delta = round(sb - sa, 4) if sa is not None and sb is not None else None
                                rows.append({
                                    "Metric": m,
                                    "Report A": round(sa, 4) if sa is not None else "—",
                                    "Report B": round(sb, 4) if sb is not None else "—",
                                    "Delta (B − A)": (f"+{delta}" if delta and delta > 0 else str(delta)) if delta is not None else "—",
                                })
                            st.dataframe(rows, use_container_width=True)
                        else:
                            st.info("No summary metrics available to compare.")

                        st.markdown("---")
                        st.markdown(f"**Report A** — {rep_a['timestamp']} · {rep_a['label']}")
                        render_report(rep_a["report"], agentic_mode=("Agentic" in rep_a["label"]), min_item_score=0.0, key_prefix=f"cmp_a_{idx_a}")
                        st.markdown("---")
                        st.markdown(f"**Report B** — {rep_b['timestamp']} · {rep_b['label']}")
                        render_report(rep_b["report"], agentic_mode=("Agentic" in rep_b["label"]), min_item_score=0.0, key_prefix=f"cmp_b_{idx_b}")

                    else:
                        col_sel, col_del, col_delall = st.columns([4, 1, 1])
                        with col_sel:
                            choice = st.selectbox(f"Select a report ({len(history)} saved, {len(filtered_labels)} shown):", filtered_labels, key="history_select")
                        idx = filtered_indices[filtered_labels.index(choice)]
                        selected = history[idx]

                        with col_del:
                            st.write("")  # vertical align
                            if st.button("🗑 Delete this", key="del_one", use_container_width=True):
                                history.pop(idx)
                                with open(REPORTS_PATH, "w", encoding="utf-8") as f:
                                    json.dump(history, f, ensure_ascii=False, indent=2)
                                st.success("Report deleted.")
                                st.rerun()

                        with col_delall:
                            st.write("")  # vertical align
                            if st.button("🗑 Delete all", key="del_all", use_container_width=True):
                                with open(REPORTS_PATH, "w", encoding="utf-8") as f:
                                    json.dump([], f)
                                st.success("All reports deleted.")
                                st.rerun()

                        st.caption(f"Mode: **{'Agentic' if 'Agentic' in selected['label'] else 'Manual'}** · Saved: {selected['timestamp']}")
                        render_report(
                            selected["report"],
                            agentic_mode=("Agentic" in selected["label"]),
                            min_item_score=0.0,
                            key_prefix=f"reports_{idx}",
                        )
        except Exception as e:
            st.error(f"Could not load report history: {e}")

# ============================== LONGITUDINAL TAB =============================
with tab_longitudinal:
    st.markdown('<h3 class="tab-heading">📈 Longitudinal Tracking</h3>', unsafe_allow_html=True)
    if not RUN_REGISTRY_PATH.exists():
        st.info("No runs recorded yet. Run an evaluation first.")
    else:
        try:
            with open(RUN_REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)
            if not registry:
                st.info("No runs recorded yet. Run an evaluation first.")
            else:
                import pandas as pd

                # ── 1. Run registry table ─────────────────────────────────
                st.markdown("### 🗃 Run Registry")
                reg_cols = ["#", "Timestamp", "Label", "Version", "Notes"]
                reg_rows = [
                    {
                        "#": i + 1,
                        "Timestamp": e["timestamp"],
                        "Label": e["label"],
                        "Version": e.get("version", ""),
                        "Notes": e.get("notes", ""),
                    }
                    for i, e in enumerate(registry)
                ]
                _ths = "".join(f"<th>{c}</th>" for c in reg_cols)
                _rows_html = "".join(
                    "<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in reg_cols) + "</tr>"
                    for r in reg_rows
                )
                st.markdown(
                    f'<div><table class="themed-table"><thead><tr>{_ths}</tr></thead>'
                    f'<tbody>{_rows_html}</tbody></table></div>',
                    unsafe_allow_html=True,
                )

                col_del_run, col_del_all_runs, _ = st.columns([2, 2, 4])
                with col_del_run:
                    run_labels = [f"#{i+1} — {e['timestamp']} · {e['label']}" for i, e in enumerate(registry)]
                    del_choice = st.selectbox("Select run to delete:", run_labels, key="lng_del_select")
                    if st.button("🗑 Delete this run", key="lng_del_one", use_container_width=True):
                        del_idx = run_labels.index(del_choice)
                        registry.pop(del_idx)
                        with open(RUN_REGISTRY_PATH, "w", encoding="utf-8") as f:
                            json.dump(registry, f, ensure_ascii=False, indent=2)
                        st.success("Run deleted.")
                        st.rerun()
                with col_del_all_runs:
                    st.write("")
                    st.write("")
                    if st.button("🗑 Delete all runs", key="lng_del_all", use_container_width=True):
                        with open(RUN_REGISTRY_PATH, "w", encoding="utf-8") as f:
                            json.dump([], f)
                        st.success("All runs deleted.")
                        st.rerun()

                # ── 2. Trend line chart ───────────────────────────────────
                st.markdown("### 📉 Metric Trends")
                all_metrics = sorted({k for e in registry for k in e.get("summary", {}).keys()})
                if all_metrics:
                    selected_trend = st.multiselect(
                        "Select metrics to plot:",
                        all_metrics,
                        default=all_metrics[:5] if len(all_metrics) > 5 else all_metrics,
                        key="trend_metrics",
                    )
                    if selected_trend:
                        chart_rows = []
                        for i, e in enumerate(registry):
                            row = {"Run": f"#{i+1} {e['timestamp'][:10]}"}
                            for m in selected_trend:
                                row[m] = e.get("summary", {}).get(m)
                            chart_rows.append(row)
                        df_chart = pd.DataFrame(chart_rows)

                        _tc = THEMES[st.session_state.get("theme", "Light")]
                        import plotly.graph_objects as go
                        _accent_colors = [
                            "#818cf8", "#34d399", "#f97316", "#f43f5e",
                            "#facc15", "#38bdf8", "#a78bfa", "#4ade80",
                        ]
                        fig = go.Figure()
                        for idx_m, metric in enumerate(selected_trend):
                            fig.add_trace(go.Scatter(
                                x=df_chart["Run"],
                                y=df_chart[metric],
                                mode="lines+markers",
                                name=metric,
                                line=dict(
                                    color=_accent_colors[idx_m % len(_accent_colors)],
                                    width=2,
                                ),
                                marker=dict(size=7),
                            ))
                        fig.update_layout(
                            paper_bgcolor=_tc["--bg"],
                            plot_bgcolor=_tc["--card"],
                            font=dict(
                                color=_tc["--text"],
                                family="Plus Jakarta Sans, system-ui, sans-serif",
                            ),
                            xaxis=dict(
                                gridcolor=_tc["--card-border"],
                                linecolor=_tc["--card-border"],
                                tickfont=dict(color=_tc["--muted"]),
                                tickangle=-30,
                            ),
                            yaxis=dict(
                                gridcolor=_tc["--card-border"],
                                linecolor=_tc["--card-border"],
                                tickfont=dict(color=_tc["--muted"]),
                                range=[0, 1],
                                tickformat=".2f",
                            ),
                            legend=dict(
                                bgcolor=_tc["--bg-alt"],
                                bordercolor=_tc["--card-border"],
                                borderwidth=1,
                                font=dict(color=_tc["--text"]),
                            ),
                            margin=dict(l=50, r=20, t=20, b=70),
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No summary metrics found in registry.")

                # ── 3. Regression detection ───────────────────────────────
                if len(registry) >= 2:
                    st.markdown('<h3 class="tab-heading">🔍 Regression Detection</h3>', unsafe_allow_html=True)
                    st.caption("Comparing the two most recent runs.")
                    threshold = st.slider(
                        "Flag drops larger than:", 0.01, 0.20, 0.05, 0.01, key="reg_threshold"
                    )
                    latest = registry[-1].get("summary", {})
                    previous = registry[-2].get("summary", {})
                    overlap = sorted(set(latest) & set(previous))
                    if overlap:
                        rdet_rows = []
                        for m in overlap:
                            curr = latest[m]
                            prev = previous[m]
                            delta = round(curr - prev, 4)
                            rdet_rows.append({
                                "Metric": m,
                                f"Previous ({registry[-2]['timestamp'][:10]})": prev,
                                f"Current ({registry[-1]['timestamp'][:10]})": curr,
                                "Delta": f"+{delta}" if delta > 0 else str(delta),
                                "Status": "🔴 Regression" if delta < -threshold else ("🟡 Watch" if delta < 0 else "🟢 OK"),
                            })
                        st.dataframe(rdet_rows, use_container_width=True)
                    else:
                        st.info("No overlapping metrics between the last two runs.")
        except Exception as e:
            st.error(f"Could not load run registry: {e}")

# ============================== YOUR RAG ADVISOR TAB =========================
with tab_advisor:
    st.markdown('<h3 class="tab-heading">🤖 Your RAG Advisor</h3>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:var(--muted); font-size:0.82rem; font-style:italic; margin-bottom:0.6rem;'>"
        "Research Thinking partner, not ground truth — suggestions are hypotheses to validate empirically, not guaranteed fixes."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:center; margin-bottom:1rem;'>"
        "<span style='display:inline-flex; align-items:center; gap:0.45rem; "
        "background:var(--chip-bg); border:1px solid var(--chip-border); "
        "border-radius:20px; padding:0.28rem 1rem; font-size:0.8rem; color:var(--muted);'>"
        "🧪 <strong style='color:var(--accent);'>Early Access</strong> — "
        "not yet validated on real datasets. Advice may be incomplete or off. "
        "Suggestions &amp; feedback welcome: "
        "<a href='mailto:ragvue.license@gmail.com' style='color:var(--accent);'>ragvue.license@gmail.com</a>"
        "</span></div>",
        unsafe_allow_html=True,
    )

    # ── Shared state (needed across all sub-tabs) ─────────────────────────────
    if "advisor_messages" not in st.session_state:
        st.session_state.advisor_messages = _load_advisor_history()
    _adv_store   = _load_profile_store()
    _adv_profile = _active_profile(_adv_store)
    _adv_system_prompt = _build_advisor_system_prompt(_adv_profile)

    if not st.session_state.advisor_messages:
        if _adv_profile:
            _welcome = (
                f"Hi! I'm **Your RAG Advisor** — your research thinking partner for diagnosing and improving your RAG pipeline.\n\n"
                f"I can see your **{_adv_profile.get('name', 'active')}** profile is loaded "
                f"({', '.join(v for k, v in _adv_profile.items() if v and k not in ('name', 'saved_at', 'notes'))[:120]}…), "
                f"so I'll tailor everything to your specific setup.\n\n"
                f"The best place to start is sharing your latest evaluation scores — use the **Share with advisor** banner "
                f"at the top of this tab after running an evaluation, or paste your scores directly. "
                f"Then tell me what you're trying to understand and we'll dig in together."
            )
        else:
            _welcome = (
                "Hi! I'm **Your RAG Advisor** — your research thinking partner for diagnosing and improving your RAG pipeline.\n\n"
                "Before we start, I'd suggest doing two things:\n\n"
                "**① Go to My Profile** and save your pipeline setup (retriever type, chunk size, top-k, embedding model, LLM). "
                "Without it, my advice will be generic rather than specific to your architecture.\n\n"
                "**② Run a RAGVue evaluation** and share the scores here — that gives us something concrete to reason about.\n\n"
                "If you're not sure where to begin, check the **Getting Started** tab for a quick walkthrough. "
                "Or just ask me anything — I'm happy to start with a question."
            )
        st.session_state.advisor_messages = [{"role": "assistant", "content": _welcome}]
        _save_advisor_history(st.session_state.advisor_messages)

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    _atab_start, _atab_chat, _atab_profile, _atab_tools = st.tabs(["🚀 Getting Started", "💬 Chat", "🗂 My Profile", "🔬 Analysis Tools"])

    # ═══════════════════════════ GETTING STARTED SUB-TAB ═════════════════════
    with _atab_start:

        st.markdown("""
<style>
/* ── Getting Started page styles ───────────────────────────────────────── */
.gs-hero {
    background: linear-gradient(135deg, var(--card) 0%, var(--bg-alt) 100%);
    border: 1px solid var(--card-border);
    border-left: 5px solid var(--accent);
    border-radius: 14px;
    padding: 2rem 2.2rem 1.8rem;
    margin-bottom: 2rem;
    text-align: center;
}
.gs-hero h2 {
    margin: 0 0 0.5rem;
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--accent);
}
.gs-hero p {
    margin: 0;
    font-size: 1rem;
    color: var(--muted);
    max-width: 660px;
    margin: 0 auto;
    line-height: 1.6;
}
.gs-section-label {
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--accent);
    margin: 2rem 0 1rem;
}
.gs-steps-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 2rem;
}
.gs-step {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
}
.gs-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    background: var(--accent);
    color: var(--accent-contrast);
    border-radius: 50%;
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
}
.gs-step h4 {
    margin: 0 0 0.5rem;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
}
.gs-step p {
    margin: 0;
    font-size: 0.95rem;
    color: var(--muted);
    line-height: 1.6;
}
.gs-compare-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-bottom: 2rem;
}
.gs-compare-bad {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 1rem 1.1rem;
}
.gs-compare-good {
    background: var(--chip-bg);
    border: 1px solid var(--accent);
    border-radius: 10px;
    padding: 1rem 1.1rem;
}
.gs-compare-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    color: var(--muted);
}
.gs-compare-good .gs-compare-label { color: var(--accent); }
.gs-compare-text {
    font-size: 0.95rem;
    color: var(--muted);
    font-style: italic;
    line-height: 1.55;
}
.gs-can-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 2rem;
}
.gs-can-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
}
.gs-can-title {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.gs-can-yes { color: #22c55e; }
.gs-cant-no { color: #ef4444; }
.gs-can-item {
    font-size: 0.94rem;
    color: var(--muted);
    padding: 0.35rem 0;
    border-bottom: 1px solid var(--card-border);
    line-height: 1.5;
}
.gs-can-item:last-child { border-bottom: none; }
.gs-can-badge {
    display: inline-block;
    width: 1.1rem;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 0.25rem;
}
.gs-can-badge.yes { color: #22c55e; }
.gs-can-badge.no  { color: #ef4444; }
.gs-golden {
    background: var(--chip-bg);
    border: 1px solid var(--chip-border);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 2rem;
    font-size: 0.97rem;
    color: var(--muted);
    text-align: center;
    line-height: 1.55;
}
.gs-golden strong { color: var(--accent); }
.gs-jump-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
}
.gs-jump-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1.15rem 1.2rem;
    text-align: center;
    cursor: default;
    transition: border-color 0.2s;
}
.gs-jump-card:hover { border-color: var(--accent); }
.gs-jump-icon {
    font-size: 1.6rem;
    margin-bottom: 0.5rem;
}
.gs-jump-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.35rem;
}
.gs-jump-desc {
    font-size: 0.9rem;
    color: var(--muted);
    line-height: 1.5;
}
</style>

<!-- HERO -->
<div class="gs-hero">
  <h2>🧠 Your RAG Advisor</h2>
  <p>
    A research thinking partner built into your evaluation dashboard.<br>
    It doesn't run evaluations — it helps you reason through what your scores
    and diagnostic fields mean, and figure out the right experiment to try next.
  </p>
</div>

<!-- WHAT IS IT -->
<div class="gs-section-label">What is it?</div>
<div class="card" style="margin-bottom:1.5rem; padding:1.2rem 1.4rem; font-size:0.97rem; color:var(--muted); line-height:1.65;">
RAGVue gives you <strong style="color:var(--text)">metric scores, per-item breakdowns, hallucinated claims, diagnostic fields,
and detailed explanations</strong> for every evaluation run.
The scores tell you <em>what</em> happened — the Advisor helps you figure out <em>why</em> it happened in your specific pipeline
and <em>what</em> to do about it.<br><br>
A <code>strict_faithfulness</code> of 0.41 with hallucinated claims listed is useful evidence, but it still leaves open
questions: retrieval problem? prompt problem? wrong LLM?
The Advisor knows every RAGVue metric and its diagnostic fields in detail, understands how pipeline choices affect scores,
and helps you form one focused hypothesis to test at a time — moving you from <em>"my scores are low"</em> to
<em>"here is the most likely cause and here is exactly how to test it."</em>
</div>

<!-- HOW IT WORKS -->
<div class="gs-section-label">How a typical session works</div>
<div class="gs-steps-grid">
  <div class="gs-step">
    <div class="gs-step-num">1</div>
    <h4>Save your pipeline setup</h4>
    <p>Go to <strong>My Profile</strong> and fill in your retriever type, chunk size, top-k, embedding model,
    and generation LLM. The advisor references your actual values in every response —
    so instead of "try increasing top-k", it says "try increasing your top-k from 3 to 7".</p>
  </div>
  <div class="gs-step">
    <div class="gs-step-num">2</div>
    <h4>Run a RAGVue evaluation</h4>
    <p>Use the <strong>Evaluate</strong> tab to run your data through the metrics. When it's done,
    a banner appears at the top of the Chat tab —
    one click shares your results with the advisor.</p>
  </div>
  <div class="gs-step">
    <div class="gs-step-num">3</div>
    <h4>Ask what the scores mean</h4>
    <p>Describe what you're seeing or just ask directly. The advisor maps your score pattern
    to likely root causes in your specific pipeline and tells you which layer —
    retrieval, grounding, or generation — to investigate first.</p>
  </div>
  <div class="gs-step">
    <div class="gs-step-num">4</div>
    <h4>Get one experiment to run</h4>
    <p>The advisor gives you one specific change to make, which metrics it expects to improve,
    and what result would confirm the hypothesis.
    Make the change, re-run RAGVue, share the new scores, and repeat.</p>
  </div>
</div>

<!-- WHAT TO SHARE -->
<div class="gs-section-label">What to share for the best advice</div>
<div class="gs-compare-grid">
  <div class="gs-compare-bad">
    <div class="gs-compare-label">❌ Too vague — generic reply</div>
    <div class="gs-compare-text">"My RAG isn't working well"</div>
  </div>
  <div class="gs-compare-good">
    <div class="gs-compare-label">✅ Specific — actionable reply</div>
    <div class="gs-compare-text">"strict_faithfulness=0.38, context_utilization=0.71, retrieval_relevance=0.82 — what's going on?"</div>
  </div>
  <div class="gs-compare-bad">
    <div class="gs-compare-label">❌ Too vague</div>
    <div class="gs-compare-text">"Should I change my chunk size?"</div>
  </div>
  <div class="gs-compare-good">
    <div class="gs-compare-label">✅ Specific</div>
    <div class="gs-compare-text">"retrieval_coverage=0.54, chunks=512 tokens, top-k=3. Would reducing chunk size help?"</div>
  </div>
  <div class="gs-compare-bad">
    <div class="gs-compare-label">❌ Too vague</div>
    <div class="gs-compare-text">"My scores dropped after a change"</div>
  </div>
  <div class="gs-compare-good">
    <div class="gs-compare-label">✅ Specific</div>
    <div class="gs-compare-text">"After increasing top-k from 3 to 8, coverage improved but faithfulness dropped 0.72→0.61. Why?"</div>
  </div>
</div>

<!-- CAN / CAN'T -->
<div class="gs-section-label">What it can and can't do</div>
<div class="gs-can-grid">
  <div class="gs-can-card">
    <div class="gs-can-title gs-can-yes">✅ It can</div>
    <div class="gs-can-item"><span class="gs-can-badge yes">✓</span>Interpret RAGVue metric scores and map them to root causes</div>
    <div class="gs-can-item"><span class="gs-can-badge yes">✓</span>Tailor advice to your retriever, chunk size, LLM, and domain</div>
    <div class="gs-can-item"><span class="gs-can-badge yes">✓</span>Predict which metrics a planned change will affect — before you run it</div>
    <div class="gs-can-item"><span class="gs-can-badge yes">✓</span>Explain what each metric measures and what its diagnostic fields contain</div>
    <div class="gs-can-item"><span class="gs-can-badge yes">✓</span>Compare two evaluation runs and explain what changed and why</div>
    <div class="gs-can-item"><span class="gs-can-badge yes">✓</span>Walk you through a structured diagnosis step by step</div>
    <div class="gs-can-item"><span class="gs-can-badge yes">✓</span>Answer general ML, NLP, and RAG questions using its own knowledge</div>
  </div>
  <div class="gs-can-card">
    <div class="gs-can-title gs-cant-no">❌ It can't</div>
    <div class="gs-can-item"><span class="gs-can-badge no">✗</span>Access your documents, chunks, or prompt templates — only what you share</div>
    <div class="gs-can-item"><span class="gs-can-badge no">✗</span>Run evaluations on your behalf</div>
    <div class="gs-can-item"><span class="gs-can-badge no">✗</span>Guarantee any suggestion will improve your scores</div>
    <div class="gs-can-item"><span class="gs-can-badge no">✗</span>Replace empirical testing — its suggestions are hypotheses, not verdicts</div>
    <div class="gs-can-item"><span class="gs-can-badge no">✗</span>Debug your code directly</div>
  </div>
</div>

<!-- GOLDEN RULE -->
<div class="gs-golden">
  <strong>The golden rule:</strong> run a RAGVue evaluation after every change.
  The advisor helps you plan <em>what</em> to change — the metrics tell you <em>whether it worked</em>.
</div>

<!-- JUMP CARDS -->
<div class="gs-section-label">Ready to start? Pick your first step</div>
<div class="gs-jump-grid">
  <div class="gs-jump-card">
    <div class="gs-jump-icon">👤</div>
    <div class="gs-jump-title">① Set up My Profile</div>
    <div class="gs-jump-desc">Save your pipeline configuration so advice is tailored to your exact setup.</div>
  </div>
  <div class="gs-jump-card">
    <div class="gs-jump-icon">💬</div>
    <div class="gs-jump-title">② Open the Chat</div>
    <div class="gs-jump-desc">Share your evaluation scores and ask what they mean for your pipeline.</div>
  </div>
  <div class="gs-jump-card">
    <div class="gs-jump-icon">🔬</div>
    <div class="gs-jump-title">③ Try Analysis Tools</div>
    <div class="gs-jump-desc">Run the Failure Mode Scanner or Suggest Next Experiment if you already have results.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ═══════════════════════════ CHAT SUB-TAB ════════════════════════════════
    with _atab_chat:

        # ── Auto-inject latest eval results banner ────────────────────────────
        if "advisor_last_offered_ts" not in st.session_state:
            st.session_state.advisor_last_offered_ts = ""
        _newest_report = None
        if REPORTS_PATH.exists():
            try:
                with open(REPORTS_PATH, "r", encoding="utf-8") as _nf:
                    _all_saved = json.load(_nf)
                if _all_saved:
                    _newest_report = _all_saved[0]
            except Exception:
                pass
        if _newest_report and _newest_report.get("timestamp", "") != st.session_state.advisor_last_offered_ts:
            _nr_label = _newest_report.get("label", "Untitled")
            _nr_ts    = _newest_report.get("timestamp", "")
            _bnr_c1, _bnr_c2, _bnr_c3 = st.columns([7, 2.5, 0.5])
            with _bnr_c1:
                st.info(f"**New evaluation ready:** {_nr_label} · {_nr_ts[:16]} — share with your advisor?")
            with _bnr_c2:
                if st.button("Share with advisor", key="auto_share_btn", use_container_width=True):
                    _nr_sum = compute_summary_from_results(_newest_report.get("report", {}).get("results", []))
                    if _nr_sum:
                        _auto_inject = (
                            f"Here are my latest evaluation results from **{_nr_label}** ({_nr_ts}):\n"
                            + "\n".join(f"  - {k}: {v:.3f}" for k, v in sorted(_nr_sum.items()))
                        )
                        _last_msg = st.session_state.advisor_messages[-1] if st.session_state.advisor_messages else {}
                        if _last_msg.get("content") != _auto_inject:
                            st.session_state.advisor_messages.append({"role": "user", "content": _auto_inject})
                            _save_advisor_history(st.session_state.advisor_messages)
                    st.session_state.advisor_last_offered_ts = _nr_ts
                    st.rerun()
            with _bnr_c3:
                if st.button("✕", key="auto_share_dismiss", use_container_width=True):
                    st.session_state.advisor_last_offered_ts = _nr_ts
                    st.rerun()

        # Thin controls bar — row 1: model / export / clear
        _cc1, _cc3, _cc4 = st.columns([4, 2, 2])
        with _cc1:
            _ADV_MODEL_OPTIONS = [
                "gpt-4o-mini  ·  OpenAI  ·  fast",
                "gpt-4o  ·  OpenAI  ·  capable",
                "gpt-3.5-turbo  ·  OpenAI  ·  budget",
                "claude-haiku-4-5-20251001  ·  Anthropic  ·  fast",
                "claude-sonnet-4-6  ·  Anthropic  ·  balanced",
                "claude-opus-4-6  ·  Anthropic  ·  powerful",
            ]
            _adv_model_choice = st.selectbox(
                "Model:", _ADV_MODEL_OPTIONS,
                key="advisor_model_choice", label_visibility="collapsed",
            )
            _adv_model    = _adv_model_choice.split("  ·  ")[0].strip()
            _adv_provider = "anthropic" if "Anthropic" in _adv_model_choice else "openai"
        with _cc3:
            if len(st.session_state.advisor_messages) > 1:
                _export_md = "\n\n".join(
                    f"**{'You' if m['role'] == 'user' else 'Your RAG Advisor'}:** {m['content']}"
                    for m in st.session_state.advisor_messages
                )
                st.download_button(
                    "⬇ Export", data=_export_md,
                    file_name="rag_advisor_conversation.md", mime="text/markdown",
                    key="advisor_export", use_container_width=True,
                )
        with _cc4:
            if st.button("🗑 Clear", key="advisor_clear", use_container_width=True):
                st.session_state.advisor_messages = []
                _save_advisor_history([])
                st.rerun()

        # ── Share panel: Summary vs Case Inspector ─────────────────────────────
        # Load saved reports once (used by both modes)
        _adv_reports: list = []
        if REPORTS_PATH.exists():
            try:
                with open(REPORTS_PATH, "r", encoding="utf-8") as _f:
                    _adv_reports = json.load(_f)
            except Exception:
                pass
        _adv_report_labels = [f"{e['timestamp']} · {e['label']}" for e in _adv_reports]

        _share_mode = st.radio(
            "Share with advisor:",
            ["📊 Summary", "🔍 Case Inspector"],
            horizontal=True,
            key="advisor_share_mode",
            label_visibility="collapsed",
        )

        if _share_mode == "📊 Summary":
            _adv_report_opts = ["📋 Select a report to share…"] + _adv_report_labels
            _adv_report_sel = st.selectbox(
                "Report:", _adv_report_opts, key="advisor_report_sel", label_visibility="collapsed"
            )
            if _adv_report_sel != "📋 Select a report to share…":
                _sel_idx   = _adv_report_opts.index(_adv_report_sel) - 1
                _sel_entry = _adv_reports[_sel_idx]
                _sel_sum   = compute_summary_from_results(_sel_entry.get("report", {}).get("results", []))
                if _sel_sum:
                    _inject_text = (
                        f"Here are my evaluation results from **{_sel_entry['label']}** ({_sel_entry['timestamp']}):\n"
                        + "\n".join(f"  - {k}: {v:.3f}" for k, v in sorted(_sel_sum.items()))
                    )
                    _last_adv = st.session_state.advisor_messages[-1] if st.session_state.advisor_messages else {}
                    if _last_adv.get("content") != _inject_text:
                        st.session_state.advisor_messages.append({"role": "user", "content": _inject_text})
                        _save_advisor_history(st.session_state.advisor_messages)
                        st.rerun()

        else:  # Case Inspector
            if not _adv_reports:
                st.caption("No saved reports found. Run an evaluation first.")
            else:
                _ci_report_opts = ["Select a report…"] + _adv_report_labels
                _ci_report_sel = st.selectbox(
                    "Report:", _ci_report_opts, key="ci_report_sel", label_visibility="collapsed"
                )

                if _ci_report_sel != "Select a report…":
                    _ci_idx     = _ci_report_opts.index(_ci_report_sel) - 1
                    _ci_entry   = _adv_reports[_ci_idx]
                    _ci_results = _ci_entry.get("report", {}).get("results", [])
                    _ci_total   = len(_ci_results)

                    if _ci_total == 0:
                        st.caption("This report has no item-level results.")
                    else:
                        # Build item options from actual questions — limited to report size
                        def _ci_label(i, item):
                            q = item.get("question") or item.get("item", {}).get("question", "")
                            preview = str(q)[:80] + "…" if len(str(q)) > 80 else str(q)
                            return f"Item {i+1}: {preview}" if preview else f"Item {i+1}"
                        _ci_item_opts = [_ci_label(i, r) for i, r in enumerate(_ci_results)]
                        _ci_item_sel = st.selectbox(
                            "Item:", _ci_item_opts, key="ci_item_sel", label_visibility="collapsed"
                        )
                        _ci_item_idx = _ci_item_opts.index(_ci_item_sel)
                        _ci_item     = _ci_results[_ci_item_idx]

                        if st.button("Share case with advisor", key="ci_share_btn", use_container_width=True):
                            # Build a rich injection: Q + A + contexts + all metric diagnostics
                            _ci_q   = _ci_item.get("question") or _ci_item.get("item", {}).get("question", "")
                            _ci_a   = _ci_item.get("answer")   or _ci_item.get("item", {}).get("answer", "")
                            _ci_ctx = _ci_item.get("contexts") or _ci_item.get("item", {}).get("contexts", [])
                            _ci_metrics = _ci_item.get("metrics", [])

                            _lines = [
                                f"**Case Inspector — Item {_ci_item_idx+1} of {_ci_total}**",
                                f"*Report: {_ci_entry['label']} ({_ci_entry['timestamp']})*",
                                "",
                            ]
                            if _ci_q:
                                _lines += [f"**Question:** {_ci_q}", ""]
                            if _ci_a:
                                _lines += [f"**Answer:** {_ci_a}", ""]
                            if _ci_ctx:
                                _lines.append(f"**Contexts:** {len(_ci_ctx)} chunk(s)")
                                for _ci, _c in enumerate(_ci_ctx):
                                    _lines.append(f"  - Chunk {_ci+1}: {str(_c)[:300]}{'…' if len(str(_c)) > 300 else ''}")
                                _lines.append("")

                            if _ci_metrics:
                                _lines.append("**Metric results:**")
                                _SKIP_KEYS = {"name", "score"}
                                for _m in _ci_metrics:
                                    _mn = _m.get("name", "?")
                                    _ms = _m.get("score")
                                    _score_str = f"{_ms:.3f}" if isinstance(_ms, (int, float)) else str(_ms)
                                    _lines.append(f"• **{_mn}**: {_score_str}")
                                    for _dk, _dv in _m.items():
                                        if _dk in _SKIP_KEYS or _dv is None:
                                            continue
                                        if isinstance(_dv, list):
                                            if _dv:
                                                _lines.append(f"  - {_dk}: {', '.join(str(x) for x in _dv[:6])}"
                                                               + (" …" if len(_dv) > 6 else ""))
                                        elif isinstance(_dv, dict):
                                            _lines.append(f"  - {_dk}: {json.dumps(_dv)[:200]}")
                                        else:
                                            _lines.append(f"  - {_dk}: {str(_dv)[:300]}")

                            _ci_inject = "\n".join(_lines)
                            _last_adv = st.session_state.advisor_messages[-1] if st.session_state.advisor_messages else {}
                            if _last_adv.get("content") != _ci_inject:
                                st.session_state.advisor_messages.append({"role": "user", "content": _ci_inject})
                                _save_advisor_history(st.session_state.advisor_messages)
                                st.rerun()

        # ── Active profile indicator ──────────────────────────────────────────
        if _adv_profile:
            _pf_name = _adv_profile.get("name", "Active Profile")
            _pf_detail_parts = []
            for _k in ("retriever", "chunk_size", "top_k", "generation_llm"):
                _v = _adv_profile.get(_k)
                if _v:
                    _pf_detail_parts.append(str(_v))
            _pf_detail = " · ".join(_pf_detail_parts[:4])
            st.markdown(
                f"<div style='display:inline-flex;align-items:center;gap:0.5rem;"
                f"background:var(--chip-bg);border:1px solid var(--chip-border);"
                f"border-radius:20px;padding:0.25rem 0.85rem;font-size:0.82rem;"
                f"color:var(--muted);margin-bottom:0.6rem;'>"
                f"<span style='color:#22c55e;font-size:0.7rem;'>●</span>"
                f"<strong style='color:var(--text);'>{_pf_name}</strong>"
                + (f"<span style='opacity:0.6;'>— {_pf_detail}</span>" if _pf_detail else "")
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='display:inline-flex;align-items:center;gap:0.5rem;"
                "background:var(--chip-bg);border:1px solid var(--chip-border);"
                "border-radius:20px;padding:0.25rem 0.85rem;font-size:0.82rem;"
                "color:var(--muted);margin-bottom:0.6rem;'>"
                "<span style='color:#f59e0b;font-size:0.7rem;'>●</span>"
                "No active profile — advice will be generic. Set one up in the <strong>My Profile</strong> tab."
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # File upload (collapsed by default)
        with st.expander("📎 Upload diagram or document", expanded=False):
            _adv_upload_col, _adv_btn_col = st.columns([4, 2])
            with _adv_upload_col:
                _adv_file = st.file_uploader(
                    "PNG / JPG / WEBP · PDF · TXT / MD",
                    type=["png", "jpg", "jpeg", "webp", "pdf", "txt", "md"],
                    key="advisor_file_upload", label_visibility="collapsed",
                )
            with _adv_btn_col:
                _adv_file_prompt = st.text_input(
                    "Question:", placeholder="What could be improved?",
                    key="advisor_file_prompt", label_visibility="collapsed",
                )
                _adv_send_file = st.button("Send to advisor", key="advisor_send_file", use_container_width=True)

        if _adv_send_file and _adv_file is not None:
            import base64
            _adv_fname     = _adv_file.name.lower()
            _adv_file_bytes = _adv_file.read()
            _file_question  = _adv_file_prompt.strip() or "Please review this and give feedback relevant to my RAG pipeline."

            if _adv_fname.endswith((".png", ".jpg", ".jpeg", ".webp")):
                _media_map  = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
                _media_type = _media_map.get(_adv_fname.rsplit(".", 1)[-1], "image/png")
                _img_b64    = base64.b64encode(_adv_file_bytes).decode("utf-8")
                st.session_state.advisor_messages.append({"role": "user", "content": f"📎 *Image: `{_adv_file.name}`*\n\n{_file_question}"})
                with st.chat_message("user"):
                    st.image(_adv_file_bytes, caption=_adv_file.name, use_column_width=True)
                    st.markdown(_file_question)
                _llm_msgs_v = [{"role": "system", "content": _adv_system_prompt}]
                for _m in st.session_state.advisor_messages[:-1]:
                    _llm_msgs_v.append({"role": _m["role"], "content": _m["content"]})
                with st.chat_message("assistant"):
                    with st.spinner("Analysing image…"):
                        try:
                            from ragvue.src.core.llm_judge import call_judge_vision, ensure_env
                            ensure_env()
                            _prev_p = os.environ.get("RAGVUE_JUDGE_PROVIDER")
                            os.environ["RAGVUE_JUDGE_PROVIDER"] = _adv_provider
                            try:
                                _response = call_judge_vision(_llm_msgs_v, _img_b64, _media_type, _file_question, model=_adv_model, temperature=0.7)
                            finally:
                                if _prev_p is None: os.environ.pop("RAGVUE_JUDGE_PROVIDER", None)
                                else: os.environ["RAGVUE_JUDGE_PROVIDER"] = _prev_p
                        except Exception as _e:
                            _response = f"Vision analysis failed. Error: `{_e}`"
                    st.markdown(_response)
                st.session_state.advisor_messages.append({"role": "assistant", "content": _response})
                _save_advisor_history(st.session_state.advisor_messages)
                st.rerun()

            elif _adv_fname.endswith(".pdf"):
                _pdf_text = ""
                try:
                    import pypdf, io as _io
                    _pdf_text = "\n\n".join(p.extract_text() or "" for p in pypdf.PdfReader(_io.BytesIO(_adv_file_bytes)).pages).strip()
                except ImportError:
                    try:
                        import PyPDF2, io as _io
                        _pdf_text = "\n\n".join(p.extract_text() or "" for p in PyPDF2.PdfReader(_io.BytesIO(_adv_file_bytes)).pages).strip()
                    except ImportError:
                        pass
                _inject = (
                    f"📎 *Document: `{_adv_file.name}`*\n\n{_file_question}\n\n---\n{_pdf_text[:6000]}" if _pdf_text
                    else f"I uploaded `{_adv_file.name}` but PDF text extraction is unavailable. Install `pypdf` or paste the text directly."
                )
                st.session_state.advisor_messages.append({"role": "user", "content": _inject})
                _save_advisor_history(st.session_state.advisor_messages)
                st.rerun()

            else:
                try: _txt = _adv_file_bytes.decode("utf-8")
                except Exception: _txt = _adv_file_bytes.decode("latin-1", errors="replace")
                st.session_state.advisor_messages.append({"role": "user", "content": f"📎 *Document: `{_adv_file.name}`*\n\n{_file_question}\n\n---\n{_txt[:6000]}"})
                _save_advisor_history(st.session_state.advisor_messages)
                st.rerun()

        # Quick starters — only on first open
        if len(st.session_state.advisor_messages) <= 1:
            st.markdown("**Try asking:**")
            _starters = [
                "What can you help me with?",
                "My faithfulness is low — could it be a chunking problem?",
                "How do I choose between dense and sparse retrieval?",
                "My context utilization score is low — what does that mean?",
                "What chunk size would you recommend for technical documentation?",
            ]
            _sc1, _sc2 = st.columns(2)
            for _i, _s in enumerate(_starters):
                with (_sc1 if _i % 2 == 0 else _sc2):
                    if st.button(_s, key=f"starter_{_i}", use_container_width=True,
                                 type="primary" if _i == 0 else "secondary"):
                        st.session_state.advisor_messages.append({"role": "user", "content": _s})
                        _save_advisor_history(st.session_state.advisor_messages)
                        st.rerun()
            st.markdown("")

        # Chat history
        for _msg in st.session_state.advisor_messages:
            with st.chat_message(_msg["role"]):
                st.markdown(_msg["content"])

        # Chat input
        if _user_input := st.chat_input("Ask a question or describe your setup…", key="advisor_input"):
            st.session_state.advisor_messages.append({"role": "user", "content": _user_input})
            with st.chat_message("user"):
                st.markdown(_user_input)
            _llm_msgs = [{"role": "system", "content": _adv_system_prompt}]
            for _m in st.session_state.advisor_messages:
                _llm_msgs.append({"role": _m["role"], "content": _m["content"]})
            with st.chat_message("assistant"):
                try:
                    from ragvue.src.core.llm_judge import call_judge_text_stream, ensure_env
                    ensure_env()
                    _prev_p = os.environ.get("RAGVUE_JUDGE_PROVIDER")
                    os.environ["RAGVUE_JUDGE_PROVIDER"] = _adv_provider
                    def _safe_stream(_gen):
                        try:
                            yield from _gen
                        except Exception as _se:
                            yield f"\n\n⚠️ Stream interrupted: `{_se}`"
                    try:
                        _response = st.write_stream(_safe_stream(
                            call_judge_text_stream(_llm_msgs, model=_adv_model, temperature=0.7)
                        ))
                    finally:
                        if _prev_p is None: os.environ.pop("RAGVUE_JUDGE_PROVIDER", None)
                        else: os.environ["RAGVUE_JUDGE_PROVIDER"] = _prev_p
                except Exception as _e:
                    _response = f"Sorry, I couldn't reach the LLM. Error: `{_e}`\n\nMake sure your API key is set in the sidebar."
                    st.markdown(_response)
            st.session_state.advisor_messages.append({"role": "assistant", "content": _response})
            _save_advisor_history(st.session_state.advisor_messages)

    # ═══════════════════════════ PROFILE SUB-TAB ═════════════════════════════
    with _atab_profile:
        st.markdown('<h4 class="tab-heading">🗂 My Architecture Profiles</h4>', unsafe_allow_html=True)
        st.caption(
            "Save one profile per pipeline configuration. Set one as **active** — "
            "Your RAG Advisor uses it automatically in every conversation."
        )

        _pf_profiles  = _adv_store.get("profiles", [])
        _pf_active    = _adv_store.get("active", -1)

        # ── Saved profiles list ───────────────────────────────────────────────
        if "pf_edit_idx" not in st.session_state:
            st.session_state.pf_edit_idx = -1

        if _pf_profiles:
            st.markdown("**Saved profiles:**")
            for _pi, _pp in enumerate(_pf_profiles):
                _is_active = (_pi == _pf_active)
                _pc1, _pc2, _pc3, _pc4 = st.columns([5, 2, 1.5, 1.5])
                with _pc1:
                    _badge = " ✅ active" if _is_active else ""
                    _saved = f"  ·  saved {_pp.get('saved_at', '')[:10]}" if _pp.get("saved_at") else ""
                    st.markdown(f"**{_pp.get('name', f'Profile {_pi+1}')}**{_badge}{_saved}")
                    _summary_parts = [
                        _pp.get("retriever"), _pp.get("generation_llm"),
                        (_pp.get("chunk_size") and f"chunk={_pp['chunk_size']}"),
                        (_pp.get("top_k") and f"k={_pp['top_k']}"),
                    ]
                    st.caption("  ·  ".join(p for p in _summary_parts if p))
                with _pc2:
                    if not _is_active:
                        if st.button("Set active", key=f"pf_activate_{_pi}", use_container_width=True):
                            _adv_store["active"] = _pi
                            _save_profile_store(_adv_store)
                            st.success(f"**{_pp.get('name', f'Profile {_pi+1}')}** is now active.")
                            st.rerun()
                    else:
                        st.markdown("<div style='padding:6px 0; color:var(--muted)'>Active</div>", unsafe_allow_html=True)
                with _pc3:
                    if st.button("✏️ Edit", key=f"pf_edit_{_pi}", use_container_width=True):
                        st.session_state.pf_edit_idx = _pi
                        st.rerun()
                with _pc4:
                    if st.button("🗑 Delete", key=f"pf_delete_{_pi}", use_container_width=True):
                        _pf_profiles.pop(_pi)
                        _adv_store["profiles"] = _pf_profiles
                        if _pf_active >= len(_pf_profiles):
                            _adv_store["active"] = len(_pf_profiles) - 1
                        elif _pf_active == _pi:
                            _adv_store["active"] = -1
                        if st.session_state.pf_edit_idx == _pi:
                            st.session_state.pf_edit_idx = -1
                        _save_profile_store(_adv_store)
                        st.rerun()
            st.markdown("---")

        # ── Add / edit form ───────────────────────────────────────────────────
        _edit_idx = st.session_state.pf_edit_idx
        _edit_pf  = _pf_profiles[_edit_idx] if 0 <= _edit_idx < len(_pf_profiles) else {}
        if _edit_pf:
            _form_title = f"✏️ Editing: **{_edit_pf.get('name', f'Profile {_edit_idx + 1}')}**"
        elif _pf_profiles:
            _form_title = "➕ Add a new profile"
        else:
            _form_title = "Create your first profile"
        st.markdown(_form_title)

        _pf_name = st.text_input("Profile name *", value=_edit_pf.get("name", ""), placeholder="e.g. BM25 baseline · Dense 256-chunk · Hybrid v2", key="pf_name")
        _pf_c1, _pf_c2 = st.columns(2)
        with _pf_c1:
            _pf_retriever  = st.text_input("Retriever type",      value=_edit_pf.get("retriever", ""),       placeholder="e.g. FAISS dense, BM25, hybrid",    key="pf_retriever")
            _pf_chunk_size = st.text_input("Chunk size (tokens)", value=_edit_pf.get("chunk_size", ""),      placeholder="e.g. 512",                          key="pf_chunk_size")
            _pf_overlap    = st.text_input("Chunk overlap",       value=_edit_pf.get("chunk_overlap", ""),   placeholder="e.g. 50 tokens / 10%",              key="pf_overlap")
            _pf_k          = st.text_input("Top-k retrieved",     value=_edit_pf.get("top_k", ""),           placeholder="e.g. 5",                            key="pf_k")
        with _pf_c2:
            _pf_embedding  = st.text_input("Embedding model",     value=_edit_pf.get("embedding_model", ""), placeholder="e.g. text-embedding-3-small",       key="pf_embedding")
            _pf_llm        = st.text_input("Generation LLM",      value=_edit_pf.get("generation_llm", ""),  placeholder="e.g. GPT-4o, LLaMA-3",              key="pf_llm")
            _pf_framework  = st.text_input("Framework / stack",   value=_edit_pf.get("framework", ""),       placeholder="e.g. LlamaIndex, LangChain",        key="pf_framework")
            _pf_domain     = st.text_input("Domain / use case",   value=_edit_pf.get("domain", ""),          placeholder="e.g. medical Q&A, legal documents", key="pf_domain")
        _pf_notes = st.text_area("Additional notes", value=_edit_pf.get("notes", ""), placeholder="Reranker, prompt template, special constraints…", key="pf_notes", height=70)

        _save_label = "💾 Update profile" if _edit_pf else "💾 Save & set active"
        _save_col, _cancel_col = (st.columns([3, 1]) if _edit_pf else (st.columns([1])[0], None))
        with _save_col:
            _do_save = st.button(_save_label, key="pf_save", type="primary", use_container_width=True)
        if _cancel_col:
            with _cancel_col:
                if st.button("Cancel", key="pf_cancel", use_container_width=True):
                    st.session_state.pf_edit_idx = -1
                    st.rerun()

        if _do_save:
            if not _pf_name.strip():
                st.warning("Give this profile a name so you can tell them apart.")
            else:
                _new_pf = {
                    "name": _pf_name.strip(),
                    "retriever": _pf_retriever, "chunk_size": _pf_chunk_size,
                    "chunk_overlap": _pf_overlap, "top_k": _pf_k,
                    "embedding_model": _pf_embedding, "generation_llm": _pf_llm,
                    "framework": _pf_framework, "domain": _pf_domain, "notes": _pf_notes,
                    "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                if 0 <= _edit_idx < len(_pf_profiles):
                    _pf_profiles[_edit_idx] = _new_pf
                    _adv_store["profiles"] = _pf_profiles
                    _save_profile_store(_adv_store)
                    st.session_state.pf_edit_idx = -1
                    st.success(f"Profile **{_pf_name.strip()}** updated.")
                else:
                    _pf_profiles.append(_new_pf)
                    _adv_store["profiles"] = _pf_profiles
                    _adv_store["active"]   = len(_pf_profiles) - 1
                    _save_profile_store(_adv_store)
                    st.success(f"Profile **{_pf_name.strip()}** saved and set as active.")
                st.rerun()

    # ═══════════════════════════ ANALYSIS TOOLS SUB-TAB ══════════════════════
    with _atab_tools:
        st.markdown('<h4 class="tab-heading">🔬 Analysis Tools</h4>', unsafe_allow_html=True)
        st.caption("Each tool sends a structured message to the Chat — switch to the Chat tab to see the response.")

        # ── Before / After Comparison ─────────────────────────────────────────
        st.markdown("**📊 Before / After Report Comparison**")
        st.caption("Select two saved reports. Your advisor explains what changed and hypothesises why.")
        _ba_report_opts = ["(select a report)"]
        _ba_reports: list = []
        if REPORTS_PATH.exists():
            try:
                with open(REPORTS_PATH, "r", encoding="utf-8") as _f:
                    _ba_reports = json.load(_f)
                _ba_report_opts += [f"{e['timestamp']} · {e['label']}" for e in _ba_reports]
            except Exception:
                pass
        _ba_c1, _ba_c2 = st.columns(2)
        with _ba_c1:
            _ba_sel_a = st.selectbox("Report A (baseline):", _ba_report_opts, key="ba_sel_a")
        with _ba_c2:
            _ba_sel_b = st.selectbox("Report B (new run):",  _ba_report_opts, key="ba_sel_b")
        if st.button("Send comparison to chat →", key="ba_compare", use_container_width=True):
            if _ba_sel_a != "(select a report)" and _ba_sel_b != "(select a report)" and _ba_sel_a != _ba_sel_b:
                _ba_idx_a  = _ba_report_opts.index(_ba_sel_a) - 1
                _ba_idx_b  = _ba_report_opts.index(_ba_sel_b) - 1
                _ba_sum_a  = compute_summary_from_results(_ba_reports[_ba_idx_a].get("report", {}).get("results", []))
                _ba_sum_b  = compute_summary_from_results(_ba_reports[_ba_idx_b].get("report", {}).get("results", []))
                _ba_lines  = []
                for _k in sorted(set(_ba_sum_a) | set(_ba_sum_b)):
                    _va, _vb = _ba_sum_a.get(_k), _ba_sum_b.get(_k)
                    if _va is not None and _vb is not None:
                        _d = _vb - _va
                        _ba_lines.append(f"  - {_k}: {_va:.3f} → {_vb:.3f} ({'▲' if _d > 0.01 else '▼' if _d < -0.01 else '≈'}{_d:+.3f})")
                    elif _vb is not None:
                        _ba_lines.append(f"  - {_k}: (new) {_vb:.3f}")
                st.session_state.advisor_messages.append({"role": "user", "content": (
                    f"Comparing runs — A: **{_ba_reports[_ba_idx_a]['label']}** vs B: **{_ba_reports[_ba_idx_b]['label']}**\n\n"
                    "Metric changes (B − A):\n" + "\n".join(_ba_lines) + "\n\n"
                    "Based on my architecture, what likely caused these changes and what should I investigate?"
                )})
                _save_advisor_history(st.session_state.advisor_messages)
                st.success("Sent to chat — switch to the **Chat** tab to see the response.")
            else:
                st.warning("Select two different reports.")

        st.markdown("---")

        # ── Hypothesis Testing ────────────────────────────────────────────────
        st.markdown("**🔬 State a Hypothesis**")
        st.caption("Describe a change you're planning. Your advisor predicts which metrics will improve — then validate with a new evaluation.")
        _hyp_text = st.text_area(
            "Your hypothesis:",
            placeholder="e.g. I'm going to reduce chunk size from 512 to 256 tokens with 20% overlap.",
            key="hyp_text", height=90,
        )
        if st.button("Send hypothesis to chat →", key="hyp_submit", use_container_width=True):
            if _hyp_text.strip():
                st.session_state.advisor_messages.append({"role": "user", "content": (
                    f"**Hypothesis:** {_hyp_text.strip()}\n\n"
                    "Which RAGVue metrics do you predict will improve or degrade, and why? "
                    "I'll run the evaluation after this change and share the new results."
                )})
                _save_advisor_history(st.session_state.advisor_messages)
                st.success("Sent to chat — switch to the **Chat** tab to see the response.")
            else:
                st.warning("Enter your hypothesis first.")

        st.markdown("---")

        # ── Failure Mode Scanner ──────────────────────────────────────────────
        st.markdown("**🔴 Failure Mode Scanner**")
        st.caption("Select a report. Your advisor identifies active failure modes, explains root causes, and prioritises the top 2 issues to fix.")
        _fm_opts = ["(select a report)"]
        _fm_reports: list = []
        if REPORTS_PATH.exists():
            try:
                with open(REPORTS_PATH, "r", encoding="utf-8") as _fm_f:
                    _fm_reports = json.load(_fm_f)
                _fm_opts += [f"{e['timestamp']} · {e['label']}" for e in _fm_reports]
            except Exception:
                pass
        _fm_sel = st.selectbox("Report for scanning:", _fm_opts, key="fm_sel", label_visibility="collapsed")
        if st.button("Scan for failure modes →", key="fm_scan", use_container_width=True):
            if _fm_sel != "(select a report)":
                _fm_idx = _fm_opts.index(_fm_sel) - 1
                _fm_sum = compute_summary_from_results(_fm_reports[_fm_idx].get("report", {}).get("results", []))
                if _fm_sum:
                    _fm_scores = "\n".join(f"  - {k}: {v:.3f}" for k, v in sorted(_fm_sum.items()))
                    st.session_state.advisor_messages.append({"role": "user", "content": (
                        f"**Failure Mode Scanner** — Report: **{_fm_reports[_fm_idx]['label']}**\n\n"
                        f"Metric scores:\n{_fm_scores}\n\n"
                        "Based on these scores, please:\n"
                        "1. Identify which RAG failure modes are active (retrieval miss, context ignorance, hallucination, "
                        "over-confidence, multi-hop gap, generation drift, or others)\n"
                        "2. For each active failure mode, briefly explain what is likely happening in the pipeline\n"
                        "3. Prioritise the top 2 most critical issues to fix first\n"
                        "4. For each priority issue, suggest one concrete, testable intervention"
                    )})
                    _save_advisor_history(st.session_state.advisor_messages)
                    st.success("Sent to chat — switch to the **Chat** tab to see the failure mode analysis.")
                else:
                    st.warning("No metric scores found in this report.")
            else:
                st.warning("Select a report first.")

        st.markdown("---")

        # ── Suggest Next Experiment ───────────────────────────────────────────
        st.markdown("**🧪 Suggest Next Experiment**")
        st.caption("Given your current scores, your advisor recommends the single highest-ROI next experiment — specific, measurable, and actionable.")
        _nx_opts = ["(select a report)"]
        _nx_reports: list = []
        if REPORTS_PATH.exists():
            try:
                with open(REPORTS_PATH, "r", encoding="utf-8") as _nx_f:
                    _nx_reports = json.load(_nx_f)
                _nx_opts += [f"{e['timestamp']} · {e['label']}" for e in _nx_reports]
            except Exception:
                pass
        _nx_sel = st.selectbox("Report for experiment suggestion:", _nx_opts, key="nx_sel", label_visibility="collapsed")
        if st.button("Get next experiment →", key="nx_suggest", use_container_width=True):
            if _nx_sel != "(select a report)":
                _nx_idx = _nx_opts.index(_nx_sel) - 1
                _nx_sum = compute_summary_from_results(_nx_reports[_nx_idx].get("report", {}).get("results", []))
                if _nx_sum:
                    _nx_scores = "\n".join(f"  - {k}: {v:.3f}" for k, v in sorted(_nx_sum.items()))
                    st.session_state.advisor_messages.append({"role": "user", "content": (
                        f"**Next Experiment Advisor** — Report: **{_nx_reports[_nx_idx]['label']}**\n\n"
                        f"Current scores:\n{_nx_scores}\n\n"
                        "Based on these scores and my architecture profile, recommend the **single highest-ROI next experiment** to run. "
                        "Be specific and opinionated — one experiment only, not a menu of options:\n"
                        "1. What exactly should I change, and from what value to what value?\n"
                        "2. Which RAGVue metrics should improve, and roughly by how much?\n"
                        "3. What result (metric threshold or pattern) would confirm the experiment succeeded?\n"
                        "4. Are there any metrics that might degrade as a trade-off?"
                    )})
                    _save_advisor_history(st.session_state.advisor_messages)
                    st.success("Sent to chat — switch to the **Chat** tab to see the recommendation.")
                else:
                    st.warning("No metric scores found in this report.")
            else:
                st.warning("Select a report first.")

        st.markdown("---")

        # ── Guided Diagnosis ──────────────────────────────────────────────────
        st.markdown("**🩺 Guided Diagnosis**")
        st.caption("Your advisor walks through Retrieval → Grounding → Generation, asking one question at a time, then produces a structured diagnosis you can export.")
        if st.button("Start guided diagnosis →", key="guided_diag", use_container_width=True, type="primary"):
            st.session_state.advisor_messages.append({"role": "user", "content": (
                "Please start a structured guided diagnosis of my RAG pipeline. "
                "Walk through each layer in order: (1) Retrieval quality, (2) Grounding / faithfulness, (3) Answer generation. "
                "Ask me one focused question at a time per layer. "
                "At the end, produce a concise structured diagnosis summary I can include in my research notes."
            )})
            _save_advisor_history(st.session_state.advisor_messages)
            st.success("Diagnosis started — switch to the **Chat** tab to begin.")

# ============================== FOOTER =======================================
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; font-size:0.85rem; color: var(--muted); line-height:1.6; padding: 1rem 0 0.5rem;">
        <span style="font-size:0.75rem; letter-spacing:0.08em; text-transform:uppercase; opacity:0.7;">Don't just score your RAG, Diagnose it </span><br>
        © 2026 · Developed by <b style="color: var(--text);">Keerthana Murugaraj</b><br>
        <span style="opacity:0.8;">Doctoral Researcher · GenAI · NLP · RAG · Agentic RAG · RAG Evaluation</span>
    </div>
    """,
    unsafe_allow_html=True,
)


