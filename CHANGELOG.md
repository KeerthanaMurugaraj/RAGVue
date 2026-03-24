# RAGVue Changelog

---

## [0.3.0 – 0.6.0] — March 2026 — Major Update

> This entry consolidates four release increments into a single major update covering the Multi-LLM Judge Backend, the Your RAG Advisor feature, a full UI/Dashboard overhaul, and a suite of RAG Advisor enhancements.

---

### Part 1 — Multi-LLM Judge Backend (`ragvue/src/core/llm_judge.py`)

All 12 OpenAI-based metrics now route through a provider-agnostic judge layer:

- **OpenAI** (default): `gpt-4o-mini` via `response_format=json_object`
- **Anthropic Claude** (optional): `claude-haiku-4-5-20251001` — cheap, fast, no JSON mode needed

Switch provider via environment variable or Streamlit sidebar:
```bash
export RAGVUE_JUDGE_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

Install with Anthropic dependency:
```bash
pip install -e ".[anthropic]"   # or pip install -e ".[all]"
```

The `llm_judge.py` module provides:
- `call_judge(messages, model, temperature)` — JSON output (metric evaluation)
- `call_judge_text(messages, model, temperature)` — text output (aspect extraction)
- `call_judge_text_stream(messages, model, temperature)` — streaming generator for token-by-token output
- `default_model()` — provider-appropriate default (`gpt-4o-mini` or `claude-haiku-4-5-20251001`)
- `ensure_env()` — loads both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` from `.env`

#### Streamlit UI: Judge Provider Selector
- **Judge Provider radio** in sidebar — toggle between OpenAI and Claude
- **Anthropic API key input** shown when Claude is selected
- Evaluation pre-check validates the correct key for the chosen provider

#### Notes
- Calibration metrics always use OpenAI (their job is cross-model agreement measurement)
- Per-metric model overrides (`ANSWER_RELEVANCE_MODEL`, etc.) work with both providers — set to a Claude model ID to use Claude for a specific metric

---

### Part 2 — Your RAG Advisor (`streamlit_app.py`)

A persistent conversational assistant embedded in the Streamlit dashboard for RAG architecture advice, with three sub-tabs: Chat, My Profile, and Analysis Tools.

#### Chat
- **Chatbox UI** — full conversation interface (`st.chat_message` + `st.chat_input`)
- **Streaming responses** — token-by-token output via `st.write_stream`; no waiting for full reply; mid-stream errors caught and displayed inline
- **Persistent memory** — conversation history saved to `rag_advisor_history.json`; survives page refreshes and restarts (last 20 turns)
- **Report injection** — select any saved report from a dropdown and inject its full metric summary into the conversation with one click; no manual copy-paste
- **Auto-inject banner** — a banner appears at the top of the Chat tab whenever a newer evaluation has been run; "Share with advisor" injects the metric summary in one click; "✕" dismisses without sharing; the banner only appears once per new run
- **RAGVue-aware system prompt** — advisor knows all 22 metrics, their meaning, and common RAG failure patterns; gives architecture-specific advice rather than generic tips
- **Token-efficient design** — users control what context is shared rather than auto-loading all results
- **Export** — download full conversation as Markdown
- **Quick starters** — pre-built prompt buttons shown on first open

#### My Profile
- **Multi-profile store** — save one profile per pipeline configuration; set any profile as active
- **Fields:** retriever type, chunk size, chunk overlap, top-k, embedding model, generation LLM, framework, domain, and free-text notes
- **Edit in place** — ✏️ Edit button pre-fills the form with existing values; Save updates the profile; Cancel exits without changes
- **Auto-inject into system prompt** — the active profile is injected into every advisor conversation automatically, so the advisor always knows your current setup

#### Analysis Tools
- **Before / After Report Comparison** — select two saved reports; advisor explains what changed and hypothesises why based on your architecture
- **Hypothesis Testing** — describe a planned change; advisor predicts which metrics will improve or degrade; validate with a new evaluation run
- **Guided Diagnosis** — step-by-step walkthrough of Retrieval → Grounding → Generation, one focused question at a time; ends with a structured diagnosis summary
- **Failure Mode Scanner** — select any saved report; advisor identifies active failure modes (retrieval miss, context ignorance, hallucination, over-confidence, multi-hop gap, generation drift); explains root causes; prioritises the top 2 issues; proposes one concrete, testable intervention per issue
- **Suggest Next Experiment** — select any saved report; advisor recommends the single highest-ROI next experiment (specific change, expected metric impact, success threshold, potential trade-offs)

#### Model Selection
- Advisor model selector is independent of the global eval provider
- Expanded from 2 to 6 options across both providers:

| Model | Provider | Tier |
|-------|----------|------|
| `gpt-4o-mini` | OpenAI | fast (default) |
| `gpt-4o` | OpenAI | capable |
| `gpt-3.5-turbo` | OpenAI | budget |
| `claude-haiku-4-5-20251001` | Anthropic | fast |
| `claude-sonnet-4-6` | Anthropic | balanced |
| `claude-opus-4-6` | Anthropic | powerful |

- Provider (OpenAI vs Anthropic) is auto-detected from the model name — no separate provider toggle needed

**Why it exists:** Evaluation results live in the dashboard — the advisor brings architecture expertise directly to those results without leaving the UI or manually exporting data.

---

### Part 3 — UI / Dashboard Overhaul (`streamlit_app.py`)

#### Theming System
- **Three themes** — Light (default), Dark (soft/dim), Beige; selectable from the ⚙️ Settings sidebar
- **Light mode is now the default** — both session state and `config.toml` start in Light
- **Soft dark mode** — replaced the near-black `#0b0f19` palette with a comfortable slate-blue dim palette (`#1e2433` base); much easier on the eyes at night
- **Beige mode** — warm parchment palette for low-contrast reading environments
- All three themes apply consistently across every UI element via CSS custom properties

#### Button Fixes
- **Always-white button text** — button text is hardcoded `#ffffff` across all themes; previously dark-theme buttons had black text (`#0b0f19`)
- **Compact button size** — removed oversized padding that made buttons look like banners; children no longer inherit box-model styles (no more "box inside a button")
- **Export / download buttons visible** — fixed conflicting CSS rules that made download button text invisible (white text on white background) in light mode

#### Sidebar Collapse Arrow
- Fixed garbled characters near the sidebar open/minimize arrow caused by `font-family !important` being applied to Streamlit's internal collapse-button glyphs; added explicit exclusion for `[data-testid="collapsedControl"]`

#### Longitudinal Tab — Themed Components
- **Run Registry table** — replaced `st.dataframe()` with a hand-built HTML `<table class="themed-table">` that fully respects all three themes (background, borders, hover, alternating rows via CSS variables)
- **Metric Trends chart** — replaced `st.line_chart()` with a Plotly `go.Figure()` whose background, grid, tick, legend, and font colours are dynamically read from the active theme at render time
- Added `plotly>=5.18` to core dependencies
---

## [0.2.0] — March 2026

### New Features

#### FastAPI REST API (`ragvue/api.py`)
A full evaluation API server with 5 endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health + loaded metric count |
| `GET` | `/metrics` | List all available metric names |
| `POST` | `/evaluate` | Evaluate a single item |
| `POST` | `/evaluate/batch` | Evaluate multiple items (returns per-metric mean) |
| `POST` | `/evaluate/agentic` | Agentic evaluation — auto-selects metrics per item |

Start with `ragvue-api` (or `ragvue-api --port 9000`). Install with `pip install -e ".[all]"`.

#### 4 Local Metrics (no API calls)
Zero-cost, instant diagnostic metrics that run entirely on-device:

| Metric | Inputs | What it measures |
|--------|--------|-----------------|
| Token Overlap | Q, A, C | Lexical overlap between answer and context |
| Answer Length | A | Answer length relative to question complexity |
| Context Similarity | A, C | Semantic closeness between answer and context |
| Readability | A | Flesch-Kincaid readability score |

Install with `pip install -e ".[local]"` (requires scikit-learn).

#### Streamlit UI Improvements
- **Live progress bar** — per-item progress during evaluation
- **Custom report labels** — name a report before running
- **Pipeline version & notes** — tag each run for longitudinal comparison
- **Retrieval Only mode** — evaluate retrieval pipeline without requiring generated answers
- **Report history** — last 10 reports saved automatically; view, filter, and delete from the Reports tab
- **Report comparison** — select two saved reports side-by-side with per-metric deltas (B − A)
- **Longitudinal tracking** — persistent run registry, metric trend chart, automatic regression detection

#### Longitudinal Tracking (`run_registry.json`)
Every evaluation run is recorded in `run_registry.json` (no cap) with:
- Timestamp, label, pipeline version, notes
- Per-metric summary scores

The **Longitudinal tab** provides:
1. Full run registry table
2. Metric trend line chart (select which metrics to plot)
3. Regression detection — compares last 2 runs, flags drops above a configurable threshold (🔴 / 🟡 / 🟢)

#### Custom Metric Plugin Template
`ragvue/src/metrics/metric_template.py` — a fully documented template for writing custom metrics, with both OpenAI-based and local patterns. Copy, rename, and drop into `ragvue/src/metrics/` — auto-discovered with no registration needed.

### Bug Fixes
- Status messages in Streamlit UI now correctly clear after evaluation completes
- Fixed duplicate `download_button` key errors when viewing reports from the Reports tab

### Install

```bash
pip install -e ".[all]"   # includes FastAPI + local metric deps
```

---

## [0.1.0] — March 2026 — Performance & New Metrics Update

### New Metrics (6 reference-free metrics)

Six new evaluation metrics added to detect complex RAG failure modes that the original 7 core metrics miss. No reference answers required.

| Metric | Inputs | What it catches |
|---|---|---|
| **Context Utilization** | Q, A, C | Retrieved context is fetched but ignored — answer doesn't use the evidence |
| **Answer Conciseness** | Q, A | Verbose, repetitive, or filler-heavy answers that obscure the core response |
| **Negative Rejection** | Q, A, C | System confidently answers when context doesn't support any answer |
| **Coherence** | Q, A | Internal self-contradictions, logical fallacies, non-sequiturs, circular reasoning |
| **Multi-Hop Faithfulness** | Q, A, C | Broken reasoning chains in multi-step answers |
| **Implicit Contradiction** | Q, A, C | Subtle contradictions that strict faithfulness misses (omitted qualifiers, shifted scope, negation flips) |

**Files created:**
- `ragvue/src/metrics/context_utilization.py`
- `ragvue/src/metrics/answer_conciseness.py`
- `ragvue/src/metrics/negative_rejection.py`
- `ragvue/src/metrics/coherence.py`
- `ragvue/src/metrics/multi_hop_faithfulness.py`
- `ragvue/src/metrics/implicit_contradiction.py`

All follow the existing module-level pattern (`evaluate(item)` entry point). Auto-discovery picks them up — no registration needed.

Total metrics: **18** (6 core evaluation + 6 calibration + 6 new complex failure mode).

---

### Performance: Parallel Metric Execution

**Problem:** Metrics ran sequentially — each API call waited for the previous one to finish. With 11+ metrics per item, evaluation was slow.

**Solution:** Added `ThreadPoolExecutor`-based parallelism at two levels:

#### 1. Parallel metrics per item (`manual_mode.py`)

Before:
```
metric_1 (1.2s) → metric_2 (1.0s) → metric_3 (1.1s) → ... → metric_11 (0.9s)
Total: ~12s per item (sequential)
```

After:
```
metric_1 (1.2s) ┐
metric_2 (1.0s) ├── all run concurrently
metric_3 (1.1s) │
...             │
metric_11 (0.9s)┘
Total: ~1.5s per item (limited by slowest call)
```

#### 2. Parallel items in agentic mode (`agentic_mode.py`)

Before:
```
item_1 → item_2 → item_3 → item_4
Total: sum of all item times
```

After:
```
item_1 ┐
item_2 ├── up to 4 items concurrently
item_3 │
item_4 ┘
Total: ~max(item times) instead of sum
```

#### Configurable via environment variables

Add to your `.env` to tune:
```
RAGVUE_METRIC_WORKERS=8   # max parallel metric calls per item (default: 8)
RAGVUE_ITEM_WORKERS=4     # max parallel items in agentic mode (default: 4)
```

Lower these if you hit OpenAI rate limits. Raise them if you have a high-tier API plan.

**Files modified:**
- `ragvue/src/core/manual_mode.py` — parallel metrics per item via `ThreadPoolExecutor`
- `ragvue/src/core/agentic_mode.py` — parallel items + extracted `_run_single_item()` helper

---

### Agentic Mode Updates (`agentic_mode.py`)

**Metric auto-selection:** New metrics are automatically selected based on input availability:
- `context_utilization` — when contexts are present
- `answer_conciseness` — when answer is present
- `coherence` — when answer is present
- `negative_rejection` — when answer and contexts are present
- `implicit_contradiction` — when answer and contexts are present
- `multi_hop_faithfulness` — when question looks multi-hop and contexts are present

**Answer overall weights:** Updated `_synthesize_answer_overall()` from 4 to 9 components:

| Metric | Weight |
|---|---|
| Strict Faithfulness | 0.30 |
| Answer Relevance | 0.20 |
| Coherence | 0.10 |
| Implicit Contradiction | 0.10 |
| Answer Completeness | 0.10 |
| Clarity | 0.05 |
| Answer Conciseness | 0.05 |
| Negative Rejection | 0.05 |
| Multi-Hop Faithfulness | 0.05 |

Weights are renormalized at runtime over whichever metrics are actually present.

---

### Streamlit UI Updates (`streamlit_app.py`)

- **Diagnostic rendering:** Added `_render_metric_diagnostics()` — each new metric's diagnostic fields are rendered with structured formatting instead of raw JSON:
  - Context utilization: utilized/unused chunk indices
  - Answer conciseness: redundant parts and filler phrases listed
  - Negative rejection: context sufficiency and refusal status
  - Coherence: contradictions and logical issues listed
  - Multi-hop faithfulness: reasoning chain with per-step pass/fail icons
  - Implicit contradiction: contradiction types, claims, and severity
- **Justification display:** Per-metric expanders now show `justification` (used by all new metrics)
- **Summary column:** Metrics table includes a truncated summary for quick scanning

---

### Documentation

- **`METRICS.md`** — Reader-friendly guide to all 18 metrics with scoring details, diagnostic fields, selection guide by use case, and agentic mode behavior
- **`README.md`** — Updated metric count and added tables for the 6 new metrics and agentic auto-selection rules

---

### Install

After pulling these changes:
```bash
pip install -e .
```
This is required for the CLI (`ragvue-cli`) to pick up the new metric files.