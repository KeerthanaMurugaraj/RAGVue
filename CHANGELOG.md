# RAGVue Changelog

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

## [0.1.0] — February 2026 — Performance & New Metrics Update

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