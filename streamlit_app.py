from __future__ import annotations
import os, io, json, csv, statistics , time
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
# ──────────────────────────────────────────────────────────────────────────────
DARK = {
    "--bg": "#0b0f19",
    "--bg-alt": "#0f1422",
    "--text": "#e5e7eb",
    "--muted": "#9ca3af",
    "--accent": "#8b93ff",
    "--accent-contrast": "#0b0f19",
    "--card": "#121829",
    "--card-border": "#1f2937",
    "--chip-bg": "#1d2437",
    "--chip-border": "#2a3550",
    "--kbd": "#e5e7eb",
    "--sidebar-bg": "#0f1422",
    "--sidebar-border": "#1f2937",
    "--focus": "#fbbf24"
}

def inject_dark_theme():
    t = DARK
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
      }}

      .stApp {{
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, sans-serif;
        background: linear-gradient(140deg, var(--bg), var(--bg-alt));
        color: var(--text);
      }}
      .main .block-container {{ padding-top: 1.25rem; padding-bottom: 2rem; }}

      h1,h2,h3,h4,h5,h6, .stMarkdown p, .stCaption, .stText, .stCode {{ color: var(--text); }}

      /* SIDEBAR */
      [data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        color: var(--text) !important;
        border-right: 1px solid var(--sidebar-border);
      }}
      [data-testid="stSidebar"] * {{ color: var(--text) !important; }}

      /* Cards */
      .card {{
        border: 1px solid var(--card-border);
        background: var(--card);
        border-radius: 14px;
        padding: 1rem 1rem;
      }}

      /* Buttons */
      .stButton>button {{
        background: var(--accent) !important;
        color: var(--accent-contrast) !important;
        border: 0;
        border-radius: 10px;
        padding: .6rem 1rem;
        font-weight: 700;
        box-shadow: 0 4px 16px rgba(0,0,0,.35);
      }}
      .stButton>button:hover {{ filter: brightness(1.06); }}
      .stButton>button:focus {{ outline: 3px solid var(--focus); outline-offset: 2px; }}

      /* Inputs */
      .stTextInput > div > div > input,
      .stNumberInput input,
      .stSelectbox > div > div > div,
      .stFileUploader,
      .stDataFrame {{ color: var(--text) !important; }}

      /* Chips */
      .chip {{
        display:inline-flex; align-items:center; gap:.4rem;
        padding: .25rem .55rem; border-radius:999px;
        background: var(--chip-bg); border: 1px solid var(--chip-border);
        font-size:.85rem; color: var(--text);
      }}

      /* Sticky summary */
      #summary-card {{
        position: sticky;
        top: .5rem;
        z-index: 50;
        border: 1px solid var(--card-border);
        background: var(--card);
        border-radius: 14px;
        padding: 1rem 1rem;
        margin-bottom: .5rem;
      }}

      /* Focus ring for common inputs (a11y) */
      input:focus, select:focus, textarea:focus {{
        outline: 3px solid var(--focus) !important; outline-offset: 1px !important;
      }}

      kbd {{
        background: var(--kbd); color: #0b0f19; border-radius:6px;
        padding: 1px 6px; font-size: .8em; font-weight: 700;
      }}
      .muted {{ color: var(--muted); }}
      footer {{ text-align:center; margin-top: 1rem; color: var(--muted); }}
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

def build_download(data: str | bytes, filename: str, mime: str):
    return st.download_button(
        label=f"⬇ Download {filename}",
        data=data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8"),
        file_name=filename,
        mime=mime,
        use_container_width=True,
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


def render_report(report: Dict[str, Any], *, agentic_mode: bool, min_item_score: float):
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
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No per-metric mean table provided by the current metrics.")

    st.divider()

    # ===== Cases =====
    st.subheader("🧩 Individual Case Results")
    kept = 0
    for idx, r in enumerate(rb.results, 1):
        item_score = _compute_item_score(r)
        if (item_score is not None) and (item_score < min_item_score):
            continue
        kept += 1

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
                rows.append({
                    "Metric": m.get("name", ""),
                    "Score": float(f"{m.get('score', 0.0):.3f}"),
                })

            if rows:
                st.markdown("**Metrics**")
                st.dataframe(rows, use_container_width=True, hide_index=True)
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
                            m_details = m.get("details")

                            with st.expander(f"Metric: {m_name}", expanded=False):
                                # Score
                                if isinstance(m_score, (int, float)):
                                    st.write("**Score:**", float(f"{m_score:.3f}"))
                                else:
                                    st.write("**Score:**", m_score)

                                # Explanation
                                if m_expl:
                                    st.write("**Explanation:**")
                                    st.markdown(f"> {m_expl}")

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

    if kept == 0:
        st.warning("No cases pass the current minimum score filter.")

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
        build_download(js, "report.json", "application/json")
    with cols[1]:
        build_download(md, "report.md", "text/markdown")
    with cols[2]:
        if rows_metrics:
            build_download(csv_metrics_buf.getvalue(), "report_metrics.csv", "text/csv")
        else:
            st.button("report_metrics.csv(no rows)", disabled=True, use_container_width=True)
    with cols[3]:
        if rows_items:
            build_download(csv_items_buf.getvalue(), "report_items_flat.csv", "text/csv")
        else:
            st.button("report_items_flat.csv (no rows)", disabled=True, use_container_width=True)
    with cols[4]:
        build_download(html, "report.html", "text/html")


# ============================== PAGE CONFIG ==================================
st.set_page_config(
    page_title="RAGVue Dashboard",
    page_icon="logo/logo_sample.png",
    layout="wide"
)
inject_dark_theme()

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
    st.header("🎛️ Settings")

    st.subheader("🔐 API Key")

    # Textbox: widget manage its own value via key="api_key_input"
    ui_key = st.text_input(
        API_ENV_VAR,
        type="password",
        placeholder="Paste here (not stored)",
        help="Used only in this session. Alternatively, put it in a local `.env` file.",
        key="api_key_input",
    )

    cols = st.columns(2)
    cols[0].button("Use in this session", on_click=_use_api_key)
    cols[1].button("Forget key", on_click=_forget_api_key)

    # Feedback messages
    msg = st.session_state.get("api_key_message")
    if msg == "set":
        st.success("API key set for this session.")
    elif msg == "cleared":
        st.info("Key cleared from this session.")
    elif msg == "empty":
        st.warning("No key entered.")

    st.caption("Status: " + ("✅ Found" if get_api_key() else "❌ Missing"))

     # API hints
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")):
        st.warning("Set `OPENAI_API_KEY` or `GROQ_API_KEY` in your environment or a `.env` file.")

    st.markdown("---")
    st.subheader("📁 Data")
    upl = st.file_uploader("Upload `items.jsonl`", type=["jsonl"])

    # Parse immediately and only show count
    if upl is not None:
        try:
            raw = upl.getvalue()
            items_preview = read_jsonl_bytes(raw)
            st.session_state["uploaded_items"] = items_preview
            st.success(f"Detected {len(items_preview)} item(s).")
        except Exception as e:
            st.error(f"Could not parse file: {e}")

    max_items = st.number_input("Limit items (0 = all)", min_value=0, value=0, step=1)

    st.subheader("⚙️ Evaluation Mode")
    mode = st.radio("Choose how to evaluate:", ["Manual (select metrics)", "Agentic (auto-select)"], index=1, help="Manual = you pick metrics. Agentic = orchestrator chooses metrics and aggregates scores.")

    selected_metrics: List[str] = []
    if mode.startswith("Manual"):
        st.caption("Select metrics to run:")
        discovered = sorted(load_metrics().keys())
        selected_metrics = st.multiselect("Metrics", discovered, default=discovered)

    st.markdown("---")
    st.subheader("🔎 Filters")
    min_item_score = st.slider("Min item score to display", 0.0, 1.0, 0.0, 0.01)

    run_btn = st.button("▶ Run Evaluation", use_container_width=True)

    st.markdown("---")
    if Path("saved_report.json").exists():
        st.success("Saved report found on disk.")
        if st.button("📂 Load saved report"):
            try:
                with open("saved_report.json", "r", encoding="utf-8") as f:
                    st.session_state["last_report"] = json.load(f)
                st.info("Loaded saved report.")
            except Exception as e:
                st.error(f"Could not load: {e}")

    if "last_report" in st.session_state:
        if st.button("💾 Save current report"):
            with open("saved_report.json", "w", encoding="utf-8") as f:
                json.dump(st.session_state["last_report"], f, ensure_ascii=False, indent=2)
            st.success("Report saved.")


# ============================== HEADER / OVERVIEW ============================
# Title
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap" rel="stylesheet">

    <div style="text-align:center; margin-top:-40px;">
        <h1 style="
            font-family: 'Dancing Script', cursive;
            font-size:4rem;
            font-weight:800;
            margin-bottom:0.2rem;
            letter-spacing:-0.10em;
        ">
            <span style="color:#ff6b6b;">R</span>
            <span style="color:#f97316;">A</span>
            <span style="color:#facc15;">G</span>
            <span style="color:#22c55e;">V</span>
            <span style="color:#0ea5e9;">u</span>
            <span style="color:#a855f7;">e</span>
        </h1>
        <p style="
            font-size:1.5rem;
            color:#6b7280;
            margin-top:0;
            font-style: italic;
        ">
            Explainable and Reference-free RAG evaluation dashboard
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Intro cards
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(
        """
<div class="card">
  <h3>Introduction</h3>
   <p>
    <strong>RAGVue</strong> is a lightweight, production-friendly dashboard to evaluate
    Retrieval-Augmented Generation systems. <p>
    <p>
    It supports two modes:
    <span class="chip">Manual</span> where you select metrics, and
    <span class="chip">Agentic</span> where an orchestrator auto-selects relevant metrics and synthesizes overall scores.
</div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
<div class="card">
  <h3 style="margin-top:0;">Key At-a-Glance</h3>
  <div class="chip">One-click run</div>
  <div class="chip">Per-item drill-down</div>
  <div class="chip">CSV/MD/HTML export</div>
  <div class="chip">Agentic orchestration</div>
</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# Feature & Benefits section
fc1, fc2 = st.columns(2)
with fc1:
    st.subheader("✨ Key Features")
    st.markdown(
        """
- **Manual & Agentic modes**: Pick metrics yourself or let the orchestrator decide.
- **Per-item drill-down**: Questions, answers, contexts, aggregate score, and metric-wise explanations.
- **Instant exports**: Download **JSON**, **CSV**, **Markdown**, or **HTML** reports for papers & repos.
- **Session resilience**: Auto-save and re-load the last report (`saved_report.json`).
        """
    )

with fc2:
    st.subheader("🎯 How It Benefits Users")
    st.markdown(
        """
    - **Researchers**: Get *explainable* metrics, not black-box scores. Compare RAG variants fast.
    - **Engineers**: Plug-and-play. Fits straight into existing pipelines and API keys.
    - **Demo audiences**: Clear visuals, expandable reasoning, easy exports.
    - **Reviewers**: Transparent, reproducible results with concise explanations.
        """
    )

st.markdown("---")

# Tabs
tab_overview, tab_eval = st.tabs(["**Overview**", "**Evaluate**"])

# ============================== OVERVIEW TAB ================================
with tab_overview:
    st.markdown(
        """
**How to use:**
1. Upload an `items.jsonl` in the sidebar.
2. Choose **Manual** or **Agentic** mode of your choice.
3. (Optional) Adjust **Min item score** filter.
4. Click **Run Evaluation**.
5. Go to **Evaluate** → view **Summary** and **Item Results**.
6. **Export** results for your paper or repo.
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
        start_time = time.perf_counter()
        status_box.info("Starting evaluation... this may take a while depending on your dataset and API speed.")
        # 🔐 Make sure we actually have a key before doing anything
        if not get_api_key():
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

            if not items:
                st.error("No items available. Upload a `.jsonl` first from the sidebar.")
            else:
                st.info(f"Running evaluation on {len(items)} item(s).")

                if mode.startswith("Manual"):
                    if not selected_metrics:
                        st.warning("No metrics selected; nothing to run.")
                    else:
                        results = []

                        for i, item in enumerate(items, start=1):
                            t0 = time.perf_counter()
                            # run evaluation for this single item
                            single_report = pkg_evaluate([item], metrics=list(selected_metrics))
                            elapsed = time.perf_counter() - t0

                            if single_report.get("results"):
                                res = single_report["results"][0]
                                # attach per-item time
                                res["eval_time_sec"] = round(elapsed,2)
                                results.append(res)
                        summary = compute_summary_from_results(results)
                        # rebuild a combined report from all single-item results
                        rb = ReportBuilder({"results": results})
                        report = {
                            "results": results,

                            "summary": summary,  # per-metric means
                        }

                        st.session_state["last_report"] = report
                        with open("saved_report.json", "w", encoding="utf-8") as f:
                            json.dump(report, f, ensure_ascii=False, indent=2)

                        render_report(report, agentic_mode=False, min_item_score=min_item_score)
                else:
                    orch = AgenticOrchestrator()
                    results = []
                    for item in items:
                        t0 = time.perf_counter()
                        single_report = orch.run([item])
                        elapsed = time.perf_counter() - t0
                        if single_report.get("results"):
                            res = single_report["results"][0]
                            res["eval_time_sec"] = round(elapsed,2)
                            results.append(res)
                    summary = compute_summary_from_results(results)
                    rb = ReportBuilder({"results": results})

                    report = {"results": results, "summary": summary}
                    st.session_state["last_report"] = report
                    with open("saved_report.json", "w", encoding="utf-8") as f:
                        json.dump(report, f, ensure_ascii=False, indent=2)
                    render_report(report, agentic_mode=True, min_item_score=min_item_score)

                    # Final status
                    elapsed = time.perf_counter() - start_time
                    status_box.success(f"✅ Evaluation completed in {elapsed:.1f} seconds.")
        except Exception as e:
            status_box.error("❌ Evaluation failed.")
            st.exception(e)

    elif "last_report" in st.session_state:
        st.info("Showing last report from session memory.")
        render_report(st.session_state["last_report"], agentic_mode=(mode.startswith("Agentic")), min_item_score=min_item_score)

    elif Path("saved_report.json").exists():
        try:
            with open("saved_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)
            st.session_state["last_report"] = report
            st.info("Loaded previously saved report from disk.")
            render_report(report, agentic_mode=(mode.startswith("Agentic")), min_item_score=min_item_score)
        except Exception as e:
            st.error(f"Could not load saved_report.json: {e}")
    else:
        st.info("Upload a `.jsonl` in the sidebar and click **Run Evaluation** to see the Summary here.")

# ============================== FOOTER =======================================
st.markdown("---")
st.markdown(
    "<footer>© 2025 · Developed by Keerthana Murugaraj· </footer>",
    unsafe_allow_html=True,
)

