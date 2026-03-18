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
    "--focus": "#fbbf24",
    "--input-bg": "#1d2437",
    "--input-border": "#2a3550",
    "--shadow": "rgba(0,0,0,.35)",
    "--expander-bg": "#121829",
    "--divider": "#1f2937",
    "--tab-bg": "#1d2437",
}

LIGHT = {
    "--bg": "#ffffff",
    "--bg-alt": "#f8f9fa",
    "--text": "#1a1a2e",
    "--muted": "#6b7280",
    "--accent": "#6366f1",
    "--accent-contrast": "#ffffff",
    "--card": "#ffffff",
    "--card-border": "#e5e7eb",
    "--chip-bg": "#e8eaed",
    "--chip-border": "#d1d5db",
    "--kbd": "#1a1a2e",
    "--sidebar-bg": "#f0f2f5",
    "--sidebar-border": "#e5e7eb",
    "--focus": "#d97706",
    "--input-bg": "#ffffff",
    "--input-border": "#d1d5db",
    "--shadow": "rgba(0,0,0,.08)",
    "--expander-bg": "#f8f9fa",
    "--divider": "#e5e7eb",
    "--tab-bg": "#e8eaed",
}

BEIGE = {
    "--bg": "#f5f5dc",
    "--bg-alt": "#ede6d6",
    "--text": "#1a1a1a",
    "--muted": "#8b7e6a",
    "--accent": "#a0785a",
    "--accent-contrast": "#fff8f0",
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

THEMES = {"Dark": DARK, "Light": LIGHT, "Beige": BEIGE}

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

      /* ══════════════════  Base / Global  ══════════════════ */
      .stApp,
      [data-testid="stAppViewContainer"],
      [data-testid="stAppViewBlockContainer"],
      .main {{
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, sans-serif;
        background: linear-gradient(140deg, var(--bg), var(--bg-alt)) !important;
        color: var(--text) !important;
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
      }}
      [data-testid="stSidebar"],
      [data-testid="stSidebar"] *,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] span,
      [data-testid="stSidebar"] div {{
        color: var(--text) !important;
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
      .stButton>button {{
        background: var(--accent) !important;
        color: var(--accent-contrast) !important;
        border: 0;
        border-radius: 10px;
        padding: .6rem 1rem;
        font-weight: 700;
        box-shadow: 0 4px 16px var(--shadow);
      }}
      .stButton>button:hover {{ filter: brightness(1.06); }}
      .stButton>button:focus {{ outline: 3px solid var(--focus); outline-offset: 2px; }}

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
      .stDownloadButton > button {{
        background: var(--card) !important;
        color: var(--text) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px var(--shadow);
        font-weight: 600 !important;
        transition: all 0.15s;
      }}
      .stDownloadButton > button:hover {{
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        box-shadow: 0 3px 12px var(--shadow);
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
inject_theme(THEMES[st.session_state.get("theme", "Dark")])

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

    st.subheader("🎨 Theme")
    theme_choice = st.radio(
        "Choose theme:",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.get("theme", "Dark")),
        key="theme",
        horizontal=True,
    )

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

    report_name = st.text_input("Report label (optional)", placeholder="e.g. v2-pipeline", key="report_name_input")

    run_btn = st.button("▶ Run Evaluation", use_container_width=True)

    st.markdown("---")
    st.caption("📋 Reports are saved automatically. View history in the **Reports** tab.")


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
            color: var(--muted);
            margin-top:0;
            font-style: italic;
        ">
            Explainable and Reference-free RAG evaluation
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
  </p>
  <p class="muted">Designed for fast demos and reproducible experiments for the EACL Demo Track.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
<div class="card">
  <h3 style="margin-top:0;">Key At-a-Glance</h3>
  <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:0.5rem;">
    <div class="chip">One-click run</div>
    <div class="chip">Per-item drill-down</div>
    <div class="chip">CSV/MD/HTML export</div>
    <div class="chip">Agentic orchestration</div>
  </div>
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
- **Session resilience**: Auto-saves the last 10 reports — view and reload any of them in the **Reports** tab.
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
tab_overview, tab_eval, tab_reports = st.tabs(["**Overview**", "**Evaluate**", "**Reports**"])

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
        progress_bar = st.empty()
        progress_text = st.empty()
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
                status_box.info(f"Running evaluation on {len(items)} item(s)...")

                if mode.startswith("Manual"):
                    if not selected_metrics:
                        st.warning("No metrics selected; nothing to run.")
                    else:
                        results = []

                        for i, item in enumerate(items, start=1):
                            progress_bar.progress(i / len(items))
                            progress_text.text(f"Evaluating item {i} of {len(items)}...")
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
                        _label = report_name.strip() if report_name.strip() else (upl.name if upl else "unknown")
                        _save_to_history(report, f"Manual · {_label} · {len(items)} items")
                        progress_bar.empty()
                        progress_text.empty()
                        elapsed = time.perf_counter() - start_time
                        status_box.success(f"✅ Evaluation completed in {elapsed:.1f} seconds.")
                        render_report(report, agentic_mode=False, min_item_score=min_item_score, key_prefix="eval")
                else:
                    orch = AgenticOrchestrator()
                    results = []
                    for i, item in enumerate(items, start=1):
                        progress_bar.progress(i / len(items))
                        progress_text.text(f"Evaluating item {i} of {len(items)}...")
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
                    _label = report_name.strip() if report_name.strip() else (upl.name if upl else "unknown")
                    _save_to_history(report, f"Agentic · {_label} · {len(items)} items")
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
    st.subheader("📋 Report History")
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

                        st.markdown("#### Metric Comparison")
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


