"""ACCT 3343 - Overvalued Stock Screener (Streamlit).

Run locally:  streamlit run app.py
All data is pre-built by the scripts in /pipeline; this app never calls a data API.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import importlib

import analytics

# Streamlit Cloud keeps imported modules in memory across code updates; reload so app.py and
# analytics.py are always the same version.
importlib.reload(analytics)
from analytics import (ALTMAN_COEFS, BENEISH, KNOWN_SEPARATIONS, LABEL_COLORS, LABEL_ORDER, M_SCORE_CUTOFF,
                       QUALITY_METRICS, SELL_FLAGS, SELL_PROFITABLE, SELL_SPECULATIVE, STANDARD_ALTMAN,
                       VALUE_METRICS, Settings, compute_metrics, explain, flag_text, pct_rank, run_screen)

st.set_page_config(page_title="Overvalued Stock Screener", page_icon="📉", layout="wide")
DATA = Path(__file__).parent / "data"
# Chart colours follow the viewer's theme (dark by default; switchable in the app menu).
IS_DARK = getattr(getattr(st.context, "theme", None), "type", "dark") != "light"
SERIES_1, SERIES_2, SERIES_3 = ("#3987e5", "#d95926", "#199e70") if IS_DARK else ("#2a78d6", "#eb6834", "#1baf7a")
MARKER_RING = "rgba(26,26,25,0.8)" if IS_DARK else "rgba(255,255,255,0.6)"
SNAP_LABEL = {"bt2024": "Formed 2024-10-15", "bt2025": "Formed 2025-10-15"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def data_version() -> tuple:
    """Changes whenever a data file or the analytics code changes, so cached results refresh."""
    files = sorted(DATA.glob("*")) + [Path(__file__).parent / "analytics.py"]
    return tuple((f.name, f.stat().st_mtime_ns) for f in files)


# Nasdaq's sector labels are wrong for a few well-known firms; these are corrected by hand.
SECTOR_FIX = {"MO": "Consumer Staples", "PM": "Consumer Staples", "PG": "Consumer Staples",
              "CL": "Consumer Staples", "KVUE": "Consumer Staples", "CHD": "Consumer Staples",
              "EL": "Consumer Staples", "COTY": "Consumer Staples", "ELF": "Consumer Staples",
              "EPC": "Consumer Staples", "IPAR": "Consumer Staples", "TMO": "Health Care",
              "FDXF": "Industrials", "FISV": "Technology"}


def clean_names(names: pd.Series) -> pd.Series:
    """Strip share-class wording from exchange listing names ("... Class A Common Stock")."""
    n = (names.str.replace(r"\s*\(Name to be changed.*\)\s*$", "", regex=True)
         .str.replace(r"\s+American Deposit[ao]ry Shares.*$", "", regex=True, case=False)
         .str.replace(r"\s+Class [A-Z]\b.*(Shares|Units|Interests|Stock).*$", "", regex=True, case=False)
         .str.replace(r"\s*(Common|Ordinary)\s+(Stock|Shares?)\b.*$", "", regex=True, case=False)
         .str.replace(r"\s+Subordinate Voting Shares.*$", "", regex=True, case=False)
         .str.replace(r"\s+(Class [A-Z]\s+)?(Non\s+)?Voting$", "", regex=True, case=False)
         .str.replace(r"^DBA\s+", "", regex=True)
         .str.replace(r"\s+REIT$", "", regex=True)
         .str.replace(r"\s{2,}", " ", regex=True).str.strip()
         .str.replace(r"^(.+?)\s+\(The\)$", r"The \1", regex=True))

    def dedupe(s: str) -> str:  # "STARWOOD PROPERTY TRUST INC. Starwood Property Trust Inc."
        w = s.split()
        h = len(w) // 2
        if len(w) % 2 == 0 and h and " ".join(w[:h]).lower() == " ".join(w[h:]).lower():
            return " ".join(w[h:])
        return s

    return n.map(dedupe)


@st.cache_data(show_spinner=False)
def load_data(version: tuple):
    firms = pd.read_parquet(DATA / "firms.parquet")
    firms["name"] = clean_names(firms["name"])
    firms.loc[firms["ticker"] == "TNL", "name"] = "Travel + Leisure Co."
    firms["sector"] = firms["ticker"].map(SECTOR_FIX).fillna(firms["sector"])
    panels = {k: pd.read_parquet(DATA / f"panel_{k}.parquet") for k in ("current", "bt2025", "bt2024")}
    rets = pd.read_parquet(DATA / "returns.parquet")
    bench = pd.read_parquet(DATA / "benchmark.parquet")
    latest = pd.read_parquet(DATA / "latest_filing.parquet")
    meta = json.loads((DATA / "meta.json").read_text())
    return firms, panels, rets, bench, latest, meta


@st.cache_data(show_spinner="Computing financial ratios for every firm...")
def metrics_for(snapshot: str, lookback: int, version: tuple) -> pd.DataFrame:
    firms, panels, rets, _, latest, _ = load_data(version)
    if snapshot == "current":
        mcap = firms.set_index("cik")["mcap"]
    else:
        mcap = firms.merge(rets, on="ticker").set_index("cik")[f"mcap_{snapshot}"]
    lf = latest[latest["snapshot"] == snapshot].set_index("cik")
    m = compute_metrics(panels[snapshot], firms, mcap, lookback, latest=lf,
                        known_separations=KNOWN_SEPARATIONS if snapshot == "current" else None)
    if snapshot != "current":
        r = rets[["ticker", f"ret_{snapshot}"]].rename(columns={f"ret_{snapshot}": "fwd_ret"})
        m = m.merge(r, on="ticker", how="left")
    return m


VERSION = data_version()
firms, panels, rets, bench, latest_filing, meta = load_data(VERSION)
ALL_SECTORS = sorted(firms["sector"].dropna().unique())
DEFAULT_SECTORS = [s for s in ALL_SECTORS if s not in ("Finance", "Real Estate")]


# ---------------------------------------------------------------------------
# Sidebar: every screening control
# ---------------------------------------------------------------------------
def pct_slider(label: str, lo: int, hi: int, default: int, key: str, help: str | None = None) -> float:
    return st.slider(label, lo, hi, default, step=1, format="%d%%", key=key, help=help) / 100


with st.sidebar:
    st.header("Screen controls")
    st.caption("Each section is one step of the course process. Open a section to change its cutoffs; "
               "every chart, table and recommendation updates as you move them.")
    if st.button("Reset all filters", width="stretch"):
        st.session_state.clear()
        st.rerun()

    with st.expander("Universe", expanded=True):
        sectors = st.multiselect("Sectors", ALL_SECTORS, DEFAULT_SECTORS, key="sectors",
                                 help="Finance and Real Estate are excluded by default: banks, insurers "
                                      "and REITs have no COGS or current assets, so most course ratios "
                                      "do not apply to them.")
        min_mcap = st.select_slider("Minimum market cap", options=[0, 1, 2, 5, 10, 25, 50, 100], value=0,
                                    format_func=lambda v: "No minimum" if v == 0 else f"${v}B", key="min_mcap")
        relative = st.radio("Ranking basis", ["Raw values", "Within sector"], horizontal=True,
                            key="basis", help="Within sector ranks each measure against the firm's own sector "
                                              "(percentile within sector), so firms are compared with peers. "
                                              "Ranks are used instead of subtracting sector means because a "
                                              "single outlier can move a sector mean a long way.") != "Raw values"

    with st.expander("Step 1 · Manipulation (Ch 3)", expanded=False):
        accrual_cut = pct_slider("Exclude worst STA + SNOA", 0, 50, 10, "accrual_cut",
                                 "Ranks scaled total accruals and scaled net operating assets (highest = "
                                 "worst), adds the ranks and removes the worst share.")
        use_probm = st.toggle("Also screen on Beneish PROBM", True, key="use_probm")
        probm_cut = pct_slider("Exclude worst PROBM", 0, 30, 5, "probm_cut") if use_probm else 0.0

    with st.expander("Step 2 · Distress (Ch 4)", expanded=False):
        altman = st.radio("Altman Z coefficients", list(ALTMAN_COEFS), key="altman",
                          help="Standard Altman (1968) weights work on ratio inputs and are what the "
                               "1.81 / 2.99 zones assume. The slide weights reproduce the class examples "
                               "(e.g. Empire Z = 1.89).")
        if altman != STANDARD_ALTMAN:
            st.caption("Slide weights rank the firms in steps 2a/2b. The distress zones (cutoff, red flag, "
                       "safe/grey/distress label) still use the standard Z, because 1.81 is calibrated for it.")
        z_raw_cut = pct_slider("Exclude worst raw Altman Z", 0, 50, 10, "z_raw_cut")
        z_rel_cut = pct_slider("Then exclude worst Z within sector", 0, 50, 10, "z_rel_cut")
        z_threshold = st.number_input("Distress cutoff (Z below this = distress zone)", value=1.81,
                                      step=0.1, format="%.2f", key="z_thr")
        z_hard = st.toggle("Also remove every firm below the cutoff", False, key="z_hard")

    with st.expander("Step 3 · Price (Ch 2)", expanded=False):
        value_labels = st.multiselect("Value measures (ranked, added, re-ranked)", list(VALUE_METRICS),
                                      ["Earnings yield (EBIT/TEV)", "Book-to-market (BM)"], key="value_m")
        excess = st.radio("Excess cash in TEV", ["Course slides: Cash + CA - CL",
                                                 "Greenblatt: Cash - max(0, CL - (CA - Cash))"], key="excess",
                          help="The slide formula counts cash twice (current assets already include cash), "
                               "which lowers TEV for firms with large working capital. Greenblatt's version "
                               "only subtracts cash not needed to cover current liabilities.")
        value_keep = pct_slider("Keep the cheapest", 5, 100, 30, "value_keep")

    with st.expander("Step 4 · Quality (Ch 2, 5, 6)", expanded=False):
        quality_labels = st.multiselect("Quality measures", list(QUALITY_METRICS),
                                        ["Franchise power (Ch 5)", "F_SCORE (Ch 6)", "Gross profitability GPA (Ch 2)"],
                                        key="quality_m")
        lookback = st.slider("Franchise look-back (fiscal years)", 3, 10, 8, key="lookback",
                             help="The book uses 8 years. Firms with shorter histories have no franchise score; "
                                  "a shorter window keeps more firms but measures persistence less well.")
        quality_keep = pct_slider("Keep the highest-quality", 5, 100, 50, "quality_keep")
        min_fscore = st.slider("Minimum F_SCORE", 0, 9, 5, key="min_f")

    with st.expander("SELL rules · overvalued", expanded=False):
        expensive_pct = pct_slider("'Expensive' = most expensive", 5, 50, 20, "exp_pct",
                                   "Profitable firms (EBIT > 0): the most expensive share of profitable firms. "
                                   "Loss-makers: the most expensive share of the whole universe.")
        flag_labels = st.multiselect("Red flags that count", list(SELL_FLAGS.values()), list(SELL_FLAGS.values()),
                                     key="flags")
        min_flags = st.slider("SELL needs at least this many red flags", 1, 6, 2, key="min_flags")
        if len(flag_labels) < min_flags:
            st.warning(f"Only {len(flag_labels)} red flag(s) are enabled, so no firm can reach {min_flags}: "
                       "the SELL list will be empty.")
        sell_scope_label = st.radio(
            "SELL groups to include", ["Both groups", "Overvalued profitable firms only",
                                       "Speculative unprofitable firms only"], key="sell_scope",
            help="Profitable firms (EBIT > 0) count as expensive when they are in the most expensive share of "
                 "*profitable* firms. Firms with an operating loss last year and no multi-year record of profits "
                 "have negative earnings yields, so they are compared with the whole universe and form a "
                 "separate, speculative group.")
        flag_accrual_pct = pct_slider("Accrual flag: worst STA + SNOA", 5, 50, 20, "flag_acc")
        flag_fscore_max = st.slider("F_SCORE flag: score at or below", 0, 6, 3, key="flag_f")
        flag_franchise_pct = pct_slider("Franchise flag: bottom", 5, 50, 25, "flag_fr")

settings = Settings(
    sectors=sectors, min_mcap=min_mcap * 1e9, relative=relative,
    accrual_cut=accrual_cut, use_probm=use_probm, probm_cut=probm_cut,
    altman=altman, z_raw_cut=z_raw_cut, z_rel_cut=z_rel_cut, z_hard=z_hard, z_threshold=z_threshold,
    excess_cash="slides" if excess.startswith("Course") else "gb",
    value_metrics=[VALUE_METRICS[v] for v in value_labels], value_keep=value_keep,
    quality_metrics=[QUALITY_METRICS[q] for q in quality_labels], quality_keep=quality_keep,
    min_fscore=min_fscore, expensive_pct=expensive_pct,
    sell_flags=[k for k, v in SELL_FLAGS.items() if v in flag_labels], flag_accrual_pct=flag_accrual_pct,
    flag_fscore_max=flag_fscore_max, flag_franchise_pct=flag_franchise_pct, min_flags=min_flags,
    sell_scope={"Both groups": "both", "Overvalued profitable firms only": "profitable",
                "Speculative unprofitable firms only": "speculative"}[sell_scope_label],
)

if not sectors:
    st.warning("Pick at least one sector in the sidebar.")
    st.stop()
if not value_labels or not quality_labels:
    st.warning("Pick at least one value measure and one quality measure in the sidebar.")
    st.stop()

metrics = metrics_for("current", lookback, VERSION)
res, steps = run_screen(metrics, settings)
counts = res["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
sells, buys = res[res["label"] == "SELL"], res[res["label"] == "BUY"]
sells_prof = sells[sells["sell_type"] == SELL_PROFITABLE]
sells_spec = sells[sells["sell_type"] == SELL_SPECULATIVE]

with st.sidebar:
    st.divider()
    st.subheader("Live funnel")
    lines = [f"Universe: **{len(res):,}**"]
    lines += [f"↓ {s['step']}: −{s['missing'] + s['excluded']:,} → {s['remaining']:,}" for s in steps]
    st.markdown("  \n".join(lines))
    st.markdown(f"**BUY {counts['BUY']} · SELL {counts['SELL']} ({len(sells_prof)} profitable + "
                f"{len(sells_spec)} speculative) · AVOID {counts['AVOID']}**")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_money(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    a = abs(x)
    if a >= 1e12:
        return f"${x / 1e12:,.2f}T"
    if a >= 1e9:
        return f"${x / 1e9:,.2f}B"
    return f"${x / 1e6:,.1f}M"


def fmt(x: float, pct: bool = False, d: int = 2) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:,.{d}f}"


TABLE_COLS = {
    "ticker": st.column_config.TextColumn("Ticker", pinned=True),
    "name": st.column_config.TextColumn("Company", width="medium"),
    "sector": "Sector",
    "mcap": st.column_config.NumberColumn("Mkt cap", format="compact"),
    "label": "Signal",
    "value_score": st.column_config.ProgressColumn("Cheapness", min_value=0, max_value=1, format="%.2f",
                                                   help="Value composite percentile (1 = cheapest)."),
    "quality_score": st.column_config.ProgressColumn("Quality", min_value=0, max_value=1, format="%.2f",
                                                     help="Quality composite percentile (1 = best)."),
    "overvaluation_score": st.column_config.NumberColumn(
        "Overvaluation score", format="%.0f",
        help="0-100: average of (1 - cheapness percentile) and (1 - quality percentile). Cheapness is ranked "
             "among profitable firms for EBIT > 0 and against the universe for loss-makers, so compare scores "
             "within a SELL group."),
    "n_flags": st.column_config.NumberColumn("Red flags", format="%d"),
    "ey": st.column_config.NumberColumn("EBIT/TEV", format="percent",
                                        help="Blank when TEV is zero or negative (ranked as the cheapest case)."),
    "bm": st.column_config.NumberColumn("B/M", format="%.2f"),
    "ep": st.column_config.NumberColumn("E/P", format="percent"),
    "fcfy": st.column_config.NumberColumn("FCF yield", format="percent"),
    "gpa": st.column_config.NumberColumn("GPA", format="%.2f"),
    "roc_gb": st.column_config.NumberColumn("ROC", format="percent"),
    "sta": st.column_config.NumberColumn("STA", format="percent"),
    "snoa": st.column_config.NumberColumn("SNOA", format="percent"),
    "mscore": st.column_config.NumberColumn("M-score", format="%.2f"),
    "altman_z": st.column_config.NumberColumn("Altman Z", format="%.2f"),
    "fscore": st.column_config.NumberColumn("F_SCORE", format="%d"),
    "franchise": st.column_config.NumberColumn("Franchise pct", format="%.2f"),
    "cfoa": st.column_config.NumberColumn("CFOA", format="%.2f"),
    "mg": st.column_config.NumberColumn("Margin growth", format="%.3f"),
    "ms": st.column_config.NumberColumn("Margin stability", format="%.1f"),
    "stage": "Screen outcome",
    "why": st.column_config.TextColumn("Why", width="large"),
    "fwd_ret": st.column_config.NumberColumn("12-mo return", format="percent"),
}
for k, v in SELL_FLAGS.items():
    TABLE_COLS[f"flag_{k}"] = st.column_config.CheckboxColumn(k.replace("_", " ").title(), help=v)


def show_table(df: pd.DataFrame, cols: list[str], height: int = 520) -> None:
    df = df.assign(ey=df["ey"].mask(df["ey"] >= 10)) if "ey" in df else df  # hide the TEV<=0 ranking sentinel
    st.dataframe(df[cols], column_config={c: TABLE_COLS.get(c, c) for c in cols}, hide_index=True,
                 width="stretch", height=height)


def csv_button(df: pd.DataFrame, name: str, key: str) -> None:
    st.download_button(f"Download {name} (CSV)", df.to_csv(index=False).encode(), f"{name}.csv",
                       "text/csv", key=key)


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=90, b=10),
                      title=dict(y=0.98, yanchor="top", x=0, xanchor="left"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title_text=""))
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("**Callum Whitehouse** &nbsp;·&nbsp; **ACCT 3343**")
st.title("Overvalued Stock Screener")
st.markdown(
    f"Applies the ACCT 3343 financial-statement techniques (magic formula, accruals, Altman Z, franchise "
    f"power, F_SCORE) to **{meta['n_firms']:,} U.S. public companies**. Fundamentals come from SEC EDGAR "
    f"10-K filings (fiscal years ending {meta['fy_end_min']} to {meta['fy_end_max']}); market caps are "
    f"from {meta['generated']}. Move any control in the sidebar to re-run the analysis."
)
with st.expander("👋 How to use this app (30 seconds)", expanded=True):
    st.markdown("""
1. **Start with ⭐ Recommendations**: the top long (BUY) and short (SELL) ideas under the current settings,
   each with a written, number-backed thesis.
2. **Change the screen in the sidebar.** Each section is one step of the course process (Ch 2–6); open it to
   adjust the cutoffs and everything recalculates instantly. *Reset all filters* restores the defaults.
3. **Check the evidence**: *Step-by-step screen* shows what each test removes, *Company analysis* shows every
   formula with a firm's real numbers, and *Backtest* applies the same rules to 2024 and 2025 data.
4. **Signals**: 🟢 BUY = cheap, high-quality survivors · 🔴 SELL = expensive with red flags (profitable or
   speculative) · 🟠 AVOID = cheap but failed the accounting or distress tests (value trap) · ⚪ HOLD = no signal.
""")
with st.expander("📖 Glossary of measures"):
    st.markdown("""
| Measure | Chapter | What it tells you | Better when |
|---|---|---|---|
| **EBIT/TEV** (earnings yield) | 2 | Operating profit per dollar of enterprise value | Higher = cheaper |
| **B/M** (book-to-market) | 2 | Book equity per dollar of market value | Higher = cheaper |
| **GPA** (gross profitability) | 2 | Gross profit / total assets: economic profitability | Higher |
| **ROC** (Greenblatt) | 2 | EBIT / (net PPE + working capital) | Higher |
| **STA** (scaled total accruals) | 3 | (Net income − cash from operations) / assets: how much profit is non-cash | Lower |
| **SNOA** (scaled net operating assets) | 3 | Accumulated operating assets less operating liabilities: a bloated balance sheet | Lower |
| **M-score / PROBM** (Beneish) | 3 | Probability of earnings manipulation; M > −1.78 is a warning | Lower |
| **Altman Z** | 4 | Bankruptcy risk: < 1.81 distress, 1.81–2.99 grey, > 2.99 safe | Higher |
| **CFOA** | 5 | Sum of the look-back years' free cash flow (8 by default) / assets | Higher |
| **MG / MS** (margin growth / stability) | 5 | Geometric growth and stability (mean ÷ std) of gross margin | Higher |
| **Franchise power** | 5 | Rank of ROA, ROC, CFOA and the better of MG/MS | Higher |
| **F_SCORE** (Piotroski) | 6 | 9 pass/fail tests of profitability, leverage/liquidity and efficiency | Higher (0–9) |
""")

k = st.columns(5)
k[0].metric("Firms screened", f"{len(res):,}")
k[1].metric("🟢 BUY candidates", int(counts["BUY"]))
k[2].metric("🔴 SELL candidates", int(counts["SELL"]),
            help=f"{len(sells_prof)} overvalued profitable firms + {len(sells_spec)} speculative loss-makers")
k[3].metric("🟠 AVOID (value traps)", int(counts["AVOID"]))
k[4].metric("⚪ HOLD / no signal", int(counts["HOLD"]))

(t_rec, t_dash, t_sell, t_buy, t_steps, t_company, t_backtest, t_method) = st.tabs(
    ["⭐ Recommendations", "Dashboard", "🔴 SELL list", "🟢 BUY list", "Step-by-step screen",
     "Company analysis", "Backtest", "Methodology"])

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
F_NAMES = {"F_ROA": "positive ROA", "F_CFO": "positive cash flow", "F_dROA": "improving ROA",
           "F_ACCRUAL": "cash flow above net income", "F_dLEVER": "falling leverage",
           "F_dLIQUID": "improving liquidity", "F_EQ_OFFER": "no net share issuance",
           "F_dMARGIN": "improving gross margin", "F_dTURN": "improving asset turnover"}
VAL_NAME = {"ey": "EBIT/TEV", "bm": "book-to-market", "ep": "E/P", "fcfy": "FCF yield"}
LONG_RANKS = ["Cheapness + quality (average percentile)", "Magic formula (rank EBIT/TEV + rank ROC)",
              "Quality only", "Cheapness only"]
SHORT_RANKS = ["Red flags, then overvaluation score", "Overvaluation score", "Beneish M-score (highest first)",
               "Accrual quality (worst first)"]


def top_pct(p: float) -> str:
    """'top/bottom X%' phrasing that never prints 0%."""
    return f"{max(1, math.ceil(100 * p))}%"


def share_pct(p: float) -> str:
    """'more than X% of firms' phrasing that never prints 0% or 100%."""
    return f"{min(99, max(1, math.floor(100 * p)))}%"


def share_below(ref: pd.DataFrame, col: str, x: float) -> float:
    s = ref[col].dropna()
    return float((s < x).mean()) if pd.notna(x) and len(s) else np.nan


def z_zone(z: float) -> str:
    return "n/a" if pd.isna(z) else "distress" if z < settings.z_threshold else "grey" if z < 2.99 else "safe"


def long_score(df: pd.DataFrame, how: str) -> pd.Series:
    if how.startswith("Magic"):
        return pct_rank(df["ey"]) + pct_rank(df["roc_gb"])  # Greenblatt: add the two ranks
    if how == "Quality only":
        return df["quality_score"]
    if how == "Cheapness only":
        return df["value_score"]
    return (df["value_score"] + df["quality_score"]) / 2


def rank_longs(df: pd.DataFrame, how: str, n: int) -> pd.DataFrame:
    b = df[df["label"] == "BUY"].assign(_s=long_score(df, how))
    return b.sort_values(["_s", "ticker"], ascending=[False, True], na_position="last").head(n)


def rank_shorts(df: pd.DataFrame, how: str, n: int, group: str) -> pd.DataFrame:
    s = df[df["sell_type"] == group]
    if how == "Overvaluation score":
        s = s.sort_values(["overvaluation_score", "ticker"], ascending=[False, True], na_position="last")
    elif how.startswith("Beneish"):
        s = s.sort_values(["mscore", "ticker"], ascending=[False, True], na_position="last")
    elif how.startswith("Accrual"):
        s = s.sort_values(["accrual_pct", "ticker"], ascending=[True, True], na_position="last")
    else:
        s = s.sort_values(["n_flags", "overvaluation_score", "ticker"], ascending=[False, False, True])
    return s.head(n)


def ref_group(r: pd.Series) -> tuple[pd.DataFrame, str]:
    """Comparison group for a thesis: the firm's sector under 'Within sector' ranking, else the universe."""
    if settings.relative:
        return res[res["sector"] == r["sector"]], f"{r['sector']} median"
    return res, "universe median"


def valuation_parts(r: pd.Series, ref: pd.DataFrame, lab: str) -> list[str]:
    """The measures that actually drive the value composite, each against the comparison median."""
    fmt_v = lambda c, x: f"{x:.2f}" if c == "bm" else f"{x:.1%}"
    parts = []
    for c in settings.value_metrics:
        x = r.get(c)
        if pd.isna(x):
            continue
        if c == "ey" and x >= 10:
            parts.append("TEV is zero or negative (cash and working capital exceed market value)")
            continue
        parts.append(f"{VAL_NAME[c]} {fmt_v(c, x)} vs a {fmt_v(c, ref[c].median())} {lab}")
    return parts


def thesis(r: pd.Series, side: str) -> list[str]:
    """Number-backed bullet points for a long or short idea. Every claim is checked against its numbers."""
    ref, lab = ref_group(r)
    med = ref[["ey", "gpa", "snoa"]].median()
    f, z, fr = r["fscore"], r["altman_z_std"], r["franchise"]
    failed = [F_NAMES[c] for c in F_NAMES if r.get(c) == 0]
    na = [F_NAMES[c] for c in F_NAMES if pd.isna(r.get(c))]
    f_note = (f"; rescaled from {int(r['fscore_raw'])}/{int(r['fscore_avail'])}, not computable: {', '.join(na)}"
              if pd.notna(f) and na else "")
    vparts = valuation_parts(r, ref, lab)
    ey = r["ey"]
    pts: list[str] = []
    risks: list[str] = []
    young = pd.notna(r["years_available"]) and r["years_available"] < 3
    if side == "long":
        head = "Cheap" if r["value_score"] >= 0.5 else "Valuation"
        txt = f"**{head}:** cheaper than {share_pct(r['value_score'])} of screened firms on the value composite"
        if vparts:
            txt += " (" + "; ".join(vparts) + ")"
        if pd.notna(ey) and 0 < ey < 10:
            txt += f", i.e. about {1 / ey:.0f}× operating profit"
        pts.append(txt + ".")
        if pd.notna(r["bm"]) and r["bm"] < 0:
            risks.append("negative common equity")
        q = []
        if pd.notna(f):
            q.append(f"F_SCORE {f:.0f}/9" + (f" (misses: {', '.join(failed)}{f_note})" if failed
                                              else f" (passes every {'computable ' if na else ''}test{f_note})"))
        if pd.notna(fr):
            if fr >= 0.5:
                q.append(f"franchise power in the top {top_pct(1 - fr)}"
                         + (f" with a {lookback}-year geometric ROA of {r['roa_geo']:.1%}" if pd.notna(r["roa_geo"]) else ""))
            else:
                risks.append(f"franchise power only in the bottom {top_pct(fr)}")
        if pd.notna(r["gpa"]):
            if r["gpa"] > med["gpa"]:
                q.append(f"gross profitability {r['gpa']:.2f} vs a {med['gpa']:.2f} {lab}")
            else:
                risks.append(f"gross profitability {r['gpa']:.2f} is below the {med['gpa']:.2f} {lab}")
        if q:
            qhead = "High quality" if pd.notna(r["quality_score"]) and r["quality_score"] >= 0.5 else "Quality"
            pts.append(f"**{qhead}:** " + "; ".join(q) + ".")
        e = []
        if pd.notna(r["sta"]):
            e.append(f"STA {r['sta']:+.1%} (cash flow {'exceeds' if r['sta'] < 0 else 'trails'} net income)")
        if pd.notna(r["snoa"]):
            e.append(f"SNOA {r['snoa']:.0%} vs a {med['snoa']:.0%} {lab}")
        if pd.notna(r["mscore"]):
            e.append(f"Beneish M-score {r['mscore']:.2f}, below the {M_SCORE_CUTOFF} warning level"
                     if r["mscore"] <= M_SCORE_CUTOFF else f"Beneish M-score {r['mscore']:.2f}")
        if e:
            pts.append("**Earnings quality:** " + "; ".join(e) + ".")
        zone = z_zone(z)
        if zone == "safe":
            pts.append(f"**Financial strength:** Altman Z {z:.2f} (safe zone).")
        elif zone in ("grey", "distress"):
            pts.append(f"**Balance sheet:** Altman Z {z:.2f} ({zone} zone).")
            risks.append(f"Altman Z {z:.2f} is in the {zone} zone")
        if pd.notna(f) and f <= 6:
            risks.append(f"F_SCORE is only {f:.0f}")
        if pd.notna(r["mg"]) and r["mg"] < 1:
            risks.append(f"gross margins have been slipping (MG {r['mg']:.3f})")
        if pd.notna(r["sta"]) and r["sta"] > 0:
            risks.append("net income runs ahead of cash flow")
        if young:
            risks.append(f"only {int(r['years_available'])} years of statements")
        risks.append("cheap stocks can stay cheap, so check the latest quarter for one-off items")
        pts.append("*Key risks:* " + "; ".join(risks) + ".")
        return pts

    # ---- short ----
    if r.get("sell_type") == SELL_PROFITABLE:
        vsp = r["value_score_profitable"]
        txt = (f"**Expensive:** in the most expensive {top_pct(vsp)} of profitable firms on the value composite"
               if pd.notna(vsp) else "**Expensive:** among the most expensive profitable firms")
        if vparts:
            txt += " (" + "; ".join(vparts) + ")"
        if pd.notna(ey) and pd.notna(med["ey"]) and 0 < ey < med["ey"]:
            txt += f"; about {1 / ey:.0f}× operating profit vs {1 / med['ey']:.0f}× for the median firm"
        pts.append(txt + ".")
    else:
        pts.append(f"**Priced on hope:** an operating loss of {fmt_money(-r['ebit'])} last year and no multi-year "
                   f"record of profits, yet the market values the company at {fmt_money(r['mcap'])}.")
    flags = []
    if r.get("flag_accruals"):
        parts = []
        if pd.notna(r["sta"]) and r["sta"] > 0:
            parts.append(f"net income above cash flow (STA {r['sta']:+.1%})")
        if pd.notna(r["snoa"]) and share_below(ref, "snoa", r["snoa"]) >= 0.5:
            parts.append(f"bloated net operating assets (SNOA {r['snoa']:.0%}, above "
                         f"{share_pct(share_below(ref, 'snoa', r['snoa']))} of firms)")
        flags.append(f"poor accrual quality, in the worst {settings.flag_accrual_pct:.0%} on STA + SNOA"
                     + (": " + " and ".join(parts) if parts else
                        f" (STA {r['sta']:+.1%}, SNOA {r['snoa']:.0%})"))
    if r.get("flag_manipulation"):
        flags.append(f"Beneish M-score {r['mscore']:.2f}, above the {M_SCORE_CUTOFF} manipulation threshold")
    if r.get("flag_distress"):
        flags.append(f"Altman Z of {z:.2f}, in the distress zone")
    if r.get("flag_weak_fscore"):
        flags.append(f"F_SCORE of only {f:.0f}/9")
    if r.get("flag_weak_franchise"):
        flags.append(f"franchise power in the bottom {top_pct(fr)}")
    if r.get("flag_cash_burn"):
        flags.append(f"burned {fmt_money(-r['fcf'])} of free cash flow last year")
    pts.append(f"**Red flags ({int(r['n_flags'])}):** " + "; ".join(flags) + ".")
    if pd.notna(f) and failed:
        pts.append(f"**F_SCORE {f:.0f}/9** fails on: {', '.join(failed)}{f_note}.")
    if (pd.notna(fr) and fr >= 0.5) or (pd.notna(r["roa_geo"]) and r["roa_geo"] > 0.10):
        risks.append("a high-return franchise can justify a premium multiple")
    if z_zone(z) == "safe":
        risks.append("no balance-sheet stress (Altman Z in the safe zone)")
    if young:
        risks.append(f"only {int(r['years_available'])} fiscal years of statements (recent IPO or spin-off), so "
                     "accrual signals may reflect carve-out accounting")
    if r["mcap"] > 1e11:
        risks.append("a mega-cap with heavy index ownership")
    if pd.notna(r.get("b_sgi")) and r["b_sgi"] > 1.25:
        risks.append(f"sales grew {r['b_sgi'] - 1:.0%} last year and may grow into the valuation")
    if pd.notna(f) and f >= 7:
        risks.append(f"operating trends are still healthy (F_SCORE {f:.0f})")
    risks.append("expensive stocks can keep rising and a short's loss is unlimited (see the Backtest tab), so "
                 "position size matters")
    pts.append("*Key risks:* " + "; ".join(risks) + ".")
    return pts


def idea_card(r: pd.Series, side: str, rank: int, compact: bool = False) -> None:
    with st.container(border=True):
        badge = "🟢 LONG" if side == "long" else "🔴 SHORT"
        st.markdown(f"**{rank}. {r['ticker']} · {r['name']}**  \n{badge} · {r['sector']} · market cap "
                    f"{fmt_money(r['mcap'])}")
        if not compact:
            c = st.columns(4)
            c[0].metric("EBIT/TEV", "TEV ≤ 0" if pd.notna(r["ey"]) and r["ey"] >= 10 else fmt(r["ey"], True))
            c[1].metric("F_SCORE", fmt(r["fscore"], d=0))
            c[2].metric("Altman Z", fmt(r["altman_z_std"]))
            c[3].metric("Red flags" if side == "short" else "Quality pct",
                        int(r["n_flags"]) if side == "short" else fmt(r["quality_score"]))
        st.markdown("\n".join(f"- {x}" for x in thesis(r, side)))


_BT_CACHE: dict = {}


def backtest_frames(floor_b: float) -> dict[str, pd.DataFrame]:
    """Screen results on the two point-in-time snapshots under the current settings (memoised per run)."""
    if floor_b not in _BT_CACHE:
        out = {}
        for snap in ("bt2024", "bt2025"):
            bm_ = metrics_for(snap, lookback, VERSION)
            out[snap], _ = run_screen(bm_, dataclasses.replace(settings, min_mcap=max(floor_b * 1e9, settings.min_mcap)))
        _BT_CACHE[floor_b] = out
    return _BT_CACHE[floor_b]


def rule_results(frames: dict[str, pd.DataFrame], rank: dict) -> list[dict]:
    """Backtest of the exact recommendation rule: picks are made before looking at returns."""
    rows = []
    for snap, full in frames.items():
        L = rank_longs(full, rank["long"], rank["n"])
        S = rank_shorts(full, rank["short"], rank["n"], SELL_PROFITABLE)
        lr, sr = full.loc[L.index, "fwd_ret"], full.loc[S.index, "fwd_ret"]
        rows.append({
            "Formation": SNAP_LABEL[snap], "Longs": len(L), "Shorts": len(S),
            "Avg long return": lr.mean(), "Avg return of shorted stocks": sr.mean(),
            "Long minus short": lr.mean() - sr.mean() if len(L) and len(S) else np.nan,
            "Long picks": ", ".join(f"{tk} {v:+.0%}" if pd.notna(v) else f"{tk} n/a" for tk, v in zip(L["ticker"], lr)),
            "Short picks": ", ".join(f"{tk} {v:+.0%}" if pd.notna(v) else f"{tk} n/a" for tk, v in zip(S["ticker"], sr)),
        })
    return rows


def recommendations_tab() -> dict:
    st.subheader("Recommendations")
    rc = st.columns([3, 3, 2])
    how_long = rc[0].selectbox("Rank long ideas by", LONG_RANKS, key="rank_long")
    how_short = rc[1].selectbox("Rank short ideas by", SHORT_RANKS, key="rank_short")
    n = int(rc[2].number_input("Ideas per side", min_value=3, max_value=10, value=5, step=1, key="n_ideas"))
    rank = {"long": how_long, "short": how_short, "n": n}
    st.markdown(
        f"Under the current sidebar settings the screen recommends going **long** the top {n} BUY candidates "
        f"(ranked by *{how_long.lower()}*) and **short** the top {n} overvalued profitable SELL candidates "
        f"(ranked by *{how_short.lower()}*). Every number below is from the firm's latest 10-K and today's market "
        f"value, compared with the {len(res):,} screened firms"
        + (" in the same sector" if settings.relative else "") + ". Change any setting and this page re-ranks."
    )
    rr = rule_results(backtest_frames(st.session_state.get("bt_floor", 2)), rank)
    if rr:
        bits = [f"{x['Formation'].lower()}: longs {x['Avg long return']:+.1%}, shorted stocks "
                f"{x['Avg return of shorted stocks']:+.1%}" for x in rr
                if pd.notna(x["Avg long return"]) and pd.notna(x["Avg return of shorted stocks"])]
        if bits:
            st.caption("Backtest of this exact rule over the next 12 months (a short gains when its stock falls): "
                       + "; ".join(bits) + ". Details on the Backtest tab.")

    longs = rank_longs(res, how_long, n)
    shorts = rank_shorts(res, how_short, n, SELL_PROFITABLE)
    spec = rank_shorts(res, how_short, 3, SELL_SPECULATIVE)
    baskets = {"Long ideas": longs, "Short ideas (profitable)": shorts, "Speculative shorts": spec, "Universe": res}
    cols = {"ey": "EBIT/TEV", "bm": "B/M", "gpa": "GPA", "fscore": "F_SCORE", "altman_z_std": "Altman Z",
            "sta": "STA", "snoa": "SNOA", "franchise": "Franchise pct"}
    summ = pd.DataFrame({k: v.assign(ey=v["ey"].mask(v["ey"] >= 10))[list(cols)].median()
                         for k, v in baskets.items() if len(v)}).T.rename(columns=cols)
    st.markdown("**Median profile of each basket**")
    st.dataframe(summ.style.format({"EBIT/TEV": "{:.1%}", "STA": "{:+.1%}", "SNOA": "{:.0%}", "B/M": "{:.2f}",
                                    "GPA": "{:.2f}", "F_SCORE": "{:.0f}", "Altman Z": "{:.2f}",
                                    "Franchise pct": "{:.2f}"}, na_rep="–"), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown(f"### 🟢 Top {len(longs)} long idea{'s' if len(longs) > 1 else ''}" if len(longs)
                    else "### 🟢 Long ideas")
        if longs.empty:
            st.info("No BUY candidates under the current settings. Loosen Step 3 or Step 4 in the sidebar.")
        for i, (_, r) in enumerate(longs.iterrows(), 1):
            idea_card(r, "long", i)
    with right:
        st.markdown(f"### 🔴 Top {len(shorts)} short idea{'s' if len(shorts) > 1 else ''}: overvalued profitable firms"
                    if len(shorts)
                    else "### 🔴 Short ideas: overvalued profitable firms")
        if shorts.empty:
            st.info('Profitable shorts are hidden by the "SELL groups to include" setting in the sidebar.'
                    if settings.sell_scope == "speculative" else
                    "No profitable SELL candidates under the current settings. Lower the red-flag requirement or "
                    "widen the 'expensive' share under SELL rules.")
        for i, (_, r) in enumerate(shorts.iterrows(), 1):
            idea_card(r, "short", i)
    if len(spec):
        st.markdown(f"### 🔴 Speculative shorts: {len(spec)} loss-making firm{'s' if len(spec) > 1 else ''} "
                    "priced on hope")
        st.caption("Higher risk: these can rise sharply on news, so they suit only small positions.")
        sc = st.columns(len(spec))
        for i, (_, r) in enumerate(spec.iterrows(), 1):
            with sc[i - 1]:
                idea_card(r, "short", i, compact=True)
    elif settings.sell_scope == "profitable":
        st.caption('Speculative shorts are hidden by the "SELL groups to include" setting in the sidebar.')
    st.caption("These are screening outputs from public financial statements, produced for an ACCT 3343 course "
               "project. They are not investment advice.")
    return rank


with t_rec:
    RANK = recommendations_tab()

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
with t_dash:
    c1, c2 = st.columns([3, 2])
    with c1:
        plot = res.dropna(subset=["value_score", "quality_score"])
        fig = px.scatter(
            plot, x="value_score", y="quality_score", color="label",
            color_discrete_map=LABEL_COLORS, category_orders={"label": LABEL_ORDER},
            hover_name="ticker",
            hover_data={"name": True, "sector": True, "value_score": ":.2f", "quality_score": ":.2f",
                        "n_flags": True, "fscore": True, "altman_z": ":.2f", "label": False},
            labels={"value_score": "Cheapness percentile  (expensive → cheap)",
                    "quality_score": "Quality percentile  (weak → strong)", "label": "Signal"},
            title="Where every firm sits: price vs. quality",
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color=MARKER_RING)))
        fig.add_vrect(x0=0, x1=settings.expensive_pct, fillcolor=LABEL_COLORS["SELL"], opacity=0.06, line_width=0)
        fig.add_vrect(x0=1 - settings.value_keep, x1=1, fillcolor=LABEL_COLORS["BUY"], opacity=0.06, line_width=0)
        st.plotly_chart(style_fig(fig, 520), width="stretch")
        st.caption("Shaded left band: the most expensive firms in the whole universe, where speculative SELL "
                   "candidates come from. Red points to the right of it are profitable firms that are expensive "
                   "relative to other profitable firms. Shaded right band: the cheap zone BUY candidates are drawn "
                   "from (after the accrual and distress screens). Orange AVOID points look cheap but failed the "
                   "manipulation or distress screens.")
    with c2:
        fun = pd.DataFrame([{"stage": "Universe", "n": len(res)}] +
                           [{"stage": s["step"], "n": s["remaining"]} for s in steps])
        ffig = go.Figure(go.Funnel(y=fun["stage"], x=fun["n"], textinfo="value+percent initial",
                                   marker=dict(color=SERIES_1)))
        ffig.update_layout(title="Sequential BUY funnel")
        st.plotly_chart(style_fig(ffig, 520), width="stretch")

    st.subheader("What separates SELL from BUY candidates")
    prof_cols = {"ey": "EBIT/TEV", "bm": "B/M", "gpa": "GPA", "fscore": "F_SCORE", "altman_z": "Altman Z",
                 "sta": "STA", "snoa": "SNOA", "mscore": "M-score", "cfoa": "CFOA (8y FCF / TA)",
                 "roa_geo": "Geo-avg ROA", "mg": "Margin growth", "ms": "Margin stability"}
    grp = res["sell_type"].fillna(res["label"].astype(str)).replace(
        {SELL_PROFITABLE: "SELL · profitable", SELL_SPECULATIVE: "SELL · speculative"})
    prof = res.assign(ey=res["ey"].mask(res["ey"] >= 10)).groupby(grp)[list(prof_cols)].median().T
    prof["Universe"] = res.assign(ey=res["ey"].mask(res["ey"] >= 10))[list(prof_cols)].median()
    prof.index = [prof_cols[i] for i in prof.index]
    order = ["BUY", "SELL · profitable", "SELL · speculative", "AVOID", "HOLD", "Universe"]
    prof = prof[[c for c in order if c in prof.columns]]
    pct_rows = [r for r in ("EBIT/TEV", "STA", "SNOA", "Geo-avg ROA") if r in prof.index]
    sty = (prof.style.format("{:.2f}", na_rep="–")
           .format("{:.1%}", na_rep="–", subset=pd.IndexSlice[pct_rows, :])
           .format("{:.0f}", na_rep="–", subset=pd.IndexSlice[["F_SCORE"], :]))
    st.dataframe(sty, width="stretch")

    if len(sells) or len(buys):
        med_ey = res["ey"].median()
        bullets = []
        if len(sells_prof):
            ey_p = sells_prof["ey"].median()
            mult = f" (about {1 / ey_p:.0f}× operating profit vs {1 / med_ey:.0f}× for the median firm)" if ey_p > 0 and med_ey > 0 else ""
            bullets.append(
                f"- **{len(sells_prof)} overvalued profitable firms** have a median EBIT/TEV of **{ey_p:.1%}** vs "
                f"**{med_ey:.1%}** for the universe{mult}, together with red flags such as accrual-heavy earnings "
                f"({sells_prof['flag_accruals'].mean():.0%} of them), a high Beneish M-score "
                f"({sells_prof['flag_manipulation'].mean():.0%}) or negative free cash flow "
                f"({sells_prof['flag_cash_burn'].mean():.0%}). A high price on weakening fundamentals is the "
                "classic overvaluation profile.")
        if len(sells_spec):
            bullets.append(
                f"- **{len(sells_spec)} speculative SELL candidates** made an operating loss last year yet have a "
                f"median market cap of **{fmt_money(sells_spec['mcap'].median())}**: their value rests on expected "
                f"future profits. {(sells_spec['fcf'] < 0).mean():.0%} burned free cash flow and their median "
                f"F_SCORE is {sells_spec['fscore'].median():.0f}.")
        if len(buys):
            bullets.append(
                f"- **{len(buys)} BUY candidates** have a median EBIT/TEV of **{buys['ey'].median():.1%}**, median "
                f"F_SCORE **{buys['fscore'].median():.0f}**, and median franchise percentile "
                f"**{buys['franchise'].median():.2f}**: cheap, financially sound and not dependent on accruals.")
        bullets.append(
            f"- **{counts['AVOID']} firms look cheap but failed the accrual or distress screens**. These are the "
            "value traps the Quantitative Value process is designed to remove before buying.")
        st.markdown("**Reading the result (updates with the filters):**\n" + "\n".join(bullets))

    sec = (res.groupby(["sector", "label"], observed=False).size().unstack(fill_value=0)
           .reindex(columns=LABEL_ORDER, fill_value=0))
    sec_share = sec.div(sec.sum(axis=1), axis=0).reset_index().melt("sector", var_name="label", value_name="share")
    sec_share = sec_share[sec_share["label"].isin(["BUY", "SELL"])]
    sfig = px.bar(sec_share, y="sector", x="share", color="label", barmode="group", orientation="h",
                  color_discrete_map=LABEL_COLORS, labels={"share": "Share of sector's firms", "sector": "",
                                                           "label": "Signal"},
                  title="BUY and SELL candidates as a share of each sector")
    sfig.update_xaxes(tickformat=".0%")
    sfig.update_layout(bargap=0.25, bargroupgap=0.1)
    st.plotly_chart(style_fig(sfig, 460), width="stretch")
    csv_button(res.drop(columns=["z_rel_step"]), "full_screen_results", "dl_all")

# ---------------------------------------------------------------------------
# SELL list
# ---------------------------------------------------------------------------
SELL_COLS = ["ticker", "name", "sector", "mcap", "n_flags", "overvaluation_score", "value_score",
             "quality_score", "ey", "bm", "fscore", "altman_z", "sta", "snoa", "mscore", "franchise"]
with t_sell:
    st.subheader(f"🔴 SELL candidates: {len(sells)} potentially overvalued stocks")
    st.markdown(
        f"A SELL candidate needs at least **{settings.min_flags}** of the red flags selected in the sidebar and a "
        "price that is hard to justify. The list has two groups, because loss-making firms always look "
        "expensive on earnings and would otherwise crowd out everything else:\n"
        f"- **Overvalued (profitable)**: positive EBIT, but in the most expensive {settings.expensive_pct:.0%} of "
        "*profitable* firms. These are the more conventional short ideas.\n"
        f"- **Speculative (unprofitable)**: an operating loss last year, no multi-year record of profits, and in "
        f"the most expensive {settings.expensive_pct:.0%} "
        "of the universe. The price rests on future profits that may not arrive.\n\n"
        "Each group is sorted by number of red flags, then by overvaluation score."
    )
    for group, frame in ((SELL_PROFITABLE, sells_prof), (SELL_SPECULATIVE, sells_spec)):
        st.markdown(f"#### {group}: {len(frame)}")
        if frame.empty:
            hidden = ((group == SELL_PROFITABLE and settings.sell_scope == "speculative")
                      or (group == SELL_SPECULATIVE and settings.sell_scope == "profitable"))
            st.info('This group is hidden by the "SELL groups to include" setting in the sidebar.' if hidden
                    else "No firms in this group under the current settings.")
            continue
        sl = frame.sort_values(["n_flags", "overvaluation_score"], ascending=False).copy()
        sl["why"] = [explain(r, settings) for _, r in sl.iterrows()]
        show_table(sl, SELL_COLS + [f"flag_{k}" for k in SELL_FLAGS] + ["why"], height=min(80 + 35 * len(sl), 460))
    csv_button(sells.sort_values(["sell_type", "n_flags", "overvaluation_score"], ascending=[True, False, False]),
               "sell_candidates", "dl_sell")
    in_scope = {"both": pd.Series(True, index=res.index), "profitable": res["ebit"] > 0,
                "speculative": ~(res["ebit"] > 0)}[settings.sell_scope]
    near = (res[res["expensive"] & in_scope & (res["n_flags"] == settings.min_flags - 1) & (res["label"] != "SELL")]
            if settings.sell_flags and settings.min_flags <= len(settings.sell_flags) else res.iloc[0:0])
    with st.expander(f"Watch list: {len(near)} expensive firms one red flag short of SELL"):
        show_table(near.sort_values("overvaluation_score", ascending=False),
                   ["ticker", "name", "sector", "mcap", "n_flags", "value_score", "quality_score", "ey",
                    "fscore", "altman_z"] + [f"flag_{k}" for k in SELL_FLAGS], height=360)

# ---------------------------------------------------------------------------
# BUY list
# ---------------------------------------------------------------------------
with t_buy:
    st.subheader(f"🟢 BUY candidates: {len(buys)} firms that survive every screen")
    st.markdown(
        "Survivors of the sequential Quantitative Value process: no extreme accruals (Ch 3), no distress "
        "(Ch 4), cheap on price (Ch 2) and high quality on franchise power, F_SCORE and gross profitability "
        "(Ch 2, 5, 6). Sorted by quality."
    )
    bl = buys.sort_values("quality_score", ascending=False)
    show_table(bl, ["ticker", "name", "sector", "mcap", "quality_score", "value_score", "ey", "bm", "gpa",
                    "roc_gb", "fscore", "franchise", "cfoa", "mg", "ms", "altman_z", "sta", "snoa"])
    csv_button(bl, "buy_candidates", "dl_buy")
    avoid = res[res["label"] == "AVOID"].sort_values("value_score", ascending=False)
    with st.expander(f"🟠 AVOID: {len(avoid)} cheap-looking value traps removed by the risk screens"):
        show_table(avoid, ["ticker", "name", "sector", "mcap", "value_score", "ey", "bm", "stage", "sta",
                           "snoa", "mscore", "altman_z", "fscore"], height=360)

# ---------------------------------------------------------------------------
# Step-by-step screen
# ---------------------------------------------------------------------------
STEP_HELP = {
    "accrual_rank_sum": ("STA = (NI − CFO) / Total assets and SNOA = (Operating assets − Operating liabilities) "
                         "/ Total assets. Both are ranked descending (rank 1 = most accrual-heavy), the ranks are "
                         "added and the worst combined ranks are removed."),
    "mscore": (f"Beneish M-score from eight indices (DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA). PROBM = Φ(M). "
               f"M above {M_SCORE_CUTOFF} is the usual manipulator threshold."),
    "altman_z": "Z = a·X1 + b·X2 + c·X3 + d·X4 + e·X5 using the selected coefficients. Lower is more distressed.",
    "z_rel_step": "Altman Z percentile within each sector among the firms still in the screen (0 = weakest in "
                  "its sector); removes the weakest firms of every sector rather than whole weak sectors.",
    "value_score": "Rank each selected value measure, average the ranks and re-rank (1 = cheapest).",
    "quality_score": "Rank each selected quality measure, average and re-rank (1 = best). Then apply the minimum F_SCORE.",
}
AXIS = {"mscore": "Beneish M-score", "altman_z": "Altman Z", "z_rel_step": "Altman Z percentile within sector",
        "value_score": "Cheapness percentile", "quality_score": "Quality percentile"}
with t_steps:
    st.markdown("Each screen acts only on the firms that survive the preceding screen. Firms missing a required "
                "metric are removed at that step and counted separately from the percentile cutoff.")
    st.dataframe(pd.DataFrame(steps)[["step", "chapter", "entering", "missing", "excluded", "remaining"]]
                 .rename(columns=str.title), hide_index=True, width="stretch")
    for s_ in steps:
        st.markdown(f"#### {s_['step']}  ·  {s_['chapter']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Entering", f"{s_['entering']:,}")
        m2.metric("Missing data", f"{s_['missing']:,}")
        m3.metric("Excluded by cutoff", f"{s_['excluded']:,}")
        m4.metric("Remaining", f"{s_['remaining']:,}")
        st.caption(STEP_HELP[s_["metric"]])
        sub = res.set_index("ticker").loc[s_["entering_idx"]]
        sub = sub.assign(outcome=np.where(sub.index.isin(s_["excluded_idx"]), "Excluded", "Retained"))
        cmap = {"Excluded": LABEL_COLORS["SELL"], "Retained": SERIES_1}
        if s_["metric"] == "accrual_rank_sum":
            d = sub.dropna(subset=["sta", "snoa"])
            hfig = px.scatter(d.reset_index(), x="sta", y="snoa", color="outcome", color_discrete_map=cmap,
                              hover_name="ticker", labels={"sta": "STA", "snoa": "SNOA", "outcome": ""})
            hfig.update_xaxes(range=[-0.3, 0.3], tickformat=".0%")
            hfig.update_yaxes(range=[-0.5, 1.5], tickformat=".0%")
            hfig.update_traces(marker=dict(size=7, opacity=0.75))
        else:
            col = s_["metric"]
            d = sub.dropna(subset=[col]).reset_index()
            lo, hi = d[col].quantile([0.01, 0.99])
            d[col] = d[col].clip(lo, hi)
            hfig = px.histogram(d, x=col, color="outcome", color_discrete_map=cmap, nbins=60,
                                barmode="overlay", labels={col: AXIS.get(col, col), "outcome": ""},
                                category_orders={"outcome": ["Retained", "Excluded"]})
            hfig.update_traces(opacity=0.8)
        st.plotly_chart(style_fig(hfig, 300), width="stretch", key=f"step_{s_['step']}")
        if s_["excluded"]:
            with st.expander("Cutoff inspection: excluded firms"):
                ex = sub[sub["outcome"] == "Excluded"].reset_index()
                show_table(ex, ["ticker", "name", "sector", "sta", "snoa", "mscore", "altman_z", "value_score",
                                "quality_score", "fscore"], height=260)

# ---------------------------------------------------------------------------
# Company analysis
# ---------------------------------------------------------------------------
def company_tab() -> None:
    if res.empty:
        st.info("No firms match the current filters.")
        return
    options = res.sort_values("mcap", ascending=False)
    tick_opts = options["ticker"].tolist()
    label_of = {t: f"{t} — {n}  [{lab}]" for t, n, lab in zip(options["ticker"], options["name"], options["label"])}
    if st.session_state.get("company") not in tick_opts:
        first = rank_shorts(res, RANK["short"], 1, SELL_PROFITABLE)
        if first.empty:
            first = rank_longs(res, RANK["long"], 1)
        st.session_state["company"] = first["ticker"].iloc[0] if len(first) else tick_opts[0]
    pick = st.selectbox("Choose a company", tick_opts, format_func=label_of.get, key="company")
    r = res.set_index("ticker").loc[pick]
    hist = panels["current"][panels["current"]["cik"] == r["cik"]].sort_values("fy_end")
    badge = {"BUY": "🟢", "SELL": "🔴", "AVOID": "🟠", "HOLD": "⚪"}[r["label"]]
    st.markdown(f"### {badge} {r['name']} ({pick}) — **{r['label']}**")
    st.markdown(f"{r['sector']} · {r['industry']} · Market cap {fmt_money(r['mcap'])} · "
                f"Latest 10-K: fiscal year ended {pd.Timestamp(r['fy_end']).date()}")
    st.info(explain(r, settings))
    if r.get("major_change", False):
        ratio_txt = ("" if pd.isna(r["post_fy_asset_ratio"]) else
                     f" The latest quarterly balance sheet shows total assets at {r['post_fy_asset_ratio']:.0%} of the 10-K figure.")
        st.warning("This company changed substantially after its last 10-K (spin-off, divestiture or merger), so "
                   "today's price and the 10-K describe different businesses." + ratio_txt +
                   " Price-based ratios are left blank for this firm.")

    excess_cash = r[f"excess_cash_{settings.excess_cash}"]
    excess_cash = r["cash"] if pd.isna(excess_cash) else excess_cash
    ey_txt = "n/a (TEV ≤ 0, ranked cheapest)" if pd.notna(r["ey"]) and r["ey"] >= 10 else fmt(r["ey"], True)
    a, b = st.columns(2)
    with a:
        st.markdown("**Ch 2 · Price and quality**")
        st.markdown(
            f"- GPA = (Revenue − COGS) / TA = {fmt_money(r['gross_profit'])} / {fmt_money(r['total_assets'])} = **{fmt(r['gpa'])}**\n"
            f"- BM = Common book equity / Market cap = {fmt_money(r['common_equity'])} / {fmt_money(r['mcap'])} = **{fmt(r['bm'])}**\n"
            f"- TEV = Market cap + Debt − Excess cash + Preferred + Minority interest = {fmt_money(r['mcap'])} + "
            f"{fmt_money(r['debt'])} − {fmt_money(excess_cash)} + {fmt_money(r['preferred'])} + "
            f"{fmt_money(r['minority_interest'])} = **{fmt_money(r['tev'])}**\n"
            f"- Earnings yield = EBIT / TEV = {fmt_money(r['ebit'])} / {fmt_money(r['tev'])} = **{ey_txt}**\n"
            f"- Greenblatt ROC = EBIT / (Net PPE + NWC) = {fmt_money(r['ebit'])} / ({fmt_money(r['ppe_net'])} + "
            f"{fmt_money(r['nwc'])}) = **{fmt(r['roc_gb'], True)}**\n"
            f"- Cheapness percentile **{fmt(r['value_score'])}**, quality percentile **{fmt(r['quality_score'])}**"
        )
        st.markdown("**Ch 3 · Accruals and manipulation**")
        st.markdown(
            f"- STA = (NI − CFO) / TA = ({fmt_money(r['net_income'])} − {fmt_money(r['cfo'])}) / "
            f"{fmt_money(r['total_assets'])} = **{fmt(r['sta'], True)}**\n"
            f"- SNOA = (Op. assets − Op. liabilities) / TA = ({fmt_money(r['op_assets'])} − {fmt_money(r['op_liab'])}) / "
            f"{fmt_money(r['total_assets'])} = **{fmt(r['snoa'], True)}**\n"
            f"- Beneish M-score = **{fmt(r['mscore'])}** (PROBM {fmt(r['probm'], True)}; flag if M > {M_SCORE_CUTOFF})"
        )
        bn = pd.DataFrame({
            "Index": ["DSRI receivables/sales", "GMI gross-margin decline", "AQI asset quality", "SGI sales growth",
                      "DEPI depreciation rate", "SGAI SG&A/sales", "LVGI leverage", "TATA total accruals/assets"],
            "Value": [r[f"b_{c}"] for c in ("dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata")],
            "Coefficient": [BENEISH[c] for c in ("dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata")],
        })
        bn["Contribution"] = bn["Value"] * bn["Coefficient"]
        with st.expander(f"Beneish M-score build-up: {BENEISH['const']} + Σ contributions"):
            st.dataframe(bn.style.format({"Value": "{:.3f}", "Coefficient": "{:+.3f}", "Contribution": "{:+.3f}"},
                                         na_rep="–"), hide_index=True, width="stretch")
        st.markdown("**Ch 4 · Altman Z**")
        coefs = ALTMAN_COEFS[settings.altman]
        az = pd.DataFrame({
            "Component": ["X1 Working capital / TA", "X2 Retained earnings / TA", "X3 EBIT / TA",
                          "X4 Market value equity / Total liabilities", "X5 Sales / TA"],
            "Ratio": [r[f"x{i}"] for i in range(1, 6)], "Weight": coefs,
        })
        az["Contribution"] = az["Ratio"] * az["Weight"]
        st.dataframe(az.style.format({"Ratio": "{:.3f}", "Weight": "{:.3f}", "Contribution": "{:.3f}"}, na_rep="–"),
                     hide_index=True, width="stretch")
        zstd = r["altman_z_std"]
        zone = ("n/a" if pd.isna(zstd) else "distress" if zstd < settings.z_threshold
                else "grey" if zstd < 2.99 else "safe")
        extra = "" if settings.altman == STANDARD_ALTMAN else f" (slide weights: {fmt(r['altman_z'])})"
        st.markdown(f"Standard Z = **{fmt(zstd)}** → {zone} zone{extra}")
    with b:
        st.markdown(f"**Ch 5 · Franchise power ({lookback}-year window)**")
        n_fcf = int(hist["fcf"].tail(lookback).isna().sum())
        cfoa_note = (f" (FCF missing for {n_fcf} of the last {lookback} years)" if pd.isna(r["cfoa"]) and n_fcf
                     else f" (only {len(hist)} years of statements)" if pd.isna(r["cfoa"]) and len(hist) < lookback
                     else "")
        st.markdown(
            f"- Geometric-average ROA **{fmt(r['roa_geo'], True)}**, geometric-average ROC **{fmt(r['roc_geo'], True)}**\n"
            f"- CFOA = Σ FCF / TA = **{fmt(r['cfoa'])}**{cfoa_note}\n"
            f"- Margin growth MG = [Π GMₜ₊₁/GMₜ]^(1/T) = **{fmt(r['mg'], d=3)}**\n"
            f"- Margin stability MS = mean(GM) / std(GM) = **{fmt(r['ms'], d=1)}**\n"
            f"- Franchise percentile **{fmt(r['franchise'])}** (years of data: {int(r['years_available'])})"
        )
        st.markdown("**Ch 6 · F_SCORE**")
        fsc = pd.DataFrame({
            "Signal": ["ROA > 0", "CFO > 0", "ΔROA > 0", "CFO > ROA (accrual)", "Leverage fell",
                       "Liquidity rose", "No net equity issued", "Gross margin rose", "Asset turnover rose"],
            "Point": [r[c] for c in ("F_ROA", "F_CFO", "F_dROA", "F_ACCRUAL", "F_dLEVER", "F_dLIQUID",
                                     "F_EQ_OFFER", "F_dMARGIN", "F_dTURN")],
        })
        fsc["Point"] = fsc["Point"].map(lambda v: "–" if pd.isna(v) else str(int(v)))
        st.dataframe(fsc, hide_index=True, width="stretch")
        st.markdown(f"F_SCORE = **{fmt(r['fscore'], d=0)}** / 9")

    if len(hist):
        h = hist.assign(year=hist["fy_end"].dt.strftime("%Y-%m"),
                        gross_margin=hist["gross_profit"] / hist["revenue"])
        long = h.melt("year", ["revenue", "net_income", "fcf"], var_name="series", value_name="usd")
        long["series"] = long["series"].map({"revenue": "Revenue", "net_income": "Net income", "fcf": "Free cash flow"})
        long["usd_b"] = long["usd"] / 1e9
        c1, c2 = st.columns(2)
        f1 = px.line(long, x="year", y="usd_b", color="series", markers=True,
                     color_discrete_sequence=[SERIES_1, SERIES_2, SERIES_3],
                     labels={"usd_b": "USD billions", "year": "Fiscal year end", "series": ""},
                     title="Revenue, net income and free cash flow")
        f1.update_traces(line=dict(width=2), marker=dict(size=8))
        c1.plotly_chart(style_fig(f1, 340), width="stretch")
        f2 = px.line(h, x="year", y="gross_margin", markers=True, title="Gross margin (drives MG and MS)",
                     labels={"gross_margin": "Gross margin", "year": "Fiscal year end"},
                     color_discrete_sequence=[SERIES_1])
        f2.update_traces(line=dict(width=2), marker=dict(size=8))
        f2.update_yaxes(tickformat=".0%")
        c2.plotly_chart(style_fig(f2, 340), width="stretch")
        with st.expander("Underlying 10-K data (USD)"):
            raw = hist.drop(columns=["cik"]).set_index("fy_end")
            raw.index = raw.index.strftime("%Y-%m-%d")
            st.dataframe(raw.select_dtypes("number").T, width="stretch")
            src = raw["gp_source"].fillna("n/a") if "gp_source" in raw else pd.Series("sec", index=raw.index)
            st.caption("Latest filing reporting each year (restated values replace originals): "
                       + ", ".join(f"FY{fy}: {d}" for fy, d in zip(raw.index.str[:7], raw["filed"].dt.strftime("%Y-%m-%d"))) +
                       ". Gross profit source by year: " + ", ".join(src) +
                       " (yahoo = Yahoo Finance standardized statement, used where SEC data has no gross profit).")


with t_company:
    company_tab()

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def interpret_backtest(notes: list[dict]) -> None:
    """Plain-English reading of the backtest. Every sentence is conditional on the numbers shown."""
    if not notes:
        return
    pct = lambda x: "n/a" if pd.isna(x) else f"{(0.0 if abs(x) < 0.0005 else x):+.1%}"
    lines = []
    for n in notes:
        if pd.isna(n["buy"]):
            buy = "no BUY candidates"
        else:
            med_win, avg_win = n["buy"] > n["uni"], n["buy_avg"] > n["uni_avg"]
            verdict = ("beat on both median and average" if med_win and avg_win else
                       "lagged on both median and average" if not med_win and not avg_win else
                       "beat on median, lagged on average" if med_win else "lagged on median, beat on average")
            buy = f"BUY median {pct(n['buy'])} vs {pct(n['uni'])} for the median firm ({verdict})"
        parts = [buy]
        if pd.notna(n["prof"]):
            parts.append(f"profitable SELLs median {pct(n['prof'])}")
        if pd.notna(n["spec"]):
            parts.append(f"speculative SELLs median {pct(n['spec'])}")
        if pd.notna(n["q5"]) and pd.notna(n["q1"]):
            parts.append(f"cheapest fifth {pct(n['q5'])} vs most expensive fifth {pct(n['q1'])} (average)")
        lines.append(f"- **{n['label']}** (S&P 500 {pct(n['spy'])}): " + "; ".join(parts) + ".")
    st.markdown("#### How to read these results\n" + "\n".join(lines))

    para = []
    years = "both test years" if len(notes) == 2 else "the test year"
    sold_up = [n for n in notes if pd.notna(n["sell_avg"]) and n["sell_avg"] > max(n["uni_avg"], 0)
               and pd.notna(n["sell_med"]) and n["sell_med"] > n["uni"]]
    if len(sold_up) == len(notes):
        txt = (f"In {years} (S&P 500 " + " and ".join(pct(n["spy"]) for n in notes) + ") the SELL candidates rose "
               "more than the average stock on both mean and median, so a short book built from them would have "
               "lost money.")
        if all(pd.notna(n["loss_avg"]) and n["loss_avg"] > n["uni_avg"] and n["loss_med"] > n["uni"] for n in notes):
            txt += " Loss-making stocks as a group were rewarded"
            txt += (", which is exactly what the speculative SELL group bets against." if settings.sell_scope != "profitable"
                    else ".")
        para.append(txt + " That is a genuine result, not a bug, and the main reason short positions need tight "
                    "size limits.")
    else:
        for n in notes:
            if pd.notna(n["sell_med"]) and n["sell_med"] < n["uni"]:
                para.append(f"For the portfolio {n['label'].lower()}, the median SELL candidate ({pct(n['sell_med'])}) "
                            f"lagged the median firm ({pct(n['uni'])}), so the warning was right more often than not.")
    for n in notes:
        if not (pd.notna(n["spec"]) and n["spec_share_below"] > 0.5 and n["spec_mean"] > n["spec"]):
            continue
        winners = [(tk, raw, w) for tk, raw, w in n["spec_top"] if raw > max(n["uni"], 0) + 0.5]
        if not winners:
            continue
        top = ", ".join(f"{tk} {raw:+.0%}" + (f" ({w:+.0%} after winsorizing)" if abs(raw - w) > 0.005 else "")
                        for tk, raw, w in winners)
        verb = "lifted the group's average to" if n["spec_mean"] > 0 else "pulled the group's average up to"
        para.append(
            f"The speculative SELL group shows the classic risk of shorting. For the portfolio {n['label'].lower()}, "
            f"{n['spec_share_below']:.0%} of these stocks did worse than the median firm (group median "
            f"{pct(n['spec'])}), so the screen was right about most of them, but a few extreme winners ({top}) "
            f"{verb} {pct(n['spec_mean'])}.")
    good = [n["label"].lower() for n in notes if pd.notna(n["q5"]) and n["q5"] > n["q1"]]
    bad = [n["label"].lower() for n in notes if pd.notna(n["q5"]) and n["q5"] <= n["q1"]]
    if good:
        para.append("The value composite sorted returns in the right direction for the portfolio "
                    + " and ".join(good) + (f", but not for the one {' and '.join(bad)}." if bad else "."))
    elif bad:
        para.append("The value composite did not sort returns in the right direction in any test year.")
    para.append(
        "Quantitative Value is built for multi-year holding periods, and value strategies have had multi-year "
        "droughts before (for example during the late-1990s technology boom). Two 12-month windows are too short "
        "to prove or disprove the method, so read this tab as a check on how the screen behaves, not as a forecast.")
    st.markdown("\n\n".join(para))


def backtest_tab() -> None:
    st.markdown(
        "**Did these rules work out of sample?** The same sidebar settings are applied to point-in-time data: "
        "only 10-K facts filed on or before each formation date, and market caps on that date. Each firm is "
        "then held for 12 months. Returns are equal-weighted total returns (dividends included)."
    )
    bt_floor = st.select_slider(
        "Backtest universe: minimum market cap on the formation date", options=[0.5, 1, 2, 5, 10], value=2,
        format_func=lambda v: f"${v}B", key="bt_floor",
        help="The firm list is today's (all above $1B now). Firms that were tiny at formation are only here "
             "because they grew, which biases results upward. A formation-date floor removes most of them.")
    frames = backtest_frames(bt_floor)
    groups = ["BUY", "SELL", "SELL · profitable", "SELL · speculative", "AVOID", "HOLD"]
    rows, quint, notes = [], [], []
    for snap, full in frames.items():
        bres = full.dropna(subset=["fwd_ret"]).copy()
        if bres.empty:
            continue
        lo, hi = bres["fwd_ret"].quantile([0.01, 0.99])
        bres["fwd_ret_w"] = bres["fwd_ret"].clip(lo, hi)
        spy = float(bench.loc[bench["snapshot"] == snap, "spy_ret"].iloc[0])
        uni_med, uni_avg = bres["fwd_ret"].median(), bres["fwd_ret_w"].mean()
        member = {
            "SELL · profitable": bres["sell_type"] == SELL_PROFITABLE,
            "SELL · speculative": bres["sell_type"] == SELL_SPECULATIVE,
        }
        for lab in groups:
            g_ = bres.loc[member[lab] if lab in member else bres["label"] == lab]
            rows.append({"Formation": SNAP_LABEL[snap], "Signal": lab, "Firms": len(g_),
                         "Avg return (winsorized)": g_["fwd_ret_w"].mean(), "Median return": g_["fwd_ret"].median(),
                         "Raw mean return": g_["fwd_ret"].mean(),
                         "Beat universe median": (g_["fwd_ret"] > uni_med).mean() if len(g_) else np.nan,
                         "Universe avg (winsorized)": uni_avg, "S&P 500 (SPY)": spy})
        quint.append(bres.assign(Formation=SNAP_LABEL[snap]))
        vq = bres.dropna(subset=["value_score"])
        vq = vq.assign(_q=pd.qcut(vq["value_score"].rank(method="first"), 5, labels=False)) if len(vq) >= 5 else None
        spec_g = bres[member["SELL · speculative"]]
        sell_g = bres[bres["label"] == "SELL"]
        loss_g = bres[~(bres["ebit"] > 0)]
        buy_g = bres[bres["label"] == "BUY"]
        notes.append({
            "label": SNAP_LABEL[snap], "spy": spy, "uni": uni_med, "uni_avg": uni_avg,
            "buy": buy_g["fwd_ret"].median(), "buy_avg": buy_g["fwd_ret_w"].mean(),
            "prof": bres.loc[member["SELL · profitable"], "fwd_ret"].median(),
            "spec": spec_g["fwd_ret"].median(), "spec_mean": spec_g["fwd_ret_w"].mean(),
            "spec_share_below": (spec_g["fwd_ret"] < uni_med).mean() if len(spec_g) else np.nan,
            "spec_top": spec_g.nlargest(3, "fwd_ret")[["ticker", "fwd_ret", "fwd_ret_w"]].values.tolist(),
            "sell_avg": sell_g["fwd_ret_w"].mean(), "sell_med": sell_g["fwd_ret"].median(),
            "loss_avg": loss_g["fwd_ret_w"].mean(), "loss_med": loss_g["fwd_ret"].median(),
            "q1": vq.loc[vq["_q"] == 0, "fwd_ret_w"].mean() if vq is not None else np.nan,
            "q5": vq.loc[vq["_q"] == 4, "fwd_ret_w"].mean() if vq is not None else np.nan,
        })
    bt = pd.DataFrame(rows)
    if bt.empty:
        st.info("No firms with returns match the current filters.")
        return
    bfig = px.bar(bt, x="Signal", y="Avg return (winsorized)", color="Formation", barmode="group",
                  color_discrete_sequence=[SERIES_1, SERIES_2], text_auto=".1%",
                  category_orders={"Signal": groups},
                  title="Average 12-month return by signal (returns winsorized at 1% / 99%)")
    bfig.update_yaxes(tickformat=".0%")
    bfig.update_layout(bargap=0.3, bargroupgap=0.08)
    st.plotly_chart(style_fig(bfig, 400), width="stretch")
    st.dataframe(bt, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(c, format="percent")
        for c in ("Avg return (winsorized)", "Median return", "Raw mean return", "Beat universe median",
                  "Universe avg (winsorized)", "S&P 500 (SPY)")})
    piv = bt.pivot(index="Formation", columns="Signal", values="Avg return (winsorized)")
    if {"BUY", "SELL"} <= set(piv.columns):
        spread = (piv["BUY"] - piv["SELL"]).dropna()
        st.markdown("**BUY minus SELL spread (average):** " + " · ".join(f"{k}: **{v:+.1%}**" for k, v in spread.items()))

    st.markdown(f"#### The recommendation rule itself (top {RANK['n']} per side)")
    rr = pd.DataFrame(rule_results(frames, RANK))
    st.dataframe(rr, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(c, format="percent")
        for c in ("Avg long return", "Avg return of shorted stocks", "Long minus short")} | {
        "Long picks": st.column_config.TextColumn("Long picks", width="large"),
        "Short picks": st.column_config.TextColumn("Short picks", width="large")})
    st.caption(f"Picks use the Recommendations tab's ranking (longs: {RANK['long'].lower()}; shorts: "
               f"{RANK['short'].lower()}) on each formation date, before any returns are known. A short gains when "
               "its stock falls, so a positive return for the shorted stocks is a loss on the short side. Raw, "
               "unwinsorized returns; 'n/a' = no price history after formation.")
    interpret_backtest(notes)

    st.markdown("#### Does each signal sort future returns?")
    qmetric_label = st.selectbox("Sort firms into quintiles by", [
        "Cheapness percentile (value composite)", "Quality percentile (quality composite)",
        "Overvaluation score", "STA (accruals)", "SNOA", "Altman Z", "F_SCORE", "Franchise percentile",
        "Beneish M-score"])
    qcol = {"Cheapness percentile (value composite)": "value_score", "Quality percentile (quality composite)":
            "quality_score", "Overvaluation score": "overvaluation_score_univ", "STA (accruals)": "sta",
            "SNOA": "snoa", "Altman Z": "altman_z", "F_SCORE": "fscore", "Franchise percentile": "franchise",
            "Beneish M-score": "mscore"}[qmetric_label]
    q = pd.concat(quint)
    # Shuffle before ranking so ties (e.g. equal F_SCOREs) are not split by market-cap order.
    q = q.dropna(subset=[qcol]).sample(frac=1, random_state=0)
    qnames = ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
    q["Quintile"] = q.groupby("Formation")[qcol].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if s.size >= 5 else pd.Series(np.nan, index=s.index))
    q = q.dropna(subset=["Quintile"])
    q["Quintile"] = q["Quintile"].map(lambda i: qnames[int(i)])
    if q.empty:
        st.info("Too few firms for a quintile sort.")
        return
    qq = q.groupby(["Formation", "Quintile"], observed=True)["fwd_ret_w"].mean().reset_index()
    qfig = px.bar(qq, x="Quintile", y="fwd_ret_w", color="Formation", barmode="group",
                  color_discrete_sequence=[SERIES_1, SERIES_2], text_auto=".1%",
                  labels={"fwd_ret_w": "Average 12-month return (winsorized)",
                          "Quintile": f"Quintile of {qmetric_label}"})
    qfig.update_yaxes(tickformat=".0%")
    qfig.update_layout(bargap=0.3, bargroupgap=0.08)
    st.plotly_chart(style_fig(qfig, 380), width="stretch")
    st.caption(
        "Caveats: the firm list is today's, so companies that went bankrupt, were acquired or shrank below "
        "$1B since formation are missing, and small firms that later boomed are included (survivorship and "
        "look-ahead bias; the formation-date size floor reduces but does not remove it). Formation-date market "
        "caps use the SEC cover-page share count from the last 10-K/10-Q filed before the formation date when "
        "it is recent and plausible; otherwise "
        f"({int((rets['mcap_src_bt2025'] == 'price_ratio').sum())} firms for 2025, "
        f"{int((rets['mcap_src_bt2024'] == 'price_ratio').sum())} for 2024) today's market cap scaled by the "
        "price change. Gross profit backfilled from Yahoo uses today's statements, a small look-ahead for "
        "those firm-years. Two one-year periods is a short sample: treat this as a sanity check, not proof."
    )


with t_backtest:
    backtest_tab()

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
with t_method:
    st.markdown(f"""
### Process
1. **Universe**: the {meta['n_operating']:,} largest U.S.-listed operating companies by market cap
   that file 10-Ks with the SEC, plus financial and real-estate firms above the same size cutoff
   (smallest firm about {fmt_money(meta['min_mcap'])}). One share class per company.
2. **Step 1 · Manipulation (Ch 3)**: remove the worst STA + SNOA combined ranks, then optionally the highest
   Beneish PROBM.
3. **Step 2 · Distress (Ch 4)**: remove the lowest Altman Z, then the lowest Z *within each sector* among
   survivors (optionally every firm under the distress cutoff).
4. **Step 3 · Price (Ch 2)**: keep the cheapest share on the value composite.
5. **Step 4 · Quality (Ch 2, 5, 6)**: keep the best share on the quality composite with a minimum F_SCORE. → **BUY**
6. **SELL** (at least *k* red flags) has two groups: **Overvalued (profitable)**, firms with EBIT > 0 in the
   most expensive share of *profitable* firms, and **Speculative (unprofitable)**, firms with an operating loss
   last year and no multi-year record of profits in the most expensive share of the universe. **AVOID**: cheap
   but removed in Step 1 or 2. Everything else is **HOLD**.
7. **Recommendations**: the top *n* BUY candidates (longs) and top *n* profitable SELL candidates (shorts), plus
   the top 3 speculative SELLs, ranked by the method chosen on the Recommendations tab: cheapness + quality,
   the Greenblatt magic formula (rank of EBIT/TEV + rank of ROC), quality or cheapness for longs; red flags,
   overvaluation score, Beneish M-score or accrual quality for shorts. The Backtest tab applies the same rule on
   each formation date.

### Formulas
""")
    st.latex(r"\text{ROC} = \frac{EBIT}{\text{Net PPE} + (CA - CL)} \qquad "
             r"\text{EY} = \frac{EBIT}{TEV},\; TEV = \text{MktCap} + \text{Debt} - \text{Excess cash} + \text{Pref} + \text{MI}")
    st.latex(r"GPA = \frac{Revenue - COGS}{Total\ Assets} \qquad BM = \frac{Common\ Book\ Equity}{Market\ Cap}")
    st.latex(r"STA = \frac{NI - CFO}{Total\ Assets} \qquad SNOA = \frac{(TA - Cash) - (TA - STD - LTD - MI - Pref - CE)}{TA}")
    st.latex(r"M = -4.84 + 0.920\,DSRI + 0.528\,GMI + 0.404\,AQI + 0.892\,SGI + 0.115\,DEPI - 0.172\,SGAI + 4.679\,TATA - 0.327\,LVGI")
    st.latex(r"Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5 \quad (\text{slides: } 0.012, 0.014, 0.033, 0.006, 0.999)")
    st.latex(r"CFOA = \frac{\sum_{t=1}^{T} FCF_t}{Total\ Assets} \qquad MG = \left[\prod_{t=1}^{T}\frac{GM_{t+1}}{GM_t}\right]^{1/T}"
             r"\qquad MS = \frac{\overline{GM}}{\sigma(GM)} \qquad MM = \max(\text{pct}(MG), \text{pct}(MS))")
    st.latex(r"\text{Franchise} = \text{rank}\big(\text{avg}[\text{pct}(ROA_{geo}), \text{pct}(ROC_{geo}), \text{pct}(CFOA), MM]\big)")
    st.markdown("""
**F_SCORE** (Ch 6): one point each for ROA > 0, CFO > 0, ΔROA > 0, CFO > ROA, lower LTD/TA, higher current ratio,
no net equity issuance, higher gross margin and higher asset turnover (ROA, CFO and turnover use beginning-of-year
assets). "Leverage fell" follows the slide literally: no point when LTD/TA is unchanged (including zero debt in both
years). If one or two signals cannot be computed, the score is rescaled to 9; with three or more missing it is blank.
Year-over-year signals require consecutive fiscal years (330-400 days apart).

### Data and missing-data rules
- **Fundamentals**: SEC EDGAR XBRL company facts, 10-K annual values only (duration 340-380 days); restated values
  replace originals. Each line item uses the first available standard us-gaap tag (e.g. `Revenues`, then
  `RevenueFromContractWithCustomerExcludingAssessedTax`, ...). Market cap, price, sector and industry come from
  the Nasdaq stock screener.
- A firm missing a metric required by a step is removed at that step and counted as *missing*, separately from
  the percentile cutoff. Percentile cuts remove floor(n × %) firms; ties are broken by ticker.
- Preferred stock, minority interest and equity issuance are taken as zero when the firm reports no such tag.
  Debt is zero only if no debt tag exists **and** interest expense is immaterial (< 0.5% of assets); capex is zero
  only if depreciation is immaterial (< 1% of assets). Otherwise the item is missing rather than zero.
- Gross profit: SEC data omits cost of sales that firms tag only by product/service dimension (e.g. TMO, GD). For
  those years the app uses Yahoo Finance's standardized gross profit, rescaled to the SEC revenue when Yahoo's
  revenue differs by 3-50% (a larger gap means a different perimeter, and the year stays blank). Margin trends
  (ΔMargin, GMI, MG, MS) only compare years from the same source.
- BM and SNOA use common equity (stockholders' equity minus preferred stock).
- Composites tolerate one gap: franchise power needs at least 3 of its 4 parts, and the quality composite at
  most one missing measure, so a single missing year no longer removes a firm.
- The Beneish manipulation flag is not applied to firms with fewer than 3 years of statements (recent IPOs and
  spin-offs), where carve-out accounting dominates the year-over-year indices.
- Sector labels come from the Nasdaq screener, with hand corrections for a few clearly mislabeled firms
  (e.g. PG, MO, PM and CL to Consumer Staples, TMO to Health Care).
- Greenblatt ROC and EBIT/TEV are blank when the denominator is zero or negative, except that positive EBIT on a
  zero/negative TEV ranks as the cheapest. Ch 5 ROC treats zero/negative capital as +100% (profitable) or −99%.
- If the company changed shape after its last 10-K, price-based ratios are blanked: latest-quarter total assets
  below 70% of the 10-K figure (spin-off/divestiture), above 150% with more than 15% more shares (stock-funded
  merger), or above 300%; plus separations announced after the last filing (CTVA, HON, CMCSA, manually reviewed).
- Firms with an unclassified balance sheet (no current assets/liabilities, e.g. some utilities) have no
  working capital, so no Altman Z, Greenblatt ROC or current-ratio F_SCORE signal.
- EBIT is operating income; if not tagged, pre-tax income + interest, else net income + tax + interest.
- "Within sector" ranking uses percentile ranks inside each Nasdaq sector (sectors with < 3 valid firms rank
  against the universe).
- Beneish indices that cannot be computed are set to 1 (neutral) and each index is bounded to [0.2, 5].

### Limitations
Accounting screens are signals, not verdicts. SELL means "price looks hard to justify by the fundamentals",
not a prediction that the stock will fall. Financial firms are excluded by default because these ratios do
not describe banks or insurers. This app is for educational use and is not investment advice.
""")
