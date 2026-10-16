"""ACCT 3343 - Overvalued Stock Screener (Streamlit).

Run locally:  streamlit run app.py
All data is pre-built by the scripts in /pipeline; this app never calls a data API.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (ALTMAN_COEFS, LABEL_COLORS, LABEL_ORDER, M_SCORE_CUTOFF, QUALITY_METRICS,
                       SELL_FLAGS, STANDARD_ALTMAN, VALUE_METRICS, Settings, compute_metrics, explain,
                       run_screen)

st.set_page_config(page_title="Overvalued Stock Screener", page_icon="📉", layout="wide")
DATA = Path(__file__).parent / "data"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"  # categorical slots for non-label charts
SNAP_LABEL = {"bt2024": "Formed 2024-10-15", "bt2025": "Formed 2025-10-15"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def data_version() -> tuple:
    """Changes whenever a data file or the analytics code changes, so cached results refresh."""
    files = sorted(DATA.glob("*")) + [Path(__file__).parent / "analytics.py"]
    return tuple((f.name, f.stat().st_mtime_ns) for f in files)


@st.cache_data(show_spinner=False)
def load_data(version: tuple):
    firms = pd.read_parquet(DATA / "firms.parquet")
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
    m = compute_metrics(panels[snapshot], firms, mcap, lookback, latest=lf)
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
    st.caption("Every chart, table and label updates as you move these.")
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

    with st.expander("Step 1 · Manipulation (Ch 3)", expanded=True):
        accrual_cut = pct_slider("Exclude worst STA + SNOA", 0, 50, 10, "accrual_cut",
                                 "Ranks scaled total accruals and scaled net operating assets (highest = "
                                 "worst), adds the ranks and removes the worst share.")
        use_probm = st.toggle("Also screen on Beneish PROBM", True, key="use_probm")
        probm_cut = pct_slider("Exclude worst PROBM", 0, 30, 5, "probm_cut") if use_probm else 0.0

    with st.expander("Step 2 · Distress (Ch 4)", expanded=True):
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

    with st.expander("Step 3 · Price (Ch 2)", expanded=True):
        value_labels = st.multiselect("Value measures (ranked, added, re-ranked)", list(VALUE_METRICS),
                                      ["Earnings yield (EBIT/TEV)", "Book-to-market (BM)"], key="value_m")
        excess = st.radio("Excess cash in TEV", ["Course slides: Cash + CA - CL",
                                                 "Greenblatt: Cash - max(0, CL - (CA - Cash))"], key="excess",
                          help="The slide formula counts cash twice (current assets already include cash), "
                               "which lowers TEV for firms with large working capital. Greenblatt's version "
                               "only subtracts cash not needed to cover current liabilities.")
        value_keep = pct_slider("Keep the cheapest", 5, 100, 30, "value_keep")

    with st.expander("Step 4 · Quality (Ch 2, 5, 6)", expanded=True):
        quality_labels = st.multiselect("Quality measures", list(QUALITY_METRICS),
                                        ["Franchise power (Ch 5)", "F_SCORE (Ch 6)", "Gross profitability GPA (Ch 2)"],
                                        key="quality_m")
        lookback = st.slider("Franchise look-back (fiscal years)", 3, 10, 8, key="lookback",
                             help="The book uses 8 years. Firms with shorter histories have no franchise score; "
                                  "a shorter window keeps more firms but measures persistence less well.")
        quality_keep = pct_slider("Keep the highest-quality", 5, 100, 50, "quality_keep")
        min_fscore = st.slider("Minimum F_SCORE", 0, 9, 5, key="min_f")

    with st.expander("SELL rules · overvalued", expanded=True):
        expensive_pct = pct_slider("'Expensive' = most expensive", 5, 50, 20, "exp_pct",
                                   "Share of the universe with the lowest value composite.")
        flag_labels = st.multiselect("Red flags that count", list(SELL_FLAGS.values()), list(SELL_FLAGS.values()),
                                     key="flags")
        min_flags = st.slider("SELL needs at least this many red flags", 1, 6, 2, key="min_flags")
        if len(flag_labels) < min_flags:
            st.warning(f"Only {len(flag_labels)} red flag(s) are enabled, so no firm can reach {min_flags}: "
                       "the SELL list will be empty.")
        sell_profitable_only = st.toggle("Only profitable firms (EBIT > 0) can be SELL", False, key="sell_prof",
                                         help="Off: firms with negative EBIT have a negative earnings yield and rank "
                                              "as the most expensive. On: SELL focuses on firms that are expensive "
                                              "relative to real, positive earnings.")
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
    sell_profitable_only=sell_profitable_only,
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

with st.sidebar:
    st.divider()
    st.subheader("Live funnel")
    lines = [f"Universe: **{len(res):,}**"]
    lines += [f"↓ {s['step']}: −{s['missing'] + s['excluded']:,} → {s['remaining']:,}" for s in steps]
    st.markdown("  \n".join(lines))
    st.markdown(f"**BUY {counts['BUY']} · SELL {counts['SELL']} · AVOID {counts['AVOID']}**")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_money(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    a = abs(x)
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
    "overvaluation_score": st.column_config.NumberColumn("Overvaluation score", format="%.0f",
                                                         help="0-100: average of expensiveness and weak quality."),
    "n_flags": st.column_config.NumberColumn("Red flags", format="%d"),
    "ey": st.column_config.NumberColumn("EBIT/TEV", format="percent"),
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
st.title("Overvalued Stock Screener")
st.markdown(
    f"Applies the ACCT 3343 financial-statement techniques (magic formula, accruals, Altman Z, franchise "
    f"power, F_SCORE) to **{meta['n_firms']:,} U.S. public companies**. Fundamentals come from SEC EDGAR "
    f"10-K filings (fiscal years ending {meta['fy_end_min']} to {meta['fy_end_max']}); market caps are "
    f"from {meta['generated']}. Move any control in the sidebar to re-run the analysis."
)
k = st.columns(5)
k[0].metric("Firms screened", f"{len(res):,}")
k[1].metric("🟢 BUY candidates", int(counts["BUY"]))
k[2].metric("🔴 SELL candidates", int(counts["SELL"]))
k[3].metric("🟠 AVOID (value traps)", int(counts["AVOID"]))
k[4].metric("⚪ HOLD / no signal", int(counts["HOLD"]))

tabs = st.tabs(["Dashboard", "🔴 SELL list", "🟢 BUY list", "Step-by-step screen", "Company analysis",
                "Backtest", "Methodology"])

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
with tabs[0]:
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
        fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color="rgba(255,255,255,0.6)")))
        fig.add_vrect(x0=0, x1=settings.expensive_pct, fillcolor=LABEL_COLORS["SELL"], opacity=0.06, line_width=0)
        fig.add_vrect(x0=1 - settings.value_keep, x1=1, fillcolor=LABEL_COLORS["BUY"], opacity=0.06, line_width=0)
        st.plotly_chart(style_fig(fig, 520), width="stretch")
        st.caption("Shaded left band: the 'expensive' zone where SELL candidates come from. Shaded right band: "
                   "the cheap zone BUY candidates are drawn from (after the accrual and distress screens). "
                   "Orange AVOID points look cheap but failed the manipulation or distress screens.")
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
    prof = res.groupby("label", observed=False)[list(prof_cols)].median().T
    prof["Universe"] = res[list(prof_cols)].median()
    prof.index = [prof_cols[i] for i in prof.index]
    prof = prof[[c for c in LABEL_ORDER + ["Universe"] if c in prof.columns]]
    st.dataframe(prof.style.format("{:.3f}", na_rep="–"), width="stretch")

    sells, buys = res[res["label"] == "SELL"], res[res["label"] == "BUY"]
    if len(sells) and len(buys):
        loss_share = (sells["ebit"] <= 0).mean()
        price_line = (
            f"**{loss_share:.0%}** of them have zero or negative operating profit, so their market value rests on "
            "expectations rather than current earnings"
            if loss_share >= 0.5 else
            "investors pay far more per dollar of operating profit than for the typical firm"
        )
        st.markdown(
            f"""
**Reading the result (updates with the filters):**
- **SELL candidates ({len(sells)})** have a median earnings yield (EBIT/TEV) of **{sells['ey'].median():.1%}** vs.
  **{res['ey'].median():.1%}** for the universe; {price_line}.
  Their median F_SCORE is **{sells['fscore'].median():.0f}**, median Altman Z is **{sells['altman_z'].median():.2f}**,
  and **{(sells['fcf'] < 0).mean():.0%}** burned free cash flow last year. A high price combined with weak or
  deteriorating fundamentals is the profile of a potentially overvalued stock. (Turn on *Only profitable firms*
  under SELL rules to focus on firms that are expensive relative to positive earnings.)
- **BUY candidates ({len(buys)})** have a median earnings yield of **{buys['ey'].median():.1%}**, median F_SCORE
  **{buys['fscore'].median():.0f}**, and median franchise percentile **{buys['franchise'].median():.2f}** —
  cheap, financially sound and not dependent on accruals.
- **{counts['AVOID']} firms look cheap but failed the accrual or distress screens**. These are the value traps
  the Quantitative Value process is designed to remove before buying.
"""
        )

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
with tabs[1]:
    st.subheader(f"🔴 SELL candidates: {len(sells)} potentially overvalued stocks")
    st.markdown(
        f"A firm is a **SELL** candidate when it is in the **most expensive {settings.expensive_pct:.0%}** of the "
        f"universe on the value composite **and** shows at least **{settings.min_flags}** of the red flags "
        "selected in the sidebar. Sorted by number of red flags, then by overvaluation score."
    )
    sl = sells.sort_values(["n_flags", "overvaluation_score"], ascending=False).copy()
    sl["why"] = [explain(r, settings) for _, r in sl.iterrows()]
    show_table(sl, ["ticker", "name", "sector", "mcap", "n_flags", "overvaluation_score", "value_score",
                    "quality_score", "ey", "bm", "fscore", "altman_z", "sta", "snoa", "mscore", "franchise"]
               + [f"flag_{k}" for k in SELL_FLAGS] + ["why"])
    csv_button(sl, "sell_candidates", "dl_sell")
    near = (res[res["expensive"] & (res["n_flags"] == settings.min_flags - 1) & (res["label"] != "SELL")]
            if settings.sell_flags and settings.min_flags <= len(settings.sell_flags) else res.iloc[0:0])
    with st.expander(f"Watch list: {len(near)} expensive firms one red flag short of SELL"):
        show_table(near.sort_values("overvaluation_score", ascending=False),
                   ["ticker", "name", "sector", "mcap", "n_flags", "value_score", "quality_score", "ey",
                    "fscore", "altman_z"] + [f"flag_{k}" for k in SELL_FLAGS], height=360)

# ---------------------------------------------------------------------------
# BUY list
# ---------------------------------------------------------------------------
with tabs[2]:
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
with tabs[3]:
    st.markdown("Each screen acts only on the firms that survive the preceding screen. Firms missing a required "
                "metric are removed at that step and counted separately from the percentile cutoff.")
    st.dataframe(pd.DataFrame(steps)[["step", "chapter", "entering", "missing", "excluded", "remaining"]],
                 hide_index=True, width="stretch")
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
                                barmode="overlay", labels={col: col.replace("_", " "), "outcome": ""},
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
        st.session_state["company"] = sells["ticker"].iloc[0] if len(sells) else tick_opts[0]
    pick = st.selectbox("Choose a company", tick_opts, format_func=label_of.get, key="company")
    r = res.set_index("ticker").loc[pick]
    hist = panels["current"][panels["current"]["cik"] == r["cik"]].sort_values("fy_end")
    badge = {"BUY": "🟢", "SELL": "🔴", "AVOID": "🟠", "HOLD": "⚪"}[r["label"]]
    st.markdown(f"### {badge} {r['name']} ({pick}) — **{r['label']}**")
    st.markdown(f"{r['sector']} · {r['industry']} · Market cap {fmt_money(r['mcap'])} · "
                f"Latest 10-K: fiscal year ended {pd.Timestamp(r['fy_end']).date()}")
    st.info(explain(r, settings))
    if r.get("major_change", False):
        st.warning(f"The latest quarterly balance sheet shows total assets at {r['post_fy_asset_ratio']:.0%} of the "
                   "10-K figure (likely a spin-off, divestiture or merger), so today's price and the 10-K describe "
                   "different companies. Price-based ratios are left blank for this firm.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Ch 2 · Price and quality**")
        st.markdown(
            f"- GPA = (Revenue − COGS) / TA = {fmt_money(r['gross_profit'])} / {fmt_money(r['total_assets'])} = **{fmt(r['gpa'])}**\n"
            f"- BM = Common book equity / Market cap = {fmt_money(r['common_equity'])} / {fmt_money(r['mcap'])} = **{fmt(r['bm'])}**\n"
            f"- Earnings yield = EBIT / TEV = {fmt_money(r['ebit'])} / {fmt_money(r['tev'])} = **{fmt(r['ey'], True)}**\n"
            f"- Greenblatt ROC = EBIT / (Net PPE + NWC) = **{fmt(r['roc_gb'], True)}**\n"
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
        st.markdown(
            f"- Geometric-average ROA **{fmt(r['roa_geo'], True)}**, geometric-average ROC **{fmt(r['roc_geo'], True)}**\n"
            f"- CFOA = Σ FCF / TA = **{fmt(r['cfoa'])}**\n"
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
        fsc["Point"] = fsc["Point"].map(lambda v: "–" if pd.isna(v) else int(v))
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
                     color_discrete_sequence=[SERIES_1, SERIES_2, "#1baf7a"],
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
            st.caption("10-K filed: " + ", ".join(raw["filed"].dt.strftime("%Y-%m-%d")) +
                       ". Gross profit source by year: " + ", ".join(src) +
                       " (yahoo = Yahoo Finance standardized statement, used where SEC data has no gross profit).")


with tabs[4]:
    company_tab()

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
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
    rows, quint = [], []
    for snap in ("bt2024", "bt2025"):
        bm_ = metrics_for(snap, lookback, VERSION)
        bres, _ = run_screen(bm_, dataclasses.replace(settings, min_mcap=max(bt_floor * 1e9, settings.min_mcap)))
        bres = bres.dropna(subset=["fwd_ret"])
        if bres.empty:
            continue
        lo, hi = bres["fwd_ret"].quantile([0.01, 0.99])
        bres["fwd_ret_w"] = bres["fwd_ret"].clip(lo, hi)
        spy = float(bench.loc[bench["snapshot"] == snap, "spy_ret"].iloc[0])
        uni_med = bres["fwd_ret"].median()
        for lab in LABEL_ORDER:
            g_ = bres.loc[bres["label"] == lab]
            rows.append({"Formation": SNAP_LABEL[snap], "Signal": lab, "Firms": len(g_),
                         "Avg return (winsorized)": g_["fwd_ret_w"].mean(), "Median return": g_["fwd_ret"].median(),
                         "Raw mean return": g_["fwd_ret"].mean(),
                         "Beat universe median": (g_["fwd_ret"] > uni_med).mean() if len(g_) else np.nan,
                         "Universe avg (winsorized)": bres["fwd_ret_w"].mean(), "S&P 500 (SPY)": spy})
        quint.append(bres.assign(Formation=SNAP_LABEL[snap]))
    bt = pd.DataFrame(rows)
    if bt.empty:
        st.info("No firms with returns match the current filters.")
        return
    bfig = px.bar(bt, x="Signal", y="Avg return (winsorized)", color="Formation", barmode="group",
                  color_discrete_sequence=[SERIES_1, SERIES_2], text_auto=".1%",
                  category_orders={"Signal": LABEL_ORDER},
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
        st.markdown("**BUY minus SELL spread:** " + " · ".join(f"{k}: **{v:+.1%}**" for k, v in spread.items()))

    st.markdown("#### Does each signal sort future returns?")
    qmetric_label = st.selectbox("Sort firms into quintiles by", [
        "Cheapness percentile (value composite)", "Quality percentile (quality composite)",
        "Overvaluation score", "STA (accruals)", "SNOA", "Altman Z", "F_SCORE", "Franchise percentile",
        "Beneish M-score"])
    qcol = {"Cheapness percentile (value composite)": "value_score", "Quality percentile (quality composite)":
            "quality_score", "Overvaluation score": "overvaluation_score", "STA (accruals)": "sta", "SNOA": "snoa",
            "Altman Z": "altman_z", "F_SCORE": "fscore", "Franchise percentile": "franchise",
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
        "caps use SEC cover-page share counts from the last filing before the formation date. Gross profit "
        "backfilled from Yahoo uses today's statements, a small look-ahead for those firm-years. Two one-year "
        "periods is a short sample: treat this as a sanity check, not proof."
    )


with tabs[5]:
    backtest_tab()

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
with tabs[6]:
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
6. **SELL**: most expensive share of the universe **and** at least *k* red flags. **AVOID**: cheap but removed in
   Step 1 or 2. Everything else is **HOLD**.

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
  revenue differs by more than 3%. Margin trends (ΔMargin, GMI, MG, MS) only compare years from the same source.
- BM and SNOA use common equity (stockholders' equity minus preferred stock).
- Greenblatt ROC and EBIT/TEV are blank when the denominator is zero or negative, except that positive EBIT on a
  zero/negative TEV ranks as the cheapest. Ch 5 ROC treats zero/negative capital as +100% (profitable) or −99%.
- If the latest 10-Q shows total assets below 70% or above 150% of the 10-K figure (spin-off, divestiture or
  merger), price-based ratios are blanked for that firm.
- "Within sector" ranking uses percentile ranks inside each Nasdaq sector (sectors with < 3 valid firms rank
  against the universe).
- Beneish indices that cannot be computed are set to 1 (neutral) and each index is bounded to [0.2, 5].

### Limitations
Accounting screens are signals, not verdicts. SELL means "price looks hard to justify by the fundamentals",
not a prediction that the stock will fall. Financial firms are excluded by default because these ratios do
not describe banks or insurers. This app is for educational use and is not investment advice.
""")
