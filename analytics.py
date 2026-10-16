"""Financial-statement analytics for the ACCT 3343 screening app.

compute_metrics() turns a firm-year panel into one row of course metrics per firm
(Ch 2 magic formula / quality-price, Ch 3 accruals, Ch 4 distress, Ch 5 franchise,
Ch 6 F_SCORE). run_screen() applies the user's interactive cutoffs and labels
each firm BUY / SELL / AVOID / HOLD.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STANDARD_ALTMAN = "Standard (1.2, 1.4, 3.3, 0.6, 1.0)"
ALTMAN_COEFS = {
    STANDARD_ALTMAN: (1.2, 1.4, 3.3, 0.6, 1.0),
    "Course slides (0.012, 0.014, 0.033, 0.006, 0.999)": (0.012, 0.014, 0.033, 0.006, 0.999),
}
BENEISH = {"const": -4.84, "dsri": 0.920, "gmi": 0.528, "aqi": 0.404, "sgi": 0.892,
           "depi": 0.115, "sgai": -0.172, "tata": 4.679, "lvgi": -0.327}
M_SCORE_CUTOFF = -1.78
ADJACENT_DAYS = (330, 400)  # consecutive fiscal years (52/53-week years included)
MAJOR_CHANGE = (0.7, 1.5)  # latest-quarter assets / 10-K assets outside this band -> flagged

# Value measures: higher = cheaper.  Quality measures: higher = better.
VALUE_METRICS = {
    "Earnings yield (EBIT/TEV)": "ey",
    "Book-to-market (BM)": "bm",
    "Earnings-to-price (E/P)": "ep",
    "FCF yield (FCF/Market cap)": "fcfy",
}
QUALITY_METRICS = {
    "Franchise power (Ch 5)": "franchise",
    "F_SCORE (Ch 6)": "fscore",
    "Gross profitability GPA (Ch 2)": "gpa",
    "Greenblatt ROC (Ch 2)": "roc_gb",
}
LABEL_ORDER = ["BUY", "SELL", "AVOID", "HOLD"]
LABEL_COLORS = {"BUY": "#0ca30c", "SELL": "#d03b3b", "AVOID": "#ec835a", "HOLD": "#c3c2b7"}


def _div(a: pd.Series, b: pd.Series, positive_denominator: bool = False) -> pd.Series:
    b = b.where(b > 0) if positive_denominator else b.where(b != 0)
    return (a / b).replace([np.inf, -np.inf], np.nan)


def _norm_cdf(x: pd.Series) -> pd.Series:
    return x.map(lambda v: 0.5 * (1 + math.erf(v / math.sqrt(2))) if pd.notna(v) else np.nan)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(panel: pd.DataFrame, firms: pd.DataFrame, mcap: pd.Series,
                    lookback: int = 8, latest: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per firm.

    mcap   : market cap indexed by cik (today or the formation date).
    latest : optional frame indexed by cik with q_end / q_assets (most recent balance
             sheet from any 10-Q/10-K) used to flag spin-offs and mergers after the 10-K.
    """
    p = panel.sort_values(["cik", "fy_end"]).copy()
    if "gp_source" not in p:
        p["gp_source"] = np.where(p["gross_profit"].notna(), "sec", None)
    g = p.groupby("cik", sort=False)
    p["gap_days"] = g["fy_end"].diff().dt.days
    p["adjacent"] = p["gap_days"].between(*ADJACENT_DAYS)
    p["ta_lag"] = g["total_assets"].shift(1).where(p["adjacent"])
    ta_begin = p["ta_lag"].fillna(p["total_assets"])
    p["gm"] = _div(p["gross_profit"], p["revenue"], positive_denominator=True)
    p["roa_y"] = _div(p["net_income"], p["ta_lag"], positive_denominator=True)
    p["cfo_y"] = _div(p["cfo"], p["ta_lag"], positive_denominator=True)
    p["lev_y"] = _div(p["lt_debt"], p["total_assets"], positive_denominator=True)
    p["liq_y"] = _div(p["current_assets"], p["current_liab"], positive_denominator=True)
    p["turn_y"] = _div(p["revenue"], p["ta_lag"], positive_denominator=True)
    # Ch 5 ROC = EBIT / (debt + equity - cash). Firms with zero/negative invested
    # capital are ranked at the extremes (+100% if profitable, -99% if not) instead of dropped.
    p["capital"] = p["st_debt"] + p["lt_debt"] + p["equity"] - p["cash"]
    roc = _div(p["ebit"], p["capital"], positive_denominator=True)
    roc = roc.where(~(p["capital"] <= 0), np.where(p["ebit"] > 0, 1.0, -0.99))
    p["roc_y"] = roc.where(p["ebit"].notna() & p["capital"].notna()).clip(lower=-0.99)
    p["roa_8"] = _div(p["net_income"], ta_begin, positive_denominator=True).clip(lower=-0.99)
    g = p.groupby("cik", sort=False)  # regroup so the new columns are visible

    cur = g.tail(1).set_index("cik")
    prev = g.nth(-2).set_index("cik").reindex(cur.index)
    # A prior year only counts if it is the immediately preceding fiscal year.
    prev.loc[~cur["adjacent"].fillna(False).to_numpy(dtype=bool)] = np.nan
    m = pd.DataFrame(index=cur.index)
    m["fy_end"] = cur["fy_end"]
    m["filed"] = cur["filed"]
    m["mcap"] = mcap.reindex(cur.index)
    ta, ca, cl = cur["total_assets"], cur["current_assets"], cur["current_liab"]

    # Spin-offs / mergers after the 10-K: price reflects a different company than the statements.
    m["post_fy_asset_ratio"] = np.nan
    if latest is not None:
        lt_ = latest.reindex(cur.index)
        later = lt_["q_end"] > cur["fy_end"]
        m["post_fy_asset_ratio"] = _div(lt_["q_assets"], ta, True).where(later)
    m["major_change"] = ~m["post_fy_asset_ratio"].between(*MAJOR_CHANGE) & m["post_fy_asset_ratio"].notna()

    # ---- Ch 2: magic formula + quality/price --------------------------------
    common_eq = cur["equity"] - cur["preferred"]
    m["common_equity"] = common_eq
    m["nwc"] = ca - cl
    m["roc_gb"] = _div(cur["ebit"], cur["ppe_net"] + m["nwc"], positive_denominator=True)
    m["excess_cash_slides"] = cur["cash"] + ca - cl
    m["excess_cash_gb"] = cur["cash"] - np.maximum(0, cl - (ca - cur["cash"]))
    debt = cur["st_debt"] + cur["lt_debt"]
    for k in ("slides", "gb"):
        tev = m["mcap"] + debt - m[f"excess_cash_{k}"].fillna(cur["cash"]) + cur["preferred"] + cur["minority_interest"]
        m[f"tev_{k}"] = tev
        ey = _div(cur["ebit"], tev, positive_denominator=True)
        # Positive EBIT on a zero/negative TEV is the cheapest possible case, not missing data.
        m[f"ey_{k}"] = ey.mask((tev <= 0) & (cur["ebit"] > 0), 10.0)
    m["gpa"] = _div(cur["gross_profit"], ta, positive_denominator=True)
    m["bm"] = _div(common_eq, m["mcap"], positive_denominator=True)
    m["ep"] = _div(cur["net_income"], m["mcap"], positive_denominator=True)
    m["fcfy"] = _div(cur["fcf"], m["mcap"], positive_denominator=True)

    # ---- Ch 3: accruals --------------------------------------------------------
    m["sta"] = _div(cur["net_income"] - cur["cfo"], ta, positive_denominator=True)
    m["op_assets"] = ta - cur["cash"]
    m["op_liab"] = ta - cur["st_debt"] - cur["lt_debt"] - cur["minority_interest"] - cur["preferred"] - common_eq
    m["snoa"] = _div(m["op_assets"] - m["op_liab"], ta, positive_denominator=True)

    # Beneish PROBM (book Ch 3). Missing component indices are set to 1 (neutral)
    # and every index is bounded to [0.2, 5] so tiny denominators cannot dominate.
    def idx(s: pd.Series) -> pd.Series:
        return s.replace([np.inf, -np.inf], np.nan).clip(0.2, 5)
    same_gp_src = cur["gp_source"] == prev["gp_source"]
    rev, rev0 = cur["revenue"], prev["revenue"]
    comp = pd.DataFrame(index=m.index)
    comp["dsri"] = idx(_div(_div(cur["receivables"], rev, True), _div(prev["receivables"], rev0, True)))
    comp["gmi"] = idx(_div(prev["gm"], cur["gm"], True)).where(same_gp_src)
    hard = lambda d: 1 - _div(d["current_assets"] + d["ppe_net"], d["total_assets"], True)
    comp["aqi"] = idx(_div(hard(cur), hard(prev), True))
    comp["sgi"] = idx(_div(rev, rev0, True))
    dep_rate = lambda d: _div(d["dep_amort"], d["dep_amort"] + d["ppe_net"], True)
    comp["depi"] = idx(_div(dep_rate(prev), dep_rate(cur), True))
    comp["sgai"] = idx(_div(_div(cur["sga"], rev, True), _div(prev["sga"], rev0, True)))
    lvg = lambda d: _div(d["current_liab"] + d["lt_debt"], d["total_assets"], True)
    comp["lvgi"] = idx(_div(lvg(cur), lvg(prev), True))
    comp["tata"] = _div(cur["ni_cont_ops"] - cur["cfo"], ta, True)
    required = comp["sgi"].notna() & comp["tata"].notna()
    comp = comp.fillna({c: 1.0 for c in ("dsri", "gmi", "aqi", "depi", "sgai", "lvgi")})
    mscore = BENEISH["const"] + sum(BENEISH[c] * comp[c] for c in comp.columns)
    m["mscore"] = mscore.where(required)
    m["probm"] = _norm_cdf(m["mscore"])
    for c in comp.columns:
        m[f"b_{c}"] = comp[c].where(required)

    # ---- Ch 4: Altman Z components ---------------------------------------------
    m["x1"] = _div(ca - cl, ta, True)
    m["x2"] = _div(cur["retained_earnings"], ta, True)
    m["x3"] = _div(cur["ebit"], ta, True)
    m["x4"] = _div(m["mcap"], cur["total_liab"], True)
    m["x5"] = _div(rev, ta, True)

    # ---- Ch 5: franchise power over `lookback` consecutive fiscal years --------
    w = g.tail(lookback)
    wg = w.groupby("cik")
    n_years = wg.size()
    consecutive = wg["adjacent"].apply(lambda s: bool(s.iloc[1:].all()))
    full = n_years[(n_years >= lookback) & consecutive.reindex(n_years.index)].index

    def geo_mean(col: str) -> pd.Series:
        lr = np.log1p(w[col])
        s = lr.groupby(w["cik"]).agg(["mean", "count"])
        return (np.exp(s["mean"]) - 1).where(s["count"] >= lookback)

    m["roa_geo"] = geo_mean("roa_8").reindex(m.index)
    m["roc_geo"] = geo_mean("roc_y").reindex(m.index)
    fcf_sum = wg["fcf"].sum(min_count=lookback)
    m["cfoa"] = _div(fcf_sum.reindex(m.index), ta, True)
    gm_ok = wg.apply(lambda d: bool(d["gm"].notna().all() and (d["gm"] > 0).all()
                                    and d["gp_source"].nunique() == 1), include_groups=False)
    gm_first, gm_last = wg["gm"].first(), wg["gm"].last()
    mg = (gm_last / gm_first) ** (1 / max(lookback - 1, 1))  # T = lookback - 1 growth periods
    m["mg"] = mg.where(gm_ok).reindex(m.index)
    gm_stats = wg["gm"].agg(["mean", "std", "count"])
    ms = (gm_stats["mean"] / gm_stats["std"]).replace(np.inf, 1e3)
    m["ms"] = ms.where(gm_ok & (gm_stats["count"] >= lookback)).reindex(m.index)
    for c in ("roa_geo", "roc_geo", "cfoa", "mg", "ms"):
        m[c] = m[c].where(m.index.isin(full))
    m["years_available"] = n_years.reindex(m.index)

    # ---- Ch 6: Piotroski F_SCORE (latest year vs prior year) -------------------
    f = pd.DataFrame(index=m.index)
    roa, cfo_s = cur["roa_y"], cur["cfo_y"]
    f["F_ROA"] = (roa > 0).where(roa.notna())
    f["F_CFO"] = (cfo_s > 0).where(cfo_s.notna())
    d_roa = roa - prev["roa_y"]
    f["F_dROA"] = (d_roa > 0).where(d_roa.notna())
    f["F_ACCRUAL"] = (cfo_s > roa).where(cfo_s.notna() & roa.notna())
    lev, lev0 = cur["lev_y"], prev["lev_y"]
    f["F_dLEVER"] = (lev < lev0).where(lev.notna() & lev0.notna())
    liq, liq0 = cur["liq_y"], prev["liq_y"]
    f["F_dLIQUID"] = (liq > liq0).where(liq.notna() & liq0.notna())
    f["F_EQ_OFFER"] = (cur["net_equity_issued"] <= 0).where(cur["net_equity_issued"].notna())
    d_gm = (cur["gm"] - prev["gm"]).where(same_gp_src)
    f["F_dMARGIN"] = (d_gm > 0).where(d_gm.notna())
    d_turn = cur["turn_y"] - prev["turn_y"]
    f["F_dTURN"] = (d_turn > 0).where(d_turn.notna())
    f = f.astype(float)
    avail = f.notna().sum(axis=1)
    m["fscore_raw"] = f.sum(axis=1, min_count=1)
    m["fscore_avail"] = avail
    m["fscore"] = (m["fscore_raw"] * 9 / avail).round().where(avail >= 7)
    m = m.join(f)
    m["d_roa"] = d_roa

    # Price-based measures are not meaningful when the company changed shape after its 10-K.
    for c in ("ey_slides", "ey_gb", "bm", "ep", "fcfy", "x4"):
        m.loc[m["major_change"], c] = np.nan

    # ---- Reference values for display ------------------------------------------
    for c in ("revenue", "gross_profit", "ebit", "net_income", "cfo", "fcf", "total_assets", "equity",
              "preferred", "gp_source"):
        m[c] = cur[c]
    m["gm"] = cur["gm"]

    info = firms.set_index("cik")[["ticker", "name", "sector", "industry"]]
    return info.join(m, how="inner").reset_index()


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    sectors: list[str]
    min_mcap: float = 0.0  # USD
    relative: bool = False  # rank within sector instead of across the universe
    accrual_cut: float = 0.10
    use_probm: bool = True
    probm_cut: float = 0.05
    altman: str = STANDARD_ALTMAN
    z_raw_cut: float = 0.10
    z_rel_cut: float = 0.10
    z_hard: bool = False
    z_threshold: float = 1.81
    excess_cash: str = "slides"
    value_metrics: list[str] = field(default_factory=lambda: ["ey", "bm"])
    value_keep: float = 0.30
    quality_metrics: list[str] = field(default_factory=lambda: ["franchise", "fscore", "gpa"])
    quality_keep: float = 0.50
    min_fscore: int = 5
    expensive_pct: float = 0.20
    sell_flags: list[str] = field(default_factory=lambda: list(SELL_FLAGS))
    flag_accrual_pct: float = 0.20
    flag_fscore_max: int = 3
    flag_franchise_pct: float = 0.25
    min_flags: int = 2
    sell_profitable_only: bool = False


SELL_FLAGS = {
    "accruals": "High accruals (STA + SNOA in worst x%)",
    "manipulation": "Beneish M-score above -1.78",
    "distress": "Altman Z below distress cutoff",
    "weak_fscore": "Weak F_SCORE (at or below threshold)",
    "weak_franchise": "Weak franchise (bottom x% franchise power)",
    "cash_burn": "Negative free cash flow (latest year)",
}


def pct_rank(s: pd.Series) -> pd.Series:
    """Percentile rank in (0, 1], higher value -> higher rank. NaN stays NaN."""
    return s.rank(method="min", pct=True)


def sector_pct(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentile rank within sector (sectors with < 3 valid firms rank against the universe).

    Ranks are robust to the extreme outliers in ratios such as Altman Z's X4, which make
    sector means meaningless (one firm with Z = 1,000 would move a whole sector).
    """
    s = df[col]
    grp = s.groupby(df["sector"])
    enough = grp.transform("count") >= 3
    return grp.rank(pct=True, method="min").where(enough, s.rank(pct=True, method="min"))


def cut_worst(df: pd.DataFrame, score: pd.Series, pct: float, higher_is_worse: bool) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Remove floor(n x pct) worst firms. Returns (missing, excluded, kept) index sets."""
    valid = score.notna()
    missing = df.index[~valid]
    elig = df.loc[valid].assign(_s=score[valid])
    k = int(math.floor(len(elig) * pct + 1e-9))
    order = elig.sort_values(["_s", "ticker"], ascending=[not higher_is_worse, True])
    return missing, order.index[:k], order.index[k:]


def run_screen(metrics: pd.DataFrame, s: Settings) -> tuple[pd.DataFrame, list[dict]]:
    df = metrics[metrics["sector"].isin(s.sectors) & (metrics["mcap"] >= s.min_mcap)].copy()
    df = df.set_index("ticker", drop=False).rename_axis(None)
    xs = [df[f"x{i}"] for i in range(1, 6)]
    df["altman_z"] = sum(c * x for c, x in zip(ALTMAN_COEFS[s.altman], xs))
    # Distress zones (1.81 / 2.99) are calibrated for the standard weights, so zone tests
    # always use the standard Z even when the slide weights drive the ranking.
    df["altman_z_std"] = sum(c * x for c, x in zip(ALTMAN_COEFS[STANDARD_ALTMAN], xs))
    df["ey"] = df[f"ey_{s.excess_cash}"]
    df["tev"] = df[f"tev_{s.excess_cash}"]
    basis = (lambda c: sector_pct(df, c)) if s.relative else (lambda c: df[c])

    # Universe-wide percentile ranks (used by every step and by the SELL rules).
    df["mm"] = np.fmax(pct_rank(basis("mg")), pct_rank(basis("ms"))).where(df["mg"].notna() & df["ms"].notna())
    fr_parts = pd.concat([pct_rank(basis(c)) for c in ("roa_geo", "roc_geo", "cfoa")] + [df["mm"]], axis=1)
    # Composites follow the course recipe: rank each measure, add the ranks, re-rank.
    df["franchise"] = pct_rank(fr_parts.mean(axis=1, skipna=False))
    value_metrics = s.value_metrics or ["ey"]
    quality_metrics = s.quality_metrics or ["fscore"]
    df["value_score"] = pct_rank(pd.concat([pct_rank(basis(c)) for c in value_metrics], axis=1).mean(axis=1, skipna=False))
    df["quality_score"] = pct_rank(pd.concat([pct_rank(basis(c)) for c in quality_metrics], axis=1).mean(axis=1, skipna=False))
    # Accrual risk: rank STA and SNOA descending (rank 1 = most accrual-heavy) and add ranks,
    # so a SMALL rank sum is the WORST accrual quality.
    df["accrual_rank_sum"] = basis("sta").rank(ascending=False, method="min") + basis("snoa").rank(ascending=False, method="min")
    df["accrual_pct"] = pct_rank(df["accrual_rank_sum"])
    df["z_rel"] = sector_pct(df, "altman_z")
    df["z_rel_step"] = np.nan

    steps: list[dict] = []
    df["stage"] = "Passed all screens"
    alive = df.index

    def record(name: str, chapter: str, metric: str, entering: pd.Index, missing, excluded, kept) -> pd.Index:
        df.loc[missing, "stage"] = f"{name}: missing data"
        df.loc[excluded, "stage"] = f"{name}: excluded"
        steps.append({"step": name, "chapter": chapter, "metric": metric, "entering": len(entering),
                      "missing": len(missing), "excluded": len(excluded), "remaining": len(kept),
                      "entering_idx": list(entering), "excluded_idx": list(excluded)})
        return kept

    # Step 1: manipulation / earnings quality.
    sub = df.loc[alive]
    mi, ex, kept = cut_worst(sub, sub["accrual_rank_sum"], s.accrual_cut, higher_is_worse=False)
    alive = record("1a Accruals (STA + SNOA)", "Ch 3", "accrual_rank_sum", sub.index, mi, ex, kept)
    if s.use_probm:
        sub = df.loc[alive]
        mi, ex, kept = cut_worst(sub, sub["probm"], s.probm_cut, higher_is_worse=True)
        alive = record("1b Beneish PROBM", "Ch 3", "mscore", sub.index, mi, ex, kept)

    # Step 2: financial distress.
    sub = df.loc[alive]
    mi, ex, kept = cut_worst(sub, sub["altman_z"], s.z_raw_cut, higher_is_worse=False)
    if s.z_hard:
        hard = sub.loc[kept].index[sub.loc[kept, "altman_z_std"] < s.z_threshold]
        ex, kept = ex.append(hard), kept.difference(hard, sort=False)
    alive = record("2a Altman Z (raw)", "Ch 4", "altman_z", sub.index, mi, ex, kept)
    sub = df.loc[alive]
    df.loc[sub.index, "z_rel_step"] = sector_pct(sub, "altman_z")
    mi, ex, kept = cut_worst(sub, df.loc[sub.index, "z_rel_step"], s.z_rel_cut, higher_is_worse=False)
    alive = record("2b Altman Z (within sector)", "Ch 4", "z_rel_step", sub.index, mi, ex, kept)

    # Step 3: price - keep the cheapest value_keep share of survivors.
    sub = df.loc[alive]
    mi, ex, kept = cut_worst(sub, sub["value_score"], 1 - s.value_keep, higher_is_worse=False)
    alive = record("3 Cheapness (value composite)", "Ch 2", "value_score", sub.index, mi, ex, kept)

    # Step 4: quality - keep the best quality_keep share, then require a minimum F_SCORE.
    sub = df.loc[alive]
    mi, ex, kept = cut_worst(sub, sub["quality_score"], 1 - s.quality_keep, higher_is_worse=False)
    if s.min_fscore > 0:
        weak = sub.loc[kept].index[~(sub.loc[kept, "fscore"] >= s.min_fscore)]
        ex, kept = ex.append(weak), kept.difference(weak, sort=False)
    alive = record("4 Quality composite + min F_SCORE", "Ch 2, 5, 6", "quality_score", sub.index, mi, ex, kept)

    # ---- SELL rules (overvaluation) -----------------------------------------
    flags = pd.DataFrame(index=df.index)
    on = set(s.sell_flags)
    if "accruals" in on:
        flags["accruals"] = df["accrual_pct"] <= s.flag_accrual_pct
    if "manipulation" in on:
        flags["manipulation"] = df["mscore"] > M_SCORE_CUTOFF
    if "distress" in on:
        flags["distress"] = df["altman_z_std"] < s.z_threshold
    if "weak_fscore" in on:
        flags["weak_fscore"] = df["fscore"] <= s.flag_fscore_max
    if "weak_franchise" in on:
        flags["weak_franchise"] = df["franchise"] <= s.flag_franchise_pct
    if "cash_burn" in on:
        flags["cash_burn"] = df["fcf"] < 0
    flags = flags.fillna(False).astype(bool)
    for c in SELL_FLAGS:
        df[f"flag_{c}"] = flags[c] if c in flags else False
    df["n_flags"] = flags.sum(axis=1) if len(flags.columns) else 0
    df["expensive"] = df["value_score"] <= s.expensive_pct
    if s.sell_profitable_only:
        df["expensive"] &= df["ebit"] > 0
    df["cheap_universe"] = df["value_score"] >= 1 - min(s.value_keep, 0.5)
    df["overvaluation_score"] = 100 * ((1 - df["value_score"].fillna(0.5)) + (1 - df["quality_score"].fillna(0.5))) / 2

    failed_risk = df.index[df["stage"].str.startswith(("1a", "1b", "2a", "2b")) & df["stage"].str.endswith("excluded")]
    sell = df["expensive"] & (df["n_flags"] >= s.min_flags)
    df["label"] = "HOLD"
    df.loc[df["cheap_universe"] & df.index.isin(failed_risk), "label"] = "AVOID"
    df.loc[sell, "label"] = "SELL"
    df.loc[df.index.isin(alive) & ~sell, "label"] = "BUY"
    df["label"] = pd.Categorical(df["label"], categories=LABEL_ORDER)
    return df.reset_index(drop=True), steps


def explain(row: pd.Series, s: Settings) -> str:
    """Plain-English reason for a firm's label."""
    lab = row["label"]
    if lab == "BUY":
        return ("Passed the accrual and distress screens, sits in the cheapest "
                f"{s.value_keep:.0%} of survivors and the top {s.quality_keep:.0%} on quality "
                f"with F_SCORE >= {s.min_fscore}.")
    if lab == "SELL":
        hit = [SELL_FLAGS[c] for c in SELL_FLAGS if row.get(f"flag_{c}", False)]
        return (f"Priced in the most expensive {s.expensive_pct:.0%} of the universe (value score "
                f"{row['value_score']:.2f}) with {int(row['n_flags'])} red flag(s): " + "; ".join(hit) + ".")
    if lab == "AVOID":
        return (f"Looks cheap (value score {row['value_score']:.2f}) but was removed at "
                f"'{row['stage']}' - a potential value trap.")
    note = " Valuation measures are blank because the latest quarterly balance sheet differs sharply from the 10-K (spin-off or merger)." if row.get("major_change", False) else ""
    return f"No BUY or SELL signal. Screen outcome: {row['stage']}.{note}"
