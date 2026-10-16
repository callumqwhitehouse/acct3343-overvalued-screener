"""Step 4: fill gross profit that SEC company facts cannot provide.

Many large firms (e.g. TMO, GD, RTX) tag cost of sales only by product/service
dimension; SEC's company-facts API drops dimensional facts, so their gross
profit is blank. For those firm-years this step uses Yahoo Finance's
standardized annual income statement (the data source used in class), matched
to the 10-K fiscal-year end within 15 days. When Yahoo's revenue (gross profit +
cost of revenue) differs from the SEC revenue by more than 3%, Yahoo's gross
margin is applied to the SEC revenue instead. Filled rows get gp_source="yahoo".
Yahoo serves today's statements, so backfilled years in the backtest snapshots
may include later restatements (a small look-ahead, noted in the app).

Always starts from the untouched step-2 panels in _cache/, so it can be re-run.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf

from common import CACHE, DATA

SNAPSHOTS = ("current", "bt2025", "bt2024")
CACHE_FILE = CACHE / "yahoo_gross_profit.parquet"


def fetch(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers):
        for attempt in range(3):
            try:
                inc = yf.Ticker(t).income_stmt
                break
            except Exception as e:
                print(f"  {t} attempt {attempt + 1}: {e}")
                time.sleep(5)
        else:
            continue
        if inc is None or inc.empty:
            continue
        for end in inc.columns:
            col = inc[end]
            gp = col.get("Gross Profit")
            if gp is None or pd.isna(gp):
                continue
            cogs = col.get("Cost Of Revenue")
            rows.append({"ticker": t, "period_end": pd.Timestamp(end), "gross_profit_yf": float(gp),
                         "cogs_yf": float(cogs) if cogs is not None and pd.notna(cogs) else np.nan})
        if (i + 1) % 50 == 0:
            print(f"  fetched {i + 1}/{len(tickers)}", flush=True)
        time.sleep(0.4)
    return pd.DataFrame(rows)


def main() -> None:
    firms = pd.read_parquet(DATA / "firms.parquet")
    panels = {k: pd.read_parquet(CACHE / f"panel_{k}_sec.parquet") for k in SNAPSHOTS}
    need = set()
    for p in panels.values():
        recent = p.sort_values("fy_end").groupby("cik").tail(5)
        need |= set(recent.loc[recent["gross_profit"].isna(), "cik"])
    tick = firms[firms["cik"].isin(need)][["cik", "ticker"]]
    print(f"{len(tick)} firms have recent years without gross profit")

    cached = pd.read_parquet(CACHE_FILE) if CACHE_FILE.exists() else pd.DataFrame(columns=["ticker"])
    todo = sorted(set(tick["ticker"]) - set(cached["ticker"]))
    if todo:
        print(f"fetching {len(todo)} tickers from Yahoo")
        cached = pd.concat([cached, fetch(todo)], ignore_index=True)
        cached.to_parquet(CACHE_FILE, index=False)
    yf_gp = cached.merge(tick, on="ticker")
    yf_gp["period_end"] = yf_gp["period_end"].astype("datetime64[ns]")
    print(f"Yahoo gross profit rows: {len(yf_gp)} for {yf_gp['ticker'].nunique()} firms")

    for k, p in panels.items():
        p = p.copy()
        p["fy_end"] = p["fy_end"].astype("datetime64[ns]")
        p["gp_source"] = pd.Series("sec", index=p.index).where(p["gross_profit"].notna())
        m = pd.merge_asof(
            p.reset_index().sort_values("fy_end"), yf_gp.sort_values("period_end"),
            left_on="fy_end", right_on="period_end", by="cik", tolerance=pd.Timedelta(days=15),
            direction="nearest",
        ).set_index("index").sort_index()
        fill = p["gross_profit"].isna() & m["gross_profit_yf"].notna() & (p["revenue"] > 0)
        implied_rev = m["gross_profit_yf"] + m["cogs_yf"]
        consistent = ((implied_rev / p["revenue"]) - 1).abs() <= 0.03
        # Rescale only small perimeter differences; a large gap means Yahoo's statement covers a
        # different business (e.g. restated for discontinued operations), so the year stays blank.
        rescale = fill & ~consistent & (implied_rev > 0) & (p["revenue"] / implied_rev).between(2 / 3, 1.5)
        as_is = fill & (consistent | m["cogs_yf"].isna())
        p.loc[as_is, "gross_profit"] = m.loc[as_is, "gross_profit_yf"]
        p.loc[rescale, "gross_profit"] = (m["gross_profit_yf"] / implied_rev * p["revenue"])[rescale]
        p.loc[as_is | rescale, "gp_source"] = "yahoo"
        p.to_parquet(DATA / f"panel_{k}.parquet", index=False)
        print(f"panel_{k}: filled {int((as_is | rescale).sum())} firm-years ({int(rescale.sum())} rescaled to SEC "
              f"revenue); gross profit still missing in {p['gross_profit'].isna().mean():.1%} of rows")


if __name__ == "__main__":
    main()
