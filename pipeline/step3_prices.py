"""Step 3: prices for the two out-of-sample backtest years (Yahoo Finance).

For each formation date F (2024-10-15, 2025-10-15) this stores
  mcap_<snap> : market cap on F = cover-page shares outstanding from the last
                10-K/10-Q filed by F (SEC) x the unadjusted close on F. Falls back to
                today's market cap x Close(F) / today's price when the SEC share count
                is missing or implausible (flagged in mcap_src_<snap>).
  ret_<snap>  : 12-month total return from F, using dividend-adjusted closes
Outputs: data/returns.parquet, data/benchmark.parquet
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf

from common import DATA, FORMATION_DATES, HOLD_END

PERIODS = {  # snapshot -> (formation date, end of 12-month holding period)
    "bt2024": (FORMATION_DATES["bt2024"], FORMATION_DATES["bt2025"]),
    "bt2025": (FORMATION_DATES["bt2025"], HOLD_END),
}
START = "2023-09-01"  # early enough to cover share counts reported up to a year before formation


def download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {"Close": [], "Adj Close": [], "Stock Splits": []}
    for i in range(0, len(tickers), 150):
        chunk = tickers[i:i + 150]
        for attempt in range(3):
            try:
                df = yf.download(chunk, start=START, end=pd.Timestamp(HOLD_END) + pd.Timedelta(days=2),
                                 auto_adjust=False, actions=True, progress=False, threads=True,
                                 group_by="column")
                for k in parts:
                    parts[k].append(df[k])
                break
            except Exception as e:  # transient Yahoo errors / rate limits
                print(f"  chunk {i} attempt {attempt + 1} failed: {e}")
                time.sleep(10)
        print(f"  downloaded {min(i + 150, len(tickers)):,}/{len(tickers):,}", flush=True)
        time.sleep(1.5)
    out = {}
    for k, lst in parts.items():
        df = pd.concat(lst, axis=1)
        out[k] = df.loc[:, ~df.columns.duplicated()]
    return out


def price_on(px: pd.DataFrame, date) -> pd.Series:
    """Last close on or before `date`; NaN if the stock had no trade in the prior 7 days."""
    d = pd.Timestamp(date)
    win = px.loc[d - pd.Timedelta(days=7):d]
    return win.ffill().iloc[-1] if len(win) else pd.Series(index=px.columns, dtype=float)


def split_factor(splits: pd.DataFrame, after, upto=None) -> pd.DataFrame:
    """Product of split ratios with ex-date in (after, upto] for every ticker."""
    s = splits.loc[pd.Timestamp(after) + pd.Timedelta(days=1):upto].replace(0, np.nan)
    return s.prod(min_count=1).fillna(1.0)


def main() -> None:
    firms = pd.read_parquet(DATA / "firms.parquet")
    lf = pd.read_parquet(DATA / "latest_filing.parquet")
    tickers = firms["ticker"].tolist()
    px = download(tickers + ["SPY"])
    close, adj, splits = px["Close"], px["Adj Close"], px["Stock Splits"].fillna(0.0)

    out = firms[["ticker", "cik", "mcap", "price"]].set_index("ticker")
    for snap, (f, e) in PERIODS.items():
        c_f = price_on(close, f).reindex(out.index)
        fallback = out["mcap"] * c_f / out["price"]
        # Yahoo closes are split-adjusted to today's basis: undo splits after F.
        unadj_f = c_f * split_factor(splits, f).reindex(out.index).fillna(1.0)
        sh = lf[lf["snapshot"] == snap].set_index("cik")[["shares", "shares_end"]]
        sh = sh.reindex(out["cik"]).set_index(out.index)
        # Shares are as of the cover date; adjust for splits between that date and F.
        adj_sh = []
        for t, row in sh.iterrows():
            if pd.isna(row["shares"]) or t not in splits.columns:
                adj_sh.append(np.nan)
                continue
            s = splits[t].loc[row["shares_end"] + pd.Timedelta(days=1):f].replace(0, np.nan)
            adj_sh.append(row["shares"] * (s.prod() if s.notna().any() else 1.0))
        sec_mcap = pd.Series(adj_sh, index=out.index) * unadj_f
        ratio = sec_mcap / fallback
        ok = sec_mcap.notna() & ratio.between(0.5, 2.0)
        out[f"mcap_{snap}"] = sec_mcap.where(ok, fallback)
        out[f"mcap_src_{snap}"] = np.where(ok, "sec_shares", np.where(fallback.notna(), "price_ratio", None))
        out[f"ret_{snap}"] = (price_on(adj, e) / price_on(adj, f) - 1).reindex(out.index)
        print(f"{snap}: {int(ok.sum())} firms use SEC share counts, "
              f"{int((~ok & fallback.notna()).sum())} use the price-ratio fallback")
    out = out.drop(columns=["mcap", "price", "cik"]).reset_index()
    out.to_parquet(DATA / "returns.parquet", index=False)

    bench = pd.DataFrame([
        {"snapshot": snap, "formation": f, "end": e,
         "spy_ret": float(price_on(adj[["SPY"]], e).iloc[0] / price_on(adj[["SPY"]], f).iloc[0] - 1)}
        for snap, (f, e) in PERIODS.items()
    ])
    bench.to_parquet(DATA / "benchmark.parquet", index=False)
    print(bench.to_string(index=False))
    for snap in PERIODS:
        print(f"{snap}: returns for {out[f'ret_{snap}'].notna().sum():,} of {len(out):,} firms")


if __name__ == "__main__":
    main()
