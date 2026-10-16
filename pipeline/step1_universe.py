"""Step 1: candidate universe = every Nasdaq-screener stock that maps to an SEC CIK.

Output: _cache/candidates.parquet (one row per company, ranked by market cap).
"""
from __future__ import annotations

import re

import pandas as pd
import requests

from common import CACHE, NASDAQ_HEADERS, SEC_HEADERS

# Listings that are not a company's common equity.
NON_COMMON = re.compile(
    r"\b(?:notes?|bonds?|debentures?|preferred|pfd|depositary|warrants?|units?|rights?|"
    r"subordinated|trust preferred|series [a-z]\b|due \d{4})|%",
    re.IGNORECASE,
)

# Rough SIC -> Nasdaq-sector mapping, used only when Nasdaq leaves the sector blank.
def sic_to_sector(sic: int | None) -> str:
    if sic is None or pd.isna(sic):
        return "Miscellaneous"
    s = int(sic)
    if 6500 <= s <= 6599 or s == 6798:
        return "Real Estate"
    if 6000 <= s <= 6799:
        return "Finance"
    if 4900 <= s <= 4999:
        return "Utilities"
    if 4800 <= s <= 4899:
        return "Telecommunications"
    if 1300 <= s <= 1399 or 2900 <= s <= 2999:
        return "Energy"
    if 2830 <= s <= 2836 or 3841 <= s <= 3851 or 8000 <= s <= 8099:
        return "Health Care"
    if 3570 <= s <= 3579 or 3670 <= s <= 3679 or 7370 <= s <= 7379:
        return "Technology"
    if 2000 <= s <= 2199 or 5400 <= s <= 5499:
        return "Consumer Staples"
    if 1000 <= s <= 1499 or 2600 <= s <= 2699 or 2800 <= s <= 2899 or 3300 <= s <= 3399:
        return "Basic Materials"
    if 5000 <= s <= 5999 or 7000 <= s <= 7999 or 2300 <= s <= 2399 or 3700 <= s <= 3719:
        return "Consumer Discretionary"
    return "Industrials"


def money(x: str) -> float:
    try:
        return float(str(x).replace("$", "").replace(",", ""))
    except ValueError:
        return float("nan")


def main() -> None:
    r = requests.get(
        "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true",
        headers=NASDAQ_HEADERS,
        timeout=120,
    )
    r.raise_for_status()
    nq = pd.DataFrame(r.json()["data"]["rows"])
    nq["ticker"] = nq["symbol"].str.strip().str.replace("/", "-", regex=False)
    nq["mcap"] = nq["marketCap"].map(money)
    nq["price"] = nq["lastsale"].map(money)
    nq["volume"] = pd.to_numeric(nq["volume"], errors="coerce")
    nq = nq[~nq["name"].fillna("").str.contains(NON_COMMON)]
    nq = nq[(nq["mcap"] > 0) & (nq["price"] > 0)]

    s = requests.get("https://www.sec.gov/files/company_tickers_exchange.json", headers=SEC_HEADERS, timeout=120)
    s.raise_for_status()
    js = s.json()
    sec = pd.DataFrame(js["data"], columns=js["fields"])
    sec["ticker"] = sec["ticker"].str.upper().str.replace(".", "-", regex=False)

    df = nq.merge(sec[["ticker", "cik", "exchange"]], on="ticker", how="inner")
    # Dual-class firms (GOOG/GOOGL): Nasdaq reports company-level market cap on
    # every class, so keep a single row per CIK - the most traded line.
    df["dollar_vol"] = df["volume"] * df["price"]
    df = df.sort_values("dollar_vol", ascending=False).drop_duplicates("cik")
    df = df.sort_values("mcap", ascending=False).reset_index(drop=True)

    df["sector"] = df["sector"].replace("", pd.NA)
    df["industry"] = df["industry"].replace("", pd.NA)
    df["name"] = (
        df["name"]
        .str.replace(r"\s+(Class [A-Z] )?(Common|Ordinary|Capital) (Stock|Shares).*$", "", regex=True)
        .str.strip()
    )
    out = df[["ticker", "cik", "name", "exchange", "sector", "industry", "country", "mcap", "price"]].head(3500)
    out.to_parquet(CACHE / "candidates.parquet", index=False)
    print(f"Nasdaq rows: {len(nq):,}  mapped to CIK: {len(df):,}  kept top: {len(out):,}")
    print("blank sectors in kept rows:", int(out["sector"].isna().sum()))


if __name__ == "__main__":
    main()
