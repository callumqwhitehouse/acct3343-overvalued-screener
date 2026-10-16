"""Shared paths and HTTP helpers for the offline data pipeline.

The pipeline runs once (locally) and writes small parquet files into ../data,
which the Streamlit app reads. The app itself never calls a data API.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = Path(os.environ.get("APP_CACHE", ROOT / "pipeline" / "_cache"))
DATA.mkdir(exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# SEC asks automated clients to identify themselves with a descriptive User-Agent.
SEC_HEADERS = {
    "User-Agent": "ACCT3343 Analysis App research contact@acct3343app.com",
    "Accept-Encoding": "gzip, deflate",
}
NASDAQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

# Point-in-time snapshots. "current" uses every filing available today; the two
# backtest snapshots only use facts filed on or before the formation date.
FORMATION_DATES = {"bt2024": "2024-10-15", "bt2025": "2025-10-15"}
HOLD_END = "2026-10-14"

# How many operating (non-financial, non-real-estate) firms to keep.
N_OPERATING = 1500
