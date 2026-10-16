# Overvalued Stock Screener (ACCT 3343 Analysis App)

An interactive Streamlit app that applies the course's financial-statement analysis techniques to about
2,000 U.S. public companies and flags potential **SELL** (overvalued), **BUY**, and **AVOID** (value-trap)
candidates. Every result updates as you move the filters.

| Course chapter | Technique in the app |
|---|---|
| Ch 2 Magic formula & variations | Greenblatt ROC, EBIT/TEV earnings yield, GPA, book-to-market |
| Ch 3 Manipulators & frauds | Scaled total accruals (STA), scaled net operating assets (SNOA), Beneish PROBM |
| Ch 4 Distress | Altman Z (standard or slide coefficients), raw and sector-relative |
| Ch 5 Franchise | Geometric-average ROA and ROC, CFOA, margin growth, margin stability, MaxMargin |
| Ch 6 Financial strength | Piotroski F_SCORE (9 signals) |

The app also runs an **out-of-sample backtest** that applies the current settings to point-in-time data
from 2024-10-15 and 2025-10-15 and reports the following 12-month returns of each signal group.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the data (optional)

The app reads the parquet files in `data/`. To refresh them:

```bash
pip install -r pipeline/requirements.txt
cd pipeline
python step1_universe.py                      # Nasdaq screener -> SEC CIK map
python step2_fundamentals.py companyfacts.zip # SEC bulk XBRL: https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
python step3_prices.py                        # Yahoo Finance prices for the backtest
```

Educational project; not investment advice.
