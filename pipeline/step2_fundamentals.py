"""Step 2: annual fundamentals from SEC EDGAR XBRL (bulk companyfacts.zip).

For every candidate (largest first) this builds a firm-year panel of 10-K data.
Three point-in-time snapshots are produced:
  current : every fact filed up to today (restated values win)
  bt2025  : only facts filed on or before 2025-10-15
  bt2024  : only facts filed on or before 2024-10-15
The backtest snapshots therefore contain only information an investor could
actually have had on the formation date (no look-ahead).

Usage: python step2_fundamentals.py <path to companyfacts.zip>
Outputs: data/firms.parquet, data/panel_<snapshot>.parquet (+ an untouched copy in
_cache/ that step 4 starts from), data/latest_filing.parquet, data/meta.json
"""
from __future__ import annotations

import json
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from common import CACHE, DATA, FORMATION_DATES, N_OPERATING, SEC_HEADERS
from step1_universe import sic_to_sector

# field -> (period type, XBRL tags in priority order). The first tag with a
# value for a given fiscal year wins, so firms that switched tags (e.g.
# SalesRevenueNet -> RevenueFromContractWithCustomer... in 2018) stay continuous.
FIELDS: dict[str, tuple[str, list[str]]] = {
    "revenue": ("flow", [
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
        "SalesRevenueGoodsNet", "SalesRevenueServicesNet", "RevenuesNetOfInterestExpense",
        "RegulatedAndUnregulatedOperatingRevenue", "OperatingRevenue", "OperatingLeaseLeaseIncome"]),
    "cogs": ("flow", [
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"]),
    "gross_profit_tag": ("flow", ["GrossProfit"]),
    "sga": ("flow", ["SellingGeneralAndAdministrativeExpense"]),
    "ga": ("flow", ["GeneralAndAdministrativeExpense"]),
    "sm": ("flow", ["SellingAndMarketingExpense", "MarketingAndAdvertisingExpense"]),
    "op_income": ("flow", ["OperatingIncomeLoss"]),
    "pretax": ("flow", [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"]),
    "interest_exp": ("flow", ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt",
                              "InterestAndDebtExpense"]),
    "net_income": ("flow", ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]),
    "ni_cont_ops": ("flow", ["IncomeLossFromContinuingOperations"]),
    "dep_amort": ("flow", ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                           "DepreciationAmortizationAndAccretionNet", "Depreciation",
                           "DepreciationNonproduction"]),
    "cfo": ("flow", ["NetCashProvidedByUsedInOperatingActivities",
                     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]),
    "capex": ("flow", ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
                       "PaymentsForCapitalImprovements", "PaymentsToAcquireOtherPropertyPlantAndEquipment",
                       "PaymentsToAcquireOtherProductiveAssets", "PaymentsToAcquireOilAndGasPropertyAndEquipment",
                       "PaymentsToAcquireOilAndGasProperty", "PaymentsToExploreAndDevelopOilAndGasProperties",
                       "PaymentsToAcquireMachineryAndEquipment"]),
    "equity_issued": ("flow", ["ProceedsFromIssuanceOfCommonStock", "ProceedsFromIssuanceOrSaleOfEquity"]),
    "equity_repurchased": ("flow", ["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity"]),
    "total_assets": ("stock", ["Assets", "LiabilitiesAndStockholdersEquity"]),
    "current_assets": ("stock", ["AssetsCurrent"]),
    "current_liab": ("stock", ["LiabilitiesCurrent"]),
    "cash": ("stock", ["CashAndCashEquivalentsAtCarryingValue",
                       "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"]),
    "receivables": ("stock", ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"]),
    "inventory": ("stock", ["InventoryNet"]),
    "ppe_net": ("stock", ["PropertyPlantAndEquipmentNet",
                          "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization"]),
    "retained_earnings": ("stock", ["RetainedEarningsAccumulatedDeficit"]),
    "liabilities_tag": ("stock", ["Liabilities"]),
    "liab_and_equity": ("stock", ["LiabilitiesAndStockholdersEquity"]),
    "equity_tag": ("stock", ["StockholdersEquity", "PartnersCapital"]),
    "equity_incl_nci": ("stock", ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                                  "PartnersCapitalIncludingPortionAttributableToNoncontrollingInterest"]),
    "minority_interest_tag": ("stock", ["MinorityInterest"]),
    "preferred_tag": ("stock", ["PreferredStockValue", "PreferredStockValueOutstanding"]),
    "debt_current_tag": ("stock", ["DebtCurrent"]),
    "ltd_current": ("stock", ["LongTermDebtAndCapitalLeaseObligationsCurrent", "LongTermDebtCurrent",
                              "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"]),
    "st_borrowings": ("stock", ["ShortTermBorrowings", "CommercialPaper", "NotesPayableCurrent",
                                "ConvertibleNotesPayableCurrent"]),
    "ltd_noncurrent": ("stock", [
        "LongTermDebtAndCapitalLeaseObligations", "LongTermDebtNoncurrent", "LongTermNotesPayable",
        "LongTermNotesAndLoans", "SeniorLongTermNotes", "UnsecuredLongTermDebt", "SecuredLongTermDebt",
        "ConvertibleLongTermNotesPayable", "ConvertibleDebtNoncurrent", "LongTermLineOfCredit",
        "OtherLongTermDebtNoncurrent"]),
    "ltd_total": ("stock", ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
                            "DebtAndCapitalLeaseObligations", "SeniorNotes", "ConvertibleNotesPayable",
                            "NotesPayable"]),
    "debt_all": ("stock", ["DebtLongtermAndShorttermCombinedAmount"]),
}
TAG_KIND = {t: kind for kind, tags in FIELDS.values() for t in tags}
N_YEARS = 12  # fiscal years kept per firm and snapshot
SNAPSHOTS = {"current": None, **FORMATION_DATES}
# A snapshot's latest 10-K must be reasonably fresh to be used.
MAX_STALENESS_DAYS = 640
# Registrants that moved to a new CIK (e.g. a new holding company) keep their
# history under the old one; both files are read.
CIK_PREDECESSOR = {2115436: 34088}  # Exxon Mobil


def fact_records(j: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(annual 10-K facts for the mapped tags, post-fiscal-year facts for Assets and shares)."""
    gaap = j.get("facts", {}).get("us-gaap", {})
    rows = []
    for tag, kind in TAG_KIND.items():
        units = gaap.get(tag, {}).get("units", {})
        for f in units.get("USD", []):
            if not str(f.get("form", "")).startswith("10-K"):
                continue
            end = f.get("end")
            if kind == "flow":
                start = f.get("start")
                if not start:
                    continue
                days = (np.datetime64(end) - np.datetime64(start)).astype(int)
                if not 340 <= days <= 380:
                    continue
            elif f.get("start"):
                continue
            rows.append((tag, end, f.get("filed"), float(f["val"])))
    df = pd.DataFrame(rows, columns=["tag", "end", "filed", "val"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])

    # Latest balance sheet from any 10-Q/10-K and the cover-page share count.
    extra = []
    for f in gaap.get("Assets", {}).get("units", {}).get("USD", []):
        if str(f.get("form", "")).startswith(("10-Q", "10-K")) and not f.get("start"):
            extra.append(("q_assets", f["end"], f["filed"], f.get("accn"), float(f["val"])))
    dei = j.get("facts", {}).get("dei", {})
    for f in dei.get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", []):
        if str(f.get("form", "")).startswith(("10-Q", "10-K")):
            extra.append(("shares", f["end"], f["filed"], f.get("accn"), float(f["val"])))
    ex = pd.DataFrame(extra, columns=["item", "end", "filed", "accn", "val"])
    ex["end"] = pd.to_datetime(ex["end"])
    ex["filed"] = pd.to_datetime(ex["filed"])
    return df, ex


def latest_filing(ex: pd.DataFrame, asof: pd.Timestamp) -> dict:
    """Most recent total assets (any 10-Q/10-K) and cover-page shares known on `asof`."""
    ex = ex[ex["filed"] <= asof]
    out = {"q_end": pd.NaT, "q_assets": np.nan, "shares": np.nan, "shares_end": pd.NaT}
    qa = ex[ex["item"] == "q_assets"].sort_values(["end", "filed"])
    if len(qa):
        out["q_end"], out["q_assets"] = qa["end"].iloc[-1], qa["val"].iloc[-1]
    sh = ex[ex["item"] == "shares"]
    if len(sh):
        last = sh.sort_values(["filed", "end"]).iloc[-1]
        same = sh[(sh["accn"] == last["accn"]) & (sh["end"] == last["end"])]
        out["shares"], out["shares_end"] = same["val"].sum(), last["end"]  # sum share classes
    return out


def build_panel(rec: pd.DataFrame, asof: pd.Timestamp | None, sector: str | None) -> pd.DataFrame | None:
    if asof is not None:
        rec = rec[rec["filed"] <= asof]
    if rec.empty:
        return None
    # Latest filing wins for each (tag, period end): restatements replace originals.
    rec = rec.sort_values("filed").drop_duplicates(["tag", "end"], keep="last")
    wide = rec.pivot(index="end", columns="tag", values="val")
    filed = rec.groupby("end")["filed"].max()

    out = pd.DataFrame(index=wide.index)
    for field, (_, tags) in FIELDS.items():
        cols = [t for t in tags if t in wide.columns]
        if not cols:
            out[field] = np.nan
        elif field == "revenue":
            # A zero or segment-sized 'Revenues' must not hide the full top line.
            vals = wide[cols].where(wide[cols] > 0)
            first, mx = vals.bfill(axis=1).iloc[:, 0], vals.max(axis=1)
            out[field] = first.where(first >= 0.5 * mx, mx)
        else:
            out[field] = wide[cols].bfill(axis=1).iloc[:, 0]

    # Fiscal-year ends: a balance sheet and an income/cash-flow statement on the same date.
    flows_present = out[["revenue", "net_income", "cfo", "op_income"]].notna().any(axis=1)
    out = out[out["total_assets"].notna() & flows_present].sort_index()
    if out.empty:
        return None
    # Drop transition-period ends that sit < 300 days before the next fiscal-year end.
    ends = out.index.to_series()
    gap_to_next = ends.shift(-1) - ends
    out = out[~(gap_to_next < pd.Timedelta(days=300))]
    out = out.tail(N_YEARS).copy()
    out["filed"] = filed.reindex(out.index).values
    return derive(out, sector)


def derive(p: pd.DataFrame, sector: str | None) -> pd.DataFrame:
    """Turn raw tag columns into the line items the course formulas use."""
    zero = lambda s: s.fillna(0.0)  # structural absence (no tag reported) -> 0
    # Utilities and oil & gas often tag only part of their operating costs as
    # cost of revenue; a derived margin from a partial COGS is not a gross margin.
    min_cogs_share = 0.2 if sector in ("Utilities", "Energy") else 0.0
    derived_gp = (p["revenue"] - p["cogs"]).where(p["cogs"] >= min_cogs_share * p["revenue"])
    p["gross_profit"] = p["gross_profit_tag"].fillna(derived_gp)
    p["sga"] = p["sga"].fillna(p["ga"] + p["sm"]).fillna(p["ga"])
    p["ebit"] = p["op_income"].fillna(p["pretax"] + zero(p["interest_exp"]))
    p["minority_interest"] = zero(p["minority_interest_tag"]).where(
        p["minority_interest_tag"].notna() | p["equity_tag"].notna())
    p["preferred"] = zero(p["preferred_tag"])
    p["equity"] = p["equity_tag"].fillna(p["equity_incl_nci"] - p["minority_interest"])
    eq_total = p["equity_incl_nci"].fillna(p["equity"] + p["minority_interest"])
    p["total_liab"] = p["liabilities_tag"].fillna(p["liab_and_equity"] - eq_total).fillna(
        p["total_assets"] - eq_total)

    # Debt. Missing tags mean "no debt" only when interest expense is immaterial.
    ltd_cur = zero(p["ltd_current"])
    stb = zero(p["st_borrowings"])
    stb = stb.where((stb - ltd_cur).abs() > 5e5, 0.0)  # same balance-sheet line tagged twice
    st = p["debt_current_tag"].fillna(ltd_cur + stb)
    p["st_debt"] = st.where(~(st > p["current_liab"]), p["current_liab"])
    lt = (p["ltd_noncurrent"].fillna(p["ltd_total"] - ltd_cur)
          .fillna(p["debt_all"] - p["st_debt"]).clip(lower=0))
    material_interest = p["interest_exp"].abs() > 0.005 * p["total_assets"]
    p["lt_debt"] = lt.fillna(0.0).where(lt.notna() | ~material_interest)

    # Capex. Missing capex is 0 only for firms with immaterial depreciation.
    material_da = p["dep_amort"] > 0.01 * p["total_assets"]
    p["capex"] = p["capex"].abs().fillna(0.0).where(p["capex"].notna() | ~material_da)
    p["fcf"] = p["cfo"] - p["capex"]
    p["net_equity_issued"] = zero(p["equity_issued"]) - zero(p["equity_repurchased"])
    p["ni_cont_ops"] = p["ni_cont_ops"].fillna(p["net_income"])
    keep = [
        "filed", "revenue", "cogs", "gross_profit", "sga", "ebit", "interest_exp", "net_income",
        "ni_cont_ops", "dep_amort", "cfo", "capex", "fcf", "net_equity_issued", "total_assets",
        "current_assets", "current_liab", "cash", "receivables", "inventory", "ppe_net",
        "retained_earnings", "total_liab", "equity", "minority_interest", "preferred", "st_debt",
        "lt_debt",
    ]
    return p[keep]


def fetch_sic(cik: int) -> tuple[int | None, str | None]:
    try:
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=SEC_HEADERS, timeout=30)
        r.raise_for_status()
        j = r.json()
        return int(j.get("sic") or 0) or None, j.get("sicDescription")
    except Exception:
        return None, None
    finally:
        time.sleep(0.15)


def read_company(z: zipfile.ZipFile, names: set[str], cik: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    recs, exs = [], []
    for c in (CIK_PREDECESSOR.get(cik), cik):
        fn = f"CIK{c:010d}.json" if c else None
        if fn and fn in names:
            r, e = fact_records(json.loads(z.read(fn)))
            recs.append(r)
            exs.append(e)
    if not recs:
        return None
    return pd.concat(recs, ignore_index=True), pd.concat(exs, ignore_index=True)


def main(zip_path: str) -> None:
    cand = pd.read_parquet(CACHE / "candidates.parquet")
    today = pd.Timestamp.today().normalize()
    asofs = {k: (pd.Timestamp(v) if v else None) for k, v in SNAPSHOTS.items()}
    panels: dict[str, list[pd.DataFrame]] = {k: [] for k in SNAPSHOTS}
    filings = []
    kept, reasons = [], {"no_xbrl_file": 0, "no_10k_data": 0, "stale": 0}
    n_operating = 0
    t0 = time.time()

    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for i, row in enumerate(cand.itertuples(index=False)):
            if n_operating >= N_OPERATING:
                break
            got = read_company(z, names, int(row.cik))
            if got is None:
                reasons["no_xbrl_file"] += 1
                continue
            rec, ex = got
            sector = row.sector if isinstance(row.sector, str) else None
            snap = {k: build_panel(rec, a, sector) for k, a in asofs.items()} if not rec.empty else {}
            cur = snap.get("current")
            if cur is None or cur.empty or cur["net_income"].isna().iloc[-1]:
                reasons["no_10k_data"] += 1
                if row.mcap > 1e10:
                    print(f"  skipped large firm {row.ticker} (${row.mcap / 1e9:.0f}B): no usable 10-K data")
                continue
            if (today - cur.index[-1]).days > MAX_STALENESS_DAYS:
                reasons["stale"] += 1
                continue
            for k, p in snap.items():
                a = asofs[k] or today
                if p is None or p.empty or (a - p.index[-1]).days > MAX_STALENESS_DAYS:
                    continue
                panels[k].append(p.reset_index(names="fy_end").assign(cik=int(row.cik)))
                filings.append({"cik": int(row.cik), "snapshot": k, **latest_filing(ex, a)})
            kept.append(row._asdict() | {"fy_end": cur.index[-1], "filed": cur["filed"].iloc[-1]})
            if row.sector not in ("Finance", "Real Estate"):
                n_operating += 1
            if len(kept) % 200 == 0:
                print(f"  {len(kept):,} firms kept ({n_operating:,} operating) after {i + 1:,} candidates, "
                      f"{time.time() - t0:.0f}s", flush=True)

    firms = pd.DataFrame(kept)
    blank = firms["sector"].isna() | firms["industry"].isna()
    print(f"Filling {int(blank.sum())} blank sectors/industries from SEC SIC codes...")
    for idx in firms.index[blank]:
        sic, desc = fetch_sic(int(firms.at[idx, "cik"]))
        if pd.isna(firms.at[idx, "sector"]):
            firms.at[idx, "sector"] = sic_to_sector(sic)
        if pd.isna(firms.at[idx, "industry"]):
            firms.at[idx, "industry"] = (desc or "Unclassified").title()
    firms["cik"] = firms["cik"].astype(int)
    firms.to_parquet(DATA / "firms.parquet", index=False)
    pd.DataFrame(filings).to_parquet(DATA / "latest_filing.parquet", index=False)

    for k, lst in panels.items():
        df = pd.concat(lst, ignore_index=True)
        df["fy_end"] = df["fy_end"].astype("datetime64[ns]")
        cols = ["cik", "fy_end"] + [c for c in df.columns if c not in ("cik", "fy_end")]
        df[cols].to_parquet(CACHE / f"panel_{k}_sec.parquet", index=False)
        df[cols].to_parquet(DATA / f"panel_{k}.parquet", index=False)
        print(f"panel_{k}: {df['cik'].nunique():,} firms, {len(df):,} firm-years")
    meta = {
        "generated": today.strftime("%Y-%m-%d"),
        "n_firms": int(len(firms)),
        "n_operating": int(n_operating),
        "min_mcap": float(firms["mcap"].min()),
        "fy_end_min": str(firms["fy_end"].min().date()),
        "fy_end_max": str(firms["fy_end"].max().date()),
        "skipped": reasons,
        "snapshots": {k: (v or "current") for k, v in SNAPSHOTS.items()},
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"firms kept: {len(firms):,} ({n_operating:,} operating); skipped: {reasons}")
    print(f"smallest market cap kept: ${firms['mcap'].min() / 1e9:.2f}B; elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
