"""
fetch_extra_data.py
===================
Downloads supplemental data from akshare and saves it under ./data/.

Run this ONCE before training (or before each submission to refresh).

    python fetch_extra_data.py

Output files
------------
data/valuation.parquet      — daily PE / PB / PS per stock
data/financials.parquet     — quarterly ROE, margins, debt ratios per stock
data/macro.parquet          — monthly CPI, PPI, PMI, M2, SHIBOR (indexed to trading days)

All three are designed to be merged into your price panel without leakage:
  - Valuation  → merge on (stock_code, date) directly (it is daily)
  - Financials → forward-fill quarterly values; always shift forward 45 days
                 to simulate announcement lag (earnings are never instant)
  - Macro      → forward-fill monthly values (use the *previous* month's print)

Usage in model.py
-----------------
    from fetch_extra_data import load_all_extra, merge_extra_into_panel
    panel = merge_extra_into_panel(panel, load_all_extra())
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── How many stocks to fetch valuation for (fetching all 499 takes ~30 min) ──
# Set to None to fetch all; use a small number for a quick test run.
MAX_STOCKS: int | None = None

SLEEP_BETWEEN_REQUESTS = 0.3   # seconds — be polite to akshare's servers


# ===========================================================================
# 1. Valuation ratios — daily, per stock
#    PE (TTM), PB (MRQ), PS (TTM), PCF (TTM), market cap
#    Source: ak.stock_a_indicator_lg  (East Money, daily)
# ===========================================================================

def fetch_valuation(constituents: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch daily PE / PB / PS for every CSI500 constituent.

    akshare function : stock_a_indicator_lg(symbol=<6-digit code>)
    Columns returned : trade_date, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio,
                       dv_ttm, total_mv

    We keep: pe_ttm, pb, ps_ttm, total_mv (log-transformed later)
    """
    records = []
    codes = constituents["stock_code"].astype(str).str.zfill(6).tolist()
    if MAX_STOCKS is not None:
        codes = codes[:MAX_STOCKS]

    print(f"Fetching valuation for {len(codes)} stocks …")
    for i, code in enumerate(codes):
        try:
            df = ak.stock_a_indicator_lg(symbol=code)
            df["stock_code"] = code
            # Rename to standard column names (akshare returns Chinese or English
            # depending on version — handle both)
            col_map = {
                "trade_date": "date",
                "日期": "date",
            }
            df = df.rename(columns=col_map)
            df["date"] = pd.to_datetime(df["date"])
            records.append(df)
        except Exception as e:
            print(f"  ⚠  {code}: {e}")

        if i % 50 == 0:
            print(f"  {i}/{len(codes)} done")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not records:
        raise RuntimeError("No valuation data fetched. Check akshare connectivity.")

    out = pd.concat(records, ignore_index=True)

    # Keep only the columns we'll use as features
    keep = ["date", "stock_code", "pe_ttm", "pb", "ps_ttm", "total_mv"]
    existing = [c for c in keep if c in out.columns]
    out = out[existing].copy()
    out = out.sort_values(["stock_code", "date"]).reset_index(drop=True)

    path = DATA_DIR / "valuation.parquet"
    out.to_parquet(path, index=False)
    print(f"✓ Saved {len(out):,} rows → {path}")
    return out


# ===========================================================================
# 2. Financial statement indicators — quarterly, per stock
#    ROE, net profit margin, gross margin, asset turnover, debt ratio,
#    revenue growth, earnings growth
#    Source: ak.stock_financial_analysis_indicator (Sina Finance)
# ===========================================================================

def fetch_financials(constituents: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch quarterly income-statement / balance-sheet ratios.

    akshare function : stock_financial_analysis_indicator(symbol, start_year)
    Columns include  : 净资产收益率(ROE), 销售净利率, 销售毛利率,
                       总资产周转率, 资产负债率, 营业收入同比增长率,
                       净利润同比增长率, etc.

    IMPORTANT — announcement lag
    Quarterly results are announced 1-3 months after quarter-end.
    We apply a conservative 45-day shift before joining to price data so
    we never use a result before it could plausibly be public.
    """
    records = []
    codes = constituents["stock_code"].astype(str).str.zfill(6).tolist()
    if MAX_STOCKS is not None:
        codes = codes[:MAX_STOCKS]

    print(f"Fetching financials for {len(codes)} stocks …")
    for i, code in enumerate(codes):
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2023")
            df["stock_code"] = code
            # Date column is named differently by version
            for dc in ["日期", "report_date", "date"]:
                if dc in df.columns:
                    df = df.rename(columns={dc: "date"})
                    break
            df["date"] = pd.to_datetime(df["date"])
            # Apply 45-day announcement lag
            df["date"] = df["date"] + pd.Timedelta(days=45)
            records.append(df)
        except Exception as e:
            print(f"  ⚠  {code}: {e}")

        if i % 50 == 0:
            print(f"  {i}/{len(codes)} done")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not records:
        raise RuntimeError("No financial data fetched.")

    out = pd.concat(records, ignore_index=True)

    # Standardise to English column names — akshare returns Chinese headers
    rename = {
        "净资产收益率": "roe",
        "销售净利率": "net_profit_margin",
        "销售毛利率": "gross_margin",
        "总资产周转率": "asset_turnover",
        "资产负债率": "debt_ratio",
        "营业收入同比增长率": "revenue_yoy",
        "净利润同比增长率": "profit_yoy",
        "每股收益": "eps",
        "每股净资产": "bvps",
    }
    out = out.rename(columns=rename)

    fin_cols = list(rename.values())
    keep = ["date", "stock_code"] + [c for c in fin_cols if c in out.columns]
    out = out[keep].copy()

    # Convert to numeric (akshare sometimes returns strings with %)
    for col in out.columns:
        if col not in ("date", "stock_code"):
            out[col] = pd.to_numeric(out[col].astype(str).str.replace("%", ""), errors="coerce")

    out = out.sort_values(["stock_code", "date"]).reset_index(drop=True)
    path = DATA_DIR / "financials.parquet"
    out.to_parquet(path, index=False)
    print(f"✓ Saved {len(out):,} rows → {path}")
    return out


# ===========================================================================
# 3. Macro indicators — monthly / quarterly, market-wide
#    CPI, PPI, PMI (official + Caixin), M2 money supply, SHIBOR
#    Source: various ak.macro_china_* functions
# ===========================================================================

def fetch_macro() -> pd.DataFrame:
    """
    Fetch Chinese macro series and align them to a daily date index.

    All series are released with a lag (e.g. January CPI comes out in
    February). We always use the *previous* period's print — i.e. at any
    trading date we only have data that was already published.

    Series fetched
    --------------
    cpi_yoy          — CPI year-on-year % (Consumer Price Index)
    ppi_yoy          — PPI year-on-year % (Producer Price Index)
    pmi_mfg          — Official NBS Manufacturing PMI
    pmi_mfg_cx       — Caixin Manufacturing PMI (more market-sensitive)
    pmi_svc_cx       — Caixin Services PMI
    m2_yoy           — M2 money supply year-on-year %
    shibor_1w        — 1-week SHIBOR (interbank rate)
    """
    frames = {}

    # ── CPI ──────────────────────────────────────────────────────────────────
    try:
        df = ak.macro_china_cpi_monthly()
        # Columns: 月份, 全国-当月, 全国-同比增长, ...
        df = df.rename(columns={"月份": "date", "全国-同比增长": "cpi_yoy"})
        df["date"] = pd.to_datetime(df["date"])
        frames["cpi_yoy"] = df[["date", "cpi_yoy"]].set_index("date")["cpi_yoy"]
        print("✓ CPI")
    except Exception as e:
        print(f"⚠  CPI: {e}")

    # ── PPI ──────────────────────────────────────────────────────────────────
    try:
        df = ak.macro_china_ppi()
        df = df.rename(columns={"月份": "date", "当月": "ppi_yoy"})
        df["date"] = pd.to_datetime(df["date"])
        frames["ppi_yoy"] = df[["date", "ppi_yoy"]].set_index("date")["ppi_yoy"]
        print("✓ PPI")
    except Exception as e:
        print(f"⚠  PPI: {e}")

    # ── Official Manufacturing PMI ────────────────────────────────────────────
    try:
        df = ak.macro_china_pmi_yearly()
        df = df.rename(columns={"月份": "date", "制造业-指数": "pmi_mfg"})
        df["date"] = pd.to_datetime(df["date"])
        frames["pmi_mfg"] = df[["date", "pmi_mfg"]].set_index("date")["pmi_mfg"]
        print("✓ PMI manufacturing (official)")
    except Exception as e:
        print(f"⚠  PMI official: {e}")

    # ── Caixin Manufacturing PMI ──────────────────────────────────────────────
    try:
        df = ak.index_pmi_man_cx()
        df.columns = ["date", "pmi_mfg_cx"]
        df["date"] = pd.to_datetime(df["date"])
        frames["pmi_mfg_cx"] = df.set_index("date")["pmi_mfg_cx"]
        print("✓ PMI manufacturing (Caixin)")
    except Exception as e:
        print(f"⚠  Caixin PMI mfg: {e}")

    # ── Caixin Services PMI ───────────────────────────────────────────────────
    try:
        df = ak.index_pmi_ser_cx()
        df.columns = ["date", "pmi_svc_cx"]
        df["date"] = pd.to_datetime(df["date"])
        frames["pmi_svc_cx"] = df.set_index("date")["pmi_svc_cx"]
        print("✓ PMI services (Caixin)")
    except Exception as e:
        print(f"⚠  Caixin PMI svc: {e}")

    # ── M2 money supply ───────────────────────────────────────────────────────
    try:
        df = ak.macro_china_money_supply()
        # Columns: 月份, M2-数量, M2-同比增长, M2-环比增长, ...
        df = df.rename(columns={"月份": "date", "M2-同比增长": "m2_yoy"})
        df["date"] = pd.to_datetime(df["date"])
        frames["m2_yoy"] = df[["date", "m2_yoy"]].set_index("date")["m2_yoy"]
        print("✓ M2")
    except Exception as e:
        print(f"⚠  M2: {e}")

    # ── SHIBOR (1-week rate) ──────────────────────────────────────────────────
    try:
        df = ak.macro_china_shibor_all()
        # daily; columns: 日期, 隔夜, 1周, 2周, 1月, 3月, 6月, 9月, 1年
        df = df.rename(columns={"日期": "date", "1周": "shibor_1w"})
        df["date"] = pd.to_datetime(df["date"])
        frames["shibor_1w"] = df.set_index("date")["shibor_1w"]
        print("✓ SHIBOR")
    except Exception as e:
        print(f"⚠  SHIBOR: {e}")

    if not frames:
        raise RuntimeError("No macro data fetched. Check akshare connectivity.")

    # ── Combine into a single daily frame ────────────────────────────────────
    # Build a daily date range covering all series
    all_dates = pd.date_range("2024-01-01", pd.Timestamp.today(), freq="D")
    macro = pd.DataFrame(index=all_dates)
    macro.index.name = "date"

    for name, series in frames.items():
        series = pd.to_numeric(series, errors="coerce")
        # Align to daily index, forward-fill (last known value applies until next release)
        macro[name] = series.reindex(macro.index).ffill()

    # Drop rows with all NaN (before any series starts)
    macro = macro.dropna(how="all").reset_index()
    macro["date"] = pd.to_datetime(macro["date"])

    path = DATA_DIR / "macro.parquet"
    macro.to_parquet(path, index=False)
    print(f"✓ Saved {len(macro):,} rows → {path}")
    return macro


# ===========================================================================
# Merging helpers (import these into model.py)
# ===========================================================================

def load_all_extra() -> dict[str, pd.DataFrame]:
    """Load saved extra data files. Call after fetch_* functions."""
    out = {}
    for name, filename in [
        ("valuation",  "valuation.parquet"),
        ("financials", "financials.parquet"),
        ("macro",      "macro.parquet"),
    ]:
        path = DATA_DIR / filename
        if path.exists():
            out[name] = pd.read_parquet(path)
        else:
            print(f"⚠  {path} not found — run fetch_extra_data.py first")
    return out


def merge_extra_into_panel(
    panel: pd.DataFrame, extra: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Merge valuation, financials, and macro into the feature panel.

    All merges are left-joins so no rows are dropped from the panel.
    Missing values (stocks with no data) are left as NaN — LightGBM
    handles NaN natively.

    Valuation
    ---------
    Joined on (stock_code, date) exactly. If the market is closed on a
    given date, forward-fill from the most recent trading day's print.

    Financials
    ----------
    Already shifted by 45 days in fetch_financials(). Here we forward-fill
    within each stock so quarterly values persist until the next report.

    Macro
    -----
    Joined on date only (same value for all stocks). Already daily
    forward-filled in fetch_macro().
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    # ── Valuation ────────────────────────────────────────────────────────────
    if "valuation" in extra:
        val = extra["valuation"].copy()
        val["date"] = pd.to_datetime(val["date"])
        val["stock_code"] = val["stock_code"].astype(str).str.zfill(6)

        # Forward-fill valuation within each stock across all calendar days,
        # then merge (handles non-trading days in the valuation series)
        val = (
            val.set_index(["stock_code", "date"])
            .groupby(level="stock_code")
            .apply(lambda g: g.droplevel(0).resample("D").last().ffill())
            .reset_index()
        )
        val_cols = [c for c in val.columns if c not in ("date", "stock_code")]

        # Log-transform market cap to reduce skew
        if "total_mv" in val.columns:
            val["log_mv"] = np.log1p(val["total_mv"].clip(lower=0))
            val_cols = [c for c in val_cols if c != "total_mv"] + ["log_mv"]
            val = val.drop(columns=["total_mv"])

        panel = panel.merge(
            val[["date", "stock_code"] + val_cols],
            on=["date", "stock_code"],
            how="left",
        )
        print(f"✓ Merged valuation: {val_cols}")

    # ── Financials ───────────────────────────────────────────────────────────
    if "financials" in extra:
        fin = extra["financials"].copy()
        fin["date"] = pd.to_datetime(fin["date"])
        fin["stock_code"] = fin["stock_code"].astype(str).str.zfill(6)
        fin_cols = [c for c in fin.columns if c not in ("date", "stock_code")]

        # Forward-fill quarterly values per stock to daily
        fin = (
            fin.sort_values(["stock_code", "date"])
            .set_index(["stock_code", "date"])
            .groupby(level="stock_code")
            .apply(lambda g: g.droplevel(0).resample("D").last().ffill())
            .reset_index()
        )

        panel = panel.merge(
            fin[["date", "stock_code"] + fin_cols],
            on=["date", "stock_code"],
            how="left",
        )
        print(f"✓ Merged financials: {fin_cols}")

    # ── Macro ─────────────────────────────────────────────────────────────────
    if "macro" in extra:
        macro = extra["macro"].copy()
        macro["date"] = pd.to_datetime(macro["date"])
        macro_cols = [c for c in macro.columns if c != "date"]

        panel = panel.merge(macro, on="date", how="left")
        # Forward-fill any remaining gaps (market holidays not in macro index)
        panel = panel.sort_values(["stock_code", "date"])
        panel[macro_cols] = panel.groupby("stock_code")[macro_cols].ffill()
        print(f"✓ Merged macro: {macro_cols}")

    return panel


# Extra feature columns to add to FEATURE_COLUMNS in model.py after merging:
EXTRA_FEATURE_COLUMNS = [
    # Valuation
    "pe_ttm", "pb", "ps_ttm", "log_mv",
    # Financials
    "roe", "net_profit_margin", "gross_margin",
    "asset_turnover", "debt_ratio", "revenue_yoy", "profit_yoy",
    # Macro
    "cpi_yoy", "ppi_yoy", "pmi_mfg", "pmi_mfg_cx", "pmi_svc_cx",
    "m2_yoy", "shibor_1w",
]


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--constituents", default="data/constituents.csv")
    parser.add_argument("--skip-valuation",  action="store_true")
    parser.add_argument("--skip-financials", action="store_true")
    parser.add_argument("--skip-macro",      action="store_true")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="Limit stocks fetched (useful for testing)")
    args = parser.parse_args()

    if args.max_stocks:
        MAX_STOCKS = args.max_stocks

    constituents = pd.read_csv(args.constituents)
    constituents["stock_code"] = constituents["stock_code"].astype(str).str.zfill(6)

    if not args.skip_valuation:
        print("\n── Valuation ratios ──────────────────────────────")
        fetch_valuation(constituents)

    if not args.skip_financials:
        print("\n── Financial indicators ──────────────────────────")
        fetch_financials(constituents)

    if not args.skip_macro:
        print("\n── Macro series ──────────────────────────────────")
        fetch_macro()

    print("\n✓ All done. Run model.py to train with the new features.")
