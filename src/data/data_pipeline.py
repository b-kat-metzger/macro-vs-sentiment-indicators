# FRED API KEY REQUIRED IN .ENV

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv, find_dotenv

################################################################################
# INITIAL CONFIG
################################################################################

START_DATE = "1992-10-01"

RAW_DIR = Path("data", "raw")
CLEAN_DIR = Path("data", "clean")

RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# FRED series data to pull
FRED_SERIES: Dict[str, str] = {

    # PRIMARY ECONOMIC INDICATORS:


    # Labor market (2)
    "UNRATE": "Unemployment Rate (m)",
    "ICSA": "Initial Jobless Claims (w)",

    # Consumers (2)
    "UMCSENT": "Consumer Sentiment (m)",
    "RSAFS": "Retail Sales (m)",

    # Inflation + Policy (2)
    "CPIAUCSL": "CPI (m)",              # delta computed after load
    "FEDFUNDS": "Fed Funds Rate (m)",   # delta computed after load

    # Production + Business Cycle (2)
    "INDPRO": "Industrial Production (m)",   # delta computed after load
    "AMTMNO": "Manufacturers' New Orders: Durable Goods (m)",

    # Housing forward indicator (1)
    "PERMIT": "Building Permits (m)",


    # FINANCIAL / SUPPORT SERIES:

    "BAA": "Moody's BAA Corporate Bond Yield (d)",  # for credit spread
    "DGS10": "10y Treasury Yield (d)",              # for slope + credit spread
    "DGS3MO": "3m Treasury Yield (d)",              # for slope
}

# Yahoo Finance tickers
YF_TICKERS: Dict[str, str] = {
    "^GSPC": "S&P 500 index",
    "^VIX": "VIX spot",
}

# Final feature set for modeling
FINAL_ECON_FEATURES: List[str] = [
    "ECO_dCPIAUCSL",
    "ECO_dFEDFUNDS",
    "ECO_dINDPRO",
    "FRED_UNRATE",
    "FRED_ICSA",
    "FRED_UMCSENT",
    "FRED_RSAFS",
    "FRED_AMTMNO",
    "FRED_PERMIT",
]

FINAL_FIN_FEATURES: List[str] = [
    "FIN_gspc_ret_lag1",
    "FIN_mom_12m",
    "FIN_mom_6m",
    "FIN_vix_level",
    "FIN_dVIX",
    "FIN_gspc_monthly_vol",
    "FIN_yield_curve_slope",
    "FIN_credit_spread",
    "FIN_rv30",
]

FINAL_FEATURES = FINAL_ECON_FEATURES + FINAL_FIN_FEATURES

########################################
# ENV + FRED

def load_api_key() -> str:
    env_path = find_dotenv(usecwd=True) or (Path(__file__).parent / ".env")
    load_dotenv(env_path)
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not found in .env")
    return key


def get_fred_client() -> Fred:
    return Fred(api_key=load_api_key())

########################################
# RAW DATA PULL

def pull_fred_series(series_id: str, client: Fred) -> pd.DataFrame:
    s = client.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    s = s[s.index >= pd.to_datetime(START_DATE)]
    df = s.rename(series_id).to_frame()
    df.index.name = "Date"
    return df


def pull_yf_series(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, start=START_DATE, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    df.index.name = "Date"
    return df


def save_raw_fred_and_yf(client: Fred) -> None:
    # FRED
    for sid, desc in FRED_SERIES.items():
        df = pull_fred_series(sid, client)
        path = RAW_DIR / f"fred_{sid}.csv"
        df.to_csv(path)
        print(f"Saved FRED {sid} -> {path} ({desc})")

    # Yahoo Finance
    for symbol, desc in YF_TICKERS.items():
        df = pull_yf_series(symbol)
        fname = f"yf_{symbol.replace('^','')}.csv"
        path = RAW_DIR / fname
        df.to_csv(path)
        print(f"Saved Yahoo {symbol} -> {path} ({desc})")

########################################
# REALIZED VOLATILITY (30D → MONTHLY)

def compute_realized_vol(window: int = 30, annual_factor: float = np.sqrt(252)) -> Path:
    """
    Compute 30-day realized volatility from daily GSPC log returns,
    then take end-of-month values to align with monthly macro data.
    """
    gspc_path = RAW_DIR / "yf_GSPC.csv"
    if not gspc_path.exists():
        raise FileNotFoundError("Missing yf_GSPC.csv. Run data pull first.")

    df = pd.read_csv(gspc_path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce")

    if "Adj Close" not in df.columns:
        raise KeyError("Expected 'Adj Close' in yf_GSPC.csv")

    df["log_return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    df["rv_30d"] = df["log_return"].rolling(window=window).std() * annual_factor

    monthly_rv30 = (
        df["rv_30d"]
        .resample("ME")
        .last()
        .to_frame("rv_30d")
        .dropna()
    )

    rv_path = RAW_DIR / "realized_vol_30d_monthly.csv"
    monthly_rv30.to_csv(rv_path)
    print(f"Saved monthly realized volatility (30d) -> {rv_path}")
    return rv_path

########################################
# GENERIC CSV LOADER
def load_csv(path: Path) -> pd.DataFrame:
    stem = path.stem

    df = pd.read_csv(path, index_col=0)
    # Force datetime index
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce")

    if stem.startswith("yf_"):
        ticker = stem.replace("yf_", "")
        return df.add_prefix(f"{ticker}_")

    if stem.startswith("realized_vol"):
        return df

    if stem.startswith("fred_"):
        series_id = stem.replace("fred_", "")
        df.columns = [f"FRED_{series_id}"]
        return df

    return df

########################################
# MERGE RAW CSVs
def build_merged_df() -> pd.DataFrame:
    """
    Merge all raw CSVs into a single daily/mixed-frequency DataFrame.
    """
    valid_fred_stems = {f"fred_{sid}" for sid in FRED_SERIES.keys()}
    valid_yf_stems = {f"yf_{symbol.replace('^','')}" for symbol in YF_TICKERS.keys()}

    files = []
    for p in RAW_DIR.glob("*.csv"):
        stem = p.stem
        if stem == "merged_dataframe":
            continue
        if stem.startswith("fred_") and stem not in valid_fred_stems:
            continue
        if stem.startswith("yf_") and stem not in valid_yf_stems:
            continue
        if stem.startswith("realized_vol"):
            files.append(p)
            continue
        if stem.startswith("fred_") or stem.startswith("yf_"):
            files.append(p)

    files = sorted(files)
    if not files:
        raise FileNotFoundError("No valid CSVs found in data/raw/")

    full_df = None
    for p in files:
        df = load_csv(p)
        full_df = df if full_df is None else full_df.join(df, how="outer")

    full_df = full_df.sort_index()

    # Collapse GSPC columns to a single 'GSPC' price series
    gspc_cols = [c for c in full_df.columns if c.startswith("GSPC_")]
    if "GSPC_Adj Close" in gspc_cols:
        to_drop = [c for c in gspc_cols if c != "GSPC_Adj Close"]
        full_df = full_df.drop(columns=to_drop)
        full_df = full_df.rename(columns={"GSPC_Adj Close": "GSPC"})

    # Collapse VIX columns to a single 'VIX_Close' series
    vix_cols = [c for c in full_df.columns if c.startswith("VIX_")]
    if "VIX_Close" in vix_cols:
        to_drop = [c for c in vix_cols if c != "VIX_Close"]
        full_df = full_df.drop(columns=to_drop)

    return full_df


def save_merged_df(full_df: pd.DataFrame) -> Path:
    merged_path = RAW_DIR / "merged_dataframe.csv"
    full_df.to_csv(merged_path)
    print(f"Merged DataFrame saved to {merged_path}")
    return merged_path

########################################
# STANDARDIZATION TO MONTHLY
def standardize_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert mixed-frequency data (daily, weekly, monthly) to an end-of-month dataset.

    - GSPC/VIX: monthly return, vol, average, level
    - FRED/macro: last observation per month
    - rv_30d: already monthly; preserved as-is
    - Then add derived econ + financial features
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df.index.name = "Date"

    freq = "ME"
    monthly_index = df.resample(freq).asfreq().index

    # Daily price-based series (Yahoo)
    daily_vars = [c for c in df.columns if c.startswith("GSPC") or c.startswith("VIX")]
    macro_vars = [c for c in df.columns if c not in daily_vars]

    # Price-based monthly aggregates
    if daily_vars:
        prices = df[daily_vars]

        monthly_prices = prices.resample(freq).last()
        monthly_return = monthly_prices.pct_change().add_suffix("_monthly_return")

        daily_ret = prices.pct_change(fill_method=None)
        monthly_vol = daily_ret.resample(freq).std().add_suffix("_monthly_vol")

        monthly_avg = prices.resample(freq).mean().add_suffix("_monthly_avg")
        monthly_level = monthly_prices.add_suffix("_monthly_level")
    else:
        monthly_return = pd.DataFrame(index=monthly_index)
        monthly_vol = pd.DataFrame(index=monthly_index)
        monthly_avg = pd.DataFrame(index=monthly_index)
        monthly_level = pd.DataFrame(index=monthly_index)

    # Macro / other series
    if macro_vars:
        monthly_macro = df[macro_vars].resample(freq).last()
    else:
        monthly_macro = pd.DataFrame(index=monthly_index)

    monthly_data = pd.concat(
        [monthly_return, monthly_vol, monthly_avg, monthly_level, monthly_macro],
        axis=1
    )
    monthly_data.index.name = "Date"

    # Add derived features
    monthly_data = add_derived_features(monthly_data)

    return monthly_data

########################################
# DERIVED FEATURES (MACRO + FINANCIAL)
def add_derived_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    df = monthly_df.copy()

    # Macro deltas
    if "FRED_CPIAUCSL" in df.columns:
        df["ECO_dCPIAUCSL"] = df["FRED_CPIAUCSL"].diff()

    if "FRED_INDPRO" in df.columns:
        df["ECO_dINDPRO"] = df["FRED_INDPRO"].diff()

    if "FRED_FEDFUNDS" in df.columns:
        df["ECO_dFEDFUNDS"] = df["FRED_FEDFUNDS"].diff()

    # Financial derived
    if "GSPC_monthly_return" in df.columns:
        df["FIN_gspc_ret_lag1"] = df["GSPC_monthly_return"].shift(1)

    if "GSPC_monthly_level" in df.columns:
        df["FIN_mom_12m"] = df["GSPC_monthly_level"].pct_change(12)
        df["FIN_mom_6m"] = df["GSPC_monthly_level"].pct_change(6)

    if "VIX_Close_monthly_level" in df.columns:
        df["FIN_vix_level"] = df["VIX_Close_monthly_level"]
        df["FIN_dVIX"] = df["VIX_Close_monthly_level"].diff()

    if "GSPC_monthly_vol" in df.columns:
        df["FIN_gspc_monthly_vol"] = df["GSPC_monthly_vol"]

    if "FRED_DGS10" in df.columns and "FRED_DGS3MO" in df.columns:
        df["FIN_yield_curve_slope"] = df["FRED_DGS10"] - df["FRED_DGS3MO"]

    if "FRED_BAA" in df.columns and "FRED_DGS10" in df.columns:
        df["FIN_credit_spread"] = df["FRED_BAA"] - df["FRED_DGS10"]

    if "rv_30d" in df.columns:
        df["FIN_rv30"] = df["rv_30d"]

    return df

########################################
# SAVE STANDARDIZED DATASET
def save_standardized_dataset(df: pd.DataFrame, filename: str = "standardized_monthly_data.csv") -> Path:
    out_path = CLEAN_DIR / filename
    df.to_csv(out_path)
    print(f"Standardized monthly dataset saved to {out_path}")
    return out_path


########################################
# FINAL DATA CLEANING: TRIM EARLY NAs + LAST PARTIAL MONTH
def clean_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean standardized monthly data:
      - Trim early rows until required features have begun
      - Drop the final month if incomplete
    """
    if df is None or df.empty:
        print("clean_feature_frame(): received empty DataFrame.")
        return df

    cleaned = df.copy()

    # Required feature columns = all FRED_, ECO_, FIN_
    required_cols = [c for c in cleaned.columns if c.startswith(("FRED_", "ECO_", "FIN_"))]

    if not required_cols:
        print("clean_feature_frame(): no required feature columns found.")
        return cleaned


    # Trim begining rows until features are consistently reported
    feature_mask = cleaned[required_cols].notna().any(axis=1)

    if feature_mask.any():
        first_valid_idx = feature_mask.idxmax()
        cleaned = cleaned.loc[first_valid_idx:]
    else:
        print("clean_feature_frame(): no non-NA values for required features. Returning unchanged.")
        return cleaned

    # Drops final month if incomplete
    last_row = cleaned.tail(1)[required_cols]
    if last_row.isna().any(axis=1).iloc[0]:
        cleaned = cleaned.iloc[:-1]

    return cleaned


########################################
# FINAL FEATURE PRUNING
def prune_to_final_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [f for f in FINAL_FEATURES if f not in df.columns]
    if missing:
        print("WARNING: Missing expected features:", missing)

    existing = [f for f in FINAL_FEATURES if f in df.columns]
    return df[existing]
