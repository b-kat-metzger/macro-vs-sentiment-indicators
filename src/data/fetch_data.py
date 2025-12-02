# FETCHING DATA REQUIRES .ENV FILE WITH FRED API KEY

import os
from pathlib import Path
import math

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv, find_dotenv
import matplotlib.pyplot as plt

# ---- CONFIG ----
START = "1990-01-01"
OUTDIR = Path("data","raw")
OUTDIR.mkdir(exist_ok=True)

FRED_SERIES = {
    "GDPC1": "Real GDP (q)",
    "INDPRO": "Industrial Production (m)",
    "TCU": "Capacity Utilization (m)",
    "CPIAUCSL": "CPI (m)",
    "PCEPI": "PCE Inflation (m)",
    "UNRATE": "Unemployment Rate (m)",
    "ICSA": "Initial Jobless Claims (w)",
    "FEDFUNDS": "Fed Funds Rate (m)",
    "DGS10": "10y Treasury (d)",
    "DGS3MO": "3m Treasury (d)",
    "M2SL": "Money Supply (m)",
    "UMCSENT": "Consumer Sentiment (m)",
}

YF_TICKERS = {
    "^GSPC": "S&P 500 index",
    "^VIX": "VIX spot",
}


# ---- UTILITIES ----
def load_api_key() -> str:
    env_path = find_dotenv(usecwd=True) or (Path(__file__).parent / ".env")
    load_dotenv(env_path)
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not found in .env")
    return key


def fred_client() -> Fred:
    return Fred(api_key=load_api_key())


# ---- DATA PULL ----
def pull_fred(series_id: str, client: Fred) -> pd.DataFrame:
    if series_id in {"DGS10", "DGS3MO"}:
        s = client.get_series(series_id, frequency="d")
    elif series_id == "FEDFUNDS":
        s = client.get_series(series_id, frequency="m")
    else:
        s = client.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    s = s[s.index >= pd.to_datetime(START)]
    return s.rename(series_id).to_frame()


def pull_yf(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, start=START, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    return df


def save_raw_data():
    client = fred_client()

    for sid, desc in FRED_SERIES.items():
        df = pull_fred(sid, client)
        path = OUTDIR / f"fred_{sid}.csv"
        df.to_csv(path)
        print(f"Saved FRED {sid} -> {path}  ({desc})")

    for tkr, desc in YF_TICKERS.items():
        df = pull_yf(tkr)
        fname = f"yf_{tkr.replace('^','')}.csv"
        path = OUTDIR / fname
        df.to_csv(path)
        print(f"Saved Yahoo {tkr} -> {path}  ({desc})")


# ---- CALCULATING REALIZED ----
def compute_realized_vol(window: int = 21, annual_factor: float = np.sqrt(252)):
    print("Computing 21-day realized volatility...")
    gspc_path = OUTDIR / "yf_GSPC.csv"
    if not gspc_path.exists():
        raise FileNotFoundError(f"{gspc_path} not found. Run data pull first.")

    gspc = pd.read_csv(gspc_path, index_col=0, parse_dates=True, skiprows=[1])
    gspc.index.name = "Date"
    # Ensure numeric columns
    gspc = gspc.apply(pd.to_numeric, errors="coerce")
    gspc["log_return"] = np.log(gspc["Adj Close"] / gspc["Adj Close"].shift(1))
    gspc["rv_21d"] = gspc["log_return"].rolling(window=window).std() * annual_factor
    rv = gspc[["rv_21d"]].dropna()
    rv_path = OUTDIR / "realized_vol_21d.csv"
    rv.to_csv(rv_path)
    print(f"Saved realized volatility -> {rv_path}")


# ---- CSV LOADER ----
def load_csv(path: Path) -> pd.DataFrame:
    stem = path.stem
    if stem.startswith("yf_"):
        df = pd.read_csv(path, index_col=0, skiprows=[1])
        df = df.dropna(how="all")
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df.index.name = "Date"
        ticker = stem.replace("yf_", "")
        df = df.add_prefix(f"{ticker}_")
        df = df.apply(pd.to_numeric, errors="coerce")
        return df

    if stem.startswith("realized_vol"):
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df.index.name = "Date"
        return df

    if stem.startswith("fred_"):
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df.index.name = "Date"
        series_id = stem.replace("fred_", "")
        df.columns = [series_id]
        df = df.apply(pd.to_numeric, errors="coerce")
        return df

    # fallback
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df.index.name = "Date"
    return df


# ---- MERGE & CLEAN ----
def build_merged_df() -> pd.DataFrame:
    files = sorted(OUTDIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {OUTDIR}")

    full_df = None
    for p in files:
        df = load_csv(p)
        full_df = df if full_df is None else full_df.join(df, how="outer")

    full_df = full_df.sort_index()

    # Keep only GSPC adjusted close if present
    gspc_cols = [c for c in full_df.columns if c.startswith("GSPC_")]
    if "GSPC_Adj Close" in gspc_cols:
        cols_to_drop = [c for c in gspc_cols if c != "GSPC_Adj Close"]
        if cols_to_drop:
            full_df = full_df.drop(columns=cols_to_drop)
        full_df = full_df.rename(columns={"GSPC_Adj Close": "GSPC"})
    else:
        print("Warning: GSPC_Adj Close not found in columns:", gspc_cols)

    # Keep only VIX_Close if present
    vix_cols = [c for c in full_df.columns if c.startswith("VIX_")]
    if "VIX_Close" in vix_cols:
        cols_to_drop = [c for c in vix_cols if c != "VIX_Close"]
        if cols_to_drop:
            full_df = full_df.drop(columns=cols_to_drop)

    return full_df



# ---- PLOT ----
def plot_time_series(full_df: pd.DataFrame):
    plot_df = full_df.select_dtypes(include=[np.number]).copy()
    plot_df = plot_df.dropna(axis=1, how="all")

    TITLE_MAP = {sid: f"{desc} ({sid})" for sid, desc in FRED_SERIES.items()}
    TITLE_MAP.update({
        "GSPC": "S&P 500 index adj close (GSPC)",
        "VIX_Close": "VIX spot close (VIX_Close)",
        "rv_21d": "21-day realized volatility (rv_21d)",
    })

    if plot_df.empty:
        print("No numeric columns available to plot.")
        return

    cols = list(plot_df.columns)
    n = len(cols)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        series = plot_df[col].dropna()
        if series.empty:
            ax.set_visible(False)
            continue
        ax.plot(series.index, series.values)
        ax.set_title(TITLE_MAP.get(col, col), fontsize=10)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Time Series of All Indicators", fontsize=14, y=0.995)
    fig.tight_layout()
    plt.show()


# ---- FETCH AND PLOT RAW DATA ----
def fetch_data_and_plot():
    # Pull and save raw data
    save_raw_data()

    # Compute realized vol from saved GSPC CSV
    compute_realized_vol()

    # Merge CSVs and plot
    full_df = build_merged_df()
    print("Merged DataFrame:")
    print(full_df.head())
    print(full_df.tail())
    plot_time_series(full_df)
    print("\nDone. Raw CSVs are in:", OUTDIR.resolve())

if __name__ == "__main__":
    fetch_data_and_plot()