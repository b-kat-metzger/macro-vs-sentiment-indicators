# pip install pandas yfinance fredapi python-dotenv
# make sure you have a working SSL cert

import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv, find_dotenv

START = "1990-01-01"
OUTDIR = Path("data","raw")
OUTDIR.mkdir(exist_ok=True)

env_path = find_dotenv(usecwd=True) or (Path(__file__).parent / ".env")
load_dotenv(env_path)
fred_key = os.getenv("FRED_API_KEY")
if not fred_key:
    raise RuntimeError("FRED_API_KEY not found in .env")
fred = Fred(api_key=fred_key)

# FRED series
FRED_SERIES = {
    "GDPC1": "Real GDP (q)",
    "CPIAUCSL": "CPI (m)",
    "UNRATE": "Unemployment rate (m)",
    "ICSA": "Initial jobless claims (w)",
    "FEDFUNDS": "Effective fed funds rate (d/m)",
    "DGS10": "10y Treasury (d)",
    "DGS3MO": "3m Treasury (d)",
}

def pull_fred(series_id: str) -> pd.DataFrame:
    s = fred.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    return s.rename(series_id).to_frame()

for sid, desc in FRED_SERIES.items():
    df = pull_fred(sid)
    df.to_csv(OUTDIR / f"fred_{sid}.csv")
    print(f"Saved FRED {sid} -> {OUTDIR/f'fred_{sid}.csv'}  ({desc})")

YF_TICKERS = {
    "^GSPC": "S&P 500 index",
    "^VIX": "VIX spot",
}

def pull_yf(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, start=START, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    return df  # keep raw OHLC, Adj Close, Volume

for tkr, desc in YF_TICKERS.items():
    tkr_name = tkr.replace('^','')
    df = pull_yf(tkr)
    df.to_csv(OUTDIR / f"yf_{tkr_name}.csv")
    print(f"Saved Yahoo {tkr_name} -> {OUTDIR/f'yf_{tkr_name}'}.csv ({desc})")

print("\nDone. Raw CSVs are in:", OUTDIR.resolve())
