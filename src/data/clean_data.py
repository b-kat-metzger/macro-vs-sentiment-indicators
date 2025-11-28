import pandas as pd
from pathlib import Path

START = "1990-01-01"
END = "today"
full_index = pd.date_range(START,END,freq="D")
DATA_PATH = Path("data","raw")
NEWS_PATH = Path("data","static","news_sentiment_data.xlsx")

OUTDIR = Path("data","clean")
OUTDIR.mkdir(exist_ok=True)

GDPC1 = pd.read_csv(Path(DATA_PATH,"fred_GDPC1.csv"), index_col=0, parse_dates=True)
GDPC1 = GDPC1.reindex(full_index).ffill()

CPIAUCSL = pd.read_csv(Path(DATA_PATH,"fred_CPIAUCSL.csv"),index_col=0,parse_dates=True)
CPIAUCSL = CPIAUCSL.reindex(full_index).ffill()

UNRATE = pd.read_csv(Path(DATA_PATH,"fred_UNRATE.csv"),index_col=0,parse_dates=True)
UNRATE = UNRATE.reindex(full_index).ffill()

ICSA = pd.read_csv(Path(DATA_PATH,"fred_ICSA.csv"),index_col=0,parse_dates=True)
ICSA = ICSA.reindex(full_index).ffill()

FEDFUNDS = pd.read_csv(Path(DATA_PATH,"fred_FEDFUNDS.csv"),index_col=0,parse_dates=True)
FEDFUNDS = FEDFUNDS.reindex(full_index).ffill()

DGS10 = pd.read_csv(Path(DATA_PATH,"fred_DGS10.csv"),index_col=0,parse_dates=True)
DGS10 = DGS10.reindex(full_index).ffill()

DGS3MO = pd.read_csv(Path(DATA_PATH,"fred_DGS3MO.csv"),index_col=0,parse_dates=True)
DGS3MO = DGS3MO.reindex(full_index).ffill()

NEWS = pd.read_excel(NEWS_PATH, sheet_name="Data", parse_dates=["date"], index_col="date")
NEWS = NEWS.rename(columns={"News Sentiment": "NEWS_SENTIMENT"})
NEWS = NEWS.reindex(full_index).ffill()

GSPC = pd.read_csv(Path(DATA_PATH,"yf_GSPC.csv"), skiprows=[1,2], index_col=0, parse_dates=True)

VIX = pd.read_csv(Path(DATA_PATH,"yf_VIX.csv"), skiprows=[1,2], index_col=0, parse_dates=True)

DFS = [
    GDPC1,
    CPIAUCSL,
    UNRATE,
    ICSA,
    FEDFUNDS,
    DGS10,
    DGS3MO,
    NEWS[["NEWS_SENTIMENT"]],
    GSPC[["Open","High","Low","Close","Adj Close", "Volume"]].add_prefix("SP500_"),
    VIX[["Open","High","Low","Close","Adj Close", "Volume"]].add_prefix("VIX_"),
]

full_df = pd.concat(DFS,axis=1)
full_df.index.name = "Date"
full_df = full_df.round(2)  # round to 2 decimal places

# Drop rows with missing values in GSPC and VIX columns (non-business days)
sp500_cols = [col for col in full_df.columns if col.startswith("SP500_")]
vix_cols = [col for col in full_df.columns if col.startswith("VIX_")]
full_df = full_df.dropna(subset=sp500_cols + vix_cols)

MACRO_FEATURES = ["GDPC1", "CPIAUCSL", "UNRATE", "ICSA", "FEDFUNDS"]
SENTIMENT_FEATURES = ["DGS10", "DGS3MO", "NEWS_SENTIMENT"] + sp500_cols + vix_cols

macro_df = full_df[MACRO_FEATURES]
sentiment_df = full_df[SENTIMENT_FEATURES]

macro_df.to_csv(OUTDIR/"macro_features.csv")
sentiment_df.to_csv(OUTDIR/"sentiment_features.csv")
full_df.to_csv(OUTDIR/"cleaned_data.csv")