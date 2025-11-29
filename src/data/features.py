import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/clean/cleaned_data.csv")
OUTDIR = Path("data/features")
OUTDIR.mkdir(exist_ok=True, parents=True)

def make_macro_features(df):
    out = pd.DataFrame(index=df.index)

    # Macro features (only for columns that exist)
    if "GDPC1" in df.columns:
        out["GDP_change"] = df["GDPC1"].pct_change()

    if "CPIAUCSL" in df.columns:
        out["CPI_change"] = df["CPIAUCSL"].pct_change()

    if "UNRATE" in df.columns:
        out["UNRATE_diff"] = df["UNRATE"].diff()

    if "ICSA" in df.columns:
        out["ICSA_change"] = df["ICSA"].pct_change()

    if "FEDFUNDS" in df.columns:
        out["FEDFUNDS_diff"] = df["FEDFUNDS"].diff()

    if "DGS10" in df.columns and "DGS3MO" in df.columns:
        out["yield_curve"] = df["DGS10"] - df["DGS3MO"]

    # Add market returns + target (NEEDED for ML)
    if "SP500_Close" in df.columns:
        out["returns"] = df["SP500_Close"].pct_change()
        out["target"] = (out["returns"].shift(-1) > 0).astype(int)

    out = out.dropna()
    return out


def make_market_features(df):
    out = pd.DataFrame(index=df.index)

    # Price-based features
    if "SP500_Close" in df.columns:
        out["returns"] = df["SP500_Close"].pct_change()
        out["log_ret"] = np.log(df["SP500_Close"] / df["SP500_Close"].shift(1))

        out["ema20"] = df["SP500_Close"].ewm(span=20).mean()
        out["ema50"] = df["SP500_Close"].ewm(span=50).mean()
        out["ema_spread"] = out["ema20"] - out["ema50"]

        out["vol_21"] = out["log_ret"].rolling(21).std()

        # target = tomorrow’s direction
        out["target"] = (out["returns"].shift(-1) > 0).astype(int)

    if "VIX_Close" in df.columns:
        out["VIX"] = df["VIX_Close"]

    if "rv_21d" in df.columns:
        out["realized_vol"] = df["rv_21d"]

    out = out.dropna()
    return out


def main():
    df = pd.read_csv(INPUT, index_col=0, parse_dates=True)

    macro = make_macro_features(df)
    market = make_market_features(df)

    OUTDIR.mkdir(exist_ok=True, parents=True)
    macro.to_csv(OUTDIR / "features_macro.csv")
    market.to_csv(OUTDIR / "features_market.csv")

    print("Saved:")
    print(" - features_macro.csv")
    print(" - features_market.csv")


if __name__ == "__main__":
    main()
