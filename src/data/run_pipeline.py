from data_pipeline import (
    get_fred_client,
    save_raw_fred_and_yf,
    compute_realized_vol,
    build_merged_df,
    save_merged_df,
    standardize_to_monthly,
    clean_feature_frame,
    save_standardized_dataset,
)


def run_pipeline():
    print("\n=== STEP 1: Connecting to FRED ===")
    fred = get_fred_client()

    print("\n=== STEP 2: Fetching raw FRED + Yahoo data ===")
    save_raw_fred_and_yf(fred)

    print("\n=== STEP 3: Computing 30-day realized volatility ===")
    compute_realized_vol()

    print("\n=== STEP 4: Building merged daily dataframe ===")
    merged = build_merged_df()
    if merged is None or merged.empty:
        raise RuntimeError("build_merged_df() returned empty or None.")
    save_merged_df(merged)

    print("\n=== STEP 5: Standardizing to monthly ===")
    monthly = standardize_to_monthly(merged)
    if monthly is None or monthly.empty:
        raise RuntimeError("standardize_to_monthly() returned empty or None.")

    print("\n=== STEP 6: Cleaning feature frame (trim early NA + last partial month) ===")
    final_df = clean_feature_frame(monthly)
    if final_df is None or final_df.empty:
        raise RuntimeError("clean_feature_frame() returned empty or None.")

    print("\n=== STEP 7: Saving final cleaned dataset ===")
    save_standardized_dataset(final_df)

    print("\n=== FINAL DATAFRAME SUMMARY ===")
    print(final_df.head())
    print("...")
    print(final_df.tail())
    final_df.info()

    print("\nPipeline complete.")
    print("Raw CSVs → data_raw/")
    print("Final cleaned → data_cleaned/standardized_monthly_data.csv")


if __name__ == "__main__":
    run_pipeline()
