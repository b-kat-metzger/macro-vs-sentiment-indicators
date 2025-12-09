from data_pipeline import (
    get_fred_client,
    save_raw_fred_and_yf,
    compute_realized_vol,
    build_merged_df,
    save_merged_df,
    standardize_to_monthly,
    clean_feature_frame,
    prune_to_final_features,
    save_standardized_dataset,
)


def run_pipeline():
    fred = get_fred_client()
    save_raw_fred_and_yf(fred)
    compute_realized_vol()

    merged = build_merged_df()
    save_merged_df(merged)

    monthly = standardize_to_monthly(merged)
    cleaned = clean_feature_frame(monthly)
    final_df = prune_to_final_features(cleaned)

    save_standardized_dataset(final_df)

    print("\nPipeline complete. Displaying head/tail:\n")
    print(final_df.head())
    print(final_df.tail())


if __name__ == "__main__":
    run_pipeline()

