import pandas as pd
from pathlib import Path

# ====== CONFIGURATION ======
FILES = {
    # "ES": "SoC_proxy_TSA/es_case_study.csv",
    "BE": "SoC_proxy_TSA/be_case_study.csv",
    # "IT": "SoC_proxy_TSA/it_case_study.csv",
}

TIME_COLUMN = "timesteps"
DATA_COLUMNS = ["solar", "offshore_wind", "onshore_wind", "demand_power"]

EXPECTED_START = "2010-01-01 00:00"
EXPECTED_END   = "2019-12-31 23:00"
FREQ = "H"

OUTPUT_SUFFIX = "_cleaned.csv"
# ===========================


def clean_dataframe(df, name):
    print(f"\n=== Processing {name} ===")
    df = df.copy()

    # ---- PARSE TIMESTAMP ----
    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        dayfirst=True,
        errors="coerce"
    )
    df = df.dropna(subset=[TIME_COLUMN])
    df = df.set_index(TIME_COLUMN).sort_index()

    # ---- REMOVE DUPLICATES (DST FALL BACK) ----
    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    after = len(df)
    print(f"  Removed duplicates: {before - after}")

    # ---- EXPECTED FULL INDEX ----
    full_index = pd.date_range(
        start=EXPECTED_START,
        end=EXPECTED_END,
        freq=FREQ
    )

    # ---- REINDEX (insert missing timestamps as NaNs) ----
    df = df.reindex(full_index)
    df.index.name = TIME_COLUMN

    # ---- INTERPOLATE BETWEEN NEIGHBOURS FOR MISSING ROWS ----
    missing_before = df[DATA_COLUMNS].isna().any(axis=1).sum()
    print(f"  Missing rows before interpolation: {missing_before}")

    df[DATA_COLUMNS] = df[DATA_COLUMNS].interpolate(
        method="time",
        limit_direction="both"
    )

    missing_after = df[DATA_COLUMNS].isna().any(axis=1).sum()
    print(f"  Missing rows after interpolation: {missing_after}")
    if missing_after > 0:
        print("  WARNING: Some NaNs remain (likely at very start/end).")

    # ---- PREPARE FOR EXPORT ----
    # Move index back to a column, format as required, and reorder columns
    out_df = df.reset_index()

    # Format timestamp as "yyyy/mm/dd hh:mm"
    out_df[TIME_COLUMN] = out_df[TIME_COLUMN].dt.strftime("%Y/%m/%d %H:%M")

    # Desired column order
    desired_order = [
        TIME_COLUMN,
        "demand_power",
        "solar",
        "offshore_wind",
        "onshore_wind"
    ]
    # Keep only those that exist (just in case)
    desired_order = [c for c in desired_order if c in out_df.columns]
    out_df = out_df[desired_order]

    return out_df


def main():
    for name, filename in FILES.items():
        path = Path(filename)
        if not path.exists():
            print(f"*** WARNING: File not found: {path} ***")
            continue

        print(f"\nLoading {name} from {path} ...")
        df = pd.read_csv(path)

        cleaned_df = clean_dataframe(df, name)

        out_file = f"{name.lower()}{OUTPUT_SUFFIX}"
        # Comma delimiter, no index column in file
        cleaned_df.to_csv(out_file, index=False, sep=",")
        print(f"  → Cleaned file written to {out_file}")

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
