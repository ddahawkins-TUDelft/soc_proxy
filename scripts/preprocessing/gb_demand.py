import pandas as pd
from pathlib import Path

# Folder where your CSVs live
DATA_DIR = Path("C:/Users/danod/Downloads")  # change this

# Build list of files demanddata_2010.csv ... demanddata_2019.csv
files = [DATA_DIR / f"demanddata_{year}.csv" for year in range(2010, 2020)]

all_data = []

for file in files:
    # Read only first three columns
    df = pd.read_csv(file, usecols=[0, 1, 2])

    # Make sure columns are named as expected
    df.columns = ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]

    # Parse date (e.g. 01-Jan-10)
    df["SETTLEMENT_DATE"] = pd.to_datetime(
        df["SETTLEMENT_DATE"].astype(str).str.strip(),
        format="%d-%b-%Y"
    )

    # Convert 30 minute periods to an hour index:
    # periods 1 and 2 -> hour 0 (00:00)
    # periods 3 and 4 -> hour 1 (01:00)
    # ...
    df["hour"] = (df["SETTLEMENT_PERIOD"] - 1) // 2

    # Build hourly timestamp
    df["datetime"] = df["SETTLEMENT_DATE"] + pd.to_timedelta(df["hour"], unit="h")

    # Only keep what we need
    all_data.append(df[["datetime", "ND"]])

# Concatenate all years
combined = pd.concat(all_data, ignore_index=True)

# Aggregate to hourly: sum ND for each datetime
hourly = combined.groupby("datetime", as_index=False)["ND"].sum()

# Sort by time just to be safe
hourly = hourly.sort_values("datetime")

# Save to CSV
output_path = DATA_DIR / "gb_demand_2010_2019.csv"
hourly.to_csv(output_path, index=False)

print(f"Saved hourly data to: {output_path}")
