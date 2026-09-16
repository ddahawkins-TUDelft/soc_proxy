
#!/usr/bin/env python3
"""
shuffle_years_config.py — Click-and-run generator for "year-shuffled" Calliope time series CSVs.

Edit the CONFIG block below, then run this file once from VS Code (Run ▶) or `python shuffle_years_config.py`.

It reads your original Calliope `time_varying_parameters.csv` (5-line header + hourly `timesteps`)
and writes N shuffled variants next to it. The output **always** has strictly chronological
timestamps (default: 2010–2019), while each year block is pulled from a randomly permuted
source year, *matching leap-year status* to keep the hour counts correct.

The original CSV is never modified. The first five header lines are preserved verbatim.

Filename pattern:
    <base>__shuffle_<srcY1-...-srcYk>__as_<targetStart-targetEnd>__NN.csv
"""

from __future__ import annotations

import calendar
import csv
import datetime as dt
from pathlib import Path
from typing import List, Sequence, Tuple
import random

import pandas as pd

# =========================
# CONFIG — EDIT AS NEEDED
# =========================
CONFIG = {
    # Path to the original Calliope CSV (will NOT be modified)
    "CSV_PATH": "SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv",

    # How many shuffled files to generate
    "N_FILES": 8,

    # Output timeline: consecutive years [TARGET_START_YEAR .. TARGET_START_YEAR + HORIZON_YEARS - 1]
    "TARGET_START_YEAR": 2010,
    "HORIZON_YEARS": 10,

    # Source year pool to sample from.
    # - Use None to auto-detect all years present in the CSV.
    # - Or provide a string like "2006-2019" OR "2006,2007,2009,2012".
    "YEAR_POOL": None,

    # Optional seed for reproducibility (set to an int or None for random)
    "SEED": 42,
}
# =========================




def is_leap(y: int) -> bool:
    return calendar.isleap(y)

def hours_in_year(y: int) -> int:
    return 8784 if is_leap(y) else 8760

def consecutive_years(start_year: int, n_years: int) -> List[int]:
    return [start_year + i for i in range(n_years)]

def parse_year_pool(spec: str) -> List[int]:
    """
    Parse a year-pool spec. Accepts either a range like "2006-2019"
    or a comma-separated list like "2006,2007,2009,2012".
    """
    spec = spec.strip()
    if '-' in spec and ',' not in spec:
        a, b = spec.split('-', 1)
        return list(range(int(a), int(b) + 1))
    years = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        years.append(int(tok))
    return years

def load_calliope_csv(csv_path: Path) -> Tuple[List[str], pd.DataFrame]:
    """
    Read a Calliope time_varying_parameters.csv-like file.

    Returns:
      header_lines: list of the first 5 lines (preserved verbatim)
      df: DataFrame with columns [0..M], where col 0 is the timestamp string
          and columns 1..M are numeric series; an extra '___YEAR___' column is added.
    """
    with csv_path.open('r', newline='') as f:
        header_lines = [next(f) for _ in range(5)]
    df = pd.read_csv(csv_path, skiprows=5, header=None)
    if df.empty:
        raise ValueError("No data rows found after the first 5 header lines.")
    ts = pd.to_datetime(df.iloc[:, 0], format="%Y/%m/%d %H:%M", errors='raise')
    df.insert(1, '___YEAR___', ts.dt.year)
    return header_lines, df

def split_year_blocks(df: pd.DataFrame, available_years: Sequence[int]) -> dict[int, pd.DataFrame]:
    """
    Return dict {year: subframe} for requested available_years.
    Validates hour counts for each year and sorts by timestamp.
    """
    blocks: dict[int, pd.DataFrame] = {}
    ts = pd.to_datetime(df.iloc[:, 0], format="%Y/%m/%d %H:%M", errors='raise')

    for y in available_years:
        mask = (ts.dt.year == y)
        sub = df.loc[mask].copy()
        if sub.empty:
            continue
        expected = hours_in_year(y)
        if len(sub) != expected:
            raise ValueError(
                f"Year {y} has {len(sub)} rows but expected {expected}. "
                f"Check the input file for completeness."
            )
        sub = sub.sort_values(by=0).reset_index(drop=True)
        blocks[y] = sub
    if not blocks:
        raise ValueError("No complete year blocks found in the provided year pool.")
    return blocks

def assign_sources_to_targets(
    source_years_pool: Sequence[int],
    target_years: Sequence[int],
    rng: random.Random
) -> List[int]:
    """
    Choose |target_years| distinct source years from source_years_pool,
    assign each target year a source year with matching leap/non-leap status.
    Returns a list of source years aligned to the order of target_years.
    """
    src_leaps = [y for y in source_years_pool if is_leap(y)]
    src_nonleaps = [y for y in source_years_pool if not is_leap(y)]
    tgt_leaps = [y for y in target_years if is_leap(y)]
    tgt_nonleaps = [y for y in target_years if not is_leap(y)]

    if len(tgt_leaps) > len(src_leaps):
        raise ValueError(
            f"Target window has {len(tgt_leaps)} leap years but source pool only "
            f"contains {len(src_leaps)} leap years. Choose a different target window or pool."
        )
    if len(tgt_nonleaps) > len(src_nonleaps):
        raise ValueError(
            f"Target window needs {len(tgt_nonleaps)} non-leap years but pool has only "
            f"{len(src_nonleaps)}. Choose a different target window or pool."
        )

    rng.shuffle(src_leaps)
    rng.shuffle(src_nonleaps)

    pick_leaps = src_leaps[: len(tgt_leaps)]
    pick_nonleaps = src_nonleaps[: len(tgt_nonleaps)]

    assigned: List[int] = []
    iL = iN = 0
    for ty in target_years:
        if is_leap(ty):
            assigned.append(pick_leaps[iL]); iL += 1
        else:
            assigned.append(pick_nonleaps[iN]); iN += 1
    return assigned

def reindex_block_to_year(sub: pd.DataFrame, out_year: int) -> pd.DataFrame:
    """
    Reindex the first column (timestamp string) of a single-year subframe
    to a new hourly index covering `out_year`.
    """
    n = len(sub)
    start = pd.Timestamp(year=out_year, month=1, day=1, hour=0, minute=0)
    new_index = pd.date_range(start=start, periods=n, freq="H")
    out = sub.copy()
    out.iloc[:, 0] = new_index.strftime("%Y/%m/%d %H:%M")
    return out.drop(columns=['___YEAR___'])

def generate_shuffled_files(
    csv_path: str | Path,
    n_files: int = 1,
    target_start_year: int = 2010,
    horizon_years: int = 10,
    year_pool: Sequence[int] | None = None,
    seed: int | None = None,
):
    """
    Generate `n_files` shuffled-year variants of a Calliope time series CSV.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    header_lines, df = load_calliope_csv(csv_path)

    ts = pd.to_datetime(df.iloc[:, 0], format="%Y/%m/%d %H:%M", errors='raise')
    data_years = sorted(ts.dt.year.unique().tolist())

    if year_pool is None:
        pool_years = data_years
    else:
        pool_years = [int(y) for y in (parse_year_pool(year_pool) if isinstance(year_pool, str) else year_pool)]
        pool_years = [y for y in pool_years if y in set(data_years)]
        if not pool_years:
            raise ValueError("Provided YEAR_POOL has no overlap with years present in the CSV.")

    blocks = split_year_blocks(df, pool_years)

    target_years = consecutive_years(target_start_year, horizon_years)
    tgt_leaps = sum(1 for y in target_years if is_leap(y))
    src_leaps_avail = sum(1 for y in blocks.keys() if is_leap(y))
    src_nonleaps_avail = len(blocks) - src_leaps_avail

    if tgt_leaps > src_leaps_avail:
        raise ValueError(
            f"Target window {target_years[0]}–{target_years[-1]} needs {tgt_leaps} leap years "
            f"but source pool has only {src_leaps_avail}. Adjust TARGET_START_YEAR/HORIZON_YEARS or YEAR_POOL."
        )
    if len(target_years) - tgt_leaps > src_nonleaps_avail:
        raise ValueError(
            f"Target window needs {len(target_years) - tgt_leaps} non-leap years but pool has only "
            f"{src_nonleaps_avail}. Adjust parameters."
        )

    rng = random.Random(seed)
    out_dir = csv_path.parent
    base = csv_path.stem
    target_label = f"{target_years[0]}-{target_years[-1]}"

    written = []
    for k in range(1, n_files + 1):
        assigned_src_years = assign_sources_to_targets(list(blocks.keys()), target_years, rng)

        out_parts = []
        for ty, sy in zip(target_years, assigned_src_years):
            sub = blocks[sy]
            out_parts.append(reindex_block_to_year(sub, ty))
        out_df = pd.concat(out_parts, axis=0, ignore_index=True)

        src_order_label = "-".join(str(y) for y in assigned_src_years)
        out_name = f"{base}__shuffle_{src_order_label}__as_{target_label}__{k:02d}.csv"
        out_path = out_dir / out_name

        with out_path.open('w', newline='') as f:
            for line in header_lines:
                f.write(line)
            out_df.to_csv(f, header=False, index=False, float_format="%.10g", lineterminator="\n")

        written.append(out_path)

    return written


if __name__ == "__main__":
    paths = generate_shuffled_files(
        csv_path=CONFIG["CSV_PATH"],
        n_files=CONFIG["N_FILES"],
        target_start_year=CONFIG["TARGET_START_YEAR"],
        horizon_years=CONFIG["HORIZON_YEARS"],
        year_pool=CONFIG["YEAR_POOL"],
        seed=CONFIG["SEED"],
    )
    print("Wrote:")
    for p in paths:
        print(" -", p)
