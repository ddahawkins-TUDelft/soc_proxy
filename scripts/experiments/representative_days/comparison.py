#!/usr/bin/env python3
"""
rep_heatmap_dates.py

Scan SoC_proxy_TSA/data/cluster_maps/*.csv, parse representative days given as
DATE STRINGS (e.g., '2016-01-01') in a column like 'PeriodNum', compute the
cardinality (count of days assigned to each representative), and render a
grid/heatmap comparison:

- X (columns): calendar days from GLOBAL min(rep_date) to GLOBAL max(rep_date)
- Y (rows): one row per cluster_map CSV (row label = file stem)
- Cell color (plasma): cardinality (# of days mapped to that representative date)
- Non-representative dates are blank (NaN)

Assumptions / heuristics:
- Representative date column is named one of:
  ['PeriodNum','Rep','Representative','RepresentativePeriod','AssignedRep',
   'ClusterID','cluster_id','rep'] and contains date-like strings or datetimes.
- If the CSV has one row per day (assignment table), cardinality is computed
  as value_counts of the representative date column.
- If the CSV is a list of representative dates (k rows), we try to find a count
  column ['Count','Cardinality','Freq','Frequency','count','cardinality'].
  If missing, we assume count=1 for those reps (still useful for comparison).
"""

from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# -------- Config --------
ROOT = Path("SoC_proxy_TSA") / "data" / "cluster_maps"
CSV_GLOB = "*.csv"

ASSIGN_COLS = [
    "PeriodNum", "Rep", "Representative", "RepresentativePeriod",
    "AssignedRep", "ClusterID", "cluster_id", "rep"
]
COUNT_COLS = ["Count", "Cardinality", "Freq", "Frequency", "count", "cardinality"]

# -------- Helpers --------

def _parse_datetime_col(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Try to parse df[col] as datetime; return Series if many non-nulls, else None."""
    if col not in df.columns:
        return None
    s = pd.to_datetime(df[col], errors="coerce", utc=False, infer_datetime_format=True)
    # Consider successful if at least 80% are valid datetimes
    valid_ratio = s.notna().mean()
    return s if valid_ratio >= 0.8 else None

def _find_repdate_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Find a column that holds representative dates and return parsed datetime Series (date-only)."""
    for col in ASSIGN_COLS:
        s = _parse_datetime_col(df, col)
        if s is not None:
            return s.dt.normalize()  # strip time, keep date
    # Fallback: try any datetime-like column
    for col in df.columns:
        s = _parse_datetime_col(df, col)
        if s is not None:
            return s.dt.normalize()
    return None

def _find_count_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in COUNT_COLS:
        if col in df.columns:
            ser = df[col]
            # numeric-ish?
            if pd.api.types.is_numeric_dtype(ser):
                return ser.astype(int)
            # strings of ints?
            try:
                return pd.to_numeric(ser, errors="coerce").fillna(0).astype(int)
            except Exception:
                pass
    return None

def load_rep_counts(path: Path) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Load a CSV and return:
      rep_dates: DatetimeIndex (unique representative dates)
      counts:    np.ndarray[int] cardinalities aligned to rep_dates

    Strategy:
    - Parse a representative-date column (date strings).
    - If file appears 'assignments per day', counts = value_counts(rep_date).
    - Else if file appears 'list of reps', use count column if present; else count=1.
    """
    df = pd.read_csv(path)
    rep_series = _find_repdate_series(df)
    if rep_series is None:
        raise ValueError("No date-like representative column found")

    # If there are many rows and repeated rep dates, treat as assignment table
    vc = rep_series.value_counts(dropna=True)
    if len(df) > len(vc) or vc.max() > 1:
        rep_dates = vc.index.sort_values()
        counts = vc.sort_index().to_numpy(dtype=int)
        return pd.DatetimeIndex(rep_dates), counts

    # Otherwise: likely a list of reps (k rows)
    cnt = _find_count_series(df)
    if cnt is None:
        # default to 1 for each listed representative
        rep_dates = rep_series.dropna().drop_duplicates().sort_values()
        counts = np.ones(len(rep_dates), dtype=int)
        return pd.DatetimeIndex(rep_dates), counts
    else:
        # align counts to the rep date rows
        # (coerce length mismatch by trimming/padding with ones)
        rep_dates = rep_series.dropna().reset_index(drop=True)
        # Drop duplicate rep_dates by summing counts if needed
        grp = pd.DataFrame({"rep": rep_dates, "cnt": cnt.fillna(0).astype(int)}).groupby("rep", as_index=False)["cnt"].sum()
        rep_dates = pd.DatetimeIndex(grp["rep"].sort_values())
        counts = grp.set_index("rep").loc[rep_dates]["cnt"].to_numpy(dtype=int)
        return rep_dates, counts

def build_matrix(file_to_repinfo: Dict[str, Tuple[pd.DatetimeIndex, np.ndarray]]) -> Tuple[np.ndarray, list, pd.DatetimeIndex]:
    """
    Build a 2D array M (rows=files, cols=global calendar days).
    For each file/row, set M[row, col_idx(date)] = count(date).
    Non-representative dates remain NaN.
    """
    # Global calendar span
    all_dates = pd.DatetimeIndex(sorted({d for rep_dates, _ in file_to_repinfo.values() for d in rep_dates}))
    if len(all_dates) == 0:
        raise ValueError("No representative dates found across files.")
    start, end = all_dates.min(), all_dates.max()
    calendar = pd.date_range(start=start, end=end, freq="D")  # global x-axis
    J = len(calendar)

    row_labels = list(file_to_repinfo.keys())
    M = np.full((len(row_labels), J), np.nan, dtype=float)

    pos = {d: i for i, d in enumerate(calendar)}

    for r, name in enumerate(row_labels):
        rep_dates, counts = file_to_repinfo[name]
        # place counts at matching calendar columns
        for d, c in zip(rep_dates, counts):
            idx = pos.get(d, None)
            if idx is not None:
                if np.isnan(M[r, idx]):
                    M[r, idx] = float(c)
                else:
                    M[r, idx] += float(c)  # in case of duplicates, aggregate

    return M, row_labels, calendar

def plot_heatmap(M: np.ndarray, row_labels: list[str], calendar: pd.DatetimeIndex,
                 title: str = "Representative-day cardinalities (calendar axis)"):
    # Colormap with NaN as white
    cmap = plt.cm.get_cmap("plasma").copy()
    cmap.set_bad(color="white")

    # Figure size heuristic: wide axis for long horizons, tall for many rows
    # fig_w = 0.8*min(20, max(10, 0.015 * len(calendar)))
    # fig_h = 0.8*min(16, max(3.5, 0.5 + 0.45 * len(row_labels)))

    fig_w = 12
    fig_h = 7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cluster map file")

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # X ticks: choose ~12 ticks max
    J = len(calendar)
    step = max(1, J // 12)
    xticks = np.arange(0, J, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([calendar[i].strftime("%Y-%m-%d") for i in xticks], rotation=45, ha="right")

    ax.set_xlim(-0.5, J - 0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cardinality (# days assigned to representative)")

    fig.tight_layout()
    return fig, ax

# -------- Main --------

def main(id_dict: dict = {}):
    root = ROOT
    if not root.exists():
        raise SystemExit(f"Directory not found: {root}")

    csv_paths = sorted(root.glob(CSV_GLOB))
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {root}")

    file_to_repinfo: Dict[str, Tuple[pd.DatetimeIndex, np.ndarray]] = {}
    print(f"[INFO] Found {len(csv_paths)} cluster maps in {root}")

    for p in csv_paths:

        if id_dict == {} or p.name in id_dict:
            try:
                rep_dates, counts = load_rep_counts(p)
                # basic sanity
                if len(rep_dates) != len(counts):
                    raise ValueError("rep_dates and counts length mismatch")
                if id_dict == {}:
                    name = p.stem
                else:
                    name = id_dict[p.name]
                file_to_repinfo[name] = (rep_dates, counts)

                # Console summary (first few reps)
                preview = ", ".join(f"{d.strftime('%Y-%m-%d')}:{int(c)}"
                                    for d, c in list(zip(rep_dates, counts))[:8])
                more = " ..." if len(rep_dates) > 8 else ""
                print(f"  - {name}: reps={len(rep_dates)} | {preview}{more}")

            except Exception as e:
                print(f"[WARN] Skipping {p.name}: {e}")

    if not file_to_repinfo:
        raise SystemExit("No usable cluster maps parsed.")

    M, row_labels, calendar = build_matrix(file_to_repinfo)
    fig, ax = plot_heatmap(M, row_labels, calendar,
                           title="Representative-day cardinalities across cluster_maps (date axis)")

    out_path = ROOT / "rep_heatmap_dates.png"
    fig.savefig(out_path, dpi=200)
    print(f"[INFO] Saved figure to: {out_path.resolve()}")
    plt.show()

dict_model_path = {
    'a9b18a4a394daf72b494.csv':'W=100, k=18' ,
    # 'a452a91aabe8c3239590.csv':'W=1',
    # '36776c7bae4e67309328.csv':'W=10',
    # '216e741ca87e9fac120f.csv':'W=100',
    # 'e153dec973f957cf3cbc.csv':'Endo'
}

if __name__ == "__main__":
    main(dict_model_path)
