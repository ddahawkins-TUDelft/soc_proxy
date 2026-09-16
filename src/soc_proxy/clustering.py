import numpy as np
import pandas as pd
import tsam.timeseriesaggregation as tsam
from typing import Dict, List, Tuple, Iterable

from dataclasses import dataclass, field

@dataclass
class ClusterResult:
    representatives: pd.DatetimeIndex         # chosen rep days
    assignment: pd.Series                     # index=all days, value=rep day
    cluster_map_path: str                     # where it was saved


# =========================
# MAIN
# =========================
def cluster_tsa_with_extremes(
    df_timeseries: pd.DataFrame, 
    number_typical_periods: int, 
    hours_per_period: int, 
    cluster_method: str, 
    rep_method: str, 
    path_to_cluster_csv: str,
    path_to_original_timeseries: str,   # kept for signature parity (unused)
    path_to_new_timeseries: str, 
    soc_proxy_dict: dict,
    weightDict: dict | None = None,
    soc_features: dict | None = None,
    # --- NEW (soft extremes / option C) ---
    extremes_spec: Dict[str, dict | str] | None = None,
    soft_prune: bool = False,
    tie_breaker_feature: str = "soc_net_MWh",  # used only if we need to prioritize in pruning ties
    save_cluster_map: bool = True
):
    """
    If extremes_spec is provided and soft_prune=True, we perform a soft force-then-prune step:
      - Candidate set S = TSAM medoids ∪ extreme dates derived from extremes_spec (per proxy).
      - Prune S back to k using drop-cost in the normalized+weighted feature space.
      - Build new typical periods from the ACTUAL chosen days.

    extremes_spec examples:
        {
          "soc_energy_debt": {"how": "max", "n": 1},   # deepest drawdown
          "soc_discharge_30d": "max",                  # heaviest multi-week discharge
          "soc_discharge_MWh": {"how": "max", "n": 1}  # peak daily discharge
        }
    """

    # --- header to prepend on export (as in your pipeline) ---
    calliope_field_headings = pd.read_csv(path_to_new_timeseries, header=None, nrows=5)

    # --- copy/track ---
    weightDict = dict(weightDict or {})
    added_cols_all: List[str] = []
    df_index = df_timeseries.index

    # --- scale soc_feature weights with proxy weights  ---
    if soc_features:
        if soc_proxy_dict.get("use_soc_proxy"):
            base_proxy = soc_proxy_dict["proxy_inputs_to_consider"][0]
            if base_proxy not in weightDict:
                raise KeyError(f"Proxy column '{base_proxy}' must have a base weight in weightDict.")
            for key in list(soc_features.keys()):
                soc_features[key] = soc_features[key] * weightDict[base_proxy]

    # --- Add ONLY requested SoC features (broadcast daily -> hourly), with suffixing if multiple proxies ---
    if soc_features and soc_proxy_dict.get("use_soc_proxy"):
        proxy_cols = list(soc_proxy_dict.get("proxy_inputs_to_consider", []))
        multi_proxy = len(proxy_cols) > 1
        for proxy_col in proxy_cols:
            suffix = proxy_col if multi_proxy else None
            df_timeseries, new_weights, added_cols = _add_requested_soc_features(
                df_timeseries, delta_col=proxy_col, requested=soc_features, suffix=suffix
            )
            weightDict.update(new_weights)
            added_cols_all.extend(added_cols)

    # =========================
    # TSAM aggregation (as-is)
    # =========================
    aggregation = tsam.TimeSeriesAggregation(
        df_timeseries, 
        noTypicalPeriods=number_typical_periods, 
        hoursPerPeriod=hours_per_period, 
        clusterMethod=cluster_method,
        representationMethod=rep_method,
        representationDict=None,
        weightDict=weightDict
    )
    typPeriods = aggregation.createTypicalPeriods()
    matched_indices = aggregation.indexMatching()

    # ===================================================
    # EXTREMES (selection independent) + optional soft prune
    # ===================================================
    # Build extreme dates only if extremes_spec has items
    extreme_dates = []
    if extremes_spec:
        extremes_csv = path_to_cluster_csv.replace(".csv", "_extremes_report.csv")
        extreme_dates, extreme_report = _collect_extreme_dates(
            df_timeseries=df_timeseries,
            soc_proxy_dict=soc_proxy_dict,
            extremes_spec=extremes_spec,
            # log_csv_path=extremes_csv,  # uncomment to save a CSV
            verbose=True
        )

    # TSAM medoid dates (or fallback)
    medoid_dates = _tsam_medoids_as_dates(df_timeseries, aggregation)

    # Candidate pool S = medoids ∪ extremes (stable de-dup)
    candidate_dates = _stable_union(medoid_dates, extreme_dates)

    # If no extremes or pool size == k, keep TSAM result untouched
    pool_grew = len(candidate_dates) > number_typical_periods

    if soft_prune and pool_grew:
        # ---------- PRUNE back to k (Option C) ----------
        cols_to_use = _numeric_cols(df_timeseries)
        scaled = _minmax_scale(df_timeseries[cols_to_use])
        scaled = _apply_weights(scaled, weightDict)
        X, day_index = _stack_days(scaled, hours_per_period)
        day_to_row = {pd.Timestamp(d): i for i, d in enumerate(day_index)}

        candidate_rows = [day_to_row[d] for d in candidate_dates if d in day_to_row]
        if not candidate_rows:
            raise ValueError("No candidate dates had complete daily coverage.")

        D = _pairwise_distances(X, X[candidate_rows])
        keep_rows = _prune_candidates_by_drop_cost(D, candidate_rows, target_k=number_typical_periods)

        chosen_dates = [day_index[r] for r in keep_rows]

        # Log which extremes survived
        extreme_set = set(extreme_dates)
        kept_extremes = [d for d in chosen_dates if d in extreme_set]
        dropped_extremes = [d for d in extreme_dates if d not in set(chosen_dates)]
        if kept_extremes:
            print(">>> TSA: Kept extremes:", ", ".join(str(d.date()) for d in kept_extremes))
        if dropped_extremes:
            print(">>> TSA: Dropped extremes:", ", ".join(str(d.date()) for d in dropped_extremes))

        # Reassign days to nearest kept representative
        D_final = _pairwise_distances(X, X[keep_rows])  # N × k
        assign_idx = D_final.argmin(axis=1)             # 0..k-1

        # pid -> representative date (0..k-1)
        pid_to_date = {pid: chosen_dates[pid] for pid in range(len(chosen_dates))}

        # Override typPeriods & matched_indices
        typPeriods = _build_typical_periods_from_dates(df=df_timeseries, chosen=pid_to_date, hours_per_period=hours_per_period)
        matched_indices = _rebuild_matched_indices(
            df_index=df_timeseries.index, assign_idx=assign_idx, days=day_index, hours_per_period=hours_per_period
        )

    elif (not soft_prune) and pool_grew:
        # ---------- KEEP ALL candidates: k becomes k + E ----------
        cols_to_use = _numeric_cols(df_timeseries)
        scaled = _minmax_scale(df_timeseries[cols_to_use])
        scaled = _apply_weights(scaled, weightDict)
        X, day_index = _stack_days(scaled, hours_per_period)
        day_to_row = {pd.Timestamp(d): i for i, d in enumerate(day_index)}

        # Keep medoids first, then extremes not already in medoids (stable union did that)
        chosen_dates = [d for d in candidate_dates if d in day_to_row]
        if not chosen_dates:
            raise ValueError("No candidate dates had complete daily coverage.")

        chosen_rows = [day_to_row[d] for d in chosen_dates]

        # Assign each day to its nearest representative among the whole pool (k+E)
        D_final = _pairwise_distances(X, X[chosen_rows])    # N × (k+E)
        assign_idx = D_final.argmin(axis=1)                 # 0..(k+E-1)

        # pid -> representative date (0..k+E-1)
        pid_to_date = {pid: chosen_dates[pid] for pid in range(len(chosen_dates))}

        # Override typPeriods & matched_indices using ALL candidates
        typPeriods = _build_typical_periods_from_dates(df=df_timeseries, chosen=pid_to_date, hours_per_period=hours_per_period)
        matched_indices = _rebuild_matched_indices(
            df_index=df_timeseries.index, assign_idx=assign_idx, days=day_index, hours_per_period=hours_per_period
        )

        print(f"[TSA] Using k + E representatives = {len(chosen_dates)} "
              f"(k={number_typical_periods}, E={len(chosen_dates) - number_typical_periods}).")
    # else: either no extremes, or pool didn't grow → leave TSAM output as-is


    # =========================
    # EXPORT (unchanged shape)
    # =========================

    if matched_indices.index.duplicated().any():
        dup_count = matched_indices.index.duplicated().sum()
        print(f"[TSA] Warning: dropping {dup_count} duplicated timesteps in matched_indices.")
        matched_indices = matched_indices[~matched_indices.index.duplicated(keep="first")]
        
    # Merge to produce clustered time series
    typPeriods = typPeriods.reset_index().rename(columns={'level_0': 'PeriodNum'})
    df_new_timeseries_values = matched_indices.merge(
        typPeriods, on=['PeriodNum', 'TimeStep'], how='left'
    )

    # Drop helper cols and engineered features from export
    df_new_timeseries_values.drop(['PeriodNum', 'TimeStep'], axis=1, inplace=True)
    if soc_proxy_dict.get("use_soc_proxy"):
        df_new_timeseries_values.drop(
            columns=soc_proxy_dict.get("proxy_inputs_to_consider", []),
            errors="ignore", inplace=True
        )
        if added_cols_all:
            df_new_timeseries_values.drop(columns=added_cols_all, errors="ignore", inplace=True)

    # Final index formatting for export
    df_new_timeseries_values.index = df_index.strftime('%Y/%m/%d %H:%M')

    # --- Build Calliope-friendly cluster map: map numeric PeriodNum -> representative DATE ---

    # 1) Build rep_map: {PeriodNum (int) -> representative date (Timestamp)}
    if 'pid_to_date' in locals():
        # soft-prune branch created pid_to_date: 0..k-1 -> Timestamp
        rep_map = {int(pid): pd.Timestamp(dt).normalize() for pid, dt in pid_to_date.items()}
    elif getattr(aggregation, "clusterCenterIndices", None):
        # TSAM medoids available: use medoid date per cluster id
        medoid_dates = (
            df_timeseries
            .resample("1D").first()
            .iloc[aggregation.clusterCenterIndices]
            .index
        )
        rep_map = {int(pid): pd.Timestamp(dt).normalize() for pid, dt in enumerate(medoid_dates)}
    else:
        # Fallback: first date encountered for each PeriodNum
        tmp = matched_indices.copy()
        tmp.index = pd.to_datetime(tmp.index)
        tmp["date"] = tmp.index.floor("D")
        first_dates = (
            tmp.reset_index(names="ts")
            .groupby("PeriodNum").first()["date"]
        )
        rep_map = {int(pid): pd.Timestamp(dt).normalize() for pid, dt in first_dates.items()}

    # 2) Create cluster_days with dates in BOTH columns (Calliope expects dates)
    cluster_days = (
        matched_indices
        .resample("1D").first()[["PeriodNum"]]             # daily PeriodNum (numeric)
        .assign(PeriodNum=lambda x: x["PeriodNum"].map(rep_map))  # map to representative DATE
    )
    cluster_days.index.name = "timesteps"

    # Ensure plain dates (no time) on both columns
    cluster_days.index = pd.to_datetime(cluster_days.index).normalize()
    cluster_days["PeriodNum"] = pd.to_datetime(cluster_days["PeriodNum"]).dt.normalize()

    #ensure columns align

    df_aligned = align_by_third_row(
    df_new_timeseries_values,
    calliope_field_headings,
    header_row_index=2,      # 3rd row
    keep_extras_at_end=False # or True if you want non-matching columns appended
)

    # Write files (same as before)
    # calliope_field_headings.to_csv(path_to_new_timeseries, index=False, header=False, mode="w") #TODO: time series saving was weirdly buggy.
    # df_aligned.to_csv(path_to_new_timeseries, index=True, header=False, mode="a")
    if save_cluster_map:
        cluster_days.to_csv(path_to_cluster_csv)

    # -------- NEW: prepare ClusterResult return --------
    representatives = pd.DatetimeIndex(
        sorted(pd.unique(cluster_days["PeriodNum"])), name="timesteps"
    )
    assignment = cluster_days["PeriodNum"].copy()  # Series: index=all days, value=rep date

    print(f"[TSA] successfully applied{' with soft extremes' if soft_prune and extremes_spec else ''}. "
        f"Results saved to {path_to_cluster_csv}.")

    # If you keep meta, you can fill a small summary; else drop 'meta=...'
    return ClusterResult(
        representatives=representatives,
        assignment=assignment,
        cluster_map_path=path_to_cluster_csv,
    )


# =========================
# 
# =========================
def align_by_third_row(
    df_values: pd.DataFrame,
    df_headings: pd.DataFrame,
    header_row_index: int = 2,   # 0-based → 3rd row
    keep_extras_at_end: bool = False
) -> pd.DataFrame:
    """
    Reorder df_values columns to match the column labels found in the given row of df_headings.

    - Strips whitespace in labels.
    - Drops columns in df_values not present in the target order unless keep_extras_at_end=True.
    - Adds any missing columns (filled with NaN).
    """

    # 1) Get target order from the specified row
    target_order = (
        df_headings.iloc[header_row_index]
        .astype(str)
        .str.strip()
        .tolist()
    )

    target_order.pop(0)

    # Remove obvious junk (NaNs converted to 'nan')
    target_order = [c for c in target_order if c and c.lower() != "nan"]

    # 2) Normalize current value columns
    current_cols = df_values.columns.astype(str).str.strip().tolist()

    # 3) Ensure all target columns exist in df_values (add if missing)
    missing = [c for c in target_order if c not in current_cols]
    for c in missing:
        df_values[c] = pd.NA

    # 4) Build final order
    if keep_extras_at_end:
        extras = [c for c in df_values.columns if c not in target_order]
        final_order = target_order + extras
    else:
        final_order = target_order

    # 5) Reindex to that order (dropping extras if not kept)
    return df_values.reindex(columns=final_order)

def _add_requested_soc_features(
    df: pd.DataFrame,
    delta_col: str,
    requested: Dict[str, float],
    suffix: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    """
    Compute ONLY the requested SoC features from the hourly ΔSoC column `delta_col`,
    broadcast them to hourly resolution, and return:
        - augmented dataframe,
        - a weight dict mapping *actual column names* -> weight,
        - a list of added column names (for cleanup on export).
    If `suffix` is provided, feature columns are named '{feature}__{suffix}'.
    """
    d = df.copy()
    s = d[delta_col]
    by_day = s.resample("1D")

    # Base daily building blocks
    daily_net = by_day.sum()                                     # MWh/day (+ charge, - discharge)
    daily_dis = daily_net.clip(upper=0).abs()                    # discharge-only MWh/day

    def _name(f): return f"{f}__{suffix}" if suffix else f

    feature_builders = {
        "soc_activity_MWh":       lambda: by_day.apply(lambda x: x.abs().sum()).rename(_name("soc_activity_MWh")),
        "soc_charge_MWh":         lambda: by_day.apply(lambda x: x.clip(lower=0).sum()).rename(_name("soc_charge_MWh")),
        "soc_discharge_MWh":      lambda: by_day.apply(lambda x: (-x.clip(upper=0)).sum()).rename(_name("soc_discharge_MWh")),
        "soc_net_MWh":            lambda: daily_net.rename(_name("soc_net_MWh")),
        "soc_max_charge_rate":    lambda: by_day.max().rename(_name("soc_max_charge_rate")),
        "soc_max_discharge_rate": lambda: (-by_day.min()).rename(_name("soc_max_discharge_rate")),
        "soc_peak10h_charge":     lambda: by_day.apply(lambda x: x.rolling(10, min_periods=1).sum().max()).rename(_name("soc_peak10h_charge")),
        "soc_peak10h_discharge":  lambda: by_day.apply(lambda x: (-x).rolling(10, min_periods=1).sum().max()).rename(_name("soc_peak10h_discharge")),
        "soc_discharge_30d":      lambda: daily_dis.rolling(30,  min_periods=1).sum().rename(_name("soc_discharge_30d")),
        "soc_discharge_90d":      lambda: daily_dis.rolling(90,  min_periods=1).sum().rename(_name("soc_discharge_90d")),
        "soc_energy_debt":        lambda: _energy_debt(daily_net).rename(_name("soc_energy_debt")),
    }

    daily_feats = []
    actual_weights: Dict[str, float] = {}
    added_cols: List[str] = []

    for feat, wt in (requested or {}).items():
        if feat not in feature_builders:
            continue
        ser = feature_builders[feat]()   # daily-indexed Series
        daily_feats.append(ser)
        actual_weights[ser.name] = float(wt)
        added_cols.append(ser.name)

    if daily_feats:
        daily_df = pd.concat(daily_feats, axis=1)
        d = (
            d.assign(__date=d.index.floor("D"))
             .join(daily_df, on="__date")
             .drop(columns="__date")
        )

    return d, actual_weights, added_cols


def _energy_debt(daily_net: pd.Series) -> pd.Series:
    """Cumulative discharge 'debt' that resets when net >= 0 (simple drawdown metric)."""
    acc = 0.0
    out = []
    for x in daily_net:
        acc = max(acc + (-x), 0.0)  # x > 0 reduces debt; x < 0 increases debt
        out.append(acc)
    return pd.Series(out, index=daily_net.index)


# =========================
# NEW helpers for soft extremes
# =========================
def _collect_extreme_dates(
    df_timeseries: pd.DataFrame,
    soc_proxy_dict: dict,
    extremes_spec: Dict[str, dict | str],
    log_csv_path: str | None = None,     # NEW: set a path to save CSV, or None to skip
    verbose: bool = True                 # NEW: print to console
) -> Tuple[List[pd.Timestamp], List[dict]]:
    """
    Returns (extreme_dates, report_rows).
    report_rows is a list of dicts with keys: date, feature, how, rank, value.
    """
    proxy_cols = list(soc_proxy_dict.get("proxy_inputs_to_consider", [])) if soc_proxy_dict.get("use_soc_proxy") else []
    if not proxy_cols:
        return [], []

    all_dates: List[pd.Timestamp] = []
    all_rows: List[dict] = []
    multi = len(proxy_cols) > 1

    for proxy_col in proxy_cols:
        daily = _daily_soc_features(df_timeseries, delta_col=proxy_col)
        dates, rows = _select_extreme_dates(
            daily_feats=daily,
            extremes_spec=extremes_spec,
            top_n_default=1,
            feature_prefix=(proxy_col if multi else None),
            return_report=True
        )
        all_dates.extend(dates)
        all_rows.extend(rows)

    # stable unique on dates
    seen = set()
    all_dates = [d for d in all_dates if not (d in seen or seen.add(d))]

    # optional: print nicely
    if verbose and all_rows:
        print(">>> TSA: Extreme-day candidates (before pruning):")
        # group by date for a compact display
        rows_by_date = {}
        for r in all_rows:
            rows_by_date.setdefault(r["date"], []).append(r)
        for dt, rows in sorted(rows_by_date.items()):
            reasons = ", ".join(f"{r['feature']}({r['how']}, rank={r['rank']}, val={r['value']:.3g})" for r in rows)
            print(f">>> TSA: - {dt.date()}: {reasons}")

    # optional: save CSV
    if log_csv_path and all_rows:
        df_log = pd.DataFrame(all_rows).sort_values(["date", "feature", "rank"])
        df_log.to_csv(log_csv_path, index=False)

    return all_dates, all_rows



def _daily_soc_features(df: pd.DataFrame, delta_col: str) -> pd.DataFrame:
    """Per-day SoC-delta features (no broadcasting); index at midnight."""
    s = df[delta_col]
    by_day = s.resample("1D")

    daily_net = by_day.sum().rename("soc_net_MWh")
    daily_dis = daily_net.clip(upper=0).abs().rename("soc_discharge_MWh")
    daily_chg = by_day.apply(lambda x: x.clip(lower=0).sum()).rename("soc_charge_MWh")
    activity  = by_day.apply(lambda x: x.abs().sum()).rename("soc_activity_MWh")
    max_ch    = by_day.max().rename("soc_max_charge_rate")
    max_dis   = (-by_day.min()).rename("soc_max_discharge_rate")

    # sustained ramps (10h)
    def peak10h_pos(x): return x.rolling(10, min_periods=1).sum().max()
    def peak10h_neg(x): return (-x).rolling(10, min_periods=1).sum().max()
    peak10_c  = by_day.apply(peak10h_pos).rename("soc_peak10h_charge")
    peak10_d  = by_day.apply(peak10h_neg).rename("soc_peak10h_discharge")

    dis_30d = daily_dis.rolling(30, min_periods=1).sum().rename("soc_discharge_30d")
    dis_90d = daily_dis.rolling(90, min_periods=1).sum().rename("soc_discharge_90d")

    debt = _energy_debt(daily_net).rename("soc_energy_debt")

    return pd.concat(
        [activity, daily_chg, daily_dis, daily_net, max_ch, max_dis,
         peak10_c, peak10_d, dis_30d, dis_90d, debt],
        axis=1
    )


def _select_extreme_dates(
    daily_feats: pd.DataFrame,
    extremes_spec: Dict[str, dict | str | List[dict] | List[str]],
    top_n_default: int = 1,
    feature_prefix: str | None = None,   # e.g. proxy name
    return_report: bool = False
) -> List[pd.Timestamp] | Tuple[List[pd.Timestamp], List[dict]]:
    """
    Supports value being a dict/str *or a list* of dict/str per feature, so you can
    request both min and max, e.g.:
        {"soc_charge_MWh": [{"how":"max","n":1}, {"how":"min","n":1}]}
    Returns a de-duplicated list of dates. If return_report=True, also returns a list of dicts:
      {"date", "feature", "how", "rank", "value"}.
    """
    picks: List[pd.Timestamp] = []
    report: List[dict] = []

    for feat, rule in (extremes_spec or {}).items():
        if feat not in daily_feats.columns:
            continue

        # Normalize to a list of rule-objects
        rules = rule if isinstance(rule, list) else [rule]

        for robj in rules:
            if isinstance(robj, str):
                how, n = robj, top_n_default
            elif isinstance(robj, dict):
                how = robj.get("how", "max")
                n = int(robj.get("n", top_n_default))
            else:
                continue

            series = daily_feats[feat].dropna()
            if series.empty:
                continue

            top = series.nlargest(n) if how == "max" else series.nsmallest(n)

            for rank, (dt, val) in enumerate(top.items(), start=1):
                dt_norm = pd.Timestamp(dt).normalize()
                picks.append(dt_norm)
                if return_report:
                    rep_name = f"{feat}__{feature_prefix}" if feature_prefix else feat
                    report.append({
                        "date": dt_norm,
                        "feature": rep_name,
                        "how": how,
                        "rank": rank,
                        "value": float(val),
                    })

    # Stable unique on dates (order preserved by first appearance)
    seen = set()
    ordered_unique = [d for d in picks if not (d in seen or seen.add(d))]
    if return_report:
        return ordered_unique, report
    return ordered_unique




def _tsam_medoids_as_dates(df_timeseries: pd.DataFrame, aggregation) -> List[pd.Timestamp]:
    """Try to read TSAM's medoid dates; fallback to first date per cluster."""
    if getattr(aggregation, "clusterCenterIndices", None):
        return (
            df_timeseries
            .resample("1D")
            .first()
            .iloc[aggregation.clusterCenterIndices]
            .index
            .tolist()
        )
    # Fallback: first date representing each cluster label
    tmp = aggregation.indexMatching().copy()
    tmp.index = pd.to_datetime(tmp.index)
    tmp["date"] = tmp.index.floor("D")
    first_dates = tmp.reset_index(names="ts").groupby("PeriodNum").first()["date"]
    return list(first_dates.sort_index().values)


def _stable_union(a: Iterable[pd.Timestamp], b: Iterable[pd.Timestamp]) -> List[pd.Timestamp]:
    seen = set()
    out: List[pd.Timestamp] = []
    for seq in (a, b):
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(pd.Timestamp(x).normalize())
    return out


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _minmax_scale(df_num: pd.DataFrame) -> pd.DataFrame:
    mins = df_num.min(axis=0)
    maxs = df_num.max(axis=0)
    rng = (maxs - mins).replace(0, 1.0)
    return (df_num - mins) / rng


def _apply_weights(df_num: pd.DataFrame, weightDict: Dict[str, float]) -> pd.DataFrame:
    w = pd.Series({c: float(weightDict.get(c, 1.0)) for c in df_num.columns})
    return df_num.mul(w, axis=1)


def _stack_days(df_scaled: pd.DataFrame, hours_per_period: int) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """
    Convert scaled, weighted hourly dataframe into a matrix of day-vectors.
    Returns (X: N_days × (hours*cols), day_index: list of midnight Timestamps).
    Drops incomplete days (less than hours_per_period rows).
    """
    cols = df_scaled.columns
    day_starts = df_scaled.index.floor("D").unique()
    rows = []
    days = []
    for d in day_starts:
        block = df_scaled.loc[d : d + pd.Timedelta(hours=hours_per_period - 1)]
        if len(block) != hours_per_period:
            continue  # skip incomplete days (e.g., DST edges)
        rows.append(block[cols].to_numpy().reshape(-1))
        days.append(pd.Timestamp(d))
    X = np.vstack(rows) if rows else np.empty((0, len(cols) * hours_per_period))
    return X, days


def _pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distances between rows of X and rows of Y."""
    # ‖x - y‖^2 = ‖x‖^2 + ‖y‖^2 - 2 x·y
    X2 = np.sum(X * X, axis=1)[:, None]
    Y2 = np.sum(Y * Y, axis=1)[None, :]
    XY = X @ Y.T
    D2 = np.maximum(X2 + Y2 - 2.0 * XY, 0.0)
    return np.sqrt(D2, dtype=np.float64)


def _prune_candidates_by_drop_cost(
    D_all_to_pool: np.ndarray,
    pool_rows: List[int],
    target_k: int
) -> List[int]:
    """
    D_all_to_pool: N x m distances from all days to candidate pool (order matches pool_rows).
    pool_rows: indices (into X) of current pool (size m).
    Returns the kept pool rows (size target_k) using iterative drop-cost pruning.
    """
    N, m = D_all_to_pool.shape
    pool_idx = list(range(m))  # local indices 0..m-1

    while len(pool_idx) > target_k:
        # distances restricted to current pool
        D_S = D_all_to_pool[:, pool_idx]  # N × |S|
        order = np.argpartition(D_S, kth=1, axis=1)[:, :2]  # indices of two smallest per row (unordered pair)
        # get best & second-best distances (ensure we label them)
        best = np.take_along_axis(D_S, order, axis=1)
        # sort the two columns so [:,0] is best, [:,1] is second-best
        swap = best[:, 1] < best[:, 0]
        best[swap] = best[swap][:, ::-1]
        idx2 = order.copy()
        idx2[swap] = idx2[swap][:, ::-1]
        best_idx_local = idx2[:, 0]   # local 0..|S|-1
        best_dist = best[:, 0]
        second_dist = best[:, 1]

        # cost(j) = sum over days assigned to j of (second - best)
        # accumulate by candidate
        costs = np.zeros(len(pool_idx), dtype=np.float64)
        np.add.at(costs, best_idx_local, (second_dist - best_dist))

        # remove the candidate with smallest drop-cost
        to_remove_local = int(np.argmin(costs))
        del pool_idx[to_remove_local]

    # map back to original X row indices
    return [pool_rows[i] for i in pool_idx]


def _build_typical_periods_from_dates(
    df: pd.DataFrame,
    chosen: Dict[int, pd.Timestamp],
    hours_per_period: int
) -> pd.DataFrame:
    """
    Construct typPeriods-like DataFrame indexed by (PeriodNum, TimeStep),
    slicing original hourly data for the chosen dates.
    """
    blocks = []
    for pid, date in sorted(chosen.items()):
        day = df.loc[date: (date + pd.Timedelta(hours=hours_per_period - 1))]
        if len(day) != hours_per_period:
            raise ValueError(f"Day {date.date()} has {len(day)} rows; expected {hours_per_period}.")
        tmp = day.copy()
        tmp = tmp.assign(PeriodNum=pid, TimeStep=np.arange(hours_per_period))
        blocks.append(tmp)
    typ = pd.concat(blocks).set_index(["PeriodNum", "TimeStep"]).sort_index()
    return typ


def _rebuild_matched_indices(
    df_index: pd.DatetimeIndex,
    assign_idx: np.ndarray,           # length = number of complete days
    days: List[pd.Timestamp],         # same order as rows of X
    hours_per_period: int
) -> pd.DataFrame:
    """
    Build a new matched_indices DataFrame (index=hourly timestamps) with PeriodNum/TimeStep
    based on the assignment 'assign_idx' (0..k-1) for each complete day in 'days'.
    """
    idx = pd.Index(df_index, name="index")
    df = pd.DataFrame(index=idx)

    # Map each day start -> cluster id
    day_to_pid = {pd.Timestamp(d): int(assign_idx[i]) for i, d in enumerate(days)}
    df["PeriodNum"] = df.index.floor("D").map(day_to_pid)

    # Drop hours from days that weren't in 'days' (should be rare)
    df = df.dropna(subset=["PeriodNum"])
    df["PeriodNum"] = df["PeriodNum"].astype(int)

    # Determine the base timestep (default 1 hour). This also handles 30-min data, etc.
    if len(df.index) >= 2:
        base_step = df.index[1] - df.index[0]
        if base_step <= pd.Timedelta(0):
            base_step = pd.Timedelta(hours=1)
    else:
        base_step = pd.Timedelta(hours=1)

    # TimeStep = integer bin within the day: (time since midnight) // base_step
    offset = df.index - df.index.floor("D")
    df["TimeStep"] = (offset // base_step).astype(int)

    return df
