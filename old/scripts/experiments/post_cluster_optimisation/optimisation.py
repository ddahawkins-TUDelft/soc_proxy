# utility_functions/helper_post_cluster_opt.py
from __future__ import annotations
from dataclasses import replace
from typing import Dict, List, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd

from .targets import TargetRegistry, ReferenceTargetRegistry
from soc_proxy.clustering import ClusterResult

if TYPE_CHECKING:
    from soc_proxy.tsa_model import tsa_model


def parse_objective_spec(spec_list: List[Dict]) -> Tuple[List[str], Dict[str, float]]:
    targets, weights = [], {}
    for item in spec_list:
        t = item["target"]
        targets.append(t)
        weights[t] = float(item.get("weight", 1.0))
    return targets, weights


def build_target_cost_matrix_two(
    df_days: pd.DataFrame,  # reference (days x features)
    df_reps: pd.DataFrame,  # test representatives (reps x features)
    metric: str = "mae",
    weights: Dict[str, float] | None = None,
) -> np.ndarray:
    """
    Returns N x P cost matrix (rows = days, cols = reps) using weighted MAE/MSE.
    Assumes df_days and df_reps have identical columns in same order.
    """
    cols = list(df_days.columns)
    X = df_days[cols].to_numpy(dtype=float)  # N x F
    C = df_reps[cols].to_numpy(dtype=float)  # P x F
    diff = X[:, None, :] - C[None, :, :]  # N x P x F

    if metric == "mae":
        per_feat = np.abs(diff)
    elif metric == "mse":
        per_feat = diff**2
    else:
        raise ValueError("Unknown metric")

    if weights:
        # If features are hourly profiles (foo_h00,...), apply one weight per root target.
        wf = np.ones(len(cols))
        if any(c.endswith("_h00") for c in cols):
            base = {}
            for i, c in enumerate(cols):
                root = c.split("_h")[0]
                base.setdefault(root, weights.get(root, 1.0))
                wf[i] = base[root]
        else:
            wf = np.array([weights.get(c, 1.0) for c in cols])
        per_feat = per_feat * wf[None, None, :]

    return per_feat.mean(axis=2)  # N x P


def solve_reassignment_argmin(
    days_index: pd.Index, reps_index: pd.Index, D: np.ndarray
) -> pd.Series:
    """
    Argmin over columns with deterministic tie-break:
    - We expect callers to have applied a tiny increasing epsilon per column to D.
    """
    choice = np.argmin(D, axis=1)
    mapped = reps_index.values[choice]
    return pd.Series(index=days_index, data=mapped, name="PeriodNum")


# ----------------------- File I/O: single source of truth ---------------------


def _read_cluster_map(path: str) -> pd.DataFrame:
    """
    Read the existing cluster map CSV without altering 'timesteps' values or order.
    Returns a DataFrame with columns ['timesteps', 'PeriodNum'] as strings.
    """
    df = pd.read_csv(path, dtype={"timesteps": str, "PeriodNum": str})
    if list(df.columns) != ["timesteps", "PeriodNum"]:
        raise ValueError(
            f"Cluster map at {path} must have columns ['timesteps','PeriodNum'] in that order."
        )
    return df


def _write_cluster_map(
    df_original_strings: pd.DataFrame, new_periodnum_dt: pd.Series, out_path: str
) -> None:
    """
    Write back with identical 'timesteps' strings and PeriodNum formatted YYYY-MM-DD.
    Robust to new_periodnum_dt being a Series or a DatetimeIndex.
    """
    out = df_original_strings.copy()

    # Ensure we have a 1D array the same length as the file rows
    if hasattr(new_periodnum_dt, "to_numpy"):
        arr_dt = pd.to_datetime(new_periodnum_dt, errors="coerce").to_numpy()
    else:
        arr_dt = pd.to_datetime(pd.Series(new_periodnum_dt), errors="coerce").to_numpy()

    if len(arr_dt) != len(out):
        raise ValueError(
            f"Length mismatch in _write_cluster_map: PeriodNum len {len(arr_dt)} "
            f"!= file rows {len(out)}"
        )

    # Start from original strings; we will overwrite only where we have valid datetimes
    period_str = out["PeriodNum"].astype(str).copy()

    # Build formatted strings for valid entries; keep originals where invalid (avoid 'NaT')
    valid_mask = ~pd.isna(arr_dt)
    formatted_str = pd.Series(
        pd.to_datetime(arr_dt[valid_mask]).strftime("%Y-%m-%d"),
        index=np.flatnonzero(valid_mask),
    )

    # Assign by integer position to avoid .loc on Index vs Series issues
    period_str.iloc[valid_mask] = formatted_str.values

    out["PeriodNum"] = period_str.str.strip()
    out["timesteps"] = out["timesteps"].astype(str)  # preserve as-is
    out = out[["timesteps", "PeriodNum"]]
    out.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")


# ------------------------------- Main routine --------------------------------


def apply_optimisation_on_cluster(
    cluster_result: ClusterResult, model: "tsa_model"
) -> ClusterResult:
    """
    If post_cluster_optimisation_params.debug_shuffle == True, skip optimisation and
    simply shuffle PeriodNum deterministically with a fixed seed (shuffle_seed, default 42).
    Otherwise, run the deterministic reassignment routine.
    """
    pco = model.tsa.params.get("post_cluster_optimisation_params") or {}

    spec = pco.get("objective", [])
    if not spec:
        _ = _read_cluster_map(cluster_result.cluster_map_path)
        return cluster_result

    ref_cfg = pco.get("reference", {})
    agg_mode = pco.get("aggregation", "daily_mean")
    H = int(model.tsa.params.get("hours_per_period", 24))

    targets, weights = parse_objective_spec(spec)

    df_map_str = _read_cluster_map(cluster_result.cluster_map_path)

    # Canonical orders
    days_full_dt = pd.to_datetime(df_map_str["timesteps"], errors="coerce")
    reps_order_str = (
        df_map_str["PeriodNum"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .drop_duplicates()
    )
    reps_full_dt = pd.to_datetime(reps_order_str, errors="coerce").dropna()
    reps_full_dt = pd.DatetimeIndex(reps_full_dt)

    # Build features
    reg_model = TargetRegistry(model)
    reg_ref = ReferenceTargetRegistry(model, ref_cfg)

    if agg_mode == "hourly_profile":
        df_ref_all = reg_ref.daily_profile_stacked(targets, hours_per_period=H)
        df_test_all = reg_model.daily_profile_stacked(targets, hours_per_period=H)
    elif agg_mode == "daily_sum":
        df_ref_all = reg_ref.daily_stacked(targets, how="sum")
        df_test_all = reg_model.daily_stacked(targets, how="sum")
    else:
        df_ref_all = reg_ref.daily_stacked(targets, how="mean")
        df_test_all = reg_model.daily_stacked(targets, how="mean")

    # Align (preserve CSV order)
    df_ref_full = df_ref_all.reindex(pd.DatetimeIndex(days_full_dt))
    df_reps_full = df_test_all.reindex(reps_full_dt)

    valid_day_mask = ~df_ref_full.isna().any(axis=1)
    valid_rep_mask = ~df_reps_full.isna().any(axis=1)

    valid_days = df_ref_full.index[valid_day_mask.values]
    valid_reps = df_reps_full.index[valid_rep_mask.values]

    new_assign_full_dt = pd.to_datetime(df_map_str["PeriodNum"], errors="coerce")

    if len(valid_reps) == 0:
        _write_cluster_map(
            df_map_str, new_assign_full_dt, cluster_result.cluster_map_path
        )
        updated_assignment = pd.Series(
            index=pd.DatetimeIndex(days_full_dt),
            data=pd.DatetimeIndex(new_assign_full_dt),
            name="PeriodNum",
        )
        return replace(cluster_result, assignment=updated_assignment)

    df_ref_valid = df_ref_full.loc[valid_days]
    df_reps_valid = df_reps_full.loc[valid_reps]

    common_cols = sorted(set(df_ref_valid.columns) & set(df_reps_valid.columns))
    if not common_cols:
        _write_cluster_map(
            df_map_str, new_assign_full_dt, cluster_result.cluster_map_path
        )
        updated_assignment = pd.Series(
            index=pd.DatetimeIndex(days_full_dt),
            data=pd.DatetimeIndex(new_assign_full_dt),
            name="PeriodNum",
        )
        return replace(cluster_result, assignment=updated_assignment)

    df_ref_valid = df_ref_valid[common_cols]
    df_reps_valid = df_reps_valid[common_cols]

    if len(valid_days) > 0:
        D = build_target_cost_matrix_two(
            df_days=df_ref_valid, df_reps=df_reps_valid, metric="mae", weights=weights
        )

        P = D.shape[1]
        if P > 0:
            span = float(np.nanmax(D) - np.nanmin(D)) if np.isfinite(D).all() else 1.0
            base = span if span > 0.0 else 1.0
            eps = base * 1e-12
            D = D + eps * np.arange(P, dtype=float)[None, :]

        reassigned = solve_reassignment_argmin(valid_days, df_reps_valid.index, D)

        day_lookup = pd.Index(days_full_dt)
        is_valid_row = day_lookup.isin(valid_days)
        new_assign_full_dt.loc[is_valid_row] = pd.to_datetime(
            reassigned.reindex(valid_days).values
        )

    bad_vals = pd.DatetimeIndex(
        pd.Series(new_assign_full_dt).dropna().unique()
    ).difference(reps_full_dt)
    if len(bad_vals) > 0:
        raise ValueError(
            f"Remap produced representatives not in original set: {list(bad_vals)}"
        )

    _write_cluster_map(df_map_str, new_assign_full_dt, cluster_result.cluster_map_path)

    updated_assignment = pd.Series(
        index=pd.DatetimeIndex(days_full_dt),
        data=pd.DatetimeIndex(new_assign_full_dt),
        name="PeriodNum",
    )
    return replace(cluster_result, assignment=updated_assignment)
