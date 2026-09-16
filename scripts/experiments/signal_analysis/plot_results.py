# plot_signal_results.py
# Combined signal analyses:
#   (A) Ex-ante vs ex-post: LDES capacity error vs SoC-proxy metrics
#       - Subplot 1: Pearson r & RMSE (trendlines + R^2)
#       - Subplot 2: Peak timing & magnitude errors
#       - Subplot 3: Combined error metric (trendline + R^2)
#   (B) Segmented (monthly) RMSE analysis per model, colored by LDES error
#   (C) Month-importance analysis: correlation between monthly (normalized) RMSE
#       AND monthly Pearson r vs LDES error across models, with a mini SoC strip
#
# Conventions:
# - Proxies are built hourly, with optional resampling (mean) AFTER proxy computation (e.g., daily).
# - Time alignment uses the intersection of test vs reference timestamps.
# - Paths and helper functions mirror your existing codebase.

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import calliope
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import Normalize
from netCDF4 import Dataset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats

# --- Utilities from your codebase ---
from soc_proxy import generate_soc_proxy
from soc_proxy.calliope.timeseries import (
    calliope_ts_to_pandas,
    extrapolate_ts_from_cluster_map,
)

# ------------------------------
# Config / paths
# ------------------------------
def derive_reference_paths_from_tvp(tvp_path: str) -> tuple[str, Path]:
    """
    From a TVP string/path, return:
      - reference .nc path
      - reference TVP .csv Path
    Supports both standard and shuffle tags.
    """
    m = re.search(r"(?i)(shuffle[^/]*?)(?=\.csv\b)", str(tvp_path))
    if m:
        tag = m.group(1)
        ref_nc  = f"SoC_proxy_TSA/data/calliope_models/{tag}.nc"
        ref_tvp = Path(f"SoC_proxy_TSA/data/timeseries/time_varying_parameters__{tag}.csv")
    else:
        ref_nc  = "SoC_proxy_TSA/data/calliope_models/standard_2010_2019_reference.nc"
        ref_tvp = Path("SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv")
    return ref_nc, ref_tvp
# Resample AFTER proxy build: 'D' for daily-mean metrics (recommended for LDES), or None for hourly
RESAMPLE_FREQ: str | None = "D"

# Normalization toggle for segmented RMSE (divide each model's monthly RMSE by its own mean RMSE)
NORMALIZE_SEGMENT_RMSE: bool = True

# Storage tech and proxy settings (copy/paste from your presentation file)
STORAGE_TECH = "h2_salt_cavern"
TS_WINDOW = ["2010-01-01", "2019-12-31"]
DEMAND_FIELD = "demand_power"
SOC_PROXY_PARAMS: Dict[str, Any] = {
    "capacity_weights": {"solar": 1, "onshore_wind": 0.5, "offshore_wind": 0.5},
    "storage_process_losses": {"charging_efficiency": 0.65 * 0.99, "discharging_efficiency": 0.56 * 0.99},
    "dispatchable_techs": {"known_dispatchable_capacity": 3300},
    "soc_decomposition": {"method": "fft_lowpass", "time_horizon_hours": 24},
}

# Colors
COLOUR_R = "#0d0887"   # Pearson r
COLOUR_E = "#6a00a8"   # RMSE
COLOUR_EC = "#b12a90"  # Combined
COLOUR_TM = "#e16462"  # Timing of maxima
COLOUR_5 = "#fca636"  
COLOUR_MM = "#f0f921" # Magnitude of maxima

# ------------------------------
# Path helpers
# ------------------------------
def path_nc(model_id: str) -> str:
    return f"SoC_proxy_TSA/data/calliope_models/{model_id}.nc"

def path_cluster_map(model_id: str) -> str:
    return f"SoC_proxy_TSA/data/cluster_maps/{model_id}.csv"

def path_timeseries(model_id: str) -> str:
    return f"SoC_proxy_TSA/data/timeseries/{model_id}.csv"

def path_params(model_id: str) -> str:
    return f"SoC_proxy_TSA/data/parameters/{model_id}.json"

# ------------------------------
# Generic helpers
# ------------------------------
def _maybe_resample(series: pd.Series, freq: str | None) -> pd.Series:
    """If freq is provided (e.g., 'D'), resample with mean; otherwise return unchanged."""
    if freq is None:
        return series
    return series.resample(freq).mean()

def read_clustered_netcdf_with_attr_fix(path: str) -> calliope.Model:
    """Matches your attribute cleanup so Calliope doesn't think clustering is still 'active'."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File does not exist at: {path}")
    with Dataset(p, "a") as nc:  # append mode
        g = nc.groups["attrs"]
        cfg = yaml.safe_load(g.getncattr("config"))
        cfg.get("init", {}).pop("time_cluster", None)
        g.setncattr("config", yaml.safe_dump(cfg))
    return calliope.read_netcdf(path)

# ------------------------------
# Parameter-file filtering (10y span, etc.)
# ------------------------------
def _extract_year(val):
    """Return an int year from int/str like 2010 or '2010-01-01'."""
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        m = re.search(r"\d{4}", val)
        if m:
            return int(m.group(0))
    return None

def _read_span_from_params(params_path: str) -> Tuple[int | None, int | None]:
    """Read (start_year, end_year) from params JSON if present."""
    try:
        with open(params_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None, None

    cp = (data or {}).get("calliope_params", {})
    dr = cp.get("date_range")

    if isinstance(dr, (list, tuple)) and len(dr) >= 2:
        y0 = _extract_year(dr[0])
        y1 = _extract_year(dr[-1])
        return y0, y1

    if isinstance(dr, dict):
        for k_start, k_end in [("start", "end"), ("first_year", "last_year")]:
            if k_start in dr and k_end in dr:
                y0 = _extract_year(dr[k_start])
                y1 = _extract_year(dr[k_end])
                return y0, y1

    return None, None

def filter_ids_by_year_span(
    model_ids: Iterable[str], start_year: int = 2010, end_year: int = 2019
) -> List[str]:
    """Keep only model ids whose parameters/{id}.json date_range matches [start_year, end_year]."""
    kept: List[str] = []
    for mid in model_ids:
        p = Path(path_params(mid))
        if not p.is_file():
            continue
        y0, y1 = _read_span_from_params(str(p))
        if y0 is None or y1 is None:
            continue
        if y0 == start_year and y1 == end_year:
            kept.append(mid)
    return kept

# ------------------------------
# Capacity helpers (ported from plot_cem_results.py)
# ------------------------------
def _get_capacities(m: calliope.Model) -> Tuple[pd.Series, pd.Series]:
    df_power = (
        m.results["flow_cap"].fillna(0).to_series().dropna().to_frame("capacity").reset_index()
        .drop(columns=["nodes"], errors="ignore")
    )
    mask_drop = (
        df_power["techs"].isin(["battery", "h2_salt_cavern", "demand"])
        | df_power["techs"].str.startswith("demand")
    )
    df_power = df_power[
        ~mask_drop
        & (
            ((df_power["techs"] == "electrolyser") & (df_power["carriers"] == "hydrogen"))
            | ((df_power["techs"] != "electrolyser") & (df_power["carriers"] == "power"))
        )
    ]
    df_power.set_index("techs", inplace=True)

    df_energy = (
        m.results["storage_cap"].fillna(0).to_series().dropna().to_frame("capacity").reset_index()
        .drop(columns=["nodes"], errors="ignore")
    )
    df_energy = df_energy[df_energy["capacity"] > 0]
    df_energy.set_index("techs", inplace=True)

    return df_power["capacity"], df_energy["capacity"]

def _relative_error(ref: pd.Series, test: pd.Series) -> Tuple[pd.Series, float]:
    e = (ref - test) / ref
    e_mean_abs = float(np.mean(np.abs(e)))
    return e, e_mean_abs

# ------------------------------
# Proxy build & metrics
# ------------------------------
def _load_timeseries_reference(csv_path: Path, ts_window: List[str]) -> pd.DataFrame:
    df = calliope_ts_to_pandas(csv_path, ts_window[0], ts_window[1])
    df.set_index("timesteps", inplace=True)
    return df.sort_index()

def _load_timeseries_clustered(cluster_map_csv: str, csv_path: Path) -> pd.DataFrame:
    df, _ = extrapolate_ts_from_cluster_map(cluster_map_csv, csv_path)
    df.set_index("timesteps", inplace=True)
    return df.sort_index()

def _build_proxy(df: pd.DataFrame, demand_field: str, params: Dict[str, Any]) -> pd.Series:
    df_proxy, _, _ = generate_soc_proxy(
        df=df,
        demand_field=demand_field,
        renewables_fields_and_weights=params["capacity_weights"],
        dispatchable_techs=params["dispatchable_techs"],
        storage_process_losses=params["storage_process_losses"],
        soc_decomposition=params["soc_decomposition"],
        timestamp_col=None,
    )
    return df_proxy["soc_proxy_LDES"].rename("soc_proxy_LDES")

def _metrics_vs_reference_proxy(proxy_test: pd.Series, proxy_ref: pd.Series) -> Dict[str, float]:
    R, T = proxy_ref.align(proxy_test, join="inner")
    pearson_r = float(R.corr(T))
    nrmse = float(np.sqrt(np.mean(np.square(R - T))) / np.max(R))

    t_start, t_end = R.index.min(), R.index.max()
    horizon = t_end - t_start
    t_max_R, t_max_T = R.idxmax(), T.idxmax()
    t_max_delta = abs(t_max_R - t_max_T)
    max_R, max_T = float(R.loc[t_max_R]), float(T.loc[t_max_T])

    maxima_magnitude_error = np.abs((max_R - max_T) / max_R)
    maxima_timing_error = (t_max_delta / horizon) if horizon != pd.Timedelta(0) else np.nan

    return {
        "pearson_r": pearson_r,
        "rmse": nrmse,
        "e_timing_maxima": maxima_timing_error,
        "e_magnitude_maxima": maxima_magnitude_error,
    }

# ------------------------------
# Core baselines & per-model metrics
# ------------------------------
def compute_reference_baselines(
    reference_nc: str,
    reference_tvp_csv: Path,
    resample_freq: str | None = None,
) -> Dict[str, Any]:
    """Load the chosen reference model + build its reference proxy (optional resample)."""
    m_ref = read_clustered_netcdf_with_attr_fix(reference_nc)

    # Reference input TS (hourly) -> build proxy (hourly)
    df_ref_ts = _load_timeseries_reference(reference_tvp_csv, TS_WINDOW)
    soc_ref_proxy = _build_proxy(df_ref_ts, DEMAND_FIELD, SOC_PROXY_PARAMS)
    soc_ref_proxy = _maybe_resample(soc_ref_proxy, resample_freq)

    power_caps_ref, energy_caps_ref = _get_capacities(m_ref)
    return {
        "soc_ref_proxy": soc_ref_proxy,
        "power_caps_ref": power_caps_ref,
        "energy_caps_ref": energy_caps_ref,
    }


def compute_ldes_error(model_id: str, energy_caps_ref: pd.Series) -> float:
    """Absolute relative error for LDES energy capacity (h2_salt_cavern)."""
    m_test = read_clustered_netcdf_with_attr_fix(path_nc(model_id))
    _, energy_caps_test = _get_capacities(m_test)
    e_storage, _ = _relative_error(energy_caps_ref, energy_caps_test)
    return float(np.abs(e_storage.get(STORAGE_TECH, np.nan)))

def compute_proxy_metrics(
    model_id: str, soc_ref_proxy: pd.Series, resample_freq: str | None = None
) -> Dict[str, float]:
    """Pearson r, normalized RMSE, and peak timing/magnitude errors vs reference proxy."""
    df_test = _load_timeseries_clustered(path_cluster_map(model_id), path_timeseries(model_id))
    soc_test_proxy = _build_proxy(df_test, DEMAND_FIELD, SOC_PROXY_PARAMS)
    soc_test_proxy = _maybe_resample(soc_test_proxy, resample_freq)
    return _metrics_vs_reference_proxy(soc_test_proxy, soc_ref_proxy)

def build_results(df_in: pd.DataFrame, resample_freq: str | None = RESAMPLE_FREQ) -> pd.DataFrame:
    """
    df_in must contain:
      - 'id'         : model ids
      - 'x_axis'     : value for plotting
      - 'tvp'        : tvp path string for that model (used to derive its reference)
    """
    rows = []
    cache: Dict[str, Dict[str, Any]] = {}  # key = reference_nc path

    for model_id, x_val, tvp in zip(df_in["id"], df_in["x_axis"], df_in["tvp"]):
        print(f'Extracting results for {model_id}')
        ref_nc, ref_tvp = derive_reference_paths_from_tvp(tvp)

        # memoize baselines per reference
        key = ref_nc
        if key not in cache:
            cache[key] = compute_reference_baselines(ref_nc, ref_tvp, resample_freq=resample_freq)
        soc_ref_proxy = cache[key]["soc_ref_proxy"]
        energy_caps_ref = cache[key]["energy_caps_ref"]

        ldes_err = compute_ldes_error(model_id, energy_caps_ref)
        metrics = compute_proxy_metrics(model_id, soc_ref_proxy, resample_freq=resample_freq)

        rows.append({
            "id": model_id,
            "ldes_error": float(ldes_err),
            "pearson_r": float(metrics["pearson_r"]),
            "rmse": float(metrics["rmse"]),
            "e_timing_maxima": float(metrics["e_timing_maxima"]),
            "e_magnitude_maxima": float(metrics["e_magnitude_maxima"]),
            "e_combined": (
                float(metrics["e_magnitude_maxima"])
                + float(metrics["e_timing_maxima"])
                + float(metrics["rmse"])
                + (1.0 - float(metrics["pearson_r"]))
            ) / 4.0,
            "x_axis": x_val,
            "reference_nc": ref_nc,
            "reference_tvp": str(ref_tvp),
        })
    return pd.DataFrame(rows)


# ------------------------------
# Plot (A): Ex-ante vs ex-post metrics in subplots
# ------------------------------
def _trendline(ax, x, y, color, label_for_r2, lw=1.2):
    """Add a linear trendline + R^2 annotation for (x,y)."""
    if len(x) < 2 or len(y) < 2:
        return
    m, b, r_val, _, _ = scipy.stats.linregress(x, y)
    x_sorted = np.sort(x)
    ax.plot(x_sorted, m * x_sorted + b, color=color, linewidth=lw)
    ax.annotate(
        f"$R^2$={r_val**2:.2f}",
        xy=(np.nanmean(x), np.nanmean(y)),
        xytext=(5, 10),
        textcoords="offset points",
        color=color,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.95),
    )

def plot_ex_ante_ex_post_subplots(
    df: pd.DataFrame,
    x_label: str = r"$\epsilon^C_{\mathrm{LDES}}$",
    figtitle: str = "Ex-ante vs Ex-post: Metrics vs LDES capacity error",
    savepath: str | None = None,
    layout: str = "grid",
):
    """
    layout:
      - "grid": top (r & RMSE), middle (timing & magnitude of maxima), bottom (combined)
      - "stacked": same content but stacked equally (e.g., for single-column figures)
    """
    x = df["ldes_error"].values
    r = df["pearson_r"].values
    e = df["rmse"].values
    e_tm = df["e_timing_maxima"].values
    e_mm = df["e_magnitude_maxima"].values
    e_comb = df["e_combined"].values

    if layout == "stacked":
        fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
        (ax1, ax2, ax3) = axes
    else:
        # grid layout: top & middle normal height, bottom a bit taller
        fig = plt.figure(figsize=(7, 9))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.25)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])

    # Subplot 1: Pearson r + RMSE
    ax1.scatter(x, r, label="Pearson r", color=COLOUR_R, s=28)
    _trendline(ax1, x, r, COLOUR_R, "Pearson r")
    ax1.scatter(x, e, label="RMSE", color=COLOUR_E, s=28)
    _trendline(ax1, x, e, COLOUR_E, "RMSE")
    ax1.set_ylabel("r, RMSE")
    leg = ax1.legend(loc="best", frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_edgecolor("black")
    ax1.yaxis.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Peak timing & magnitude errors
    ax2.scatter(x, e_tm, label="Peak timing error", color=COLOUR_TM, s=28)
    ax2.scatter(x, e_mm, label="Peak magnitude error", color=COLOUR_MM, s=28)

    # --- NEW: combo trendline only (no dots) ---
    e_peak_combo = 0.5 * (e_tm + e_mm)  # or a weighted combo if you prefer
    m_c, b_c, r_c, _, _ = scipy.stats.linregress(x, e_peak_combo)
    x_sorted = np.sort(x)
    ax2.plot(
        x_sorted,
        m_c * x_sorted + b_c,
        linestyle="--",
        linewidth=1.6,
        color=COLOUR_5,
        label="(timing ⊕ magnitude)"
    )
    # R^2 annotation for the combo
    ax2.annotate(
        f"$R^2$={r_c**2:.2f}",
        xy=(np.nanmean(x), np.nanmean(e_peak_combo)),
        xytext=(6, 8), textcoords="offset points",
        fontsize=9, color=COLOUR_5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLOUR_5, lw=0.8, alpha=0.9),
    )
    ax2.set_ylabel("Peak errors")
    leg2 = ax2.legend(loc="best", frameon=True)
    leg2.get_frame().set_facecolor("white")
    leg2.get_frame().set_alpha(0.9)
    leg2.get_frame().set_edgecolor("black")
    ax2.yaxis.grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: Combined
    ax3.scatter(x, e_comb, label="Combined metric", color=COLOUR_EC, s=30)
    _trendline(ax3, x, e_comb, COLOUR_EC, "Combined")
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("Combined")
    leg3 = ax3.legend(loc="best", frameon=True)
    leg3.get_frame().set_facecolor("white")
    leg3.get_frame().set_alpha(0.9)
    leg3.get_frame().set_edgecolor("black")
    ax3.yaxis.grid(True, linestyle=":", alpha=0.6)

    # fig.suptitle(figtitle)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

# ==============================
# (B) Segmented (monthly) metrics per model
# ==============================
def segmented_metrics_by_month(proxy_test: pd.Series, proxy_ref: pd.Series) -> pd.DataFrame:
    """
    Compute per-month metrics on the overlap:
      - RMSE (per calendar Month)
      - Pearson r (per calendar Month)
    Returns columns: ['month', 'rmse', 'pearson_r_m']
    """
    ref_aligned, test_aligned = proxy_ref.align(proxy_test, join="inner")
    if ref_aligned.empty or test_aligned.empty:
        return pd.DataFrame(columns=["month", "rmse", "pearson_r_m"])

    df = pd.DataFrame({"ref": ref_aligned, "test": test_aligned})
    df["month"] = df.index.to_period("M")

    def _rmse(g):
        return float(np.sqrt(np.mean((g["ref"] - g["test"]) ** 2)))

    def _pearson(g):
        if g["ref"].notna().sum() >= 2 and g["test"].notna().sum() >= 2:
            return float(g["ref"].corr(g["test"]))
        return np.nan

    grouped = df.groupby("month").apply(lambda g: pd.Series({
        "rmse": _rmse(g),
        "pearson_r_m": _pearson(g)
    })).reset_index()

    return grouped

def build_segmented_results(
    df_in: pd.DataFrame,
    resample_freq: str | None = RESAMPLE_FREQ,
    normalize_segment_rmse: bool = NORMALIZE_SEGMENT_RMSE,
) -> pd.DataFrame:
    """
    df_in must contain: 'id', 'tvp'
    Returns tidy rows with monthly metrics + ldes_error
    """
    all_rows = []
    cache: Dict[str, Dict[str, Any]] = {}

    for mid, tvp in zip(df_in["id"], df_in["tvp"]):
        ref_nc, ref_tvp = derive_reference_paths_from_tvp(tvp)
        key = ref_nc
        if key not in cache:
            cache[key] = compute_reference_baselines(ref_nc, ref_tvp, resample_freq=resample_freq)
        soc_ref_proxy = cache[key]["soc_ref_proxy"]
        energy_caps_ref = cache[key]["energy_caps_ref"]

        # Build test proxy hourly + optional resample AFTER proxy
        df_test = _load_timeseries_clustered(path_cluster_map(mid), path_timeseries(mid))
        soc_test_proxy = _build_proxy(df_test, DEMAND_FIELD, SOC_PROXY_PARAMS)
        soc_test_proxy = _maybe_resample(soc_test_proxy, resample_freq)

        # Monthly metrics
        df_m = segmented_metrics_by_month(soc_test_proxy, soc_ref_proxy)
        if df_m.empty:
            continue

        df_m["model_id"] = mid
        if normalize_segment_rmse:
            mean_rmse = df_m["rmse"].mean()
            df_m["rmse_norm"] = df_m["rmse"] / mean_rmse if mean_rmse and not np.isnan(mean_rmse) else np.nan
        else:
            df_m["rmse_norm"] = np.nan

        # LDES capacity error (scalar per model)
        ldes_err = compute_ldes_error(mid, energy_caps_ref)
        df_m["ldes_error"] = float(ldes_err)

        ts = df_m["month"].dt.to_timestamp(how="start")
        df_m["month_midpoint_ts"] = ts + pd.to_datetime("15D") - pd.Timestamp(0)  # or pd.to_timedelta(15, "D")

        all_rows.append(df_m)

    if not all_rows:
        return pd.DataFrame(columns=["model_id","month","rmse","rmse_norm","pearson_r_m","ldes_error","month_midpoint_ts"])
    return pd.concat(all_rows, ignore_index=True)

def plot_monthly_rmse_with_ref_proxy(
    df_seg: pd.DataFrame,
    proxy_ref: pd.Series,
    title: str = "Monthly RMSE vs Reference SoC Proxy (colour = LDES capacity error)",
    savepath: str | None = None,
    cmap: str = "plasma",
    show_proxy_strip: bool = True,
    height_ratios: tuple[int, int] = (4, 1),  # main : strip → strip is ~4x smaller than main
    use_normalized: bool = NORMALIZE_SEGMENT_RMSE,
):
    """
    Top panel: monthly (normalized) RMSE scatter (color = LDES capacity error), continuous datetime x-axis.
    Bottom panel (optional): thin SoC Proxy strip sharing the x-axis (no y-axis).
    """
    if df_seg.empty:
        print("No segmented data to plot.")
        return

    y_col = "rmse_norm" if use_normalized and "rmse_norm" in df_seg.columns else "rmse"

    if show_proxy_strip:
        fig, (ax_main, ax_strip) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 6),
            gridspec_kw={"height_ratios": list(height_ratios), "hspace": 0.05}
        )
    else:
        fig, ax_main = plt.subplots(figsize=(12, 5))
        ax_strip = None

    # Prepare data
    x = pd.to_datetime(df_seg["month_midpoint_ts"].values)
    y = df_seg[y_col].values
    c = df_seg["ldes_error"].values

    # Main panel: RMSE scatter (left axis)
    norm = Normalize(vmin=np.nanmin(c), vmax=np.nanmax(c))
    sc = ax_main.scatter(x, y, c=c, cmap=cmap, norm=norm, alpha=0.9, edgecolors="none", s=24, label="Monthly RMSE")

    ax_main.set_ylabel("Monthly RMSE" + (" (normalized)" if y_col == "rmse_norm" else ""))
    ax_main.xaxis.set_major_locator(mdates.YearLocator())
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_main.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax_main.grid(True, which="major", axis="x", linestyle=":", alpha=0.5)

    # Compact colorbar inset
    cax = inset_axes(ax_main, width="2.2%", height="60%", loc="upper left", borderpad=1)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(r"LDES capacity error ($|\epsilon^C_{\mathrm{LDES}}|$)")

    # Strip panel: thin SoC proxy under the main chart
    if show_proxy_strip and ax_strip is not None:
        ref_series = proxy_ref  # scale if needed for visuals (e.g., /1e6)
        ax_strip.plot(ref_series.index, ref_series.values, color="grey", linewidth=1.0)
        ax_strip.set_xlabel("")
        ax_strip.yaxis.set_visible(False)
        ax_strip.spines["left"].set_visible(False)
        ax_strip.spines["right"].set_visible(False)
        ax_strip.spines["top"].set_visible(False)
        ax_strip.grid(False)

    # fig.suptitle(title)
    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

# ==============================
# (C) Month-importance across models
# ==============================
def month_importance_by_corr(
    df_seg: pd.DataFrame, use_normalized: bool = NORMALIZE_SEGMENT_RMSE
) -> pd.DataFrame:
    """
    For each calendar month across models, compute the Pearson correlation between:
      - (monthly RMSE or normalized RMSE) and LDES capacity error  -> corr_rmse
      - (monthly Pearson r) and LDES capacity error                -> corr_r

    Returns columns:
      ['month','corr_rmse','corr_r','n_models','month_midpoint_ts']
    """
    if df_seg.empty:
        return pd.DataFrame(columns=["month", "corr_rmse", "corr_r", "n_models", "month_midpoint_ts"])

    y_rmse_col = "rmse_norm" if use_normalized and "rmse_norm" in df_seg.columns else "rmse"
    y_r_col = "pearson_r_m"  # from segmented_metrics_by_month

    def _safe_corr(a: pd.Series, b: pd.Series) -> float:
        a, b = a.align(b, join="inner")
        if a.notna().sum() < 3 or b.notna().sum() < 3:
            return np.nan
        try:
            return float(a.corr(b))
        except Exception:
            return np.nan

    def _agg(g: pd.DataFrame) -> pd.Series:
        # corr of RMSE (or normalized) vs LDES error across models for this month
        corr_rmse = _safe_corr(g[y_rmse_col], g["ldes_error"])
        # corr of monthly Pearson r vs LDES error
        corr_r = _safe_corr(g[y_r_col], g["ldes_error"])
        return pd.Series({
            "corr_rmse": corr_rmse,
            "corr_r": corr_r,
            "n_models": g["model_id"].nunique(),
        })

    grouped = df_seg.groupby("month").apply(_agg).reset_index()
    ts = grouped["month"].dt.to_timestamp(how="start")
    grouped["month_midpoint_ts"] = ts + pd.to_timedelta(15, unit="D")
    return grouped

def plot_month_importance_bar_with_strip(
    df_imp: pd.DataFrame,
    proxy_ref: pd.Series,
    title: str = "Month importance: correlations across models",
    savepath: str | None = None,
    colors: Tuple[str, str] = (COLOUR_E, COLOUR_EC),  # RMSE (or nRMSE) and r
    show_proxy_strip: bool = True,
    height_ratios: tuple[int, int] = (4, 1),
):
    """
    Upper panel: two stem series over time:
      - corr_rmse (blue-ish)
      - corr_r    (red-ish)
    Lower panel: mini SoC proxy strip (shared x; no y-axis) as context.
    """
    if df_imp.empty:
        print("No month-importance data to plot.")
        return

    # Figure & axes
    if show_proxy_strip:
        fig, (ax, ax_strip) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 6),
            gridspec_kw={"height_ratios": list(height_ratios), "hspace": 0.05}
        )
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax_strip = None

    x = pd.to_datetime(df_imp["month_midpoint_ts"].values)
    y1 = df_imp["corr_rmse"].values
    y2 = df_imp["corr_r"].values

    # Signed colors via two calls to stem (clear & flexible)
    pos1 = y1 >= 0
    neg1 = ~pos1
    c1 = colors[0]

    cont_pos1 = ax.stem(x[pos1], y1[pos1], linefmt='-', markerfmt='o', basefmt=' ')
    plt.setp(cont_pos1.stemlines, color=c1, linewidth=1.8)
    plt.setp(cont_pos1.markerline, markerfacecolor=c1, markeredgecolor="white")

    cont_neg1 = ax.stem(x[neg1], y1[neg1], linefmt='-', markerfmt='o', basefmt=' ')
    plt.setp(cont_neg1.stemlines, color=c1, linewidth=1.0, alpha=0.5)
    plt.setp(cont_neg1.markerline, markerfacecolor=c1, markeredgecolor="white", alpha=1)

    # # Second series (Pearson r importance)
    # pos2 = y2 >= 0
    # neg2 = ~pos2
    # c2 = colors[1]

    # cont_pos2 = ax.stem(x[pos2], y2[pos2], linefmt='-', markerfmt='s', basefmt=' ')
    # plt.setp(cont_pos2.stemlines, color=c2, linewidth=1.8)
    # plt.setp(cont_pos2.markerline, markerfacecolor=c2, markeredgecolor="white")

    # cont_neg2 = ax.stem(x[neg2], y2[neg2], linefmt='-', markerfmt='s', basefmt=' ')
    # plt.setp(cont_neg2.stemlines, color=c2, linewidth=1.0, alpha=0.5)
    # plt.setp(cont_neg2.markerline, markerfacecolor=c2, markeredgecolor="white", alpha=0.5)

    # Baseline style
    cont_pos1.baseline.set_color("lightgrey")
    cont_pos1.baseline.set_linewidth(0.8)

    ax.axhline(0, color="lightgrey", linewidth=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Pearson r")
    # ax.set_title(title)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.5)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=c1, marker='o', linestyle='-', label='corr(RMSE, LDES err)'),
            # plt.Line2D([0], [0], color=c2, marker='s', linestyle='-', label='corr(Pearson r, LDES err)'),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False
    )

    # Proxy strip
    if show_proxy_strip and ax_strip is not None:
        ref_series = proxy_ref
        ax_strip.plot(ref_series.index, ref_series.values, color="grey", linewidth=1.0)
        ax_strip.set_xlabel("")
        ax_strip.yaxis.set_visible(False)
        ax_strip.spines["left"].set_visible(False)
        ax_strip.spines["right"].set_visible(False)
        ax_strip.spines["top"].set_visible(False)
        ax_strip.grid(False)

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

# ------------------------------
# Runner
# ------------------------------
if __name__ == "__main__":
    from os import walk

    # 1) Discover model ids from cluster_maps
    ids_all: List[str] = []
    for (dirpath, dirnames, filenames) in walk("SoC_proxy_TSA/data/cluster_maps"):
        ids_all.extend(fn.removesuffix(".csv") for fn in filenames if fn.endswith(".csv"))
        break

    # 2) Filter to 10-year models
    ids_10y = filter_ids_by_year_span(ids_all, start_year=2010, end_year=2019)

    # 3) Bring in per-id tvp (from your notes log)
    config_src = pd.read_csv("SoC_proxy_TSA/data/notes/log.csv")
    # Keep only needed cols; left-join to ensure we keep our filtered IDs
    df_ids = pd.DataFrame({"id": ids_10y})
    df_cfg = config_src[["id", "tvp"]]
    df_in = df_ids.merge(df_cfg, on="id", how="left")

    # 4A) Overall metrics vs LDES error
    df_in["x_axis"] = df_in["id"]  # or any x you want
    df_overall = build_results(df_in, resample_freq=RESAMPLE_FREQ)
    plot_ex_ante_ex_post_subplots(
        df_overall,
        figtitle="Ex-ante vs Ex-post: Metrics vs LDES capacity error",
        savepath="exante_expost_metrics_subplots.pdf",
        layout="grid",
    )

    # 4B) Segmented monthly metrics
    df_seg = build_segmented_results(
        df_in,
        resample_freq=RESAMPLE_FREQ,
        normalize_segment_rmse=NORMALIZE_SEGMENT_RMSE,
    )
    plot_monthly_rmse_with_ref_proxy(
        df_seg,
        # pass *any* reference for the strip; pick the standard one or the most common
        proxy_ref=compute_reference_baselines(
            *derive_reference_paths_from_tvp("SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv"),
            resample_freq=RESAMPLE_FREQ
        )["soc_ref_proxy"],
        title="Monthly RMSE vs Reference SoC Proxy (colour = LDES capacity error)"
              + (" [normalized]" if NORMALIZE_SEGMENT_RMSE else ""),
        savepath="monthly_rmse_vs_ref_proxy_coloured.pdf",
        use_normalized=NORMALIZE_SEGMENT_RMSE,
    )

    # 4C) Month-importance (same df_seg)
    df_imp = month_importance_by_corr(df_seg, use_normalized=NORMALIZE_SEGMENT_RMSE)
    plot_month_importance_bar_with_strip(
        df_imp,
        proxy_ref=compute_reference_baselines(
            *derive_reference_paths_from_tvp("SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv"),
            resample_freq=RESAMPLE_FREQ
        )["soc_ref_proxy"],
        title="Month importance across models (corr with LDES error): RMSE & Pearson r",
        savepath="month_importance_corr_with_strip.pdf",
    )

