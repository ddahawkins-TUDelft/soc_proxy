"""
signal_analysis_build_cache.py

Step 1 cache builder for signal analysis:
- Reads a log.csv listing clustered models and their TVP.
- Resolves each clustered model to its corresponding REFERENCE model (varies by horizon sample & country).
- Extracts:
    * CEM SoC signal (reference + clustered)
    * SoC Proxy signal (reference + clustered)
    * LDES capacity = max(SoC) (reference + clustered)
- Computes metrics:
    (1) Proxy Error (Reference):  proxy_ref vs cem_soc_ref  (RMSE, Pearson r, min-max error)
    (2) Proxy Error (Clustered):   proxy_clu vs cem_soc_clu  (RMSE, Pearson r, min-max error)
    (3) TSA Error:                 proxy_clu vs proxy_ref    (RMSE, Pearson r, min-max error)
    (4) Capacity Error:            signed + absolute relative error in LDES capacity (max SoC)
- Writes:
    * master cache CSV with per-model summary metrics + reference identifier for fast matching
    * optional per-model parquet signals (SoC + Proxy) for plotting later without re-running

Notes:
- Uses utilities you already have:
    soc_proxy.generate_soc_proxy
    soc_proxy.calliope.timeseries.{calliope_ts_to_pandas, extrapolate_ts_from_cluster_map}
- Clustered SoC extraction mirrors helper_signals._soc_from_clustered_model
- Reference derivation mirrors plot_signal_results.derive_reference_paths_from_tvp, but extended for country-awareness.

"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from netCDF4 import Dataset

import calliope

# --- Utilities from your codebase (as in your prior scripts) ---
from soc_proxy import generate_soc_proxy
from soc_proxy.calliope.timeseries import (
    calliope_ts_to_pandas,
    extrapolate_ts_from_cluster_map,
)

from scipy.stats import pearsonr

# =============================================================================
# Config
# =============================================================================

# Base folders (adjust if your repo paths differ)
DATA_ROOT = Path("SoC_proxy_TSA/data")
MODELS_DIR = DATA_ROOT / "calliope_models"
CLUSTER_MAPS_DIR = DATA_ROOT / "cluster_maps"
TIMESERIES_DIR = DATA_ROOT / "timeseries"
PARAMS_DIR = DATA_ROOT / "parameters"

# Inputs
# DEFAULT_LOG_CSV = DATA_ROOT / "notes" / "log_10Y_NL_BE.csv"
DEFAULT_LOG_CSV = DATA_ROOT / "notes" / "log_10Y_NL_BE.csv"


# Outputs
CACHE_DIR = DATA_ROOT / "signal_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_MASTER_CSV = CACHE_DIR / "signal_cache_master.csv"
CACHE_SIGNALS_DIR = CACHE_DIR / "signals"  # optional per-model parquet
CACHE_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

# Signal extraction settings
STORAGE_TECH = "h2_salt_cavern"
DEMAND_FIELD = "demand_power"
# TS_WINDOW = ("2010-01-01", "2019-12-31")  # keep aligned with earlier scripts

# Whether to save the full time series signals to disk (parquet).
# If False: only summary metrics saved in the master CSV.
SAVE_SIGNAL_PARQUETS = False

# Optional resampling AFTER computing signals (often helpful for LDES)
# Use None for hourly, or "D" for daily mean.
RESAMPLE_FREQ: Optional[str] = None

# Peak / window diagnostics (cyclic horizon)
# Smoothing is used ONLY to locate peak timing robustly (edge-safe via wrap).
PEAK_SMOOTH_DAYS: int = 1
# Half-window sizes (in days) around the reference CEM peak (cyclic wrap).
PEAK_WINDOW_HALF_DAYS: tuple[int, ...] = (
    15,
    30,
    45,
    90,
    180,
    int(0.75 * 365),
    365,
    548,
    730,
    int(2.5 * 365),
)

# Country-specific dispatchable capacity (from your earlier code)
DISPATCHABLE_BY_COUNTRY = {
    "NL": 3300,
    "IT": 0,
    "ES": 0,
    "BE": 2501,
    "GB": 9323,
}

# Base proxy params (country-specific known_dispatchable_capacity is filled dynamically)
SOC_PROXY_PARAMS_BASE: Dict[str, Any] = {
    "capacity_weights": {"solar": 1, "onshore_wind": 0.5, "offshore_wind": 0.5},
    "storage_process_losses": {
        "charging_efficiency": 0.65 * 0.99,
        "discharging_efficiency": 0.56 * 0.99,
    },
    "soc_decomposition": {"method": "fft_lowpass", "time_horizon_hours": 24},
}


# =============================================================================
# Small helpers
# =============================================================================
def parse_k_and_wp(model_name: str) -> tuple[float | None, float | None]:
    """
    Extract k (number of representatives) and Wp (proxy weight)
    from the model_name string.

    Expected patterns (order-independent):
      - reps=<int>
      - W_proxy=<float>

    Returns (k, Wp). If not found, returns None for that field.
    """
    if not isinstance(model_name, str):
        return None, None

    k = None
    wp = None

    m_k = re.search(r"reps\s*=\s*(\d+)", model_name)
    if m_k:
        k = int(m_k.group(1))

    m_wp = re.search(r"W[_\-]?proxy\s*=\s*([0-9]*\.?[0-9]+)", model_name)
    if m_wp:
        wp = float(m_wp.group(1))

    return k, wp


def parse_dates_window(dates_str: str) -> tuple[str, str, str]:
    """
    dates_str format: "yyyy,yyyy" (e.g., "2010,2019")
    Returns:
      (start_date_iso, end_date_iso, window_id)
    Where window_id is stable for caching, like "2010_2019".
    """
    if pd.isna(dates_str):
        raise ValueError("dates column is NaN; cannot infer window")

    parts = [p.strip() for p in str(dates_str).split(",")]
    if len(parts) != 2:
        raise ValueError(f"dates must be 'yyyy,yyyy' but got: {dates_str}")

    y0, y1 = int(parts[0]), int(parts[1])
    if y1 < y0:
        raise ValueError(f"dates end year < start year: {dates_str}")

    start = f"{y0:04d}-01-01"
    end = f"{y1:04d}-12-31"
    window_id = f"{y0:04d}_{y1:04d}"
    return start, end, window_id


def _maybe_resample(s: pd.Series, freq: Optional[str]) -> pd.Series:
    if freq is None:
        return s
    return s.resample(freq).mean()


def _safe_pearson(a: pd.Series, b: pd.Series) -> float:
    a2, b2 = a.align(b, join="inner")
    if a2.notna().sum() < 2 or b2.notna().sum() < 2:
        return np.nan
    return float(a2.corr(b2))


def _rmse(a: pd.Series, b: pd.Series) -> float:
    a2, b2 = a.align(b, join="inner")
    if a2.empty:
        return np.nan
    return float(np.sqrt(np.mean((a2 - b2) ** 2)))


def _nrmse_by_range(a: pd.Series, b: pd.Series) -> float:
    """Normalize RMSE by (max(ref)-min(ref)) on aligned range, robust for min-max comparisons."""
    a2, b2 = a.align(b, join="inner")
    if a2.empty:
        return np.nan
    denom = float(a2.max() - a2.min())
    if denom == 0:
        return np.nan
    return float(np.sqrt(np.mean((a2 - b2) ** 2)) / denom)


def _minmax_error(a: pd.Series, b: pd.Series) -> float:
    """
    Compare min+max shape by taking L1 distance between (min,max) pairs, normalized by ref range.
    This is intentionally simple/transparent; you can swap later.
    """
    a2, b2 = a.align(b, join="inner")
    if a2.empty:
        return np.nan
    ref_min, ref_max = float(a2.min()), float(a2.max())
    tst_min, tst_max = float(b2.min()), float(b2.max())
    ref_range = ref_max - ref_min
    if ref_range == 0:
        return np.nan
    return float((abs(ref_min - tst_min) + abs(ref_max - tst_max)) / ref_range)


def _signed_rel_error(ref_val: float, test_val: float) -> float:
    if ref_val == 0 or np.isnan(ref_val) or np.isnan(test_val):
        return np.nan
    return float((test_val - ref_val) / ref_val)


# -----------------------------------------------------------------------------
# Standardised diagnostics (consistent across all metric categories)
# -----------------------------------------------------------------------------
def safe_pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r for two numeric arrays with NaN/inf protection."""
    if x.size == 0 or y.size == 0:
        return np.nan
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x2 = x[m]
    y2 = y[m]
    if np.std(x2) == 0 or np.std(y2) == 0:
        return np.nan
    return float(pearsonr(x2, y2)[0])


def range_denom(y: np.ndarray) -> float:
    """Range denominator (max-min) with NaN/inf protection."""
    if y.size == 0:
        return np.nan
    m = np.isfinite(y)
    if m.sum() == 0:
        return np.nan
    yy = y[m]
    return float(np.max(yy) - np.min(yy))


def summarise_pair_metrics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    denom: float | None = None,
) -> dict[str, float]:
    """Return standardised metrics for x vs y (diff = x - y).
    Metrics (always present):
      - pearson_r
      - nrmse_range  = RMSE(diff) / denom
      - nmae_range   = MAE(diff)  / denom
      - nmbe_range   = MBE(diff)  / denom
    If denom is None, uses range(y) on the provided arrays.
    """
    if x.size == 0 or y.size == 0:
        return {
            "pearson_r": np.nan,
            "nrmse_range": np.nan,
            "nmae_range": np.nan,
            "nmbe_range": np.nan,
        }

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return {
            "pearson_r": np.nan,
            "nrmse_range": np.nan,
            "nmae_range": np.nan,
            "nmbe_range": np.nan,
        }

    xx = x[m].astype(float, copy=False)
    yy = y[m].astype(float, copy=False)

    r = safe_pearson_np(xx, yy)
    diff = xx - yy

    mbe = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    d = float(denom) if denom is not None else range_denom(yy)
    if not np.isfinite(d) or d == 0.0:
        return {
            "pearson_r": r,
            "nrmse_range": np.nan,
            "nmae_range": np.nan,
            "nmbe_range": np.nan,
        }

    return {
        "pearson_r": r,
        "nrmse_range": rmse / d,
        "nmae_range": mae / d,
        "nmbe_range": mbe / d,
    }


def align_four(
    cem_ref: pd.Series,
    proxy_ref: pd.Series,
    cem_clu: pd.Series,
    proxy_clu: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Align four series on the common timestamp intersection and drop any NaNs."""
    df = pd.concat(
        [
            cem_ref.rename("cem_ref"),
            proxy_ref.rename("proxy_ref"),
            cem_clu.rename("cem_clu"),
            proxy_clu.rename("proxy_clu"),
        ],
        axis=1,
        join="inner",
    ).dropna(how="any")

    return df["cem_ref"], df["proxy_ref"], df["cem_clu"], df["proxy_clu"]


def build_stepwise_errors_aligned(
    cem_ref: pd.Series,
    proxy_ref: pd.Series,
    cem_clu: pd.Series,
    proxy_clu: pd.Series,
) -> dict[str, pd.Series]:
    """
    Build aligned stepwise error signals on a shared index.

    Identity holds exactly on this aligned index:
      e_proxy_clu = e_proxy_ref + e_tsa_proxy - e_tsa_cem
    """
    cem_ref_a, proxy_ref_a, cem_clu_a, proxy_clu_a = align_four(
        cem_ref, proxy_ref, cem_clu, proxy_clu
    )

    d_proxy_ref = proxy_ref_a.diff().dropna()
    d_proxy_clu = proxy_clu_a.diff().dropna()
    d_tsa_proxy = (d_proxy_clu - d_proxy_ref).rename("d_tsa_proxy")

    e_proxy_ref = (proxy_ref_a - cem_ref_a).rename("e_proxy_ref")
    e_proxy_clu = (proxy_clu_a - cem_clu_a).rename("e_proxy_clu")
    e_tsa_proxy = (proxy_clu_a - proxy_ref_a).rename("e_tsa_proxy")
    e_tsa_cem = (cem_clu_a - cem_ref_a).rename("e_tsa_cem")

    d_cem_ref = cem_ref_a.diff().dropna()
    d_cem_clu = cem_clu_a.diff().dropna()

    return {
        "cem_ref": cem_ref_a,
        "proxy_ref": proxy_ref_a,
        "cem_clu": cem_clu_a,
        "proxy_clu": proxy_clu_a,
        "e_proxy_ref": e_proxy_ref,
        "e_proxy_clu": e_proxy_clu,
        "e_tsa_proxy": e_tsa_proxy,
        "e_tsa_cem": e_tsa_cem,
        "d_proxy_ref": d_proxy_ref,
        "d_proxy_clu": d_proxy_clu,
        "d_tsa_proxy": d_tsa_proxy,
        "d_cem_ref": d_cem_ref,
        "d_cem_clu": d_cem_clu,
    }


def summarise_signal_full(
    x: np.ndarray,
    y: np.ndarray,
    *,
    denom: float | None = None,
) -> dict[str, float]:
    """Backwards-compatible wrapper returning the standard metric set for x vs y."""
    return summarise_pair_metrics(x, y, denom=denom)


def _infer_steps_per_day(index: pd.DatetimeIndex) -> int:
    """Infer steps/day from a regular DateTimeIndex."""
    if len(index) < 2:
        raise ValueError("Need at least 2 timesteps to infer frequency")
    dt = index[1] - index[0]
    if dt <= pd.Timedelta(0):
        raise ValueError("Index must be strictly increasing")
    steps = int(round(pd.Timedelta(days=1) / dt))
    if steps <= 0:
        raise ValueError(f"Could not infer steps/day from dt={dt}")
    return steps


def circular_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Circular rolling mean over a 1D array."""
    if window <= 1:
        return values.astype(float, copy=True)
    n = values.size
    if n == 0:
        return values.astype(float, copy=True)
    k = window // 2
    padded = np.concatenate([values[-k:], values, values[:k]])
    s = pd.Series(padded)
    sm = s.rolling(window=window, center=True, min_periods=window).mean().to_numpy()
    # drop padding
    out = sm[k : k + n]
    # If window > n, rolling will be NaN; fall back to no smoothing
    if np.isnan(out).all():
        return values.astype(float, copy=True)
    # fill any NaNs (only possible at extremes if min_periods triggers)
    if np.isnan(out).any():
        out = pd.Series(out).fillna(method="bfill").fillna(method="ffill").to_numpy()
    return out


def circular_window_indices(center: int, half_width: int, n: int) -> np.ndarray:
    """Return indices for a cyclic window [center-half_width, ..., center+half_width] modulo n."""
    if n <= 0:
        return np.array([], dtype=int)
    offsets = np.arange(-half_width, half_width + 1)
    return (center + offsets) % n


def peak_and_windows_diagnostics(
    cem_ref: pd.Series,
    cem_clu: pd.Series,
    proxy_ref: pd.Series,
    proxy_clu: pd.Series,
    smooth_days: int,
    half_days_list: tuple[int, ...],
) -> dict[str, float]:
    """Cyclic peak timing/magnitude + windowed metrics around reference peak."""

    # All series are assumed aligned on a shared index and regular.
    idx = cem_ref.index
    n = len(idx)
    if n == 0:
        return {}

    steps_per_day = _infer_steps_per_day(idx)
    smooth_window = max(1, smooth_days * steps_per_day)

    # arrays
    a_ref = cem_ref.to_numpy(dtype=float)
    a_clu = cem_clu.to_numpy(dtype=float)
    p_ref = proxy_ref.to_numpy(dtype=float)
    p_clu = proxy_clu.to_numpy(dtype=float)

    # smoothed for peak timing (circular)
    a_ref_sm = circular_rolling_mean(a_ref, smooth_window)
    a_clu_sm = circular_rolling_mean(a_clu, smooth_window)
    p_ref_sm = circular_rolling_mean(p_ref, smooth_window)
    p_clu_sm = circular_rolling_mean(p_clu, smooth_window)

    i_peak_ref = int(np.nanargmax(a_ref_sm))
    i_peak_cem_clu = int(np.nanargmax(a_clu_sm))
    i_peak_proxy_ref = int(np.nanargmax(p_ref_sm))
    i_peak_proxy_clu = int(np.nanargmax(p_clu_sm))

    # timing diffs in days, in cyclic sense (shortest signed distance)
    def _cyc_dt_days(i_test: int, i_ref: int) -> float:
        d = i_test - i_ref
        # wrap to (-n/2, n/2]
        if d > n / 2:
            d -= n
        elif d <= -n / 2:
            d += n
        return float(d) / float(steps_per_day)

    out: dict[str, float] = {
        "peak_dt_days_cem_clu_vs_cem_ref": _cyc_dt_days(i_peak_cem_clu, i_peak_ref),
        "peak_dt_days_proxy_ref_vs_cem_ref": _cyc_dt_days(i_peak_proxy_ref, i_peak_ref),
        "peak_dt_days_proxy_clu_vs_cem_ref": _cyc_dt_days(i_peak_proxy_clu, i_peak_ref),
        "peak_dt_days_proxy_clu_vs_proxy_ref": _cyc_dt_days(
            i_peak_proxy_clu, i_peak_proxy_ref
        ),
    }

    # peak magnitudes at each series' own smoothed peak time (using RAW values at that index)
    out.update(
        {
            "peak_val_cem_ref": float(a_ref[i_peak_ref]),
            "peak_val_cem_clu": float(a_clu[i_peak_cem_clu]),
            "peak_val_proxy_ref": float(p_ref[i_peak_proxy_ref]),
            "peak_val_proxy_clu": float(p_clu[i_peak_proxy_clu]),
        }
    )

    # proxy peak magnitude error relative to corresponding CEM peak magnitude (own-peak anchored)
    # (This avoids the old "value-at-ref-peak" only view)
    denom_ref = out["peak_val_cem_ref"]
    denom_clu = out["peak_val_cem_clu"]
    out["peak_mag_err_proxy_ref_vs_cem_ref_abs"] = (
        out["peak_val_proxy_ref"] - out["peak_val_cem_ref"]
    )
    out["peak_mag_err_proxy_clu_vs_cem_clu_abs"] = (
        out["peak_val_proxy_clu"] - out["peak_val_cem_clu"]
    )
    out["peak_mag_err_proxy_ref_vs_cem_ref_rel"] = (
        (out["peak_mag_err_proxy_ref_vs_cem_ref_abs"] / denom_ref)
        if denom_ref not in (0.0, np.nan)
        else np.nan
    )
    out["peak_mag_err_proxy_clu_vs_cem_clu_rel"] = (
        (out["peak_mag_err_proxy_clu_vs_cem_clu_abs"] / denom_clu)
        if denom_clu not in (0.0, np.nan)
        else np.nan
    )

    # -------------------------------------------------------------------------
    # Windowed diagnostics centred on the REFERENCE PROXY peak (cyclic).
    # Normalisation uses FULL-HORIZON ranges (fixed denom) for comparability.
    # -------------------------------------------------------------------------
    denom_proxy_ref_full = range_denom(p_ref)
    denom_cem_ref_full = range_denom(a_ref)
    denom_cem_clu_full = range_denom(a_clu)

    d_proxy_ref_full = np.diff(p_ref)
    d_cem_ref_full = np.diff(a_ref)
    d_cem_clu_full = np.diff(a_clu)

    denom_d_proxy_ref_full = range_denom(d_proxy_ref_full)
    denom_d_cem_ref_full = range_denom(d_cem_ref_full)
    denom_d_cem_clu_full = range_denom(d_cem_clu_full)

    for half_days in half_days_list:
        half_w = int(half_days * steps_per_day)
        inds = circular_window_indices(i_peak_proxy_ref, half_w, n)

        a_ref_w = a_ref[inds]
        a_clu_w = a_clu[inds]
        p_ref_w = p_ref[inds]
        p_clu_w = p_clu[inds]

        d_p_ref_w = np.diff(p_ref_w)
        d_p_clu_w = np.diff(p_clu_w)
        d_a_ref_w = np.diff(a_ref_w)
        d_a_clu_w = np.diff(a_clu_w)

        # TSA distortion (proxy): proxy_clu vs proxy_ref (levels + deltas)
        s = summarise_signal_full(p_clu_w, p_ref_w, denom=denom_proxy_ref_full)
        for k, v in s.items():
            out[f"win{half_days}d_tsa_proxy_{k}"] = v

        s = summarise_signal_full(d_p_clu_w, d_p_ref_w, denom=denom_d_proxy_ref_full)
        for k, v in s.items():
            out[f"win{half_days}d_tsa_proxy_delta_{k}"] = v

        # TSA distortion (CEM): cem_clu vs cem_ref (levels)
        s = summarise_signal_full(a_clu_w, a_ref_w, denom=denom_cem_ref_full)
        for k, v in s.items():
            out[f"win{half_days}d_tsa_cem_{k}"] = v

        # Proxy approximation (reference): proxy_ref vs cem_ref (levels + deltas)
        s = summarise_signal_full(p_ref_w, a_ref_w, denom=denom_cem_ref_full)
        for k, v in s.items():
            out[f"win{half_days}d_proxy_ref_vs_cem_ref_{k}"] = v

        s = summarise_signal_full(d_p_ref_w, d_a_ref_w, denom=denom_d_cem_ref_full)
        for k, v in s.items():
            out[f"win{half_days}d_proxy_delta_ref_vs_cem_delta_ref_{k}"] = v

        # Proxy approximation (clustered): proxy_clu vs cem_clu (levels + deltas)
        s = summarise_signal_full(p_clu_w, a_clu_w, denom=denom_cem_clu_full)
        for k, v in s.items():
            out[f"win{half_days}d_proxy_clu_vs_cem_clu_{k}"] = v

        s = summarise_signal_full(d_p_clu_w, d_a_clu_w, denom=denom_d_cem_clu_full)
        for k, v in s.items():
            out[f"win{half_days}d_proxy_delta_clu_vs_cem_delta_clu_{k}"] = v

        # Compound diagnostic: proxy_ref vs cem_clu (levels + deltas)
        s = summarise_signal_full(p_ref_w, a_clu_w, denom=denom_cem_clu_full)
        for k, v in s.items():
            out[f"win{half_days}d_proxy_ref_vs_cem_clu_{k}"] = v

        s = summarise_signal_full(d_p_ref_w, d_a_clu_w, denom=denom_d_cem_clu_full)
        for k, v in s.items():
            out[f"win{half_days}d_proxy_delta_ref_vs_cem_delta_clu_{k}"] = v

    return out


# =============================================================================
# Reference resolution
# =============================================================================


@dataclass(frozen=True)
class ReferenceKey:
    """
    A stable identifier to connect many clustered models to one reference baseline.
    Keep it string-friendly for caching & joins.
    """

    ref_nc: str
    ref_tvp: str
    country: str
    tag: str  # e.g. "standard_2010_2019_reference" or "shuffleXYZ"

    def as_id(self) -> str:
        return f"{self.country}__{self.tag}"


def _infer_country_from_tvp(tvp: str) -> str:
    """
    Infer country from the tvp path/name.

    Rules (STRICT, but supports NL legacy convention):
    1) If tvp filename is exactly 'time_varying_parameters.csv' -> NL
       (because this is your Dutch baseline with no country suffix)
    2) Else try to parse an explicit country code from the string.
    3) Else fail (prevents BE mistakenly becoming NL).
    """
    s = str(tvp).replace("\\", "/")
    fname = Path(s).name

    # --- Legacy Dutch convention ---
    if fname == "time_varying_parameters.csv":
        return "NL"

    # --- Explicit country markers in filename/path ---
    # e.g. time_varying_parameters_BE.csv
    m = re.search(
        r"time_varying_parameters[_\-](NL|BE|GB|IT|ES)\.csv\b",
        fname,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # e.g. ...__shuffleABC_BE.csv or ..._BE.csv
    m2 = re.search(r"(?:^|[_/])(NL|BE|GB|IT|ES)(?:[_\.\-/]|$)", s)
    if m2:
        return m2.group(1).upper()

    m3 = re.search(r"_(NL|BE|GB|IT|ES)(?:\.csv\b|_)", s)
    if m3:
        return m3.group(1).upper()

    raise ValueError(f"Could not infer country from tvp string: {tvp}")


def derive_reference_from_tvp(tvp_path: str, window_id: str) -> ReferenceKey:
    """
    STRICT horizon-aware + NL-special casing.

    Standard reference naming:
      - NL:  standard_{window_id}_reference.nc
      - BE:  standard_{window_id}_reference_BE.nc
      - GB:  standard_{window_id}_reference_GB.nc
      - etc.

    Shuffle reference naming:
      - {shuffleTag}.nc or {shuffleTag}__{window_id}.nc
      - TVP: time_varying_parameters__{shuffleTag}.csv (or horizon-specific if you have it)

    NO FALLBACKS:
      - Missing reference netcdf -> raises
      - Missing reference tvp -> raises
      - No "default to NL"
    """
    country = _infer_country_from_tvp(tvp_path)

    # -------------------------
    # Shuffle case
    # -------------------------
    m = re.search(r"(?i)(shuffle[^/]*?)(?=\.csv\b)", str(tvp_path))
    if m:
        tag_base = m.group(1)

        nc_candidates = [
            MODELS_DIR / f"{tag_base}__{window_id}.nc",
            MODELS_DIR / f"{tag_base}.nc",
        ]
        ref_nc_path = next((p for p in nc_candidates if p.is_file()), None)
        if ref_nc_path is None:
            raise FileNotFoundError(
                f"Missing shuffled reference netcdf for tag '{tag_base}' window '{window_id}'. "
                f"Tried: {[str(p) for p in nc_candidates]}"
            )

        tvp_candidates = [
            TIMESERIES_DIR / f"time_varying_parameters__{tag_base}__{window_id}.csv",
            TIMESERIES_DIR / f"time_varying_parameters__{tag_base}.csv",
        ]
        ref_tvp_path = next((p for p in tvp_candidates if p.is_file()), None)
        if ref_tvp_path is None:
            raise FileNotFoundError(
                f"Missing shuffled reference TVP for tag '{tag_base}' window '{window_id}'. "
                f"Tried: {[str(p) for p in tvp_candidates]}"
            )

        tag = ref_nc_path.stem
        return ReferenceKey(
            ref_nc=str(ref_nc_path), ref_tvp=str(ref_tvp_path), country=country, tag=tag
        )

    # -------------------------
    # Standard case (NL-special)
    # -------------------------
    if country == "NL":
        # Dutch references do NOT carry _NL
        nc_candidates = [MODELS_DIR / f"standard_{window_id}_reference.nc"]
    else:
        # Non-NL MUST carry suffix (e.g., _BE)
        nc_candidates = [
            MODELS_DIR / f"standard_{window_id}_reference_{country}.nc",
            MODELS_DIR / f"standard_{window_id}_reference_{country.upper()}.nc",
        ]

    ref_nc_path = next((p for p in nc_candidates if p.is_file()), None)
    if ref_nc_path is None:
        raise FileNotFoundError(
            f"Missing STANDARD reference netcdf for country '{country}' window '{window_id}'. "
            f"Tried: {[str(p) for p in nc_candidates]}"
        )

    # TVP: if you keep a single time_varying_parameters.csv, that's fine.
    # Still strict: must exist.
    tvp_candidates = [
        TIMESERIES_DIR
        / f"time_varying_parameters_{window_id}.csv",  # if you ever made horizon-specific TVPs
        TIMESERIES_DIR / "time_varying_parameters.csv",
    ]
    ref_tvp_path = next((p for p in tvp_candidates if p.is_file()), None)
    if ref_tvp_path is None:
        raise FileNotFoundError(
            f"Missing reference TVP for window '{window_id}'. Tried: {[str(p) for p in tvp_candidates]}"
        )

    tag = ref_nc_path.stem
    return ReferenceKey(
        ref_nc=str(ref_nc_path), ref_tvp=str(ref_tvp_path), country=country, tag=tag
    )


# =============================================================================
# Calliope model loading (cluster attr fix)
# =============================================================================


def read_netcdf_with_attr_fix(path: str) -> calliope.Model:
    """
    Matches the “attr fix” pattern used previously so Calliope doesn't treat
    clustering as still active inside the stored netcdf.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"NetCDF does not exist: {path}")

    with Dataset(p, "a") as nc:
        if "attrs" not in nc.groups:
            return calliope.read_netcdf(str(p))
        g = nc.groups["attrs"]
        cfg = yaml.safe_load(g.getncattr("config"))
        cfg.get("init", {}).pop("time_cluster", None)
        g.setncattr("config", yaml.safe_dump(cfg))

    return calliope.read_netcdf(str(p))


# =============================================================================
# SoC extraction (CEM output)
# =============================================================================


def _soc_from_reference_model(m: calliope.Model) -> pd.Series:
    """
    Reference (non-clustered): SoC is in results['storage'] for STORAGE_TECH.
    Robust to column naming differences (avoids KeyError: 0).
    """
    # Convert to a tidy dataframe with an explicit value column name
    df = (
        m.results["storage"]
        .fillna(0)
        .to_series()
        .dropna()
        .rename("cem_soc")  # <-- key fix: name the values
        .reset_index()
    )

    # Filter to storage tech
    df = df[df["techs"] == STORAGE_TECH].copy()

    # Ensure datetime index
    df["timesteps"] = pd.to_datetime(df["timesteps"])

    # Build series
    out = df.set_index("timesteps")["cem_soc"].sort_index()
    out.name = "cem_soc"
    return out


def _soc_from_clustered_model(m: calliope.Model, cluster_map_csv: str) -> pd.Series:
    """
    Clustered SoC reconstruction mirrors helper_signals._soc_from_clustered_model:
    SoC(t) = storage_inter_cluster(datestep) + intra_cluster_storage(mapped_datestep, hour)
    """
    cluster_map = pd.read_csv(cluster_map_csv).rename(
        columns={"timesteps": "datesteps", "PeriodNum": "mapped_datesteps"}
    )
    cluster_map["datesteps"] = pd.to_datetime(
        cluster_map["datesteps"], format="%Y-%m-%d"
    )
    cluster_map["mapped_datesteps"] = pd.to_datetime(
        cluster_map["mapped_datesteps"], format="%Y-%m-%d"
    )

    # intra
    df_intra = (
        m.results["storage"]
        .fillna(0)
        .to_series()
        .dropna()
        .to_frame("intra_soc")
        .reset_index()
    )
    df_intra = df_intra[df_intra["techs"] == STORAGE_TECH]
    df_intra["mapped_datesteps"] = pd.to_datetime(
        df_intra["timesteps"], format="%Y-%m-%d"
    )
    df_intra["mapped_datesteps"] = df_intra["mapped_datesteps"].dt.normalize()

    # inter
    df_inter = (
        m.results["storage_inter_cluster"]
        .fillna(0)
        .to_series()
        .dropna()
        .to_frame("inter_soc")
        .reset_index()
    )
    df_inter = df_inter[df_inter["techs"] == STORAGE_TECH]

    # merge by datestep and mapped_datestep
    df = df_inter.merge(cluster_map, on="datesteps", how="left")
    df = df.merge(df_intra, on="mapped_datesteps", how="left")
    df = df[["datesteps", "timesteps", "inter_soc", "intra_soc"]]

    # reconstruct full timestamp: datestep date + hour from timesteps
    time_only = pd.to_datetime(df["timesteps"]).dt.time
    df["full_timestamp"] = df["datesteps"].dt.normalize() + pd.to_timedelta(
        time_only.astype(str)
    )
    df = df.set_index("full_timestamp").sort_index()

    soc = (df["inter_soc"].fillna(0) + df["intra_soc"].fillna(0)).rename("cem_soc")
    soc.index.name = "timesteps"
    return soc


def extract_cem_soc(model_id: str) -> pd.Series:
    nc_path = str(MODELS_DIR / f"{model_id}.nc")
    m = read_netcdf_with_attr_fix(nc_path)

    is_clustered = (
        bool(getattr(m.inputs, "clusters", None).any())
        if hasattr(m, "inputs")
        else False
    )
    if is_clustered:
        cm_path = str(CLUSTER_MAPS_DIR / f"{model_id}.csv")
        return _soc_from_clustered_model(m, cm_path)
    return _soc_from_reference_model(m)


# =============================================================================
# SoC Proxy extraction
# =============================================================================


def _soc_proxy_params_for_country(country: str) -> Dict[str, Any]:
    params = dict(SOC_PROXY_PARAMS_BASE)
    params["dispatchable_techs"] = {
        "known_dispatchable_capacity": DISPATCHABLE_BY_COUNTRY.get(country, 0)
    }
    return params


def _load_reference_tvp(tvp_csv: str, window: tuple[str, str]) -> pd.DataFrame:
    df = calliope_ts_to_pandas(Path(tvp_csv), window[0], window[1])
    df["timesteps"] = pd.to_datetime(df["timesteps"])
    df = df.set_index("timesteps").sort_index()
    return df


def _load_clustered_timeseries_from_ref_tvp(
    cluster_map_csv: str,
    ref_tvp_csv: str,
    window: tuple[str, str],
) -> pd.DataFrame:
    df, _ = extrapolate_ts_from_cluster_map(
        source_cluster_map=cluster_map_csv,
        source_original_ts=ref_tvp_csv,
    )
    df["timesteps"] = pd.to_datetime(df["timesteps"])
    df = df.set_index("timesteps").sort_index()

    # window after reconstruction (keeps logic simple + matches older behavior)
    df = df.loc[window[0] : window[1]]
    return df


def build_soc_proxy(df_ts: pd.DataFrame, country: str) -> pd.Series:
    params = _soc_proxy_params_for_country(country)
    df_proxy, _, _ = generate_soc_proxy(
        df=df_ts.reset_index(),
        demand_field=DEMAND_FIELD,
        renewables_fields_and_weights=params["capacity_weights"],
        dispatchable_techs=params["dispatchable_techs"],
        storage_process_losses=params["storage_process_losses"],
        soc_decomposition=params["soc_decomposition"],
        timestamp_col="timesteps",
        margin_value=0.04,
        margin_mode="fixed",
    )
    s = df_proxy.set_index("timesteps")["soc_proxy_LDES"].rename("soc_proxy")
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s


def extract_soc_proxy_reference(
    ref_key: ReferenceKey, window: tuple[str, str]
) -> pd.Series:
    df_ref = _load_reference_tvp(ref_key.ref_tvp, window)
    return build_soc_proxy(df_ref, country=ref_key.country)


def extract_soc_proxy_clustered(
    model_id: str, ref_key: ReferenceKey, window: tuple[str, str]
) -> pd.Series:
    cm_path = str(CLUSTER_MAPS_DIR / f"{model_id}.csv")
    df = _load_clustered_timeseries_from_ref_tvp(cm_path, ref_key.ref_tvp, window)
    return build_soc_proxy(df, country=ref_key.country)


# =============================================================================
# Metrics
# =============================================================================


def compute_signal_metrics(test: pd.Series, ref: pd.Series) -> Dict[str, float]:
    """Standardised metrics for comparing test vs reference signals.

    Returns ALWAYS:
      pearson_r, nrmse_range, nmae_range, nmbe_range

    Normalisation uses the FULL-HORIZON range of the reference signal (after alignment).
    """
    ref_a, test_a = ref.align(test, join="inner")
    if ref_a.empty:
        return {
            "pearson_r": np.nan,
            "nrmse_range": np.nan,
            "nmae_range": np.nan,
            "nmbe_range": np.nan,
        }
    denom = range_denom(ref_a.to_numpy(dtype=float))
    return summarise_pair_metrics(
        test_a.to_numpy(dtype=float),
        ref_a.to_numpy(dtype=float),
        denom=denom,
    )


def capacity_from_soc(soc: pd.Series) -> float:
    if soc.empty:
        return np.nan
    return float(soc.max())


# =============================================================================
# Cache building
# =============================================================================


@dataclass
class ReferenceBaseline:
    cem_soc_ref: pd.Series
    soc_proxy_ref: pd.Series
    ldes_cap_ref: float


def build_reference_baseline(
    ref_key: ReferenceKey, window: tuple[str, str]
) -> ReferenceBaseline:
    """
    Build (and later memoize) all reference-derived baselines once per reference.
    """
    # CEM SoC from the reference netcdf
    # IMPORTANT: reference netcdf id might not equal tag; this assumes ref_nc exists and stores SoC.
    m_ref = read_netcdf_with_attr_fix(ref_key.ref_nc)
    cem_soc_ref = _soc_from_reference_model(m_ref).loc[window[0] : window[1]]
    soc_proxy_ref = extract_soc_proxy_reference(ref_key, window).loc[
        window[0] : window[1]
    ]

    # Keep resampling consistent (if enabled)
    cem_soc_ref = _maybe_resample(cem_soc_ref, RESAMPLE_FREQ)
    soc_proxy_ref = _maybe_resample(soc_proxy_ref, RESAMPLE_FREQ)

    return ReferenceBaseline(
        cem_soc_ref=cem_soc_ref,
        soc_proxy_ref=soc_proxy_ref,
        ldes_cap_ref=capacity_from_soc(cem_soc_ref),
    )


def save_signals_parquet(
    model_id: str, ref_id: str, cem_soc: pd.Series, soc_proxy: pd.Series
) -> Tuple[str, str]:
    """
    Write two parquet files (cem_soc, soc_proxy). Returns their paths as strings.
    """
    p1 = CACHE_SIGNALS_DIR / f"{model_id}__{ref_id}__cem_soc.parquet"
    p2 = CACHE_SIGNALS_DIR / f"{model_id}__{ref_id}__soc_proxy.parquet"

    cem_soc.to_frame("cem_soc").to_parquet(p1)
    soc_proxy.to_frame("soc_proxy").to_parquet(p2)
    return str(p1), str(p2)


def build_cache(
    log_csv: Path = DEFAULT_LOG_CSV,
    output_csv: Path = CACHE_MASTER_CSV,
) -> pd.DataFrame:
    """
    Main routine:
    - read log
    - for each clustered model id -> resolve reference -> memoize reference baseline
    - extract clustered signals
    - compute metrics + write cache CSV
    """
    df_log = pd.read_csv(log_csv)

    # Minimal expectation: must include 'id' and 'tvp'
    if "id" not in df_log.columns or "tvp" not in df_log.columns:
        raise ValueError(
            f"log CSV must contain at least columns ['id','tvp']; got {df_log.columns.tolist()}"
        )

    if "dates" not in df_log.columns:
        raise ValueError("log CSV must contain a 'dates' column formatted 'yyyy,yyyy'")

    # Optionally filter to models that actually have a cluster map
    df_log = df_log[
        df_log["id"]
        .astype(str)
        .apply(lambda x: (CLUSTER_MAPS_DIR / f"{x}.csv").is_file())
    ].copy()

    # Memoize references
    ref_cache: Dict[tuple[str, str], ReferenceBaseline] = {}

    rows = []
    counter = 0
    for _, r in df_log.iterrows():
        counter += 1
        print(f"Evaluating model: {counter}")

        model_id = str(r["id"])
        tvp = str(r["tvp"])

        start, end, window_id = parse_dates_window(r["dates"])
        window = (start, end)

        ref_key = derive_reference_from_tvp(tvp, window_id)

        # Guardrail: ensure NL has no suffix and non-NL has suffix (matches your conventions)
        ref_stem = Path(ref_key.ref_nc).stem
        if ref_key.country == "NL":
            if re.search(r"_NL$", ref_stem):
                raise ValueError(
                    f"NL reference unexpectedly has _NL suffix: {ref_key.ref_nc}"
                )
        else:
            if not re.search(rf"_{ref_key.country}$", ref_stem):
                raise ValueError(
                    f"Non-NL reference is missing expected _{ref_key.country} suffix: {ref_key.ref_nc}"
                )

        ref_id = ref_key.as_id()

        cache_key = (ref_id, window_id)
        if cache_key not in ref_cache:
            ref_cache[cache_key] = build_reference_baseline(ref_key, window)

        baseline = ref_cache[cache_key]

        # Clustered CEM SoC + proxy (slice to window + resample consistently)
        cem_soc_clu = extract_cem_soc(model_id).loc[window[0] : window[1]]
        soc_proxy_clu = extract_soc_proxy_clustered(model_id, ref_key, window).loc[
            window[0] : window[1]
        ]
        cem_soc_clu = _maybe_resample(cem_soc_clu, RESAMPLE_FREQ)
        soc_proxy_clu = _maybe_resample(soc_proxy_clu, RESAMPLE_FREQ)

        # Capacities (max SoC from CEM signals)
        ldes_cap_clu = capacity_from_soc(cem_soc_clu)

        # --- Aligned stepwise errors + cyclic peak/window diagnostics ---
        aligned = build_stepwise_errors_aligned(
            cem_ref=baseline.cem_soc_ref,
            proxy_ref=baseline.soc_proxy_ref,
            cem_clu=cem_soc_clu,
            proxy_clu=soc_proxy_clu,
        )

        # ---------------------------------------------------------------------
        # Standardised whole-horizon metrics (fixed denom = full-horizon reference ranges)
        # ---------------------------------------------------------------------
        denom_proxy_ref = range_denom(aligned["proxy_ref"].to_numpy(dtype=float))
        denom_cem_ref = range_denom(aligned["cem_ref"].to_numpy(dtype=float))
        denom_cem_clu = range_denom(aligned["cem_clu"].to_numpy(dtype=float))

        d_proxy_ref = aligned["d_proxy_ref"].to_numpy(dtype=float)
        d_proxy_clu = aligned["d_proxy_clu"].to_numpy(dtype=float)
        d_cem_ref = aligned["d_cem_ref"].to_numpy(dtype=float)
        d_cem_clu = aligned["d_cem_clu"].to_numpy(dtype=float)

        denom_d_proxy_ref = range_denom(d_proxy_ref)
        denom_d_cem_ref = range_denom(d_cem_ref)
        denom_d_cem_clu = range_denom(d_cem_clu)

        # TSA distortion (proxy): proxy_clu vs proxy_ref (levels + deltas)
        e_tsa_proxy_sum = summarise_signal_full(
            aligned["proxy_clu"].to_numpy(dtype=float),
            aligned["proxy_ref"].to_numpy(dtype=float),
            denom=denom_proxy_ref,
        )
        d_tsa_proxy_sum = summarise_signal_full(
            d_proxy_clu,
            d_proxy_ref,
            denom=denom_d_proxy_ref,
        )

        # TSA distortion (CEM): cem_clu vs cem_ref (levels)
        e_tsa_cem_sum = summarise_signal_full(
            aligned["cem_clu"].to_numpy(dtype=float),
            aligned["cem_ref"].to_numpy(dtype=float),
            denom=denom_cem_ref,
        )

        # Proxy approximation (reference): proxy_ref vs cem_ref (levels + deltas)
        e_proxy_ref_sum = summarise_signal_full(
            aligned["proxy_ref"].to_numpy(dtype=float),
            aligned["cem_ref"].to_numpy(dtype=float),
            denom=denom_cem_ref,
        )
        d_base_ref_sum = summarise_signal_full(
            d_proxy_ref,
            d_cem_ref,
            denom=denom_d_cem_ref,
        )

        # Proxy approximation (clustered): proxy_clu vs cem_clu (levels + deltas)
        e_proxy_clu_sum = summarise_signal_full(
            aligned["proxy_clu"].to_numpy(dtype=float),
            aligned["cem_clu"].to_numpy(dtype=float),
            denom=denom_cem_clu,
        )
        d_base_clu_sum = summarise_signal_full(
            d_proxy_clu,
            d_cem_clu,
            denom=denom_d_cem_clu,
        )

        # Compound diagnostic: proxy_ref vs cem_clu (levels + deltas)
        comp_lvl_sum = summarise_signal_full(
            aligned["proxy_ref"].to_numpy(dtype=float),
            aligned["cem_clu"].to_numpy(dtype=float),
            denom=denom_cem_clu,
        )
        d_proxy_ref_s, d_cem_clu_s = aligned["d_proxy_ref"].align(
            aligned["d_cem_clu"], join="inner"
        )
        comp_delta_sum = summarise_signal_full(
            d_proxy_ref_s.to_numpy(dtype=float),
            d_cem_clu_s.to_numpy(dtype=float),
            denom=denom_d_cem_clu,
        )
        # Cyclic peak alignment + windowed metrics around the reference CEM peak
        peak_win = peak_and_windows_diagnostics(
            cem_ref=aligned["cem_ref"],
            cem_clu=aligned["cem_clu"],
            proxy_ref=aligned["proxy_ref"],
            proxy_clu=aligned["proxy_clu"],
            smooth_days=PEAK_SMOOTH_DAYS,
            half_days_list=PEAK_WINDOW_HALF_DAYS,
        )

        cap_err_signed = _signed_rel_error(baseline.ldes_cap_ref, ldes_cap_clu)
        cap_err_abs = (
            float(np.abs(cap_err_signed)) if not np.isnan(cap_err_signed) else np.nan
        )

        # (1) Proxy Error: reference proxy vs reference CEM SoC
        metrics_proxy_ref = compute_signal_metrics(
            test=baseline.soc_proxy_ref, ref=baseline.cem_soc_ref
        )

        # (2) Proxy Error: clustered proxy vs clustered CEM SoC
        metrics_proxy_clu = compute_signal_metrics(test=soc_proxy_clu, ref=cem_soc_clu)

        # (3) TSA Error: clustered proxy vs reference proxy
        metrics_tsa = compute_signal_metrics(
            test=soc_proxy_clu, ref=baseline.soc_proxy_ref
        )

        # Optional: store signals for later plotting
        cem_parq, proxy_parq = (None, None)
        if SAVE_SIGNAL_PARQUETS:
            cem_parq, proxy_parq = save_signals_parquet(
                model_id=model_id,
                ref_id=ref_id,
                cem_soc=cem_soc_clu,
                soc_proxy=soc_proxy_clu,
            )

        model_name = r.get("model_name", None)
        k_val, wp_val = parse_k_and_wp(model_name)

        rows.append(
            {
                "model_id": model_id,
                "tvp": tvp,
                "country": ref_key.country,
                "reference_id": ref_id,
                "reference_nc": ref_key.ref_nc,
                "reference_tvp": ref_key.ref_tvp,
                "resample_freq": RESAMPLE_FREQ if RESAMPLE_FREQ else "hourly",
                "dates": str(r["dates"]),
                "window_id": window_id,
                # log hyperparams (if present)
                "Wp": wp_val,
                "k": k_val,
                # capacities
                "ldes_cap_ref_maxsoc": baseline.ldes_cap_ref,
                "ldes_cap_clu_maxsoc": ldes_cap_clu,
                "ldes_cap_error_signed": cap_err_signed,
                "ldes_cap_error_abs": cap_err_abs,
                # proxy error (reference)
                "proxy_ref_pearson_r": metrics_proxy_ref["pearson_r"],
                "proxy_ref_nrmse_range": metrics_proxy_ref["nrmse_range"],
                "proxy_ref_nmae_range": metrics_proxy_ref["nmae_range"],
                "proxy_ref_nmbe_range": metrics_proxy_ref["nmbe_range"],
                # proxy error (clustered)
                "proxy_clu_pearson_r": metrics_proxy_clu["pearson_r"],
                "proxy_clu_nrmse_range": metrics_proxy_clu["nrmse_range"],
                "proxy_clu_nmae_range": metrics_proxy_clu["nmae_range"],
                "proxy_clu_nmbe_range": metrics_proxy_clu["nmbe_range"],
                # TSA error (proxy clustered vs proxy reference)
                "tsa_pearson_r": metrics_tsa["pearson_r"],
                "tsa_nrmse_range": metrics_tsa["nrmse_range"],
                "tsa_nmae_range": metrics_tsa["nmae_range"],
                "tsa_nmbe_range": metrics_tsa["nmbe_range"],
                # signal locations
                "cem_soc_parquet": cem_parq,
                "soc_proxy_parquet": proxy_parq,
                # aligned stepwise error summaries
                "e_proxy_ref_pearson_r": e_proxy_ref_sum["pearson_r"],
                "e_proxy_ref_nrmse_range": e_proxy_ref_sum["nrmse_range"],
                "e_proxy_ref_nmae_range": e_proxy_ref_sum["nmae_range"],
                "e_proxy_ref_nmbe_range": e_proxy_ref_sum["nmbe_range"],
                "e_proxy_clu_pearson_r": e_proxy_clu_sum["pearson_r"],
                "e_proxy_clu_nrmse_range": e_proxy_clu_sum["nrmse_range"],
                "e_proxy_clu_nmae_range": e_proxy_clu_sum["nmae_range"],
                "e_proxy_clu_nmbe_range": e_proxy_clu_sum["nmbe_range"],
                "e_tsa_proxy_pearson_r": e_tsa_proxy_sum["pearson_r"],
                "e_tsa_proxy_nrmse_range": e_tsa_proxy_sum["nrmse_range"],
                "e_tsa_proxy_nmae_range": e_tsa_proxy_sum["nmae_range"],
                "e_tsa_proxy_nmbe_range": e_tsa_proxy_sum["nmbe_range"],
                "e_tsa_cem_pearson_r": e_tsa_cem_sum["pearson_r"],
                "e_tsa_cem_nrmse_range": e_tsa_cem_sum["nrmse_range"],
                "e_tsa_cem_nmae_range": e_tsa_cem_sum["nmae_range"],
                "e_tsa_cem_nmbe_range": e_tsa_cem_sum["nmbe_range"],
                # deltas
                "e_tsa_proxy_delta_pearson_r": d_tsa_proxy_sum["pearson_r"],
                "e_tsa_proxy_delta_nrmse_range": d_tsa_proxy_sum["nrmse_range"],
                "e_tsa_proxy_delta_nmae_range": d_tsa_proxy_sum["nmae_range"],
                "e_tsa_proxy_delta_nmbe_range": d_tsa_proxy_sum["nmbe_range"],
                "e_proxy_delta_ref_vs_cem_delta_ref_pearson_r": d_base_ref_sum[
                    "pearson_r"
                ],
                "e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range": d_base_ref_sum[
                    "nrmse_range"
                ],
                "e_proxy_delta_ref_vs_cem_delta_ref_nmae_range": d_base_ref_sum[
                    "nmae_range"
                ],
                "e_proxy_delta_ref_vs_cem_delta_ref_nmbe_range": d_base_ref_sum[
                    "nmbe_range"
                ],
                "e_proxy_delta_clu_vs_cem_delta_clu_pearson_r": d_base_clu_sum[
                    "pearson_r"
                ],
                "e_proxy_delta_clu_vs_cem_delta_clu_nrmse_range": d_base_clu_sum[
                    "nrmse_range"
                ],
                "e_proxy_delta_clu_vs_cem_delta_clu_nmae_range": d_base_clu_sum[
                    "nmae_range"
                ],
                "e_proxy_delta_clu_vs_cem_delta_clu_nmbe_range": d_base_clu_sum[
                    "nmbe_range"
                ],
                # NEW (Q4 diagnostics): compound discrepancy (REF proxy vs CLU CEM)
                "e_proxy_ref_vs_cem_clu_pearson_r": comp_lvl_sum["pearson_r"],
                "e_proxy_ref_vs_cem_clu_nrmse_range": comp_lvl_sum["nrmse_range"],
                "e_proxy_ref_vs_cem_clu_nmae_range": comp_lvl_sum["nmae_range"],
                "e_proxy_ref_vs_cem_clu_nmbe_range": comp_lvl_sum["nmbe_range"],
                "e_proxy_delta_ref_vs_cem_delta_clu_pearson_r": comp_delta_sum[
                    "pearson_r"
                ],
                "e_proxy_delta_ref_vs_cem_delta_clu_nrmse_range": comp_delta_sum[
                    "nrmse_range"
                ],
                "e_proxy_delta_ref_vs_cem_delta_clu_nmae_range": comp_delta_sum[
                    "nmae_range"
                ],
                "e_proxy_delta_ref_vs_cem_delta_clu_nmbe_range": comp_delta_sum[
                    "nmbe_range"
                ],
                # cyclic peak timing/magnitude + windowed diagnostics
                **peak_win,
            }
        )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    return df_out


# =============================================================================
# CLI entrypoint
# =============================================================================

if __name__ == "__main__":
    df_cache = build_cache(DEFAULT_LOG_CSV, CACHE_MASTER_CSV)
    print(f"Wrote cache with {len(df_cache)} rows to: {CACHE_MASTER_CSV}")
