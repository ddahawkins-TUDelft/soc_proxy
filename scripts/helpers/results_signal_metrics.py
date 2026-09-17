"""Signal-level diagnostics for SoC Proxy experiments.

The extractor records the candidate metric space used to build the original
ex-post diagnostic, plus a few cheap additions which are useful for rebuilding
that diagnostic after reviewer-driven changes.

For each case it compares four chronological signals:

* reference CEM SoC
* original/reference SoC Proxy
* clustered CEM SoC
* reconstructed/clustered SoC Proxy

Five directed error families are evaluated for both level and delta signals:
reference approximation, clustered approximation, TSA distortion, RPCC, and
CEM distortion. Metrics are calculated over the full horizon and over cyclic
windows centred on the original/reference proxy peak.

Raw RMSE/MAE/MBE are retained, together with Pearson correlation and two
normalised variants:

* ``reference_proxy_full_range``: common full-horizon reference-proxy scale.
* ``comparison_reference_full_range``: the pair-specific convention used by
  the legacy signal-analysis code.

Keeping both conventions allows the old diagnostic to be reproduced while a
revised diagnostic can use a consistent common normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.helpers.results_signals import extract_proxy_signal, extract_storage_soc


DEFAULT_WINDOW_HALF_WIDTH_DAYS = (15, 30, 45, 90, 180, 273, 365, 548, 730, 912)

SIGNAL_METRIC_COLUMNS = [
    "case_id",
    "error_family",
    "signal_type",
    "metric",
    "normalisation_basis",
    "normalisation_scale",
    "window_half_width_days",
    "effective_window_steps",
    "covers_full_horizon",
    "window_center_timestamp",
    "value",
]


@dataclass(frozen=True)
class _Signals:
    index: pd.DatetimeIndex
    reference_soc: np.ndarray
    reference_proxy: np.ndarray
    clustered_soc: np.ndarray
    clustered_proxy: np.ndarray
    reference_soc_delta: np.ndarray
    reference_proxy_delta: np.ndarray
    clustered_soc_delta: np.ndarray
    clustered_proxy_delta: np.ndarray


@dataclass(frozen=True)
class _Pair:
    family: str
    test_level: np.ndarray
    reference_level: np.ndarray
    test_delta: np.ndarray
    reference_delta: np.ndarray


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def extract_signal_metrics(
    *,
    case_id: str,
    reference_model,
    clustered_model,
    original_proxy: pd.DataFrame,
    reconstructed_proxy: pd.DataFrame,
    cluster_map: pd.DataFrame,
    node: str = "netherlands",
    tech: str = "h2_salt_cavern",
    window_half_width_days: Iterable[int] = DEFAULT_WINDOW_HALF_WIDTH_DAYS,
    peak_smoothing_days: int = 1,
    validate_proxy_deltas: bool = True,
) -> pd.DataFrame:
    """Extract all candidate signal diagnostics for one solved case.

    ``original_proxy`` and ``reconstructed_proxy`` should be the proxy outputs
    already held in ``TSAArtifacts``. Proxy deltas are read directly from
    ``surplus_LDES``; CEM deltas are calculated from the cyclic SoC trajectory.

    Full-horizon metrics have ``window_half_width_days = NA``. Requested
    windows which are as long as (or longer than) the model horizon are safely
    reduced to one full-horizon sample rather than wrapping over the same data
    multiple times.
    """
    windows = _validate_windows(window_half_width_days)

    signals = _build_signals(
        reference_soc=extract_storage_soc(reference_model, node=node, tech=tech),
        clustered_soc=extract_storage_soc(
            clustered_model, node=node, tech=tech, cluster_map=cluster_map
        ),
        reference_proxy=extract_proxy_signal(original_proxy, field="soc_proxy_LDES"),
        clustered_proxy=extract_proxy_signal(
            reconstructed_proxy, field="soc_proxy_LDES"
        ),
        reference_proxy_delta=_extract_proxy_delta(original_proxy),
        clustered_proxy_delta=_extract_proxy_delta(reconstructed_proxy),
        validate_proxy_deltas=validate_proxy_deltas,
    )

    steps_per_day = _infer_steps_per_day(signals.index)
    peak_idx = _peak_index(
        signals.reference_proxy,
        steps_per_day=steps_per_day,
        smoothing_days=peak_smoothing_days,
    )
    peak_timestamp = signals.index[peak_idx]

    level_proxy_scale = _range(signals.reference_proxy)
    delta_proxy_scale = _range(signals.reference_proxy_delta)
    pairs = _pairs(signals)
    rows: list[dict[str, object]] = []

    # Full horizon.
    all_idx = np.arange(len(signals.index), dtype=int)
    for pair in pairs:
        rows.extend(
            _pair_rows(
                case_id=case_id,
                pair=pair,
                indices=all_idx,
                level_proxy_scale=level_proxy_scale,
                delta_proxy_scale=delta_proxy_scale,
                half_days=None,
                covers_full_horizon=True,
                peak_timestamp=peak_timestamp,
            )
        )

    # Peak-centred cyclic windows.
    for half_days in windows:
        indices, covers_full_horizon = _cyclic_window(
            center=peak_idx,
            half_width=half_days * steps_per_day,
            n=len(signals.index),
        )
        for pair in pairs:
            rows.extend(
                _pair_rows(
                    case_id=case_id,
                    pair=pair,
                    indices=indices,
                    level_proxy_scale=level_proxy_scale,
                    delta_proxy_scale=delta_proxy_scale,
                    half_days=half_days,
                    covers_full_horizon=covers_full_horizon,
                    peak_timestamp=peak_timestamp,
                )
            )

    # Cheap exploratory candidates retained from the legacy analysis.
    rows.extend(
        _peak_rows(
            case_id=case_id,
            signals=signals,
            steps_per_day=steps_per_day,
            smoothing_days=peak_smoothing_days,
            reference_proxy_peak_idx=peak_idx,
            reference_proxy_peak_timestamp=peak_timestamp,
        )
    )

    out = pd.DataFrame(rows, columns=SIGNAL_METRIC_COLUMNS)
    out["window_half_width_days"] = out["window_half_width_days"].astype("Int64")
    out["effective_window_steps"] = out["effective_window_steps"].astype("Int64")
    out["covers_full_horizon"] = out["covers_full_horizon"].astype("boolean")
    out["window_center_timestamp"] = pd.to_datetime(out["window_center_timestamp"])
    return out


# -----------------------------------------------------------------------------
# Signal preparation
# -----------------------------------------------------------------------------


def _extract_proxy_delta(proxy: pd.DataFrame) -> pd.Series:
    try:
        return extract_proxy_signal(proxy, field="surplus_LDES")
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            "Signal diagnostics require 'surplus_LDES' to be retained in the "
            "proxy output alongside 'soc_proxy_LDES'."
        ) from exc


def _build_signals(
    *,
    reference_soc: pd.Series,
    clustered_soc: pd.Series,
    reference_proxy: pd.Series,
    clustered_proxy: pd.Series,
    reference_proxy_delta: pd.Series,
    clustered_proxy_delta: pd.Series,
    validate_proxy_deltas: bool,
) -> _Signals:
    prepared = {
        name: _prepare_series(series, name)
        for name, series in {
            "reference_soc": reference_soc,
            "clustered_soc": clustered_soc,
            "reference_proxy": reference_proxy,
            "clustered_proxy": clustered_proxy,
            "reference_proxy_delta": reference_proxy_delta,
            "clustered_proxy_delta": clustered_proxy_delta,
        }.items()
    }

    index = prepared["reference_soc"].index
    for name, series in prepared.items():
        if not series.index.equals(index):
            raise RuntimeError(
                "Signal metrics require identical chronological indices; "
                f"{name!r} does not match reference_soc."
            )
        if series.isna().any():
            raise RuntimeError(f"Signal {name!r} contains NaN values.")

    arrays = {name: series.to_numpy(dtype=float) for name, series in prepared.items()}
    reference_soc_delta = _cyclic_diff(arrays["reference_soc"])
    clustered_soc_delta = _cyclic_diff(arrays["clustered_soc"])

    if validate_proxy_deltas:
        _validate_proxy_delta(
            arrays["reference_proxy"], arrays["reference_proxy_delta"], "reference"
        )
        _validate_proxy_delta(
            arrays["clustered_proxy"], arrays["clustered_proxy_delta"], "clustered"
        )

    return _Signals(
        index=index,
        reference_soc=arrays["reference_soc"],
        reference_proxy=arrays["reference_proxy"],
        clustered_soc=arrays["clustered_soc"],
        clustered_proxy=arrays["clustered_proxy"],
        reference_soc_delta=reference_soc_delta,
        reference_proxy_delta=arrays["reference_proxy_delta"],
        clustered_soc_delta=clustered_soc_delta,
        clustered_proxy_delta=arrays["clustered_proxy_delta"],
    )


def _prepare_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name!r} must be a pandas Series.")
    out = series.copy()
    out.index = pd.DatetimeIndex(out.index)
    out = out.sort_index().astype(float)
    if out.index.has_duplicates:
        raise RuntimeError(f"Signal {name!r} contains duplicate timestamps.")
    if len(out) < 3:
        raise RuntimeError(f"Signal {name!r} contains fewer than three timesteps.")
    return out


def _cyclic_diff(values: np.ndarray) -> np.ndarray:
    return values - np.roll(values, 1)


def _validate_proxy_delta(level: np.ndarray, delta: np.ndarray, label: str) -> None:
    derived = _cyclic_diff(level)
    scale = max(1.0, float(np.max(np.abs(derived))), float(np.max(np.abs(delta))))
    if not np.allclose(derived, delta, rtol=1e-8, atol=scale * 1e-9):
        mismatch = float(np.max(np.abs(derived - delta)))
        raise RuntimeError(
            f"{label.capitalize()} surplus_LDES is not aligned with the cyclic "
            f"difference of soc_proxy_LDES (max mismatch={mismatch:.6g})."
        )


# -----------------------------------------------------------------------------
# Error families and metrics
# -----------------------------------------------------------------------------


def _pairs(s: _Signals) -> tuple[_Pair, ...]:
    """Directed comparisons; signed errors always use test - reference."""
    return (
        _Pair(
            "reference_approximation",
            s.reference_proxy,
            s.reference_soc,
            s.reference_proxy_delta,
            s.reference_soc_delta,
        ),
        _Pair(
            "clustered_approximation",
            s.clustered_proxy,
            s.clustered_soc,
            s.clustered_proxy_delta,
            s.clustered_soc_delta,
        ),
        _Pair(
            "tsa",
            s.clustered_proxy,
            s.reference_proxy,
            s.clustered_proxy_delta,
            s.reference_proxy_delta,
        ),
        _Pair(
            "rpcc",
            s.reference_proxy,
            s.clustered_soc,
            s.reference_proxy_delta,
            s.clustered_soc_delta,
        ),
        _Pair(
            "cem",
            s.clustered_soc,
            s.reference_soc,
            s.clustered_soc_delta,
            s.reference_soc_delta,
        ),
    )


def _pair_rows(
    *,
    case_id: str,
    pair: _Pair,
    indices: np.ndarray,
    level_proxy_scale: float,
    delta_proxy_scale: float,
    half_days: int | None,
    covers_full_horizon: bool,
    peak_timestamp: pd.Timestamp,
) -> list[dict[str, object]]:
    common = dict(
        case_id=case_id,
        error_family=pair.family,
        window_half_width_days=half_days,
        effective_window_steps=len(indices),
        covers_full_horizon=covers_full_horizon,
        window_center_timestamp=peak_timestamp,
    )

    rows = _metric_rows(
        **common,
        signal_type="level",
        test=pair.test_level[indices],
        reference=pair.reference_level[indices],
        proxy_scale=level_proxy_scale,
        comparison_scale=_range(pair.reference_level),
    )
    rows.extend(
        _metric_rows(
            **common,
            signal_type="delta",
            test=pair.test_delta[indices],
            reference=pair.reference_delta[indices],
            proxy_scale=delta_proxy_scale,
            comparison_scale=_range(pair.reference_delta),
        )
    )
    return rows


def _metric_rows(
    *,
    case_id: str,
    error_family: str,
    signal_type: str,
    test: np.ndarray,
    reference: np.ndarray,
    proxy_scale: float,
    comparison_scale: float,
    window_half_width_days: int | None,
    effective_window_steps: int,
    covers_full_horizon: bool,
    window_center_timestamp: pd.Timestamp,
) -> list[dict[str, object]]:
    raw = _raw_metrics(test, reference)
    common = dict(
        case_id=case_id,
        error_family=error_family,
        signal_type=signal_type,
        window_half_width_days=window_half_width_days,
        effective_window_steps=effective_window_steps,
        covers_full_horizon=covers_full_horizon,
        window_center_timestamp=window_center_timestamp,
    )

    rows = [
        {
            **common,
            "metric": metric,
            "normalisation_basis": None,
            "normalisation_scale": np.nan,
            "value": raw[metric],
        }
        for metric in ("rmse", "mae", "mbe", "pearson")
    ]

    for basis, scale in (
        ("reference_proxy_full_range", proxy_scale),
        ("comparison_reference_full_range", comparison_scale),
    ):
        for metric, source in (("nrmse", "rmse"), ("nmae", "mae"), ("nmbe", "mbe")):
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "normalisation_basis": basis,
                    "normalisation_scale": scale,
                    "value": _normalise(raw[source], scale),
                }
            )
    return rows


def _raw_metrics(test: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(test) & np.isfinite(reference)
    if finite.sum() < 3:
        return {"rmse": np.nan, "mae": np.nan, "mbe": np.nan, "pearson": np.nan}

    test = test[finite]
    reference = reference[finite]
    diff = test - reference
    pearson = (
        np.nan
        if np.isclose(np.std(test), 0.0) or np.isclose(np.std(reference), 0.0)
        else float(np.corrcoef(test, reference)[0, 1])
    )
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "mbe": float(np.mean(diff)),
        "pearson": pearson,
    }


def _range(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return np.nan if finite.size == 0 else float(np.max(finite) - np.min(finite))


def _normalise(value: float, scale: float) -> float:
    if not np.isfinite(value) or not np.isfinite(scale) or np.isclose(scale, 0.0):
        return np.nan
    return float(value / scale)


# -----------------------------------------------------------------------------
# Windows and peak diagnostics
# -----------------------------------------------------------------------------


def _validate_windows(values: Iterable[int]) -> tuple[int, ...]:
    windows = tuple(sorted(int(v) for v in values))
    if any(v <= 0 for v in windows):
        raise ValueError("window_half_width_days must be positive.")
    if len(set(windows)) != len(windows):
        raise ValueError("window_half_width_days contains duplicates.")
    return windows


def _infer_steps_per_day(index: pd.DatetimeIndex) -> int:
    diffs = np.unique(np.diff(index.asi8))
    if len(diffs) != 1:
        raise RuntimeError("Signal metrics require a regular DatetimeIndex.")
    timestep = pd.to_timedelta(int(diffs[0]), unit="ns")
    steps = float(pd.Timedelta(days=1) / timestep)
    if not np.isclose(steps, round(steps)):
        raise RuntimeError(f"Timestep {timestep} does not divide evenly into one day.")
    return int(round(steps))


def _circular_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Legacy-compatible circular rolling mean used only for peak location."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size == 0:
        return values.copy()
    window = min(window, len(values))
    k = window // 2
    padded = np.concatenate([values[-k:], values, values[:k]])
    smoothed = (
        pd.Series(padded)
        .rolling(window=window, center=True, min_periods=window)
        .mean()
        .to_numpy()[k : k + len(values)]
    )
    if np.isnan(smoothed).all():
        return values.copy()
    return pd.Series(smoothed).bfill().ffill().to_numpy()


def _peak_index(values: np.ndarray, *, steps_per_day: int, smoothing_days: int) -> int:
    if smoothing_days < 0:
        raise ValueError("peak_smoothing_days must be >= 0.")
    smoothed = _circular_rolling_mean(
        values, max(1, smoothing_days * steps_per_day)
    )
    return int(np.nanargmax(smoothed))


def _cyclic_window(*, center: int, half_width: int, n: int) -> tuple[np.ndarray, bool]:
    requested_length = 2 * half_width + 1
    if requested_length >= n:
        return np.arange(n, dtype=int), True
    offsets = np.arange(-half_width, half_width + 1, dtype=int)
    return (center + offsets) % n, False


def _peak_rows(
    *,
    case_id: str,
    signals: _Signals,
    steps_per_day: int,
    smoothing_days: int,
    reference_proxy_peak_idx: int,
    reference_proxy_peak_timestamp: pd.Timestamp,
) -> list[dict[str, object]]:
    level_signals = {
        "reference_soc": signals.reference_soc,
        "reference_proxy": signals.reference_proxy,
        "clustered_soc": signals.clustered_soc,
        "clustered_proxy": signals.clustered_proxy,
    }
    peaks = {
        name: _peak_index(values, steps_per_day=steps_per_day, smoothing_days=smoothing_days)
        for name, values in level_signals.items()
    }
    n = len(signals.index)

    def timing_days(test: int, reference: int) -> float:
        distance = test - reference
        if distance > n / 2:
            distance -= n
        elif distance <= -n / 2:
            distance += n
        return float(distance / steps_per_day)

    values = {
        **{
            f"{name}_peak_value": float(signal[peaks[name]])
            for name, signal in level_signals.items()
        },
        "clustered_soc_vs_reference_soc_peak_timing_days": timing_days(
            peaks["clustered_soc"], peaks["reference_soc"]
        ),
        "reference_proxy_vs_reference_soc_peak_timing_days": timing_days(
            peaks["reference_proxy"], peaks["reference_soc"]
        ),
        "clustered_proxy_vs_reference_soc_peak_timing_days": timing_days(
            peaks["clustered_proxy"], peaks["reference_soc"]
        ),
        "clustered_proxy_vs_reference_proxy_peak_timing_days": timing_days(
            peaks["clustered_proxy"], reference_proxy_peak_idx
        ),
    }

    ref_soc_peak = level_signals["reference_soc"][peaks["reference_soc"]]
    ref_proxy_peak = level_signals["reference_proxy"][peaks["reference_proxy"]]
    clu_soc_peak = level_signals["clustered_soc"][peaks["clustered_soc"]]
    clu_proxy_peak = level_signals["clustered_proxy"][peaks["clustered_proxy"]]
    ref_peak_error = ref_proxy_peak - ref_soc_peak
    clu_peak_error = clu_proxy_peak - clu_soc_peak

    values.update(
        {
            "reference_proxy_peak_magnitude_error": float(ref_peak_error),
            "clustered_proxy_peak_magnitude_error": float(clu_peak_error),
            "reference_proxy_peak_magnitude_error_relative": _safe_divide(
                ref_peak_error, ref_soc_peak
            ),
            "clustered_proxy_peak_magnitude_error_relative": _safe_divide(
                clu_peak_error, clu_soc_peak
            ),
        }
    )

    return [
        {
            "case_id": case_id,
            "error_family": "peak_diagnostic",
            "signal_type": "level",
            "metric": metric,
            "normalisation_basis": None,
            "normalisation_scale": np.nan,
            "window_half_width_days": None,
            "effective_window_steps": None,
            "covers_full_horizon": None,
            "window_center_timestamp": reference_proxy_peak_timestamp,
            "value": value,
        }
        for metric, value in values.items()
    ]


def _safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    if np.isclose(denominator, 0.0):
        return np.nan
    return float(numerator / denominator)
