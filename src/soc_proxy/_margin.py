"""Automatic margin selection for the SoC Proxy.

The selector deliberately remains private. Users configure economics alongside
renewable and storage definitions in :func:`soc_proxy.generate_soc_proxy`; this
module receives validated, normalised specifications and evaluates a compact
margin sweep entirely with NumPy arrays.

Selection currently combines two independently motivated informants:

* economic saturation: the first margin capturing at least 80% of the maximum
  annual proxy-system cost saving available over the internal sweep;
* persistent terminal event: the first margin whose dominant LDES event belongs
  to the event which persists across the high-margin tail.

The selected margin is the larger of those two values. Lower-tail / delta
sparsity is recorded as a diagnostic rather than used as a hard constraint;
there is not yet sufficient evidence for a universal sparsity threshold. This
keeps the selection rule explicit and avoids hiding an unvalidated tuning
constant inside the public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ._core import DecompositionPlan, ProxyArrays, build_proxy_arrays


_AUTO_MARGINS = np.round(np.arange(0.0, 0.1500001, 0.005), 6)
_ECONOMIC_CAPTURE_TARGET = 0.80
_TERMINAL_TAIL_POINTS = 5
_EVENT_TOLERANCE_DAYS = 90
_NEAR_ZERO_FRACTIONS = (0.01, 0.05)


@dataclass(frozen=True)
class RenewableSpec:
    """Validated renewable input used internally by the proxy."""

    field: str
    weight: float
    annualised_capacity_cost: float | None = None
    variable_cost: float = 0.0


@dataclass(frozen=True)
class StorageSpec:
    """Validated effective storage-chain input used internally by the proxy."""

    charging_efficiency: float
    discharging_efficiency: float
    annualised_energy_capacity_cost: float | None = None
    annualised_charge_power_cost: float | None = None
    annualised_discharge_power_cost: float | None = None
    charge_variable_cost: float = 0.0
    discharge_variable_cost: float = 0.0


@dataclass(frozen=True)
class MarginDiagnostics:
    """Summary and compact sweep returned by automatic margin selection."""

    selected_margin: float
    economic_margin_80: float
    terminal_event_margin: float
    terminal_event_timestamp: pd.Timestamp
    terminal_event_support_fraction: float
    selected_terminal_event_prominence: float
    selected_near_zero_1pct_delta_fraction: float
    selected_near_zero_5pct_delta_fraction: float
    sweep: pd.DataFrame


@dataclass(frozen=True)
class _Candidate:
    margin: float
    curtailment_factor: float
    total_renewable_capacity: float
    annual_cost: float
    ldes_range: float
    dominant_peak_timestamp: pd.Timestamp
    terminal_event_prominence: float | None
    near_zero_1pct_delta_fraction: float
    near_zero_5pct_delta_fraction: float


def select_margin(
    *,
    chronology: pd.DatetimeIndex,
    renewable_profiles: Mapping[str, np.ndarray],
    renewables: tuple[RenewableSpec, ...],
    residual_demand: np.ndarray,
    base_gen: np.ndarray,
    base_renewable_capacity: float,
    c_star: float,
    storage: StorageSpec,
    decomposition_plan: DecompositionPlan,
    horizon_years: float,
) -> MarginDiagnostics:
    """Select an endogenous margin from a lightweight internal sweep."""
    _validate_auto_economics(renewables, storage)

    fixed_cost_rate = sum(
        spec.weight * float(spec.annualised_capacity_cost)
        for spec in renewables
    )
    variable_cost_profile = np.zeros(len(chronology), dtype=np.float64)
    for spec in renewables:
        variable_cost_profile += (
            spec.weight
            * renewable_profiles[spec.field]
            * float(spec.variable_cost)
        )

    raw_candidates: list[dict[str, object]] = []

    for margin in _AUTO_MARGINS:
        curtailment_factor = max(1e-6, c_star * (1.0 - float(margin)))
        arrays = build_proxy_arrays(
            base_gen=base_gen,
            demand=residual_demand,
            curtailment_factor=curtailment_factor,
            eta_ch=storage.charging_efficiency,
            eta_dis=storage.discharging_efficiency,
            decomposition_plan=decomposition_plan,
        )

        total_capacity = base_renewable_capacity / curtailment_factor
        annual_cost = _proxy_system_annual_cost(
            arrays=arrays,
            total_renewable_capacity=total_capacity,
            fixed_renewable_cost_rate=fixed_cost_rate,
            variable_renewable_cost_profile=variable_cost_profile,
            residual_demand=residual_demand,
            storage=storage,
            horizon_years=horizon_years,
        )

        shifted_soc = arrays.soc_ldes - float(np.min(arrays.soc_ldes))
        dominant_idx = int(np.argmax(shifted_soc))
        sparsity = _near_zero_delta_exposure(shifted_soc)

        raw_candidates.append(
            {
                "margin": float(margin),
                "curtailment_factor": float(curtailment_factor),
                "total_renewable_capacity": float(total_capacity),
                "annual_cost": float(annual_cost),
                "ldes_range": float(np.ptp(shifted_soc)),
                "dominant_peak_timestamp": chronology[dominant_idx],
                "near_zero_1pct_delta_fraction": sparsity[0.01],
                "near_zero_5pct_delta_fraction": sparsity[0.05],
                "soc_ldes": shifted_soc,
            }
        )

    terminal_timestamp, support_fraction = _terminal_event(raw_candidates)

    for row in raw_candidates:
        shifted_soc = np.asarray(row.pop("soc_ldes"), dtype=np.float64)
        row["terminal_event_prominence"] = _event_prominence(
            shifted_soc,
            chronology,
            terminal_timestamp,
        )
        row["terminal_event_match"] = _same_event(
            pd.Timestamp(row["dominant_peak_timestamp"]),
            terminal_timestamp,
        )

    sweep = pd.DataFrame(raw_candidates).sort_values("margin").reset_index(
        drop=True
    )
    sweep = _add_frontier_diagnostics(sweep)
    sweep = _add_economic_capture(sweep)

    economic_margin = float(
        sweep.loc[
            sweep["economic_saving_capture"]
            >= _ECONOMIC_CAPTURE_TARGET - 1e-12,
            "margin",
        ].iloc[0]
    )

    event_matches = sweep.loc[sweep["terminal_event_match"], "margin"]
    if event_matches.empty:
        # The tail event itself must occur somewhere, so this should only be
        # reachable through a pathological timestamp/index mismatch.
        raise RuntimeError("Terminal event was not found in the margin sweep.")
    terminal_event_margin = float(event_matches.iloc[0])

    selected_margin = max(economic_margin, terminal_event_margin)
    selected = sweep.loc[np.isclose(sweep["margin"], selected_margin)]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one selected-margin row, found {len(selected)}."
        )
    selected_row = selected.iloc[0]

    return MarginDiagnostics(
        selected_margin=selected_margin,
        economic_margin_80=economic_margin,
        terminal_event_margin=terminal_event_margin,
        terminal_event_timestamp=terminal_timestamp,
        terminal_event_support_fraction=support_fraction,
        selected_terminal_event_prominence=float(
            selected_row["terminal_event_prominence"]
        ),
        selected_near_zero_1pct_delta_fraction=float(
            selected_row["near_zero_1pct_delta_fraction"]
        ),
        selected_near_zero_5pct_delta_fraction=float(
            selected_row["near_zero_5pct_delta_fraction"]
        ),
        sweep=sweep,
    )


def _proxy_system_annual_cost(
    *,
    arrays: ProxyArrays,
    total_renewable_capacity: float,
    fixed_renewable_cost_rate: float,
    variable_renewable_cost_profile: np.ndarray,
    residual_demand: np.ndarray,
    storage: StorageSpec,
    horizon_years: float,
) -> float:
    """Price the proxy-implied VRE/LDES system for one candidate margin."""
    vre_fixed = total_renewable_capacity * fixed_renewable_cost_rate

    charging_input = (
        np.maximum(arrays.surplus, 0.0)
        / storage.charging_efficiency
    )
    positive_residual_demand = np.maximum(residual_demand, 0.0)
    background_surplus = np.maximum(-residual_demand, 0.0)

    direct_renewable_use = np.minimum(
        arrays.generation,
        positive_residual_demand,
    )
    renewable_charging = np.maximum(
        charging_input - background_surplus,
        0.0,
    )
    renewable_charging = np.minimum(
        renewable_charging,
        np.maximum(arrays.generation - direct_renewable_use, 0.0),
    )
    dispatched_generation = np.clip(
        direct_renewable_use + renewable_charging,
        0.0,
        arrays.generation,
    )
    dispatch_fraction = np.divide(
        dispatched_generation,
        arrays.generation,
        out=np.zeros_like(arrays.generation),
        where=arrays.generation > 1e-12,
    )

    # variable_renewable_cost_profile is sum(weight_i * CF_i * variable_i).
    # Multiplication by total capacity converts it to currency/hour.
    vre_variable = (
        total_renewable_capacity
        * float(np.sum(variable_renewable_cost_profile * dispatch_fraction))
        / horizon_years
    )

    ldes = arrays.surplus_ldes
    storage_energy_capacity = float(np.ptp(arrays.soc_ldes))
    charge_power_input = (
        float(np.max(np.maximum(ldes, 0.0)))
        / storage.charging_efficiency
    )
    discharge_power_output = (
        float(np.max(np.maximum(-ldes, 0.0)))
        * storage.discharging_efficiency
    )

    annual_charge_input = (
        float(np.sum(np.maximum(ldes, 0.0)))
        / storage.charging_efficiency
        / horizon_years
    )
    annual_discharge_output = (
        float(np.sum(np.maximum(-ldes, 0.0)))
        * storage.discharging_efficiency
        / horizon_years
    )

    ldes_cost = (
        storage_energy_capacity
        * float(storage.annualised_energy_capacity_cost)
        + charge_power_input
        * float(storage.annualised_charge_power_cost)
        + discharge_power_output
        * float(storage.annualised_discharge_power_cost)
        + annual_charge_input * storage.charge_variable_cost
        + annual_discharge_output * storage.discharge_variable_cost
    )

    return float(vre_fixed + vre_variable + ldes_cost)



def _add_frontier_diagnostics(sweep: pd.DataFrame) -> pd.DataFrame:
    """Add inexpensive physical diagnostics for the renewable/LDES frontier."""
    out = sweep.copy()
    baseline = out.loc[np.isclose(out["margin"], 0.0)]
    if len(baseline) != 1:
        raise RuntimeError("Automatic margin sweep must contain exactly one m=0.")

    baseline_range = float(baseline["ldes_range"].iloc[0])
    if baseline_range > 0:
        out["ldes_range_relative_to_m0"] = out["ldes_range"] / baseline_range
    else:
        out["ldes_range_relative_to_m0"] = np.nan

    out["interval_storage_saved_per_renewable_capacity"] = (
        -out["ldes_range"].diff()
        / out["total_renewable_capacity"].diff()
    )
    return out

def _add_economic_capture(sweep: pd.DataFrame) -> pd.DataFrame:
    """Add saving relative to m=0 and fraction of attainable saving captured."""
    out = sweep.copy()
    baseline = out.loc[np.isclose(out["margin"], 0.0)]
    if len(baseline) != 1:
        raise RuntimeError("Automatic margin sweep must contain exactly one m=0.")

    cost_0 = float(baseline["annual_cost"].iloc[0])
    minimum_cost = float(out["annual_cost"].min())
    maximum_saving = cost_0 - minimum_cost

    out["economic_saving_vs_m0"] = cost_0 - out["annual_cost"]
    if maximum_saving <= 1e-12:
        out["economic_saving_capture"] = 1.0
    else:
        out["economic_saving_capture"] = (
            out["economic_saving_vs_m0"] / maximum_saving
        )

    return out


def _terminal_event(
    rows: list[dict[str, object]],
) -> tuple[pd.Timestamp, float]:
    """Identify the robust dominant event in the high-margin tail."""
    tail = rows[-min(_TERMINAL_TAIL_POINTS, len(rows)) :]
    reversed_tail = list(reversed(tail))
    tolerance = pd.Timedelta(days=_EVENT_TOLERANCE_DAYS)
    clusters: list[dict[str, object]] = []

    for row in reversed_tail:
        timestamp = pd.Timestamp(row["dominant_peak_timestamp"])
        candidates: list[tuple[pd.Timedelta, int]] = []

        for cluster_id, cluster in enumerate(clusters):
            distance = abs(timestamp - pd.Timestamp(cluster["anchor"]))
            if distance <= tolerance:
                candidates.append((distance, cluster_id))

        if candidates:
            _, cluster_id = min(candidates, key=lambda item: (item[0], item[1]))
            clusters[cluster_id]["members"].append(row)
        else:
            clusters.append({"anchor": timestamp, "members": [row]})

    max_margin = float(tail[-1]["margin"])

    def cluster_key(cluster: dict[str, object]) -> tuple[int, int, float]:
        member_margins = [
            float(member["margin"])
            for member in cluster["members"]
        ]
        contains_max = int(
            any(np.isclose(value, max_margin) for value in member_margins)
        )
        return len(member_margins), contains_max, float(sum(member_margins))

    selected = max(clusters, key=cluster_key)
    timestamps = [
        pd.Timestamp(member["dominant_peak_timestamp"])
        for member in selected["members"]
    ]
    timestamp_ns = np.asarray([value.value for value in timestamps], dtype=np.int64)
    distance_sums = np.abs(timestamp_ns[:, None] - timestamp_ns[None, :]).sum(
        axis=1
    )
    terminal_timestamp = timestamps[int(np.argmin(distance_sums))]
    support = len(timestamps) / len(tail)
    return terminal_timestamp, float(support)


def _event_prominence(
    shifted_soc: np.ndarray,
    chronology: pd.DatetimeIndex,
    event_timestamp: pd.Timestamp,
) -> float:
    """Return event-window peak divided by the global proxy peak."""
    global_peak = float(np.max(shifted_soc))
    if global_peak <= 0:
        return np.nan

    distances = np.abs(chronology - event_timestamp)
    mask = distances <= pd.Timedelta(days=_EVENT_TOLERANCE_DAYS / 2)
    if not np.any(mask):
        return np.nan

    return float(np.max(shifted_soc[mask]) / global_peak)


def _same_event(left: pd.Timestamp, right: pd.Timestamp) -> bool:
    return abs(left - right) <= pd.Timedelta(days=_EVENT_TOLERANCE_DAYS)


def _near_zero_delta_exposure(shifted_soc: np.ndarray) -> dict[float, float]:
    """Measure delta chronology spent wholly inside the lower proxy tail."""
    proxy_range = float(np.ptp(shifted_soc))
    metrics: dict[float, float] = {}

    for fraction in _NEAR_ZERO_FRACTIONS:
        if proxy_range <= 0:
            mask = np.ones(len(shifted_soc), dtype=bool)
        else:
            mask = shifted_soc <= fraction * proxy_range

        if len(mask) < 2:
            metrics[fraction] = np.nan
        else:
            metrics[fraction] = float(np.mean(mask[:-1] & mask[1:]))

    return metrics


def _validate_auto_economics(
    renewables: tuple[RenewableSpec, ...],
    storage: StorageSpec,
) -> None:
    missing_renewables = [
        spec.field
        for spec in renewables
        if spec.annualised_capacity_cost is None
    ]
    if missing_renewables:
        raise ValueError(
            "margin_mode='auto' requires annualised_capacity_cost for every "
            f"renewable field; missing {missing_renewables}."
        )

    missing_storage = [
        name
        for name, value in (
            (
                "annualised_energy_capacity_cost",
                storage.annualised_energy_capacity_cost,
            ),
            (
                "annualised_charge_power_cost",
                storage.annualised_charge_power_cost,
            ),
            (
                "annualised_discharge_power_cost",
                storage.annualised_discharge_power_cost,
            ),
        )
        if value is None
    ]
    if missing_storage:
        raise ValueError(
            "margin_mode='auto' requires storage annualised costs; missing "
            f"{missing_storage}."
        )
