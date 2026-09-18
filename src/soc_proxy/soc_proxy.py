"""Public SoC Proxy API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from ._core import (
    build_decomposition_plan,
    build_proxy_arrays,
    find_min_feasible_curtailment,
)
from ._margin import (
    MarginDiagnostics,
    RenewableSpec,
    StorageSpec,
    select_margin,
)


@dataclass(frozen=True)
class SocProxyResult:
    """Result returned by :func:`generate_soc_proxy`."""

    data: pd.DataFrame
    capacity_factors: dict[str, float]
    renewable_capacities: dict[str, float]
    base_renewable_capacity: float
    margin: float
    curtailment_factor: float
    margin_diagnostics: MarginDiagnostics | None


def generate_soc_proxy(
    df: pd.DataFrame,
    *,
    renewables: Mapping[str, Mapping[str, float]],
    storage: Mapping[str, float],
    demand_field: str = "demand_power",
    dispatchable_capacity: float = 0.0,
    soc_decomposition: Mapping[str, object] | None = None,
    timestamp_col: str | None = None,
    margin_mode: Literal["auto", "fixed"] = "auto",
    margin_value: float | None = None,
) -> SocProxyResult:
    """Generate LDES/SDES storage proxies from demand and renewable profiles.

    Parameters
    ----------
    df
        Input chronology. Renewable fields are interpreted as per-unit capacity
        factors; ``demand_field`` is interpreted as power demand.
    renewables
        Mapping from input-column name to a technology description. Every
        technology requires ``weight``. Automatic margin selection additionally
        requires ``annualised_capacity_cost``. ``variable_cost`` is optional and
        defaults to zero. Costs may use any currency provided all supplied costs
        use a consistent basis.

        Example::

            {
                "solar": {
                    "weight": 2,
                    "annualised_capacity_cost": 0.067,  # M€/MW/year
                    "variable_cost": 0.0,               # M€/MWh
                },
                "wind": {
                    "weight": 1,
                    "annualised_capacity_cost": 0.124,
                    "variable_cost": 0.002,
                },
            }

    storage
        Effective storage-chain description. ``charging_efficiency`` and
        ``discharging_efficiency`` are always required. Automatic margin
        selection additionally requires annualised energy-, charge-power-, and
        discharge-power-capacity costs. The charge-power cost is interpreted per
        MW of electricity entering the effective storage chain; the
        discharge-power cost is interpreted per MW of electricity leaving it.
        Optional variable costs use the same input/output electricity basis.
    demand_field
        Demand column in ``df``.
    dispatchable_capacity
        Fixed dispatchable/background capacity subtracted from demand before
        renewable/storage balancing.
    soc_decomposition
        Mapping with ``method`` and ``time_horizon_hours``. Defaults to a
        24-hour FFT low-pass split.
    timestamp_col
        Optional timestamp column. If omitted, ``df`` must use a DatetimeIndex.
    margin_mode
        ``"auto"`` selects an endogenous margin from the economic and
        persistent-event informants. ``"fixed"`` uses ``margin_value`` and
        performs no automatic margin sweep.
    margin_value
        User-specified margin for ``margin_mode="fixed"``. Values must satisfy
        ``0 <= margin < 1``. A fixed value of zero replaces the previous
        ``margin_mode="none"`` behaviour.

    Returns
    -------
    SocProxyResult
        Augmented chronology, implied renewable capacities, selected margin,
        and automatic-selection diagnostics when applicable.

    Notes
    -----
    SoC proxy levels are intentionally *not* shifted to make their minimum
    zero. The proxy is defined only up to an additive constant and the TSA
    consumes its delta signal. Leaving the cumulative signal unshifted avoids
    giving artificial significance to repeated contacts with a zero level;
    genuine flat periods in the delta signal are preserved.
    """
    if df.empty:
        raise ValueError("df cannot be empty.")

    decomposition = _normalise_decomposition(soc_decomposition)
    renewable_specs = _normalise_renewables(renewables)
    storage_spec = _normalise_storage(storage)

    chronology = _chronology(df, timestamp_col=timestamp_col)
    timestep_hours = _regular_timestep_hours(chronology)

    renewable_fields = [spec.field for spec in renewable_specs]
    required = [demand_field, *renewable_fields]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing fields in dataframe: {missing}")

    profiles = {
        field: df[field].to_numpy(dtype=np.float64, copy=False)
        for field in renewable_fields
    }
    original_demand = df[demand_field].to_numpy(dtype=np.float64, copy=False)
    residual_demand = original_demand - float(dispatchable_capacity)

    weights = np.asarray([spec.weight for spec in renewable_specs], dtype=float)
    weighted_cf = np.zeros(len(df), dtype=np.float64)
    for spec in renewable_specs:
        weighted_cf += spec.weight * profiles[spec.field]

    capacity_factors = {
        field: float(np.mean(profiles[field]))
        for field in renewable_fields
    }
    capacity_factors["weighted_mean"] = float(np.mean(weighted_cf))

    cf_sum = float(np.sum(weighted_cf))
    if cf_sum <= 0:
        raise ValueError("Weighted renewable capacity factors must sum to > 0.")

    base_renewable_capacity = float(np.sum(residual_demand) / cf_sum)
    if base_renewable_capacity <= 0:
        raise ValueError(
            "Residual mean demand must be positive after subtracting "
            "dispatchable_capacity."
        )

    base_gen = weighted_cf * base_renewable_capacity

    c_star, lost_final, _ = find_min_feasible_curtailment(
        base_gen,
        residual_demand,
        storage_spec.charging_efficiency,
        storage_spec.discharging_efficiency,
        tol=1e-12,
    )
    if lost_final > 1e-9:
        raise RuntimeError(
            "Could not find a feasible renewable scaling within the internal "
            f"curtailment search range; remaining lost load={lost_final:.6g}."
        )

    decomposition_plan = build_decomposition_plan(
        n=len(df),
        timestep_hours=timestep_hours,
        method=str(decomposition["method"]),
        time_horizon_hours=float(decomposition["time_horizon_hours"]),
    )

    margin_diagnostics: MarginDiagnostics | None
    if margin_mode == "fixed":
        margin = _validate_fixed_margin(margin_value)
        margin_diagnostics = None
    elif margin_mode == "auto":
        horizon_years = _horizon_years(chronology)
        margin_diagnostics = select_margin(
            chronology=chronology,
            renewable_profiles=profiles,
            renewables=renewable_specs,
            residual_demand=residual_demand,
            base_gen=base_gen,
            base_renewable_capacity=base_renewable_capacity,
            c_star=c_star,
            storage=storage_spec,
            decomposition_plan=decomposition_plan,
            horizon_years=horizon_years,
        )
        margin = margin_diagnostics.selected_margin
    else:
        raise ValueError(
            f"Unknown margin_mode {margin_mode!r}; expected 'auto' or 'fixed'."
        )

    curtailment_factor = max(1e-6, c_star * (1.0 - margin))
    arrays = build_proxy_arrays(
        base_gen=base_gen,
        demand=residual_demand,
        curtailment_factor=curtailment_factor,
        eta_ch=storage_spec.charging_efficiency,
        eta_dis=storage_spec.discharging_efficiency,
        decomposition_plan=decomposition_plan,
    )
    if arrays.lost_load > 1e-8:
        raise RuntimeError(
            "Selected margin unexpectedly produced lost load: "
            f"{arrays.lost_load:.6g}."
        )

    output = df.copy()
    output["mean_capacity_factor"] = weighted_cf
    output["surplus"] = arrays.surplus
    output["surplus_LDES"] = arrays.surplus_ldes
    output["surplus_SDES"] = arrays.surplus_sdes
    output["soc_proxy_LDES"] = arrays.soc_ldes
    output["soc_proxy_SDES"] = arrays.soc_sdes

    for column in (
        "surplus",
        "surplus_LDES",
        "surplus_SDES",
        "soc_proxy_LDES",
        "soc_proxy_SDES",
    ):
        values = output[column].to_numpy(dtype=np.float64, copy=True)
        values[np.abs(values) < 1e-9] = 0.0
        output[column] = values

    # The cycle residual is the sum of the delta signal (equivalently the final
    # cumulative value from an implicit zero start), not last SoC minus first
    # SoC, which would omit the first timestep's increment.
    cycle_residual = float(np.sum(arrays.surplus_ldes))
    if abs(cycle_residual) > 1e4:
        print(
            "[SoC Proxy] Warning: LDES proxy is severely non-cyclical "
            f"(sum(delta)={cycle_residual:.3e})."
        )

    selected_total_capacity = base_renewable_capacity / curtailment_factor
    renewable_capacities = {
        spec.field: float(spec.weight * selected_total_capacity)
        for spec in renewable_specs
    }
    renewable_capacities["total"] = float(selected_total_capacity)

    print(
        f"[SoC Proxy] c*={c_star:.6f} | margin={margin:.3%} | "
        f"curtailment_factor={curtailment_factor:.6f}"
    )
    if margin_diagnostics is not None:
        print(
            "[SoC Proxy] auto margin: "
            f"economic m80={margin_diagnostics.economic_margin_80:.3%}, "
            f"terminal-event m={margin_diagnostics.terminal_event_margin:.3%}, "
            f"selected={margin:.3%}"
        )

    return SocProxyResult(
        data=output,
        capacity_factors=capacity_factors,
        renewable_capacities=renewable_capacities,
        base_renewable_capacity=base_renewable_capacity,
        margin=margin,
        curtailment_factor=curtailment_factor,
        margin_diagnostics=margin_diagnostics,
    )


def _normalise_renewables(
    renewables: Mapping[str, Mapping[str, float]],
) -> tuple[RenewableSpec, ...]:
    if not renewables:
        raise ValueError("renewables cannot be empty.")

    raw: list[tuple[str, Mapping[str, float], float]] = []
    total_weight = 0.0

    for field, definition in renewables.items():
        if "weight" not in definition:
            raise ValueError(f"Renewable {field!r} is missing 'weight'.")
        weight = float(definition["weight"])
        if weight < 0:
            raise ValueError(f"Renewable {field!r} weight cannot be negative.")
        total_weight += weight
        raw.append((field, definition, weight))

    if total_weight <= 0:
        raise ValueError("Renewable weights must sum to > 0.")

    specs: list[RenewableSpec] = []
    for field, definition, weight in raw:
        annualised_cost = definition.get("annualised_capacity_cost")
        if annualised_cost is not None and float(annualised_cost) < 0:
            raise ValueError(
                f"Renewable {field!r} annualised_capacity_cost cannot be negative."
            )
        variable_cost = float(definition.get("variable_cost", 0.0))

        specs.append(
            RenewableSpec(
                field=field,
                weight=weight / total_weight,
                annualised_capacity_cost=(
                    None if annualised_cost is None else float(annualised_cost)
                ),
                variable_cost=variable_cost,
            )
        )

    return tuple(specs)


def _normalise_storage(storage: Mapping[str, float]) -> StorageSpec:
    required = {"charging_efficiency", "discharging_efficiency"}
    missing = required - set(storage)
    if missing:
        raise ValueError(f"storage is missing required fields: {sorted(missing)}")

    eta_ch = float(storage["charging_efficiency"])
    eta_dis = float(storage["discharging_efficiency"])
    if not 0 < eta_ch <= 1:
        raise ValueError("storage charging_efficiency must be in (0, 1].")
    if not 0 < eta_dis <= 1:
        raise ValueError("storage discharging_efficiency must be in (0, 1].")

    def optional_nonnegative(name: str) -> float | None:
        value = storage.get(name)
        if value is None:
            return None
        value = float(value)
        if value < 0:
            raise ValueError(f"storage {name} cannot be negative.")
        return value

    return StorageSpec(
        charging_efficiency=eta_ch,
        discharging_efficiency=eta_dis,
        annualised_energy_capacity_cost=optional_nonnegative(
            "annualised_energy_capacity_cost"
        ),
        annualised_charge_power_cost=optional_nonnegative(
            "annualised_charge_power_cost"
        ),
        annualised_discharge_power_cost=optional_nonnegative(
            "annualised_discharge_power_cost"
        ),
        charge_variable_cost=float(storage.get("charge_variable_cost", 0.0)),
        discharge_variable_cost=float(
            storage.get("discharge_variable_cost", 0.0)
        ),
    )


def _normalise_decomposition(
    definition: Mapping[str, object] | None,
) -> dict[str, object]:
    if definition is None:
        return {
            "method": "fft_lowpass",
            "time_horizon_hours": 24.0,
        }

    if "method" not in definition or "time_horizon_hours" not in definition:
        raise ValueError(
            "soc_decomposition requires 'method' and 'time_horizon_hours'."
        )

    return {
        "method": str(definition["method"]),
        "time_horizon_hours": float(definition["time_horizon_hours"]),
    }


def _chronology(
    df: pd.DataFrame,
    *,
    timestamp_col: str | None,
) -> pd.DatetimeIndex:
    if timestamp_col is not None:
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column {timestamp_col!r} is missing.")
        chronology = pd.DatetimeIndex(pd.to_datetime(df[timestamp_col]))
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                "df must use a DatetimeIndex when timestamp_col is not supplied."
            )
        chronology = pd.DatetimeIndex(df.index)

    if chronology.has_duplicates:
        raise ValueError("Chronology contains duplicate timestamps.")
    if not chronology.is_monotonic_increasing:
        raise ValueError("Chronology must be monotonically increasing.")
    if len(chronology) < 2:
        raise ValueError("At least two timestamps are required.")

    return chronology


def _regular_timestep_hours(
    chronology: pd.DatetimeIndex,
) -> float:
    if len(chronology) < 2:
        raise ValueError(
            "At least two timestamps are required."
        )

    deltas = chronology[1:] - chronology[:-1]
    timestep = deltas[0]

    if not np.all(deltas == timestep):
        raise ValueError(
            "Timeseries data must have a regular timestep."
        )

    hours = float(
        timestep / pd.Timedelta(hours=1)
    )

    if hours <= 0:
        raise ValueError(
            "Timeseries timestep must be positive."
        )

    return hours


def _validate_fixed_margin(margin_value: float | None) -> float:
    if margin_value is None:
        raise ValueError("margin_value is required when margin_mode='fixed'.")

    margin = float(margin_value)
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin_value must satisfy 0 <= margin < 1.")
    return margin


def _horizon_years(chronology: pd.DatetimeIndex) -> float:
    """Return represented years from an evenly spaced chronology."""
    timestep = chronology[1] - chronology[0]
    represented_end = chronology[-1] + timestep
    start = chronology[0]

    years = represented_end.year - start.year
    if years > 0 and start + pd.DateOffset(years=years) == represented_end:
        return float(years)

    return float(
        (represented_end - start).total_seconds()
        / (365.2425 * 24.0 * 3600.0)
    )
