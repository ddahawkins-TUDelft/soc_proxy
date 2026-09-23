"""Extract case-level experiment parameters and solver metadata."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scripts.helpers.calliope import calliope_runtime_seconds


def extract_parameters(
    config: dict[str, Any],
    case,
    *,
    reference_model,
    case_id: str,
) -> pd.DataFrame:
    """Extract one row of configured and resolved case-level parameters."""
    model = case.calliope_model
    proxy_result = case.tsa.original_proxy

    runtime_proxy_seconds = float(case.tsa.runtime_proxy_seconds)

    runtime_tsa_seconds = float(case.tsa.runtime_tsa_seconds)

    runtime_calliope_seconds = calliope_runtime_seconds(model)

    reference_runtime_calliope_seconds = float(reference_model.runtime_calliope_seconds)

    runtime_method_seconds = (
        runtime_proxy_seconds + runtime_tsa_seconds + runtime_calliope_seconds
    )

    data_params = config["data_params"]
    soc_proxy_params = config["soc_proxy_params"]
    tsa_params = config["tsa_params"]

    start = pd.Timestamp(data_params["start_date"])
    end = pd.Timestamp(data_params["end_date"])

    proxy_use = tsa_params["soc_proxy"]
    decomposition = soc_proxy_params["soc_decomposition"]

    row = {
        "case_id": case_id,
        "experiment_name": config.get("experiment_name"),
        "country": data_params["country"],
        "dispatchable_capacity": float(soc_proxy_params["dispatchable_capacity"]),
        "start_date": start,
        "end_date": end,
        "horizon_years": _horizon_years(start, end),
        "dispatch_mode": config["dispatch_mode"],
        "scenario": config["calliope_params"]["scenario"],
        "solver": config["solver_params"]["solver"],
        "k_periods": tsa_params["k_periods"],
        "period_duration": tsa_params["period_duration"],
        "preserve_column_means": tsa_params["preserve_column_means"],
        "cluster_method": tsa_params["cluster"]["method"],
        "representation_method": tsa_params["representation"]["method"],
        "soc_proxy_enabled": proxy_use["enabled"],
        "lambda_soc": proxy_use["lambda_soc"],
        "proxy_field_name": proxy_use["proxy_field_name"],
        "proxy_decomposition_method": decomposition["method"],
        "proxy_time_horizon_hours": decomposition["time_horizon_hours"],
        "margin_mode": soc_proxy_params["margin_mode"],
        "margin_value": float(proxy_result.margin),
        "termination_condition": _termination_condition(model),
        "objective": _objective(model),
        "runtime_proxy_seconds": runtime_proxy_seconds,
        "runtime_tsa_seconds": runtime_tsa_seconds,
        "runtime_calliope_seconds": runtime_calliope_seconds,
        "runtime_method_seconds": runtime_method_seconds,
        "reference_runtime_calliope_seconds": (reference_runtime_calliope_seconds),
    }

    return pd.DataFrame([row])


def _horizon_years(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    """Return horizon duration in calendar years where possible."""
    if end <= start:
        raise ValueError("end_date must be later than start_date.")

    whole_years = end.year - start.year

    if start + pd.DateOffset(years=whole_years) == end:
        return float(whole_years)

    # Fallback for horizons which are not exact calendar-year multiples.
    return float((end - start) / pd.Timedelta(days=365.2425))


def _termination_condition(model) -> str | None:
    """Return the Calliope solver termination condition."""
    runtime = getattr(model, "runtime", None)

    if runtime is None:
        return None

    value = getattr(
        runtime,
        "termination_condition",
        None,
    )

    if value is None:
        return None

    return str(getattr(value, "value", value))


def _objective(model) -> float:
    variable = "min_cost_optimisation"

    if variable not in model.results:
        raise RuntimeError(f"Calliope result {variable!r} is unavailable.")

    result = model.results[variable]

    if result.size != 1:
        raise RuntimeError(f"{variable!r} did not resolve to one scalar.")

    return float(result.item())
