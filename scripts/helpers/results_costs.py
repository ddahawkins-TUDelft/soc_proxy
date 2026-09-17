"""Extract additive technology costs from solved Calliope models."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
import xarray as xr


class ModelData(Protocol):
    """Solved model data required for cost extraction."""

    results: xr.Dataset


COST_COLUMNS = [
    "case_id",
    "model_type",
    "node",
    "tech",
    "cost_class",
    "value",
    "unannualised_value",
]


def extract_costs(
    model: ModelData,
    *,
    case_id: str,
    model_type: str,
) -> pd.DataFrame:
    """Extract CAPEX and OPEX by technology in long format.

    ``value`` contains the additive cost values used to reproduce Calliope's
    total model cost:

    - CAPEX: annualised investment cost
    - OPEX: fixed + variable operating cost

    ``unannualised_value`` contains the underlying investment cost for CAPEX
    rows and is missing for OPEX rows.

    Therefore:

        costs["value"].sum()

    reproduces Calliope's total technology cost, while raw investment remains
    available separately for alternative weighting and reporting.
    """
    _validate_model_type(model_type)

    required = {
        "cost",
        "cost_investment",
        "cost_investment_annualised",
        "cost_operation_fixed",
        "cost_operation_variable",
    }

    missing = required.difference(model.results.data_vars)

    if missing:
        raise RuntimeError(
            "Required Calliope cost results are unavailable: "
            f"{sorted(missing)}"
        )

    annualised_capex = _monetary(
        model.results["cost_investment_annualised"]
    )

    unannualised_capex = _monetary(
        model.results["cost_investment"]
    )

    fixed_opex = _monetary(
        model.results["cost_operation_fixed"]
    )

    variable_opex = _aggregate_variable_opex(
        _monetary(
            model.results["cost_operation_variable"]
        )
    )

    capex = _capex_frame(
        annualised_capex,
        unannualised_capex,
        case_id=case_id,
        model_type=model_type,
    )

    opex = _opex_frame(
        fixed_opex,
        variable_opex,
        case_id=case_id,
        model_type=model_type,
    )

    costs = pd.concat(
        [capex, opex],
        ignore_index=True,
    )

    _validate_cost_total(
        model,
        costs,
    )

    return costs


def _aggregate_variable_opex(
    data: xr.DataArray,
) -> xr.DataArray:
    """Sum already-weighted variable OPEX across representative timesteps."""
    if "timesteps" not in data.dims:
        return data

    present = data.notnull().any(
        dim="timesteps"
    )

    return (
        data
        .fillna(0)
        .sum(dim="timesteps")
        .where(present)
    )


def _capex_frame(
    annualised: xr.DataArray,
    unannualised: xr.DataArray,
    *,
    case_id: str,
    model_type: str,
) -> pd.DataFrame:
    """Build CAPEX rows containing both investment representations."""
    annualised, unannualised = xr.align(
        annualised,
        unannualised,
        join="outer",
    )

    annualised_series = (
        annualised
        .to_series()
        .rename("value")
    )

    unannualised_series = (
        unannualised
        .to_series()
        .rename("unannualised_value")
    )

    frame = pd.concat(
        [
            annualised_series,
            unannualised_series,
        ],
        axis=1,
    ).reset_index()

    # Drop technologies for which neither investment representation exists.
    frame = frame.dropna(
        subset=[
            "value",
            "unannualised_value",
        ],
        how="all",
    )

    if frame.empty:
        return _empty_cost_frame()

    frame["case_id"] = case_id
    frame["model_type"] = model_type
    frame["cost_class"] = "capex"

    return _format_cost_frame(frame)


def _opex_frame(
    fixed: xr.DataArray,
    variable: xr.DataArray,
    *,
    case_id: str,
    model_type: str,
) -> pd.DataFrame:
    """Build additive OPEX rows from fixed and variable operating costs."""
    fixed, variable = xr.align(
        fixed,
        variable,
        join="outer",
    )

    present = (
        fixed.notnull()
        | variable.notnull()
    )

    total = (
        fixed.fillna(0)
        + variable.fillna(0)
    ).where(present)

    frame = (
        total
        .to_series()
        .dropna()
        .rename("value")
        .reset_index()
    )

    if frame.empty:
        return _empty_cost_frame()

    frame["case_id"] = case_id
    frame["model_type"] = model_type
    frame["cost_class"] = "opex"
    frame["unannualised_value"] = np.nan

    return _format_cost_frame(frame)


def _format_cost_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Standardise cost table column names and ordering."""
    frame = frame.rename(
        columns={
            "nodes": "node",
            "techs": "tech",
        }
    )

    return frame[COST_COLUMNS]


def _monetary(
    data: xr.DataArray,
) -> xr.DataArray:
    """Select the single monetary cost class used by this model."""
    if "costs" not in data.dims:
        return data

    available = [
        str(value)
        for value in data["costs"].values
    ]

    if available != ["monetary"]:
        raise RuntimeError(
            "Cost extraction currently assumes exactly one cost class "
            f"named 'monetary'. Found: {available}"
        )

    return data.sel(
        costs="monetary",
        drop=True,
    )


def _validate_cost_total(
    model: ModelData,
    costs: pd.DataFrame,
) -> None:
    """Check that additive extracted costs reproduce Calliope's total cost."""
    extracted_total = float(
        costs["value"].sum()
    )

    calliope_total = float(
        _monetary(
            model.results["cost"]
        )
        .sum(skipna=True)
        .item()
    )

    if not np.isclose(
        extracted_total,
        calliope_total,
        rtol=1e-9,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Extracted CAPEX/OPEX does not reproduce Calliope's total "
            "technology cost. "
            f"Extracted={extracted_total:.12g}, "
            f"Calliope={calliope_total:.12g}."
        )

    if "min_cost_optimisation" in model.results:
        objective = float(
            model.results[
                "min_cost_optimisation"
            ].item()
        )

        if not np.isclose(
            calliope_total,
            objective,
            rtol=1e-9,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Calliope technology costs do not reproduce the optimisation "
                "objective. This may indicate an additional objective term or "
                "penalty. "
                f"Costs={calliope_total:.12g}, "
                f"objective={objective:.12g}."
            )


def _validate_model_type(
    model_type: str,
) -> None:
    if model_type not in {
        "reference",
        "clustered",
    }:
        raise ValueError(
            "model_type must be 'reference' or 'clustered'."
        )


def _empty_cost_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=COST_COLUMNS
    )
