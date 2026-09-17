"""Extract additive technology costs from solved Calliope models."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
import xarray as xr


class ModelData(Protocol):
    results: xr.Dataset


def extract_costs(
    model: ModelData,
    *,
    case_id: str,
    model_type: str,
) -> pd.DataFrame:
    """Extract CAPEX and OPEX by technology in long format.

    The returned rows are additive: summing ``value`` for one
    ``case_id``/``model_type`` yields total system cost.
    """
    if model_type not in {"reference", "clustered"}:
        raise ValueError("model_type must be 'reference' or 'clustered'.")

    required = {
        "cost_investment_annualised",
        "cost_operation_fixed",
        "cost_operation_variable",
        "cost",
    }

    missing = required.difference(model.results.data_vars)

    if missing:
        raise RuntimeError(
            f"Required Calliope cost results are unavailable: {sorted(missing)}"
        )

    capex = _monetary(model.results["cost_investment_annualised"])

    fixed_opex = _monetary(model.results["cost_operation_fixed"])

    variable = _monetary(model.results["cost_operation_variable"])

    variable_present = variable.notnull().any(dim="timesteps")

    variable_opex = variable.fillna(0).sum(dim="timesteps").where(variable_present)

    fixed_opex, variable_opex = xr.align(
        fixed_opex,
        variable_opex,
        join="outer",
    )

    opex_present = fixed_opex.notnull() | variable_opex.notnull()

    opex = (fixed_opex.fillna(0) + variable_opex.fillna(0)).where(opex_present)

    capex_frame = _cost_frame(
        capex,
        case_id=case_id,
        model_type=model_type,
        cost_class="capex",
    )

    opex_frame = _cost_frame(
        opex,
        case_id=case_id,
        model_type=model_type,
        cost_class="opex",
    )

    costs = pd.concat(
        [capex_frame, opex_frame],
        ignore_index=True,
    )

    _validate_cost_total(
        model,
        costs,
    )

    return costs


def _monetary(
    data: xr.DataArray,
) -> xr.DataArray:
    """Select the single monetary cost class."""
    if "costs" not in data.dims:
        return data

    available = [str(value) for value in data["costs"].values]

    if available != ["monetary"]:
        raise RuntimeError(
            "This extraction currently assumes the model has exactly "
            f"one monetary cost class. Found: {available}"
        )

    return data.sel(costs="monetary", drop=True)


def _cost_frame(
    data: xr.DataArray,
    *,
    case_id: str,
    model_type: str,
    cost_class: str,
) -> pd.DataFrame:
    frame = data.to_series().dropna().rename("value").reset_index()

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "model_type",
                "node",
                "tech",
                "cost_class",
                "value",
            ]
        )

    frame["case_id"] = case_id
    frame["model_type"] = model_type
    frame["cost_class"] = cost_class

    return frame[
        [
            "case_id",
            "model_type",
            "nodes",
            "techs",
            "cost_class",
            "value",
        ]
    ].rename(
        columns={
            "nodes": "node",
            "techs": "tech",
        }
    )


def _validate_cost_total(
    model: ModelData,
    costs: pd.DataFrame,
) -> None:
    """Check that extracted rows reproduce Calliope total cost."""
    extracted_total = float(costs["value"].sum())

    calliope_cost = _monetary(model.results["cost"])

    calliope_total = float(calliope_cost.sum(skipna=True).item())

    if not np.isclose(
        extracted_total,
        calliope_total,
        rtol=1e-9,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Extracted CAPEX/OPEX does not reproduce Calliope's "
            "technology cost total. "
            f"Extracted={extracted_total:.12g}, "
            f"Calliope={calliope_total:.12g}."
        )

    # In this model, monetary cost has weight 1 and there should be no
    # unmet-demand penalty, so total technology cost should also equal
    # the optimisation objective.
    if "min_cost_optimisation" in model.results:
        objective = float(model.results["min_cost_optimisation"].item())

        if not np.isclose(
            calliope_total,
            objective,
            rtol=1e-9,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Calliope technology costs do not reproduce the "
                "optimisation objective. This may indicate an unexpected "
                "objective penalty or cost term. "
                f"Costs={calliope_total:.12g}, "
                f"objective={objective:.12g}."
            )
