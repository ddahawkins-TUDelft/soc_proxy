"""Extract installed technology capacities from solved models."""

from __future__ import annotations

from typing import Protocol

import pandas as pd
import xarray as xr


class ModelData(Protocol):
    inputs: xr.Dataset
    results: xr.Dataset


def extract_capacities(
    model: ModelData,
    *,
    case_id: str,
    model_type: str,
) -> pd.DataFrame:
    """Extract output flow and storage capacities in long format."""
    if model_type not in {"reference", "clustered"}:
        raise ValueError("model_type must be 'reference' or 'clustered'.")

    rows: list[pd.DataFrame] = []

    if "flow_cap" in model.results:
        rows.append(
            _extract_output_flow_cap(
                model,
                case_id=case_id,
                model_type=model_type,
            )
        )

    if "storage_cap" in model.results:
        rows.append(
            _extract_storage_cap(
                model,
                case_id=case_id,
                model_type=model_type,
            )
        )

    rows = [frame for frame in rows if not frame.empty]

    if not rows:
        return _empty_capacity_frame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


def _extract_output_flow_cap(
    model: ModelData,
    *,
    case_id: str,
    model_type: str,
) -> pd.DataFrame:
    """Extract flow capacity for output carriers only."""
    if "carrier_out" not in model.inputs:
        raise RuntimeError("Calliope input 'carrier_out' is unavailable.")

    flow_cap = model.results["flow_cap"]

    carrier_out = model.inputs["carrier_out"].fillna(False).astype(bool)

    # xarray aligns dimensions automatically. This removes input-side
    # capacity entries for conversion technologies.
    output_flow_cap = flow_cap.where(carrier_out)

    frame = output_flow_cap.to_series().dropna().rename("value").reset_index()

    if frame.empty:
        return _empty_capacity_frame()

    frame["case_id"] = case_id
    frame["model_type"] = model_type
    frame["capacity_type"] = "flow_cap"

    return frame[
        [
            "case_id",
            "model_type",
            "nodes",
            "techs",
            "capacity_type",
            "carriers",
            "value",
        ]
    ].rename(
        columns={
            "nodes": "node",
            "techs": "tech",
            "carriers": "carrier",
        }
    )


def _extract_storage_cap(
    model: ModelData,
    *,
    case_id: str,
    model_type: str,
) -> pd.DataFrame:
    """Extract storage energy capacity and attach its output carrier."""
    storage = (
        model.results["storage_cap"].to_series().dropna().rename("value").reset_index()
    )

    if storage.empty:
        return _empty_capacity_frame()

    records = []

    for row in storage.itertuples(index=False):
        carriers = _output_carriers(
            model,
            node=row.nodes,
            tech=row.techs,
        )

        if len(carriers) != 1:
            raise RuntimeError(
                "Storage capacity reporting assumes exactly one output "
                f"carrier per technology. {row.techs!r} at {row.nodes!r} "
                f"has output carriers {carriers}."
            )

        records.append(
            {
                "case_id": case_id,
                "model_type": model_type,
                "node": row.nodes,
                "tech": row.techs,
                "capacity_type": "storage_cap",
                "carrier": carriers[0],
                "value": float(row.value),
            }
        )

    return pd.DataFrame.from_records(records)


def _output_carriers(
    model: ModelData,
    *,
    node: str,
    tech: str,
) -> list[str]:
    """Return the output carrier(s) for one node/technology."""
    lookup = model.inputs["carrier_out"]

    selectors = {}

    if "nodes" in lookup.dims:
        selectors["nodes"] = node

    if "techs" in lookup.dims:
        selectors["techs"] = tech

    selected = lookup.sel(selectors)

    if "carriers" not in selected.dims:
        raise RuntimeError("'carrier_out' does not contain a carriers dimension.")

    selected = selected.fillna(False).astype(bool)

    return [
        str(carrier)
        for carrier in selected["carriers"].values
        if bool(selected.sel(carriers=carrier).item())
    ]


def _empty_capacity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "case_id",
            "model_type",
            "node",
            "tech",
            "capacity_type",
            "carrier",
            "value",
        ]
    )
