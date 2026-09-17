"""Helpers for locating and loading full-chronology reference models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr


DEFAULT_REFERENCE_DIR = Path("resources/calliope_models/reference")


@dataclass(frozen=True)
class ReferenceModel:
    """Numerical data loaded from a historical Calliope reference model.

    Historical reference NetCDF files may have been written by older
    Calliope 0.7 development versions whose serialized model definitions are
    no longer accepted by the current schema.

    For analysis we only require the stored numerical inputs and results, so
    these are loaded directly from their NetCDF groups without asking Calliope
    to reconstruct the original model definition.
    """

    path: Path
    inputs: xr.Dataset
    results: xr.Dataset


def resolve_reference_model_path(
    config: dict[str, Any],
    *,
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
) -> Path:
    """Find the calendar-year reference corresponding to one experiment."""
    data_params = config["data_params"]

    start = pd.Timestamp(data_params["start_date"])
    end = pd.Timestamp(data_params["end_date"])

    country = data_params["country"]

    if end <= start:
        raise ValueError("end_date must be later than start_date.")

    if not (start.month == 1 and start.day == 1 and end.month == 1 and end.day == 1):
        raise ValueError(
            "Existing reference models represent complete calendar-year "
            "horizons. No matching historical reference can be inferred for "
            f"[{start.date()}, {end.date()})."
        )

    start_year = start.year
    end_year = end.year - 1

    filename = f"standard_{start_year}_{end_year}_reference_{country}.nc"

    path = Path(reference_dir) / filename

    if not path.is_file():
        raise FileNotFoundError(f"Expected reference model does not exist: {path}")

    return path


def load_reference_model(
    config: dict[str, Any],
    *,
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
) -> ReferenceModel:
    """Load stored numerical data from the matching reference model."""
    path = resolve_reference_model_path(
        config,
        reference_dir=reference_dir,
    )

    try:
        with xr.open_dataset(
            path,
            group="inputs",
        ) as dataset:
            inputs = dataset.load()

        with xr.open_dataset(
            path,
            group="results",
        ) as dataset:
            results = dataset.load()

    except Exception as error:
        raise RuntimeError(
            f"Could not load numerical data from reference model {path}."
        ) from error

    if not results.data_vars:
        raise RuntimeError(f"Reference model contains no solved results: {path}")

    return ReferenceModel(
        path=path,
        inputs=inputs,
        results=results,
    )
