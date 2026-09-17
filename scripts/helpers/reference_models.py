"""Helpers for locating and loading full-chronology reference models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import calliope
import pandas as pd


DEFAULT_REFERENCE_DIR = Path(
    "resources/calliope_models/reference"
)


def resolve_reference_model_path(
    config: dict[str, Any],
    *,
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
) -> Path:
    """Find the reference model corresponding to one experiment.

    Existing reference models use calendar-year horizons and filenames of the
    form::

        standard_<start_year>_<end_year>_reference_<country>.nc

    Experiment horizons use a half-open ``[start_date, end_date)`` convention.

    Raises
    ------
    ValueError
        If the requested horizon is not aligned to complete calendar years.
        Non-calendar horizons (e.g. April-March) require newly generated
        reference models.
    FileNotFoundError
        If the expected reference model does not exist.
    """
    data_params = config["data_params"]

    start = pd.Timestamp(
        data_params["start_date"]
    )
    end = pd.Timestamp(
        data_params["end_date"]
    )

    country = data_params["country"]

    if end <= start:
        raise ValueError(
            "end_date must be later than start_date."
        )

    # Existing references represent complete January-December years.
    if not (
        start.month == 1
        and start.day == 1
        and end.month == 1
        and end.day == 1
    ):
        raise ValueError(
            "No historical calendar-year reference can be inferred for "
            f"[{start.date()}, {end.date()}). "
            "A matching reference model must be generated for this horizon."
        )

    start_year = start.year
    end_year = end.year - 1

    filename = (
        f"standard_{start_year}_{end_year}"
        f"_reference_{country}.nc"
    )

    path = Path(reference_dir) / filename

    if not path.is_file():
        raise FileNotFoundError(
            "Expected reference model does not exist: "
            f"{path}"
        )

    return path


def load_reference_model(
    config: dict[str, Any],
    *,
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
) -> calliope.Model:
    """Load the reference model corresponding to one experiment."""
    path = resolve_reference_model_path(
        config,
        reference_dir=reference_dir,
    )

    return calliope.read_netcdf(path)
