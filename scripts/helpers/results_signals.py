"""Extract and reconstruct chronological result signals.

This module provides a single canonical interface for timeseries-like outputs
used in SoC Proxy diagnostics and metrics.

In particular, storage state of charge is returned as a full chronological
hourly series regardless of whether the source Calliope model was solved with
the full chronology or with representative-day clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Protocol

import xarray as xr


class ModelData(Protocol):
    """Solved model data required by signal extraction helpers."""

    inputs: xr.Dataset
    results: xr.Dataset


def extract_storage_soc(
    model: ModelData,
    *,
    node: str = "netherlands",
    tech: str = "h2_salt_cavern",
    cluster_map: pd.Series | None = None,
) -> pd.Series:
    """Extract the physical storage state-of-charge chronology.

    For an unclustered Calliope model, ``storage`` already represents the
    physical chronological storage level.

    For a clustered model using Calliope's ``storage_inter_cluster`` math,
    physical storage is reconstructed from the inter-cluster storage level
    and the within-representative-day storage trajectory.

    Parameters
    ----------
    model
        Solved Calliope model.
    node
        Node containing the storage technology.
    tech
        Storage technology to extract.
    cluster_map
        Mapping from each original date to its representative date. Required
        for clustered models.

    Returns
    -------
    pandas.Series
        Chronological state of charge indexed by timestep.
    """
    _require_result(model, "storage")

    if "storage_inter_cluster" in model.results:
        if cluster_map is None:
            raise ValueError(
                "cluster_map is required to reconstruct SoC from a "
                "clustered Calliope model."
            )

        return _reconstruct_clustered_storage_soc(
            model,
            cluster_map=cluster_map,
            node=node,
            tech=tech,
        )

    return _extract_reference_storage_soc(
        model,
        node=node,
        tech=tech,
    )


def extract_proxy_signal(
    proxy: pd.DataFrame,
    *,
    field: str = "soc_proxy_LDES",
) -> pd.Series:
    """Extract one chronological signal from SoC Proxy output."""
    if not isinstance(proxy, pd.DataFrame):
        raise TypeError("proxy must be a pandas DataFrame.")

    if not isinstance(proxy.index, pd.DatetimeIndex):
        raise TypeError("proxy must use a DatetimeIndex.")

    if field not in proxy.columns:
        raise ValueError(
            f"Proxy field {field!r} is not available. "
            f"Available fields are: {list(proxy.columns)}"
        )

    signal = proxy[field].copy()

    if signal.isna().any():
        raise ValueError(f"Proxy field {field!r} contains missing values.")

    signal.name = field

    return signal


def _extract_reference_storage_soc(
    model: ModelData,
    *,
    node: str,
    tech: str,
) -> pd.Series:
    """Extract storage directly from an unclustered model."""
    storage = _select_result(
        model,
        "storage",
        node=node,
        tech=tech,
    )

    if set(storage.dims) != {"timesteps"}:
        raise RuntimeError(
            "Expected unclustered storage to be indexed only by "
            f"'timesteps', received dimensions {storage.dims}."
        )

    soc = storage.to_series().astype(float)

    soc.index = pd.DatetimeIndex(
        soc.index,
        name="timesteps",
    )

    soc = soc.sort_index()
    soc.name = "storage_soc"

    _validate_soc_bounds(
        model,
        soc,
        node=node,
        tech=tech,
    )

    return soc


def _reconstruct_clustered_storage_soc(
    model: ModelData,
    *,
    cluster_map: pd.Series,
    node: str,
    tech: str,
) -> pd.Series:
    """Reconstruct full chronology from Calliope clustered storage results.

    With zero hourly storage loss, physical storage at any timestep is:

        inter-cluster storage for the original day
        + within-cluster storage for its representative day

    The current SoC Proxy Calliope models explicitly set ``storage_loss: 0``.
    Non-zero storage loss is therefore rejected rather than approximated.
    """
    _validate_cluster_map(cluster_map)

    storage_loss = _get_input_scalar(
        model,
        "storage_loss",
        node=node,
        tech=tech,
    )

    if not np.isclose(
        storage_loss,
        0.0,
        atol=1e-12,
    ):
        raise NotImplementedError(
            "Chronological reconstruction with non-zero storage_loss is "
            "not yet implemented. The current SoC Proxy models use "
            "storage_loss=0."
        )

    intra = _select_result(
        model,
        "storage",
        node=node,
        tech=tech,
    )

    inter = _select_result(
        model,
        "storage_inter_cluster",
        node=node,
        tech=tech,
    )

    if set(intra.dims) != {"timesteps"}:
        raise RuntimeError(
            "Expected clustered intra-day storage to be indexed by "
            f"'timesteps', received dimensions {intra.dims}."
        )

    if set(inter.dims) != {"datesteps"}:
        raise RuntimeError(
            "Expected inter-cluster storage to be indexed by "
            f"'datesteps', received dimensions {inter.dims}."
        )

    intra_series = intra.to_series().astype(float)
    intra_series.index = pd.DatetimeIndex(
        intra_series.index,
        name="timesteps",
    )

    inter_series = inter.to_series().astype(float)
    inter_series.index = pd.DatetimeIndex(
        inter_series.index,
        name="datesteps",
    ).normalize()

    cluster_map = cluster_map.copy()
    cluster_map.index = pd.DatetimeIndex(cluster_map.index).normalize()

    cluster_map = pd.Series(
        pd.to_datetime(cluster_map.to_numpy()).normalize(),
        index=cluster_map.index,
        name=cluster_map.name,
    )

    if not inter_series.index.equals(cluster_map.index):
        missing = cluster_map.index.difference(inter_series.index)
        unexpected = inter_series.index.difference(cluster_map.index)

        raise RuntimeError(
            "Calliope inter-cluster storage dates do not match the "
            "original TSA dates. "
            f"Missing: {missing[:5].tolist()}; "
            f"unexpected: {unexpected[:5].tolist()}."
        )

    reconstructed_days: list[pd.Series] = []

    for original_date, representative_date in cluster_map.items():
        representative_mask = intra_series.index.normalize() == representative_date

        representative_profile = intra_series.loc[representative_mask]

        if representative_profile.empty:
            raise RuntimeError(
                "No intra-cluster storage profile exists for "
                f"representative date {representative_date.date()}."
            )

        # The current experiments use hourly representative days.
        if len(representative_profile) != 24:
            raise RuntimeError(
                "Expected 24 hourly timesteps per representative day, "
                f"received {len(representative_profile)} for "
                f"{representative_date.date()}."
            )

        offsets = (
            representative_profile.index - representative_profile.index.normalize()
        )

        chronological_index = original_date + offsets

        physical_soc = pd.Series(
            inter_series.loc[original_date] + representative_profile.to_numpy(),
            index=pd.DatetimeIndex(
                chronological_index,
                name="timesteps",
            ),
        )

        reconstructed_days.append(physical_soc)

    soc = pd.concat(reconstructed_days).sort_index()

    if soc.index.has_duplicates:
        raise RuntimeError("Reconstructed clustered SoC contains duplicate timesteps.")

    expected_timesteps = len(cluster_map) * 24

    if len(soc) != expected_timesteps:
        raise RuntimeError(
            "Reconstructed clustered SoC has an unexpected length. "
            f"Expected {expected_timesteps}, received {len(soc)}."
        )

    soc.name = "storage_soc"

    _validate_soc_bounds(
        model,
        soc,
        node=node,
        tech=tech,
    )

    return soc


def _select_result(
    model: ModelData,
    variable: str,
    *,
    node: str,
    tech: str,
):
    """Select one node/technology from a Calliope result array."""
    _require_result(
        model,
        variable,
    )

    data = model.results[variable]

    selectors = {}

    if "nodes" in data.dims:
        selectors["nodes"] = node

    if "techs" in data.dims:
        selectors["techs"] = tech

    try:
        selected = data.sel(selectors)
    except KeyError as error:
        raise ValueError(
            f"Could not select node={node!r}, tech={tech!r} "
            f"from Calliope result {variable!r}."
        ) from error

    return selected.squeeze(drop=True)


def _get_input_scalar(
    model: ModelData,
    variable: str,
    *,
    node: str,
    tech: str,
) -> float:
    """Extract one scalar model input."""
    if variable not in model.inputs:
        raise RuntimeError(f"Calliope input {variable!r} is not available.")

    data = model.inputs[variable]

    selectors = {}

    if "nodes" in data.dims:
        selectors["nodes"] = node

    if "techs" in data.dims:
        selectors["techs"] = tech

    try:
        selected = data.sel(selectors).squeeze(drop=True)
    except KeyError as error:
        raise ValueError(
            f"Could not select node={node!r}, tech={tech!r} "
            f"from Calliope input {variable!r}."
        ) from error

    if selected.size != 1:
        raise RuntimeError(
            f"Expected Calliope input {variable!r} to resolve to one "
            f"value, received shape {selected.shape}."
        )

    return float(selected.item())


def _validate_cluster_map(
    cluster_map: pd.Series,
) -> None:
    """Validate assumptions required for chronological reconstruction."""
    if not isinstance(
        cluster_map,
        pd.Series,
    ):
        raise TypeError("cluster_map must be a pandas Series.")

    if cluster_map.empty:
        raise ValueError("cluster_map cannot be empty.")

    if not isinstance(
        cluster_map.index,
        pd.DatetimeIndex,
    ):
        raise TypeError("cluster_map must use a DatetimeIndex.")

    if cluster_map.index.has_duplicates:
        raise ValueError("cluster_map dates must be unique.")

    if cluster_map.isna().any():
        raise ValueError("cluster_map contains missing representative dates.")


def _validate_soc_bounds(
    model: ModelData,
    soc: pd.Series,
    *,
    node: str,
    tech: str,
) -> None:
    """Check reconstructed physical SoC against installed storage capacity."""
    if "storage_cap" not in model.results:
        return

    storage_cap = _select_result(
        model,
        "storage_cap",
        node=node,
        tech=tech,
    )

    if storage_cap.size != 1:
        raise RuntimeError("Expected storage_cap to resolve to one scalar.")

    capacity = float(storage_cap.item())

    tolerance = (
        max(
            1.0,
            abs(capacity),
        )
        * 1e-8
    )

    minimum = float(soc.min())
    maximum = float(soc.max())

    if minimum < -tolerance:
        raise RuntimeError(
            f"Reconstructed storage SoC falls below zero. Minimum={minimum:.6g}."
        )

    if maximum > capacity + tolerance:
        raise RuntimeError(
            "Reconstructed storage SoC exceeds installed storage "
            f"capacity. Maximum={maximum:.6g}, "
            f"capacity={capacity:.6g}."
        )


def _require_result(
    model: ModelData,
    variable: str,
) -> None:
    """Require a solved model result variable."""
    if model.results is None:
        raise RuntimeError("Calliope model has no solved results.")

    if variable not in model.results:
        raise RuntimeError(f"Calliope result {variable!r} is not available.")
