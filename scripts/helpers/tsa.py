"""Helpers for temporal aggregation in the SoC Proxy experiments.

This module provides the experiment-facing interface to TSAM.

Experiment configuration is defined in YAML and resolved before reaching this
module. The helpers here translate those resolved parameters into native TSAM
configuration objects and provide the additional functionality needed to pass
TSAM results into Calliope.

Responsibilities
----------------
* build native TSAM clustering and representation configuration;
* construct SoC-proxy clustering weights;
* run temporal aggregation;
* apply an existing clustering to the full Calliope input timeseries;
* generate a Calliope-compatible representative-day mapping; and
* provide the reconstructed timeseries corresponding to the selected TSAM
  representation method.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import pandas as pd
import tsam
from tsam import (
    AggregationResult,
    ClusterConfig,
    Distribution,
    ExtremeConfig,
    KMeans,
    KMedoids,
    MinMaxMean,
    SegmentConfig,
)


_SIMPLE_CLUSTER_METHODS = {
    "hierarchical",
    "kmaxoids",
    "averaging",
    "contiguous",
}

_SIMPLE_REPRESENTATIONS = {
    "mean",
    "medoid",
    "maxoid",
}


def build_cluster_config(
    tsa_params: dict[str, Any],
    solver_params: dict[str, Any] | None = None,
) -> ClusterConfig:
    """Build a native TSAM clustering configuration.

    Parameters
    ----------
    tsa_params
        Resolved ``tsa_params`` section from the experiment configuration.
    solver_params
        Resolved solver configuration. The selected solver is used by
        k-medoids, which requires a MILP solver.

    Returns
    -------
    tsam.ClusterConfig
        Native TSAM clustering configuration.

    Notes
    -----
    Parameters that only apply to a particular clustering method are ignored
    by the others. For example, ``random_state`` is consumed only when
    ``method: kmeans``.
    """
    cluster_params = tsa_params["cluster"]
    method_name = cluster_params["method"].lower()

    if method_name == "kmeans":
        method = KMeans(
            n_init=cluster_params.get("n_init"),
            random_state=cluster_params.get("random_state"),
        )

    elif method_name == "kmedoids":
        solver = (solver_params or {}).get("solver", "highs")

        method = KMedoids(
            solver=solver,
            options=cluster_params.get("solver_options", {}),
        )

    elif method_name in _SIMPLE_CLUSTER_METHODS:
        method = method_name

    else:
        raise ValueError(
            f"Unsupported TSAM clustering method: {method_name!r}"
        )

    representation = _build_representation(
        tsa_params.get("representation")
    )

    return ClusterConfig(
        method=method,
        representation=representation,
        scale_by_column_means=cluster_params.get(
            "scale_by_column_means",
            False,
        ),
        use_duration_curves=cluster_params.get(
            "use_duration_curves",
            False,
        ),
        include_period_sums=cluster_params.get(
            "include_period_sums",
            False,
        ),
    )


def build_weights(
    columns: Collection[str],
    *,
    proxy_column: str | None = None,
    proxy_weight: float | None = None,
) -> dict[str, float] | None:
    """Build TSAM column weights for an SoC Proxy experiment.

    When no proxy column is supplied, ``None`` is returned and TSAM applies
    equal weighting to all input features.

    When a proxy column is supplied, ``proxy_weight`` is interpreted as the
    fraction of total clustering weight assigned to that proxy. The remaining
    weight is distributed equally across all other clustering features.

    Parameters
    ----------
    columns
        Columns supplied to TSAM for clustering.
    proxy_column
        Column containing the SoC-proxy feature. ``None`` indicates that the
        proxy is not used for clustering.
    proxy_weight
        Fraction of total clustering weight assigned to the proxy feature.
        Must lie strictly between 0 and 1 when both proxy and ordinary
        features are present.

    Returns
    -------
    dict[str, float] | None
        Per-column clustering weights, or ``None`` when no custom weighting
        is required.

    Notes
    -----
    TSAM enforces a small positive minimum column weight. Therefore an exact
    proxy weight of 0 or 1 should be represented by changing the clustering
    feature set instead of assigning zero weight:

    * lambda = 0: omit the proxy column;
    * lambda = 1: cluster on the proxy column alone.
    """
    columns = list(columns)

    if not columns:
        raise ValueError("At least one clustering column is required.")

    if len(columns) != len(set(columns)):
        raise ValueError("Clustering columns must be unique.")

    if proxy_column is None:
        return None

    if proxy_column not in columns:
        raise ValueError(
            f"Proxy column {proxy_column!r} is not present in clustering data."
        )

    if proxy_weight is None:
        raise ValueError(
            "proxy_weight must be provided when a proxy column is used."
        )

    if not 0 <= proxy_weight <= 1:
        raise ValueError("proxy_weight must lie between 0 and 1.")

    base_columns = [
        column
        for column in columns
        if column != proxy_column
    ]

    if not base_columns:
        if proxy_weight != 1:
            raise ValueError(
                "proxy_weight must be 1 when clustering only on the proxy."
            )
        return None

    if proxy_weight in {0, 1}:
        raise ValueError(
            "An exact proxy_weight of 0 or 1 would assign zero weight to "
            "some clustering features. For proxy_weight=0, omit the proxy "
            "column. For proxy_weight=1, cluster on the proxy column alone."
        )

    base_weight = (1 - proxy_weight) / len(base_columns)

    return {
        column: (
            proxy_weight
            if column == proxy_column
            else base_weight
        )
        for column in columns
    }


def run_tsa(
    data: pd.DataFrame,
    *,
    n_clusters: int,
    cluster: ClusterConfig,
    weights: dict[str, float] | None = None,
    period_duration: int | float | str = "1D",
    segments: SegmentConfig | None = None,
    extremes: ExtremeConfig | None = None,
    preserve_column_means: bool = True,
    rescale_exclude_columns: list[str] | None = None,
) -> AggregationResult:
    """Run a TSAM aggregation.

    This function remains intentionally thin. Clustering and representation
    behaviour is defined through native TSAM configuration objects.
    """
    _validate_timeseries(data)

    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")

    return tsam.aggregate(
        data,
        n_clusters=n_clusters,
        period_duration=period_duration,
        cluster=cluster,
        segments=segments,
        extremes=extremes,
        weights=weights,
        preserve_column_means=preserve_column_means,
        rescale_exclude_columns=rescale_exclude_columns,
    )


def apply_tsa_to_timeseries(
    tsa_result: AggregationResult,
    timeseries: pd.DataFrame,
) -> AggregationResult:
    """Apply an existing TSAM clustering to another timeseries dataset.

    This allows clustering to be fitted on the experiment features, including
    the SoC proxy, and then transferred to the complete set of input
    timeseries required by Calliope.

    The new dataset must describe the same time horizon and temporal index as
    the data used to fit the clustering.
    """
    _validate_timeseries(timeseries)

    if len(timeseries) != len(tsa_result.original):
        raise ValueError(
            "The target timeseries must have the same number of timesteps "
            "as the data used to fit the TSAM clustering."
        )

    if not timeseries.index.equals(tsa_result.original.index):
        raise ValueError(
            "The target timeseries index must exactly match the index used "
            "to fit the TSAM clustering."
        )

    return tsa_result.clustering.apply(timeseries)


def build_calliope_cluster_map(
    tsa_result: AggregationResult,
    *,
    name: str = "cluster_days",
) -> pd.Series:
    """Create a Calliope-compatible representative-day mapping.

    Calliope requires every original date to map to a representative date
    that exists in its input timeseries.

    TSAM's reconstructed timeseries places each cluster's representative
    profile at every original period assigned to that cluster. We therefore
    select the first period assigned to each cluster as a deterministic
    Calliope anchor date.

    The anchor is only a label for the representative profile. For synthetic
    representations such as ``mean`` or ``distribution``, it does not imply
    that the representative profile historically occurred on that date.
    """
    assignments = tsa_result.assignments

    required_columns = {"period_idx", "cluster_idx"}
    missing = required_columns - set(assignments.columns)

    if missing:
        raise ValueError(
            "TSAM assignments are missing required columns: "
            f"{sorted(missing)}"
        )

    periods = (
        assignments
        .reset_index(names="timestamp")
        .groupby("period_idx", sort=True, as_index=False)
        .first()
    )

    periods["date"] = (
        pd.to_datetime(periods["timestamp"])
        .dt.normalize()
    )

    if periods["date"].duplicated().any():
        raise ValueError(
            "More than one TSAM period begins on the same date. "
            "The current Calliope adapter expects daily typical periods."
        )

    representative_dates = (
        periods
        .groupby("cluster_idx", sort=True)["date"]
        .first()
    )

    cluster_map = periods["cluster_idx"].map(
        representative_dates
    )

    return pd.Series(
        cluster_map.to_numpy(),
        index=pd.DatetimeIndex(periods["date"]),
        name=name,
    )


def prepare_calliope_inputs(
    tsa_result: AggregationResult,
    calliope_timeseries: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Prepare temporal inputs for a clustered Calliope model.

    The fitted TSAM clustering is applied to the complete Calliope input
    timeseries. TSAM therefore applies the selected representation method to
    all model inputs consistently.

    Returns
    -------
    cluster_map
        Daily mapping from original dates to representative dates.
    reconstructed_timeseries
        Full-length timeseries in which each original period has been replaced
        by its TSAM representative profile.
    """
    calliope_result = apply_tsa_to_timeseries(
        tsa_result,
        calliope_timeseries,
    )

    cluster_map = build_calliope_cluster_map(
        calliope_result
    )

    return (
        cluster_map,
        calliope_result.reconstructed.copy(),
    )


def _build_representation(
    representation_params: dict[str, Any] | None,
) -> Any:
    """Build a native TSAM representation from experiment parameters."""
    if not representation_params:
        return None

    method = representation_params.get("method")

    if method is None:
        return None

    method = method.lower()

    if method in _SIMPLE_REPRESENTATIONS:
        return method

    if method == "distribution":
        options = {
            key: value
            for key, value in representation_params.items()
            if key != "method"
        }

        if not options:
            return "distribution"

        return Distribution(**options)

    if method == "distribution_minmax":
        options = {
            key: value
            for key, value in representation_params.items()
            if key != "method"
        }

        if not options:
            return "distribution_minmax"

        return Distribution(
            preserve_minmax=True,
            **options,
        )

    if method == "minmax_mean":
        max_columns = representation_params.get(
            "max_columns",
            [],
        )
        min_columns = representation_params.get(
            "min_columns",
            [],
        )

        if not max_columns and not min_columns:
            return "minmax_mean"

        return MinMaxMean(
            max_columns=max_columns,
            min_columns=min_columns,
        )

    raise ValueError(
        f"Unsupported TSAM representation method: {method!r}"
    )


def _validate_timeseries(data: pd.DataFrame) -> None:
    """Validate assumptions made by the TSA experiment helpers."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "TSAM input data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "TSAM input data cannot be empty."
        )

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(
            "TSAM input data must use a DatetimeIndex."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "TSAM input timestamps must be monotonically increasing."
        )

    if data.index.has_duplicates:
        raise ValueError(
            "TSAM input timestamps must be unique."
        )

    if data.columns.has_duplicates:
        raise ValueError(
            "TSAM input columns must be unique."
        )