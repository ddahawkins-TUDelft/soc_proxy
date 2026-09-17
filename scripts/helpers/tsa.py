"""Helpers for applying TSAM aggregations in the SoC Proxy experiments.

This module deliberately keeps TSAM configuration in TSAM itself. Experiment
scripts should construct native TSAM objects such as ``ClusterConfig``,
``KMeans``, ``KMedoids``, ``SegmentConfig``, and ``ExtremeConfig`` and pass
them into ``run_tsa``.

The helpers here provide only the additional functionality required by the
SoC Proxy experiments:

* construct SoC-proxy feature weights;
* run a TSAM aggregation;
* apply an existing clustering to the full Calliope input timeseries;
* generate a Calliope-compatible representative-day mapping; and
* provide the reconstructed timeseries corresponding to the TSAM
  representation method.
"""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd
import tsam
from tsam import AggregationResult, ClusterConfig, ExtremeConfig, SegmentConfig


def build_weights(
    columns: Collection[str],
    proxy_columns: Collection[str],
    proxy_weight: float,
) -> dict[str, float]:
    """Build TSAM column weights for an SoC Proxy experiment.

    ``proxy_weight`` is the fraction of the total clustering weight assigned
    to the SoC-proxy features. The remaining weight is distributed equally
    over all non-proxy features.

    If multiple proxy columns are supplied, the proxy weight is distributed
    equally between them.

    Parameters
    ----------
    columns
        All columns supplied to TSAM for clustering.
    proxy_columns
        Columns containing SoC-proxy features.
    proxy_weight
        Fraction of total clustering weight assigned to proxy columns.
        Must lie between 0 and 1.

    Returns
    -------
    dict[str, float]
        Per-column weights summing to 1.

    Raises
    ------
    ValueError
        If columns are duplicated, proxy columns are absent from ``columns``,
        or the requested weighting cannot be represented.
    """
    columns = list(columns)
    proxy_columns = list(proxy_columns)

    if not columns:
        raise ValueError("At least one clustering column is required.")

    if len(columns) != len(set(columns)):
        raise ValueError("Clustering columns must be unique.")

    if len(proxy_columns) != len(set(proxy_columns)):
        raise ValueError("Proxy columns must be unique.")

    unknown_proxy_columns = set(proxy_columns) - set(columns)
    if unknown_proxy_columns:
        raise ValueError(
            "Proxy columns are not present in the clustering data: "
            f"{sorted(unknown_proxy_columns)}"
        )

    if not 0 <= proxy_weight <= 1:
        raise ValueError("proxy_weight must lie between 0 and 1.")

    base_columns = [column for column in columns if column not in proxy_columns]

    if not proxy_columns:
        if proxy_weight != 0:
            raise ValueError(
                "proxy_weight must be 0 when no proxy columns are supplied."
            )
        return {column: 1 / len(columns) for column in columns}

    if not base_columns:
        if proxy_weight != 1:
            raise ValueError(
                "proxy_weight must be 1 when only proxy columns are supplied."
            )
        return {
            column: 1 / len(proxy_columns)
            for column in proxy_columns
        }

    # TSAM enforces a small positive minimum column weight. True zero-weight
    # groups are therefore represented more cleanly by excluding those
    # columns from the clustering data altogether.
    if proxy_weight in {0, 1}:
        raise ValueError(
            "A proxy_weight of exactly 0 or 1 would assign zero weight to "
            "some columns. For proxy_weight=0, omit the proxy columns from "
            "the clustering data. For proxy_weight=1, cluster on the proxy "
            "columns alone."
        )

    base_weight = (1 - proxy_weight) / len(base_columns)
    per_proxy_weight = proxy_weight / len(proxy_columns)

    return {
        column: (
            per_proxy_weight if column in proxy_columns else base_weight
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
    """Run a TSAM aggregation using native TSAM configuration objects.

    This is intentionally a thin wrapper around :func:`tsam.aggregate`.
    Clustering algorithms and representation methods should be configured
    through TSAM's own configuration objects rather than reproduced here.
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

    This is used to cluster on the selected experiment features (including
    the SoC proxy) and then apply the resulting assignments and representation
    method to the complete set of timeseries consumed by Calliope.

    The input must cover the same time horizon and temporal structure as the
    dataset used to generate ``tsa_result``.
    """
    _validate_timeseries(timeseries)

    if len(timeseries) != len(tsa_result.original):
        raise ValueError(
            "The Calliope timeseries must have the same number of timesteps "
            "as the data used to fit the TSAM clustering."
        )

    if not timeseries.index.equals(tsa_result.original.index):
        raise ValueError(
            "The Calliope timeseries index must exactly match the index of "
            "the data used to fit the TSAM clustering."
        )

    return tsa_result.clustering.apply(timeseries)


def build_calliope_cluster_map(
    tsa_result: AggregationResult,
    *,
    name: str = "cluster_days",
) -> pd.Series:
    """Create a Calliope-compatible mapping to representative dates.

    Calliope requires every original date to map to a representative date
    that exists in the supplied timeseries.

    The reconstructed TSAM timeseries contains the representative profile at
    *every* period assigned to a cluster. Therefore, we can choose the first
    original period belonging to each cluster as a deterministic anchor date,
    irrespective of whether TSAM used an observed representation (e.g.
    medoid/maxoid) or a synthetic one (e.g. mean/distribution).

    Parameters
    ----------
    tsa_result
        Aggregation result corresponding to the timeseries that will be passed
        to Calliope.
    name
        Name assigned to the returned Series.

    Returns
    -------
    pandas.Series
        Series indexed by each original date, with values giving the
        representative date used by Calliope.
    """
    assignments = tsa_result.assignments

    required_columns = {"period_idx", "cluster_idx"}
    missing = required_columns - set(assignments.columns)
    if missing:
        raise ValueError(
            "TSAM assignments are missing required columns: "
            f"{sorted(missing)}"
        )

    # One row per original period. The index of that row is the first
    # timestamp in the period.
    periods = (
        assignments
        .reset_index(names="timestamp")
        .groupby("period_idx", sort=True, as_index=False)
        .first()
    )

    # Calliope clusters dates rather than arbitrary timestep timestamps.
    periods["date"] = pd.to_datetime(periods["timestamp"]).dt.normalize()

    if periods["date"].duplicated().any():
        raise ValueError(
            "More than one TSAM period begins on the same date. "
            "The current Calliope adapter expects daily typical periods."
        )

    # Pick one deterministic anchor period per cluster. Because Calliope is
    # given tsa_result.reconstructed, every member of a cluster already
    # contains the same representative profile.
    representative_dates = (
        periods
        .groupby("cluster_idx", sort=True)["date"]
        .first()
    )

    cluster_map = periods["cluster_idx"].map(representative_dates)

    return pd.Series(
        cluster_map.to_numpy(),
        index=pd.DatetimeIndex(periods["date"]),
        name=name,
    )


def prepare_calliope_inputs(
    tsa_result: AggregationResult,
    calliope_timeseries: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Prepare the two temporal inputs required by a clustered Calliope model.

    The fitted clustering is applied to the complete Calliope timeseries.
    This ensures that whatever representation method TSAM used is also
    applied to all Calliope input variables.

    Returns
    -------
    cluster_map
        Daily mapping from original dates to Calliope representative dates.
    reconstructed_timeseries
        Full-length timeseries in which every original period has been
        replaced by its TSAM representative profile.
    """
    calliope_result = apply_tsa_to_timeseries(
        tsa_result,
        calliope_timeseries,
    )

    cluster_map = build_calliope_cluster_map(calliope_result)

    return cluster_map, calliope_result.reconstructed.copy()


def _validate_timeseries(data: pd.DataFrame) -> None:
    """Validate the minimal assumptions made by the experiment helpers."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("TSAM input data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("TSAM input data cannot be empty.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("TSAM input data must use a DatetimeIndex.")

    if not data.index.is_monotonic_increasing:
        raise ValueError("TSAM input timestamps must be monotonically increasing.")

    if data.index.has_duplicates:
        raise ValueError("TSAM input timestamps must be unique.")

    if data.columns.has_duplicates:
        raise ValueError("TSAM input columns must be unique.")