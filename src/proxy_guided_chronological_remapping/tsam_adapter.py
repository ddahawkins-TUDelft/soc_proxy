"""Adapters between TSAM aggregation results and PGCR.

The adapter is intentionally conservative. PGCR receives only frozen arrays:

* the target proxy delta from the original, pre-TSA chronology;
* one proxy-delta profile for each already-selected TSAM representative; and
* the existing period-to-cluster assignment.

It never mutates or re-runs the TSAM result. For real selected representatives
(``medoid`` / ``maxoid``), proxy profiles are taken directly from the original
proxy at the representative source period. For synthetic representations, the
proxy profile must already exist in ``cluster_representatives``; the adapter
will not silently synthesise a new proxy representation after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ObjectArray = NDArray[np.object_]

_SELECTED_REAL_REPRESENTATIONS = {"medoid", "maxoid"}


@dataclass(frozen=True)
class TsamProxyChronologyInputs:
    """Frozen inputs required by proxy-guided chronological remapping."""

    target_delta: FloatArray
    representative_delta: dict[int, FloatArray]
    initial_cluster_map: ObjectArray
    representative_period_indices: dict[int, int]


def prepare_tsam_proxy_chronology_inputs(
    tsa_result: Any,
    target_proxy_delta: pd.Series,
    *,
    proxy_column_name: str,
) -> TsamProxyChronologyInputs:
    """Extract immutable PGCR inputs from one completed TSAM aggregation.

    Parameters
    ----------
    tsa_result
        A TSAM ``AggregationResult``. The function deliberately uses duck
        typing so the proxy chronology package does not import TSAM at module
        import time.
    target_proxy_delta
        The full-resolution ex-ante proxy delta from the *original* physical
        chronology. Its index must exactly match ``tsa_result.original.index``.
    proxy_column_name
        Name of the proxy field, e.g. ``"surplus_LDES"``.

    Notes
    -----
    The target is never taken from ``tsa_result.reconstructed``.

    For medoid/maxoid representations, the representative proxy profile is the
    original target profile at the selected source period. This preserves the
    one-to-one association between a real representative day and its original
    proxy trajectory, even when the proxy was not used as a clustering feature.

    For synthetic representations, the proxy field must already have been part
    of the TSAM aggregation so that its represented profile is available in
    ``cluster_representatives``. We intentionally refuse to manufacture a new
    synthetic proxy representation here, because that could be inconsistent
    with the physical representative that PGCR is supposed to keep fixed.
    """

    if not isinstance(target_proxy_delta, pd.Series):
        raise TypeError("target_proxy_delta must be a pandas Series.")

    original = tsa_result.original
    if not target_proxy_delta.index.equals(original.index):
        raise ValueError(
            "target_proxy_delta must use exactly the original pre-TSA time index."
        )

    if target_proxy_delta.empty:
        raise ValueError("target_proxy_delta cannot be empty.")

    if target_proxy_delta.isna().any():
        raise ValueError("target_proxy_delta contains missing values.")

    if getattr(tsa_result, "n_segments", None) is not None:
        raise NotImplementedError(
            "PGCR currently supports unsegmented representative periods only."
        )

    n_steps = int(tsa_result.n_timesteps_per_period)
    n_periods = len(tsa_result.cluster_assignments)
    expected_length = n_periods * n_steps

    if len(target_proxy_delta) != expected_length:
        raise ValueError(
            "PGCR currently requires complete representative periods. "
            f"Expected {expected_length} proxy timesteps "
            f"({n_periods} periods x {n_steps}), got {len(target_proxy_delta)}."
        )

    target_matrix = target_proxy_delta.to_numpy(dtype=float).reshape(
        n_periods,
        n_steps,
    )

    initial_cluster_map = np.asarray(
        tsa_result.cluster_assignments,
        dtype=object,
    )

    representation = tsa_result.clustering.representation
    representation_name = (
        representation.lower()
        if isinstance(representation, str)
        else None
    )

    cluster_ids = [int(cluster_id) for cluster_id in tsa_result.period_index]

    if representation_name in _SELECTED_REAL_REPRESENTATIONS:
        centers = tsa_result.clustering.cluster_centers
        if centers is None:
            raise ValueError(
                f"TSAM representation {representation_name!r} selects real periods, "
                "but cluster_centers is unavailable. PGCR cannot safely attach "
                "representatives to their source periods."
            )

        if len(centers) != len(cluster_ids):
            raise ValueError(
                "cluster_centers and cluster representatives have different lengths: "
                f"{len(centers)} != {len(cluster_ids)}."
            )

        representative_period_indices = {
            cluster_id: int(period_index)
            for cluster_id, period_index in zip(cluster_ids, centers, strict=True)
        }

        representative_delta = {
            cluster_id: target_matrix[period_index].copy()
            for cluster_id, period_index in representative_period_indices.items()
        }

    else:
        representative_period_indices = {}

        representatives = tsa_result.cluster_representatives
        if proxy_column_name not in representatives.columns:
            raise ValueError(
                "Synthetic TSAM representatives do not contain the requested proxy "
                f"column {proxy_column_name!r}. For this experimental PGCR adapter, "
                "the proxy must have been represented by TSAM itself; the adapter "
                "will not construct a new synthetic proxy representation after TSA."
            )

        representative_delta = {}
        for cluster_id in cluster_ids:
            profile = representatives.xs(cluster_id, level=0)[proxy_column_name]

            if len(profile) != n_steps:
                raise ValueError(
                    f"Representative {cluster_id} contains {len(profile)} proxy "
                    f"timesteps; expected {n_steps}."
                )

            values = profile.to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Representative {cluster_id} proxy profile contains "
                    "non-finite values."
                )

            representative_delta[cluster_id] = values.copy()

    return TsamProxyChronologyInputs(
        target_delta=target_matrix,
        representative_delta=representative_delta,
        initial_cluster_map=initial_cluster_map,
        representative_period_indices=representative_period_indices,
    )
