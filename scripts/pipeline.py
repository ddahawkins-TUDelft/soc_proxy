"""Single-case execution pipeline for SoC Proxy experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from tsam import AggregationResult

from soc_proxy import generate_soc_proxy

from scripts.helpers.tsa import (
    build_cluster_config,
    build_weights,
    prepare_calliope_inputs,
    run_tsa,
)


@dataclass
class TSAArtifacts:
    """Outputs produced before the Calliope solve."""

    original_timeseries: pd.DataFrame
    original_proxy: pd.DataFrame

    tsa_result: AggregationResult

    cluster_map: pd.Series
    reconstructed_timeseries: pd.DataFrame
    reconstructed_proxy: pd.DataFrame


def run_tsa_case(
    config: dict[str, Any],
    timeseries: pd.DataFrame,
) -> TSAArtifacts:
    """Execute the TSA portion of one resolved experiment.

    Parameters
    ----------
    config
        Fully resolved experiment configuration.
    timeseries
        Raw chronological timeseries for the requested model horizon.

    Returns
    -------
    TSAArtifacts
        Intermediate results required for Calliope and subsequent analysis.
    """
    tsa_params = config["tsa_params"]
    soc_proxy_params = config["soc_proxy_params"]
    solver_params = config.get("solver_params", {})

    # ------------------------------------------------------------------
    # 1. Generate the SoC proxy on the original chronology
    # ------------------------------------------------------------------

    original_proxy = _generate_proxy(
        timeseries,
        soc_proxy_params,
    )

    # ------------------------------------------------------------------
    # 2. Construct the feature matrix used by TSAM
    # ------------------------------------------------------------------

    features, proxy_column = _build_clustering_features(
        timeseries=timeseries,
        proxy=original_proxy,
        tsa_params=tsa_params,
    )

    weights = _build_tsa_weights(
        features=features,
        proxy_column=proxy_column,
        tsa_params=tsa_params,
    )

    # ------------------------------------------------------------------
    # 3. Build native TSAM configuration and aggregate
    # ------------------------------------------------------------------

    cluster_config = build_cluster_config(
        tsa_params,
        solver_params,
    )

    tsa_result = run_tsa(
        features,
        n_clusters=tsa_params["k_periods"],
        cluster=cluster_config,
        weights=weights,
        period_duration=tsa_params.get(
            "period_duration",
            "1D",
        ),
        preserve_column_means=tsa_params.get(
            "preserve_column_means",
            True,
        ),
    )

    # ------------------------------------------------------------------
    # 4. Apply the fitted TSA to the full Calliope timeseries
    # ------------------------------------------------------------------

    cluster_map, reconstructed_timeseries = prepare_calliope_inputs(
        tsa_result,
        timeseries,
    )

    # ------------------------------------------------------------------
    # 5. Recompute the proxy implied by the TSA representation
    # ------------------------------------------------------------------

    reconstructed_proxy = _generate_proxy(
        reconstructed_timeseries,
        soc_proxy_params,
    )

    return TSAArtifacts(
        original_timeseries=timeseries,
        original_proxy=original_proxy,
        tsa_result=tsa_result,
        cluster_map=cluster_map,
        reconstructed_timeseries=reconstructed_timeseries,
        reconstructed_proxy=reconstructed_proxy,
    )


def _build_clustering_features(
    *,
    timeseries: pd.DataFrame,
    proxy: pd.DataFrame,
    tsa_params: dict[str, Any],
) -> tuple[pd.DataFrame, str | None]:
    """Construct the feature matrix passed to TSAM."""
    soc_config = tsa_params["soc_proxy"]

    # Start with the ordinary model timeseries.
    features = timeseries.copy()

    if not soc_config["enabled"]:
        return timeseries.copy(), None

    proxy_column = soc_config["proxy_field_name"]
    lambda_soc = soc_config["lambda_soc"]

    if proxy_column not in proxy.columns:
        raise ValueError(...)

    if lambda_soc == 0:
        return timeseries.copy(), None

    if lambda_soc == 1:
        return proxy[[proxy_column]].copy(), proxy_column

    features = timeseries.copy()
    features[proxy_column] = proxy[proxy_column]

    return features, proxy_column


def _build_tsa_weights(
    *,
    features: pd.DataFrame,
    proxy_column: str | None,
    tsa_params: dict[str, Any],
) -> dict[str, float] | None:
    """Build clustering weights, including lambda boundary cases."""
    if proxy_column is None:
        return None

    lambda_soc = tsa_params["soc_proxy"]["lambda_soc"]

    if lambda_soc == 0:
        # Equivalent to ordinary TSA.
        return None

    if lambda_soc == 1:
        # Handled upstream by reducing the feature set to the proxy alone.
        #
        # We will implement that explicitly once lambda=1 experiments are
        # included in the experiment portfolio.
        raise NotImplementedError(
            "lambda_soc=1 requires clustering on the proxy feature alone."
        )

    return build_weights(
        features.columns,
        proxy_column=proxy_column,
        proxy_weight=lambda_soc,
    )


def _generate_proxy(
    timeseries: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Generate proxy fields from one chronological timeseries.

    This adapter is intentionally isolated because the public
    ``generate_soc_proxy`` API is the next component we intend to clean up.
    """
    result, _, _ = generate_soc_proxy(
        df=timeseries,
        renewables_fields_and_weights=params[
            "renewable_portfolio_weights"
        ],
        soc_decomposition=params["temporal_decomposition"],
    )

    return result
