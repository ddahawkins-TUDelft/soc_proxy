"""Single-case execution pipeline for SoC Proxy experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import calliope
import pandas as pd
from tsam import AggregationResult

from proxy_guided_chronological_remapping import (
    ProxyRemapResult,
    greedy_proxy_chronology_remap,
    prepare_tsam_proxy_chronology_inputs,
)
from soc_proxy import SocProxyResult, generate_soc_proxy

from scripts.helpers.calliope import run_clustered_calliope
from scripts.helpers.timeseries import (
    EXPECTED_FEATURE_COLUMNS,
    read_calliope_timeseries,
)
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
    original_proxy: SocProxyResult

    tsa_result: AggregationResult

    cluster_map: pd.Series
    reconstructed_timeseries: pd.DataFrame
    reconstructed_proxy: SocProxyResult

    runtime_proxy_seconds: float
    runtime_tsa_seconds: float

    # Experimental post-TSA chronology remapping. These remain ``None`` / 0
    # for every existing pipeline call unless proxy_chronology_params is
    # explicitly supplied and enabled.
    proxy_chronology_result: ProxyRemapResult | None = None
    runtime_proxy_chronology_seconds: float = 0.0


@dataclass
class CaseArtifacts:
    """In-memory outputs from one complete experiment case."""

    tsa: TSAArtifacts
    calliope_model: calliope.Model


def load_case_timeseries(
    config: dict[str, Any],
    source: str | Path,
) -> pd.DataFrame:
    """Load the exact model horizon defined by one resolved experiment.

    The project source files use Calliope's multi-row CSV format. This adapter
    converts that format to the simple dataframe used by the SoC Proxy and
    TSAM, while enforcing the experiment's half-open ``[start_date, end_date)``
    horizon.
    """
    data_params = config["data_params"]

    return read_calliope_timeseries(
        source,
        start=data_params["start_date"],
        end=data_params["end_date"],
    )


def run_case(
    config: dict[str, Any],
    source_timeseries: str | Path,
    *,
    model_path: str | Path = "config/calliope/model.yaml",
    proxy_chronology_params: dict[str, Any] | None = None,
) -> CaseArtifacts:
    """Run one resolved TSA experiment through to a solved Calliope model.

    ``proxy_chronology_params`` is an opt-in experimental hook. Omitting it
    preserves the existing experiment pipeline and configuration behaviour.
    """
    timeseries = load_case_timeseries(config, source_timeseries)
    tsa = run_tsa_case(
        config,
        timeseries,
        proxy_chronology_params=proxy_chronology_params,
    )

    model = run_clustered_calliope(
        config,
        reconstructed_timeseries=tsa.reconstructed_timeseries,
        cluster_map=tsa.cluster_map,
        timeseries_template=source_timeseries,
        model_path=model_path,
    )

    return CaseArtifacts(
        tsa=tsa,
        calliope_model=model,
    )


def run_tsa_case(
    config: dict[str, Any],
    timeseries: pd.DataFrame,
    *,
    proxy_chronology_params: dict[str, Any] | None = None,
) -> TSAArtifacts:
    """Execute the TSA portion of one resolved experiment.

    Parameters
    ----------
    config
        Fully resolved experiment configuration.
    timeseries
        Raw chronological timeseries for the requested model horizon.
    proxy_chronology_params
        Optional experimental PGCR parameters. ``None`` (the default) leaves
        the existing TSA pipeline unchanged. Supported keys are ``enabled``,
        ``target_field_name``, ``start_period``, ``lookahead_periods``,
        ``max_sweeps``, ``tie_tolerance``, and ``stop_changed_fraction``.

    Returns
    -------
    TSAArtifacts
        Intermediate results required for Calliope and subsequent analysis.
    """
    tsa_params = config["tsa_params"]
    soc_proxy_params = config["soc_proxy_params"]
    solver_params = config.get("solver_params", {})

    if "dispatchable_capacity" not in soc_proxy_params:
        raise ValueError(
            "Resolved soc_proxy_params must contain "
            "'dispatchable_capacity'. Country assumptions may not "
            "have been resolved."
        )

    # ------------------------------------------------------------------
    # 1. Generate the SoC proxy on the original chronology
    # ------------------------------------------------------------------

    proxy_start = perf_counter()

    original_proxy = _generate_proxy(
        timeseries,
        soc_proxy_params,
    )

    runtime_proxy_seconds = perf_counter() - proxy_start

    # ------------------------------------------------------------------
    # 2. Construct the feature matrix used by TSAM
    # ------------------------------------------------------------------
    tsa_start = perf_counter()

    features, proxy_column = _build_clustering_features(
        timeseries=timeseries,
        proxy=original_proxy.data,
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
        verbose=False,
    )

    # ------------------------------------------------------------------
    # 4. Apply the fitted TSA to the full Calliope timeseries
    # ------------------------------------------------------------------

    # Run the established preparation path first, even when PGCR is enabled.
    # This preserves both the old behaviour and the meaning of
    # runtime_tsa_seconds. The experimental remapping is timed separately.
    cluster_map, reconstructed_timeseries = prepare_calliope_inputs(
        tsa_result,
        timeseries,
    )

    runtime_tsa_seconds = perf_counter() - tsa_start

    # ------------------------------------------------------------------
    # 4b. Optional experimental proxy-guided chronology remapping
    # ------------------------------------------------------------------

    proxy_chronology_result: ProxyRemapResult | None = None
    runtime_proxy_chronology_seconds = 0.0

    pgcr_params = proxy_chronology_params or {}
    if pgcr_params.get("enabled", False):
        pgcr_start = perf_counter()

        target_field_name = pgcr_params.get(
            "target_field_name",
            tsa_params["soc_proxy"]["proxy_field_name"],
        )

        if target_field_name not in original_proxy.data.columns:
            raise ValueError(
                "PGCR target field "
                f"{target_field_name!r} is not present in the original ex-ante "
                f"SoC proxy. Available fields: {list(original_proxy.data.columns)}"
            )

        # Crucially, the target comes directly from the original pre-TSA proxy.
        # It is never reconstructed from TSAM output.
        pgcr_inputs = prepare_tsam_proxy_chronology_inputs(
            tsa_result,
            original_proxy.data[target_field_name],
            proxy_column_name=target_field_name,
        )

        def exact_delta_evaluator(candidate_map):
            """Evaluate the physical proxy implied by one candidate PGCR map."""
            _, candidate_timeseries = prepare_calliope_inputs(
                tsa_result,
                timeseries,
                cluster_assignments=candidate_map,
            )

            candidate_proxy = _generate_proxy(
                candidate_timeseries,
                {
                    **soc_proxy_params,
                    "margin_mode": "fixed",
                    "margin_value": original_proxy.margin,
                    "verbosity": "off",
                },
            )

            return candidate_proxy.data[target_field_name]

        proxy_chronology_result = greedy_proxy_chronology_remap(
            pgcr_inputs.target_delta,
            pgcr_inputs.representative_delta,
            pgcr_inputs.initial_cluster_map,
            representative_period_indices=(
                pgcr_inputs.representative_period_indices
            ),
            start_period=pgcr_params.get("start_period", "minimum"),
            lookahead_periods=pgcr_params.get("lookahead_periods", 7),
            max_sweeps=pgcr_params.get("max_sweeps", 3),
            tie_tolerance=pgcr_params.get("tie_tolerance", 1e-12),
            stop_changed_fraction=pgcr_params.get(
                "stop_changed_fraction",
                0.0,
            ),
            exact_delta_evaluator=exact_delta_evaluator,
        )

        # Re-run only the transfer/finalisation helper. It freezes TSAM's
        # representative profiles using the original TSAM assignment and then
        # deploys those fixed profiles according to the PGCR map. It does not
        # rebuild representatives from the remapped clusters.
        cluster_map, reconstructed_timeseries = prepare_calliope_inputs(
            tsa_result,
            timeseries,
            cluster_assignments=proxy_chronology_result.cluster_map,
        )

        runtime_proxy_chronology_seconds = perf_counter() - pgcr_start

    # ------------------------------------------------------------------
    # 5. Recompute the proxy implied by the final TSA representation/map
    # ------------------------------------------------------------------

    # The margin is a property of the original chronology. If it was selected
    # automatically, do not select it again after TSA has altered the signal.
    reconstructed_proxy_params = {
        **soc_proxy_params,
        "margin_mode": "fixed",
        "margin_value": original_proxy.margin,
        # The public info/debug output describes the endogenous selection on
        # the original chronology. Do not emit a second headline when the
        # selected margin is merely reused after TSA.
        "verbosity": "off",
    }
    reconstructed_proxy = _generate_proxy(
        reconstructed_timeseries,
        reconstructed_proxy_params,
    )

    if soc_proxy_params.get("verbosity", "off") == "debug":
        print(
            "[SoC Proxy] reconstructed chronology | "
            f"fixed margin={original_proxy.margin:.1%}"
        )

    return TSAArtifacts(
        original_timeseries=timeseries,
        original_proxy=original_proxy,
        tsa_result=tsa_result,
        cluster_map=cluster_map,
        reconstructed_timeseries=reconstructed_timeseries,
        reconstructed_proxy=reconstructed_proxy,
        runtime_proxy_seconds=runtime_proxy_seconds,
        runtime_tsa_seconds=runtime_tsa_seconds,
        proxy_chronology_result=proxy_chronology_result,
        runtime_proxy_chronology_seconds=runtime_proxy_chronology_seconds,
    )


def _build_clustering_features(
    *,
    timeseries: pd.DataFrame,
    proxy: pd.DataFrame,
    tsa_params: dict[str, Any],
) -> tuple[pd.DataFrame, str | None]:
    """Construct the feature matrix passed to TSAM."""
    soc_config = tsa_params["soc_proxy"]

    base_features = timeseries.loc[:, EXPECTED_FEATURE_COLUMNS].copy()

    if not soc_config["enabled"]:
        return base_features, None

    proxy_column = soc_config["proxy_field_name"]
    lambda_soc = soc_config["lambda_soc"]

    if proxy_column not in proxy.columns:
        raise ValueError(
            f"Configured proxy field {proxy_column!r} was not generated. "
            f"Available fields are: {list(proxy.columns)}"
        )

    if lambda_soc == 0:
        return base_features, None

    if lambda_soc == 1:
        return proxy[[proxy_column]].copy(), proxy_column

    base_features[proxy_column] = proxy[proxy_column]

    return base_features, proxy_column


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

    if lambda_soc in {0, 1}:
        return None

    return build_weights(
        features.columns,
        proxy_column=proxy_column,
        proxy_weight=lambda_soc,
    )


def _generate_proxy(
    timeseries: pd.DataFrame,
    params: dict[str, Any],
) -> SocProxyResult:
    """Generate the SoC Proxy using the public contribution API."""
    return generate_soc_proxy(
        df=timeseries,
        **params,
    )
