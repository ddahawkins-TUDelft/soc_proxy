"""Helpers for running clustered Calliope models in SoC Proxy experiments.

TSAM outputs are treated as ephemeral inputs to Calliope. Representative-day
mappings and reconstructed timeseries are passed to Calliope as in-memory
pandas dataframes; they do not need to be written to persistent experiment
output directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import calliope
import pandas as pd

from scripts.helpers.timeseries import to_calliope_timeseries_table


_CLUSTER_INPUT = "cluster_days_param"
_TIMESERIES_TABLE = "time_varying_parameters_df"
_CLUSTER_TABLE = "cluster_days_df"


def build_clustered_model(
    config: dict[str, Any],
    *,
    reconstructed_timeseries: pd.DataFrame,
    cluster_map: pd.Series,
    timeseries_template: str | Path,
    model_path: str | Path = "config/calliope/model.yaml",
) -> calliope.Model:
    """Build a Calliope model definition for one clustered experiment.

    Parameters
    ----------
    config
        Fully resolved experiment configuration.
    reconstructed_timeseries
        Full-length TSAM reconstructed timeseries. Each original day contains
        the representative profile of its assigned cluster.
    cluster_map
        Mapping from each original date to the representative date Calliope
        should use for that cluster.
    timeseries_template
        Original project Calliope timeseries CSV. Its metadata rows define the
        node, technology, and input parameter associated with each of the four
        fixed timeseries columns.
    model_path
        Base Calliope model definition YAML.

    Returns
    -------
    calliope.Model
        Initialised, but not yet built or solved, Calliope model.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Calliope model definition does not exist: {model_path}"
        )

    calliope_params = config.get("calliope_params", {})
    solver_params = config.get("solver_params", {})

    timeseries_table = to_calliope_timeseries_table(
        reconstructed_timeseries,
        template_path=timeseries_template,
    )
    cluster_table = _cluster_map_to_table(cluster_map)

    overrides: dict[str, Any] = {
        "data_tables.time_varying_parameters": {
            "table": _TIMESERIES_TABLE,
            "rows": "timesteps",
            "columns": ["nodes", "techs", "inputs"],
        },
        "data_tables.cluster_days": {
            "table": _CLUSTER_TABLE,
            "rows": "timesteps",
            "add_dims": {"inputs": _CLUSTER_INPUT},
        },
        "config.init.time_cluster": _CLUSTER_INPUT,
        # Calliope's built-in inter-cluster storage formulation tracks storage
        # across the chronology of representative days and handles cyclic
        # storage without the legacy custom-math workaround.
        "config.init.extra_math": ["storage_inter_cluster"],
    }

    solver = solver_params.get("solver")
    if solver is not None:
        overrides["config.solve.solver"] = solver

    solver_io = solver_params.get("solver_io")
    if solver_io is not None:
        overrides["config.solve.solver_io"] = solver_io

    solver_options = solver_params.get("solver_options")
    if solver_options is not None:
        overrides["config.solve.solver_options"] = solver_options

    # Explicit Calliope overrides are useful for reviewer experiments without
    # requiring the runner to understand every Calliope configuration option.
    user_overrides = calliope_params.get("overrides", {})
    if not isinstance(user_overrides, dict):
        raise TypeError("calliope_params.overrides must be a mapping.")
    overrides.update(user_overrides)

    return calliope.read_yaml(
        model_path,
        scenario=calliope_params.get("scenario"),
        override_dict=overrides,
        data_table_dfs={
            _TIMESERIES_TABLE: timeseries_table,
            _CLUSTER_TABLE: cluster_table,
        },
    )


def run_clustered_calliope(
    config: dict[str, Any],
    *,
    reconstructed_timeseries: pd.DataFrame,
    cluster_map: pd.Series,
    timeseries_template: str | Path,
    model_path: str | Path = "config/calliope/model.yaml",
) -> calliope.Model:
    """Build and solve one clustered Calliope capacity-expansion model."""
    model = build_clustered_model(
        config,
        reconstructed_timeseries=reconstructed_timeseries,
        cluster_map=cluster_map,
        timeseries_template=timeseries_template,
        model_path=model_path,
    )

    model.build()
    model.solve()

    if model.results.nbytes == 0:
        raise RuntimeError("Calliope solve completed without model results.")

    return model


def _cluster_map_to_table(cluster_map: pd.Series) -> pd.DataFrame:
    """Convert a representative-day mapping to a Calliope data table."""
    if not isinstance(cluster_map, pd.Series):
        raise TypeError("cluster_map must be a pandas Series.")
    if cluster_map.empty:
        raise ValueError("cluster_map cannot be empty.")
    if not isinstance(cluster_map.index, pd.DatetimeIndex):
        raise TypeError("cluster_map must use a DatetimeIndex.")
    if cluster_map.index.has_duplicates:
        raise ValueError("cluster_map dates must be unique.")

    values = pd.to_datetime(cluster_map, errors="raise")
    frame = values.to_frame(name="cluster_days")
    frame.index = pd.DatetimeIndex(frame.index, name="timesteps")
    return frame
