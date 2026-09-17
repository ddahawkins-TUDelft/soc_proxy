"""Helpers for running clustered Calliope models in SoC Proxy experiments.

TSAM outputs are treated as ephemeral inputs to Calliope. Representative-day
mappings and reconstructed timeseries are passed directly to Calliope as
in-memory pandas dataframes.

The Calliope data-table handoff is deliberately validated before and after
model initialisation. This guards against silent data loss, overwriting, or
unexpected broadcasting during Calliope preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import calliope
import numpy as np
import pandas as pd
import yaml

from scripts.helpers.timeseries import to_calliope_timeseries_table


_CLUSTER_INPUT = "cluster_days_param"
_TIMESERIES_TABLE = "time_varying_parameters_df"
_CLUSTER_TABLE = "cluster_days_df"
_STORAGE_CLUSTER_MATH = "storage_inter_cluster"

_EXPECTED_TIMESERIES_TECHS = {
    "demand_power",
    "solar",
    "offshore_wind",
    "onshore_wind",
}


def build_clustered_model(
    config: dict[str, Any],
    *,
    reconstructed_timeseries: pd.DataFrame,
    cluster_map: pd.Series,
    timeseries_template: str | Path,
    model_path: str | Path = "config/calliope/model.yaml",
) -> calliope.Model:
    """Initialise and validate one clustered Calliope model.

    The reconstructed TSAM timeseries and representative-day mapping are
    supplied to Calliope in memory. After preprocessing, the resulting
    ``model.inputs`` dataset is checked against the supplied data before the
    optimisation model is allowed to build.

    Parameters
    ----------
    config
        Fully resolved experiment configuration.
    reconstructed_timeseries
        Full-length TSAM reconstructed timeseries. Every original day contains
        the representative profile corresponding to its cluster.
    cluster_map
        Daily mapping from every original model date to its representative
        date.
    timeseries_template
        Original project Calliope timeseries CSV. Its metadata rows define the
        node, technology, and Calliope input corresponding to each fixed
        timeseries feature.
    model_path
        Base Calliope model YAML.

    Returns
    -------
    calliope.Model
        Initialised and validated model, ready for ``build()``.
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

    # First validate our own adapters, before asking Calliope to preprocess
    # either table.
    _validate_in_memory_tables(
        reconstructed_timeseries=reconstructed_timeseries,
        cluster_map=cluster_map,
        timeseries_table=timeseries_table,
        cluster_table=cluster_table,
    )

    user_overrides = calliope_params.get("overrides", {})

    if not isinstance(user_overrides, dict):
        raise TypeError(
            "calliope_params.overrides must be a mapping."
        )

    # Start with user-defined experimental overrides, then force the temporal
    # plumbing required by this experiment. This prevents an unrelated
    # Calliope override from silently replacing the in-memory TSA data.
    overrides: dict[str, Any] = dict(user_overrides)

    overrides.update(
        {
            "config.init.subset.timesteps": None,
            "config.init.time_cluster": _CLUSTER_INPUT,
            "config.init.extra_math": _cluster_extra_math(model_path),

            "data_tables.time_varying_parameters.table":
                _TIMESERIES_TABLE,
            "data_tables.time_varying_parameters.rows":
                "timesteps",
            "data_tables.time_varying_parameters.columns": [
                "nodes",
                "techs",
                "inputs",
            ],
            "data_tables.time_varying_parameters.drop":
                None,
            "data_tables.time_varying_parameters.rename_dims":
                None,

            "data_tables.cluster_days": {
                "table": _CLUSTER_TABLE,
                "rows": "timesteps",
                "add_dims": {
                    "inputs": _CLUSTER_INPUT,
                },
            },
        }
    )

    solver = solver_params.get("solver")
    if solver is not None:
        overrides["config.solve.solver"] = solver

    solver_io = solver_params.get("solver_io")
    if solver_io is not None:
        overrides["config.solve.solver_io"] = solver_io

    solver_options = solver_params.get("solver_options")
    if solver_options is not None:
        overrides["config.solve.solver_options"] = solver_options

    model = calliope.read_yaml(
        model_path,
        scenario=calliope_params.get("scenario"),
        override_dict=overrides,
        data_table_dfs={
            _TIMESERIES_TABLE: timeseries_table,
            _CLUSTER_TABLE: cluster_table,
        },
    )

    # read_yaml() has now completed Calliope preprocessing, including temporal
    # clustering. Validate that the supplied data survived that process
    # exactly as expected.
    validate_clustered_model_inputs(
        model,
        reconstructed_timeseries=reconstructed_timeseries,
        cluster_map=cluster_map,
        timeseries_table=timeseries_table,
    )

    return model


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

    termination_condition = _get_termination_condition(model)

    if termination_condition not in {"optimal", "feasible"}:
        raise RuntimeError(
            "Calliope solve did not terminate successfully. "
            f"Termination condition: {termination_condition!r}"
        )

    if model.results is None or not model.results.data_vars:
        raise RuntimeError(
            "Calliope solve completed without model results."
        )

    return model


def validate_clustered_model_inputs(
    model: calliope.Model,
    *,
    reconstructed_timeseries: pd.DataFrame,
    cluster_map: pd.Series,
    timeseries_table: pd.DataFrame,
) -> None:
    """Validate Calliope preprocessing against the supplied TSA inputs.

    Calliope intentionally reduces the full reconstructed chronology to the
    representative days referenced by ``cluster_map``. Therefore, validation
    compares ``model.inputs`` against the corresponding representative-day
    rows of ``reconstructed_timeseries``.

    The derived cluster weights and lookup arrays are also checked against
    values reconstructed independently from ``cluster_map``.
    """
    if "timesteps" not in model.inputs.coords:
        raise RuntimeError(
            "Calliope preprocessing removed the timesteps dimension."
        )

    model_timesteps = pd.DatetimeIndex(
        model.inputs.coords["timesteps"].values,
        name="timesteps",
    )

    representative_dates = pd.DatetimeIndex(
        pd.to_datetime(cluster_map.unique())
    ).normalize().sort_values()

    expected_timesteps = reconstructed_timeseries.index[
        reconstructed_timeseries.index
        .normalize()
        .isin(representative_dates)
    ]

    if not model_timesteps.equals(expected_timesteps):
        missing = expected_timesteps.difference(model_timesteps)
        unexpected = model_timesteps.difference(expected_timesteps)

        raise RuntimeError(
            "Calliope representative timesteps do not match the expected "
            "TSAM representative days. "
            f"Missing: {missing[:5].tolist()}; "
            f"unexpected: {unexpected[:5].tolist()}."
        )

    _validate_loaded_timeseries_values(
        model,
        reconstructed_timeseries=reconstructed_timeseries,
        timeseries_table=timeseries_table,
        model_timesteps=model_timesteps,
    )

    _validate_cluster_structure(
        model,
        cluster_map=cluster_map,
        model_timesteps=model_timesteps,
    )


def _validate_loaded_timeseries_values(
    model: calliope.Model,
    *,
    reconstructed_timeseries: pd.DataFrame,
    timeseries_table: pd.DataFrame,
    model_timesteps: pd.DatetimeIndex,
) -> None:
    """Check every supplied Calliope timeseries value after preprocessing."""
    if not isinstance(timeseries_table.columns, pd.MultiIndex):
        raise RuntimeError(
            "Internal Calliope timeseries table lost its MultiIndex columns."
        )

    expected_column_names = ["nodes", "techs", "inputs"]

    if list(timeseries_table.columns.names) != expected_column_names:
        raise RuntimeError(
            "Unexpected Calliope timeseries table dimensions. "
            f"Expected {expected_column_names}, received "
            f"{list(timeseries_table.columns.names)}."
        )

    loaded_techs: set[str] = set()

    for node, tech, input_name in timeseries_table.columns:
        loaded_techs.add(str(tech))

        if input_name not in model.inputs:
            raise RuntimeError(
                f"Calliope dropped input {input_name!r} supplied for "
                f"{node!r}/{tech!r}."
            )

        if tech not in reconstructed_timeseries.columns:
            raise RuntimeError(
                f"Cannot validate Calliope technology {tech!r}: "
                "no matching reconstructed timeseries column exists."
            )

        data_array = model.inputs[input_name]

        required_dims = {
            "nodes": node,
            "techs": tech,
        }

        selectors: dict[str, Any] = {}

        for dimension, value in required_dims.items():
            if dimension not in data_array.dims:
                raise RuntimeError(
                    f"Calliope input {input_name!r} unexpectedly lacks "
                    f"dimension {dimension!r}."
                )
            selectors[dimension] = value

        loaded = data_array.sel(selectors)

        # Any remaining singleton dimensions can safely disappear, but the
        # resulting data must describe one value per representative timestep.
        loaded = loaded.squeeze(drop=True)

        if set(loaded.dims) != {"timesteps"}:
            raise RuntimeError(
                f"Calliope input {input_name!r} for {node!r}/{tech!r} "
                "has unexpected remaining dimensions after selection: "
                f"{loaded.dims}."
            )

        loaded = loaded.sel(timesteps=model_timesteps)

        expected = reconstructed_timeseries.loc[
            model_timesteps,
            tech,
        ].to_numpy(dtype=float)

        actual = loaded.to_numpy().astype(float)

        _assert_numeric_equal(
            label=f"{input_name}[{node}, {tech}]",
            actual=actual,
            expected=expected,
        )

    if loaded_techs != _EXPECTED_TIMESERIES_TECHS:
        raise RuntimeError(
            "Calliope did not retain exactly the expected timeseries "
            "technologies. "
            f"Expected {_EXPECTED_TIMESERIES_TECHS}, "
            f"received {loaded_techs}."
        )


def _validate_cluster_structure(
    model: calliope.Model,
    *,
    cluster_map: pd.Series,
    model_timesteps: pd.DatetimeIndex,
) -> None:
    """Validate Calliope's cluster weights and lookup arrays."""
    original_dates = pd.DatetimeIndex(
        cluster_map.index
    ).normalize()

    representative_dates = pd.Series(
        pd.to_datetime(cluster_map.to_numpy()).normalize(),
        index=original_dates,
        name="representative_date",
    )

    # ------------------------------------------------------------------
    # Original datesteps
    # ------------------------------------------------------------------

    if "datesteps" not in model.inputs.coords:
        raise RuntimeError(
            "Calliope clustering did not create the datesteps dimension."
        )

    loaded_datesteps = pd.DatetimeIndex(
        model.inputs.coords["datesteps"].values
    )

    if not loaded_datesteps.equals(original_dates):
        raise RuntimeError(
            "Calliope datesteps do not match the original cluster-map dates."
        )

    # ------------------------------------------------------------------
    # Cluster IDs
    # ------------------------------------------------------------------

    expected_datestep_clusters = (
        representative_dates
        .groupby(representative_dates)
        .ngroup()
        .astype(int)
    )

    if "lookup_datestep_cluster" not in model.inputs:
        raise RuntimeError(
            "Calliope clustering did not create "
            "'lookup_datestep_cluster'."
        )

    loaded_datestep_clusters = (
        model.inputs["lookup_datestep_cluster"]
        .sel(datesteps=loaded_datesteps)
        .to_numpy()
        .astype(int)
    )

    if not np.array_equal(
        loaded_datestep_clusters,
        expected_datestep_clusters.to_numpy(),
    ):
        raise RuntimeError(
            "Calliope's datestep-to-cluster lookup does not match the "
            "supplied representative-day mapping."
        )

    cluster_id_by_rep_date = (
        expected_datestep_clusters
        .groupby(representative_dates)
        .first()
    )

    if "timestep_cluster" not in model.inputs:
        raise RuntimeError(
            "Calliope clustering did not create 'timestep_cluster'."
        )

    expected_timestep_clusters = (
        pd.Series(
            model_timesteps.normalize(),
            index=model_timesteps,
        )
        .map(cluster_id_by_rep_date)
        .to_numpy()
        .astype(int)
    )

    loaded_timestep_clusters = (
        model.inputs["timestep_cluster"]
        .sel(timesteps=model_timesteps)
        .to_numpy()
        .astype(int)
    )

    if not np.array_equal(
        loaded_timestep_clusters,
        expected_timestep_clusters,
    ):
        raise RuntimeError(
            "Calliope's timestep-to-cluster lookup does not match the "
            "supplied representative-day mapping."
        )

    # ------------------------------------------------------------------
    # Representative-day weights
    # ------------------------------------------------------------------

    if "timestep_weights" not in model.inputs:
        raise RuntimeError(
            "Calliope clustering did not create 'timestep_weights'."
        )

    expected_counts = representative_dates.value_counts()

    expected_weights = (
        pd.Series(
            model_timesteps.normalize(),
            index=model_timesteps,
        )
        .map(expected_counts)
        .to_numpy(dtype=float)
    )

    loaded_weights = (
        model.inputs["timestep_weights"]
        .sel(timesteps=model_timesteps)
        .to_numpy()
        .astype(float)
    )

    _assert_numeric_equal(
        label="timestep_weights",
        actual=loaded_weights,
        expected=expected_weights,
    )

    if "clusters" not in model.inputs.coords:
        raise RuntimeError(
            "Calliope clustering did not create the clusters dimension."
        )

    n_loaded_clusters = model.inputs.coords["clusters"].size
    n_expected_clusters = representative_dates.nunique()

    if n_loaded_clusters != n_expected_clusters:
        raise RuntimeError(
            "Calliope created an unexpected number of clusters. "
            f"Expected {n_expected_clusters}, "
            f"received {n_loaded_clusters}."
        )


def _validate_in_memory_tables(
    *,
    reconstructed_timeseries: pd.DataFrame,
    cluster_map: pd.Series,
    timeseries_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
) -> None:
    """Validate the two pandas objects before passing them to Calliope."""
    if not timeseries_table.index.equals(
        reconstructed_timeseries.index
    ):
        raise RuntimeError(
            "Calliope timeseries-table index differs from the reconstructed "
            "TSAM timeseries."
        )

    if not isinstance(timeseries_table.columns, pd.MultiIndex):
        raise RuntimeError(
            "Calliope timeseries table must have MultiIndex columns."
        )

    if list(timeseries_table.columns.names) != [
        "nodes",
        "techs",
        "inputs",
    ]:
        raise RuntimeError(
            "Calliope timeseries table has unexpected column dimensions."
        )

    for _, tech, _ in timeseries_table.columns:
        expected = reconstructed_timeseries[tech].to_numpy(dtype=float)
        actual = timeseries_table.xs(
            tech,
            axis=1,
            level="techs",
        ).iloc[:, 0].to_numpy(dtype=float)

        _assert_numeric_equal(
            label=f"in-memory timeseries table: {tech}",
            actual=actual,
            expected=expected,
        )

    expected_cluster_index = pd.DatetimeIndex(
        cluster_map.index,
        name="timesteps",
    )

    if not cluster_table.index.equals(expected_cluster_index):
        raise RuntimeError(
            "In-memory Calliope cluster table index differs from "
            "the TSAM cluster map."
        )

    expected_cluster_values = (
        pd.to_datetime(cluster_map)
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )

    actual_cluster_values = (
        cluster_table.iloc[:, 0]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        actual_cluster_values,
        expected_cluster_values,
    ):
        raise RuntimeError(
            "In-memory Calliope cluster table values differ from "
            "the TSAM cluster map."
        )


def _cluster_map_to_table(
    cluster_map: pd.Series,
) -> pd.DataFrame:
    """Convert a representative-day mapping to a Calliope data table.

    Representative dates are deliberately passed as ISO date strings.

    Calliope's clustering preprocessor accepts either datetime-valued or
    string-valued mappings and parses string values internally. Using strings
    also avoids an xarray dtype-promotion issue when Calliope merges an
    in-memory datetime-valued data table into its numeric model dataset.
    """
    if not isinstance(cluster_map, pd.Series):
        raise TypeError(
            "cluster_map must be a pandas Series."
        )

    if cluster_map.empty:
        raise ValueError(
            "cluster_map cannot be empty."
        )

    if not isinstance(cluster_map.index, pd.DatetimeIndex):
        raise TypeError(
            "cluster_map must use a DatetimeIndex."
        )

    if cluster_map.index.has_duplicates:
        raise ValueError(
            "cluster_map dates must be unique."
        )

    if cluster_map.isna().any():
        raise ValueError(
            "cluster_map cannot contain missing representative dates."
        )

    original_dates = pd.DatetimeIndex(
        cluster_map.index
    )

    representative_dates = pd.DatetimeIndex(
        pd.to_datetime(cluster_map, errors="raise")
    )

    if not original_dates.equals(original_dates.normalize()):
        raise ValueError(
            "cluster_map index must contain dates at midnight."
        )

    if not representative_dates.equals(
        representative_dates.normalize()
    ):
        raise ValueError(
            "cluster_map representative values must contain dates at "
            "midnight."
        )

    original_date_set = set(original_dates)

    unknown_representatives = (
        set(representative_dates) - original_date_set
    )

    if unknown_representatives:
        raise ValueError(
            "Every representative date must also exist in the original "
            "cluster-map dates. Unknown representatives: "
            f"{sorted(unknown_representatives)[:5]}"
        )

    frame = pd.DataFrame(
        {
            "cluster_days": (
                representative_dates
                .strftime("%Y-%m-%d")
            )
        },
        index=pd.DatetimeIndex(
            original_dates,
            name="timesteps",
        ),
    )

    return frame


def _cluster_extra_math(
    model_path: Path,
) -> list[str]:
    """Preserve base-model extra math and add inter-cluster storage math."""
    with model_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        model_definition = yaml.safe_load(file) or {}

    extra_math = (
        model_definition
        .get("config", {})
        .get("init", {})
        .get("extra_math", [])
    ) or []

    if not isinstance(extra_math, list):
        raise TypeError(
            "config.init.extra_math in the Calliope model must be a list."
        )

    return list(
        dict.fromkeys(
            [
                *extra_math,
                _STORAGE_CLUSTER_MATH,
            ]
        )
    )


def _assert_numeric_equal(
    *,
    label: str,
    actual: np.ndarray,
    expected: np.ndarray,
) -> None:
    """Raise a useful error if two numeric arrays differ."""
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{label} shape changed during Calliope preprocessing: "
            f"expected {expected.shape}, received {actual.shape}."
        )

    if np.allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    ):
        return

    differences = np.abs(actual - expected)

    raise RuntimeError(
        f"{label} values changed during Calliope preprocessing. "
        f"Maximum absolute difference: "
        f"{np.nanmax(differences):.6g}."
    )


def _get_termination_condition(
    model: calliope.Model,
) -> str | None:
    """Retrieve the solve termination condition across Calliope 0.7 APIs."""
    runtime = getattr(model, "runtime", None)

    if runtime is not None:
        termination = getattr(
            runtime,
            "termination_condition",
            None,
        )

        if termination is not None:
            return str(termination)

        try:
            termination = runtime.get(
                "termination_condition"
            )
        except AttributeError:
            termination = None

        if termination is not None:
            return str(termination)

    if model.results is not None:
        termination = model.results.attrs.get(
            "termination_condition"
        )

        if termination is not None:
            return str(termination)

    return None