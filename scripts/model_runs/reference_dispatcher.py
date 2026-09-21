"""Run full-chronology Calliope reference cases from experiment configuration.

The dispatcher reuses the same resolved experiment configuration and Calliope
input table as the TSA experiments, but disables temporal clustering and solves
the complete chronology. This makes the saved reference directly comparable to
clustered runs.

Examples
--------
Run selected cases::

    pixi run python -m scripts.model_runs.reference_dispatcher \
        baseline_NL baseline_BE

Run every experiment in a configuration::

    pixi run python -m scripts.model_runs.reference_dispatcher --all

Overwrite an existing reference::

    pixi run python -m scripts.model_runs.reference_dispatcher \
        baseline_BE --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import calliope
import numpy as np
import pandas as pd

from scripts.helpers.config import load_experiment_config
from scripts.helpers.timeseries import (
    EXPECTED_FEATURE_COLUMNS,
    to_calliope_timeseries_table,
)
from scripts.pipeline import load_case_timeseries


DEFAULT_CONFIG = Path("config/experiment_config.yaml")
DEFAULT_MODEL_PATH = Path("config/calliope/model.yaml")
DEFAULT_TIMESERIES_DIR = Path("resources/raw_timeseries")
DEFAULT_REFERENCE_DIR = Path("resources/calliope_models/reference")

_TIMESERIES_TABLE = "time_varying_parameters_reference_df"


def main(argv: list[str] | None = None) -> int:
    """Run requested full-chronology reference cases."""
    args = _parse_args(argv)

    configs = load_experiment_config(args.config)
    selected = _select_experiments(
        configs,
        names=args.experiments,
        run_all=args.all,
    )

    calliope.set_log_verbosity(
        args.log_level,
        include_solver_output=args.solver_output,
    )

    failures: list[tuple[str, Exception]] = []
    completed_outputs: set[Path] = set()

    for name, config in selected.items():
        output_path = _reference_model_path(
            config,
            reference_dir=args.reference_dir,
        )

        # Multiple TSA experiments can share one country/horizon. Their full
        # chronological reference is the same, so solve it only once per run.
        if output_path in completed_outputs:
            print(f"\n=== {name}: reference already handled in this run ===")
            print(f"    {output_path}")
            continue

        try:
            source_path = _resolve_timeseries_path(
                config,
                timeseries_dir=args.timeseries_dir,
            )

            print()
            print("=" * 79)
            print(f"Reference case: {name}")
            print("=" * 79)
            print(f"country:    {config['data_params']['country']}")
            print(
                "horizon:    "
                f"{config['data_params']['start_date']} -> "
                f"{config['data_params']['end_date']}"
            )
            print(
                "scenario:   "
                f"{config.get('calliope_params', {}).get('scenario')}"
            )
            print(f"timeseries: {source_path}")
            print(f"output:     {output_path}")

            if output_path.exists() and not args.force:
                print("status:     exists; skipping (use --force to overwrite)")
                completed_outputs.add(output_path)
                continue

            timeseries = load_case_timeseries(
                config,
                source_path,
            )

            model = _run_reference_model(
                config,
                timeseries=timeseries,
                timeseries_template=source_path,
                model_path=args.model_path,
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            model.to_netcdf(str(output_path))

            print(f"status:     saved {output_path}")
            completed_outputs.add(output_path)

        except Exception as error:  # keep a long DelftBlue batch moving
            failures.append((name, error))
            print(f"status:     FAILED: {error}")

    if failures:
        print()
        print("Reference runs with failures:")
        for name, error in failures:
            print(f"  - {name}: {error!r}")
        return 1

    print()
    print("All requested reference runs completed successfully.")
    return 0


def _run_reference_model(
    config: dict[str, Any],
    *,
    timeseries: pd.DataFrame,
    timeseries_template: str | Path,
    model_path: str | Path,
) -> calliope.Model:
    """Initialise, validate, build, and solve one chronological reference."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Calliope model definition does not exist: {model_path}"
        )

    table = to_calliope_timeseries_table(
        timeseries,
        template_path=timeseries_template,
    )

    calliope_params = config.get("calliope_params", {})
    solver_params = config.get("solver_params", {})

    user_overrides = calliope_params.get("overrides", {})
    if not isinstance(user_overrides, dict):
        raise TypeError("calliope_params.overrides must be a mapping.")

    overrides: dict[str, Any] = dict(user_overrides)
    overrides.update(
        {
            # load_case_timeseries has already selected the exact half-open
            # [start_date, end_date) chronology.
            "config.init.subset.timesteps": None,
            # Never inherit clustering into a reference solve.
            "config.init.time_cluster": None,
            "data_tables.time_varying_parameters.table": _TIMESERIES_TABLE,
            "data_tables.time_varying_parameters.rows": "timesteps",
            "data_tables.time_varying_parameters.columns": [
                "nodes",
                "techs",
                "inputs",
            ],
            "data_tables.time_varying_parameters.drop": None,
            "data_tables.time_varying_parameters.rename_dims": None,
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
            _TIMESERIES_TABLE: table,
        },
    )

    _validate_reference_inputs(
        model,
        timeseries=timeseries,
        timeseries_table=table,
    )

    model.build()
    model.solve()

    termination = _termination_condition(model)
    if termination != "optimal":
        raise RuntimeError(
            "Reference Calliope solve did not terminate successfully. "
            f"Termination condition: {termination!r}"
        )

    if model.results is None or not model.results.data_vars:
        raise RuntimeError(
            "Reference Calliope solve completed without model results."
        )

    return model


def _validate_reference_inputs(
    model: calliope.Model,
    *,
    timeseries: pd.DataFrame,
    timeseries_table: pd.DataFrame,
) -> None:
    """Check that Calliope retained the complete supplied chronology."""
    if "timesteps" not in model.inputs.coords:
        raise RuntimeError("Calliope preprocessing removed the timesteps dimension.")

    actual_index = pd.DatetimeIndex(
        model.inputs.coords["timesteps"].values,
        name="timesteps",
    )
    expected_index = pd.DatetimeIndex(
        timeseries.index,
        name="timesteps",
    )

    if not actual_index.equals(expected_index):
        missing = expected_index.difference(actual_index)
        unexpected = actual_index.difference(expected_index)
        raise RuntimeError(
            "Reference model timesteps do not match the requested chronology. "
            f"Missing: {missing[:5].tolist()}; "
            f"unexpected: {unexpected[:5].tolist()}."
        )

    if "datesteps" in model.inputs.coords or "clusters" in model.inputs.coords:
        raise RuntimeError(
            "Reference model unexpectedly contains temporal-clustering dimensions."
        )

    if not isinstance(timeseries_table.columns, pd.MultiIndex):
        raise RuntimeError("Reference timeseries table lost its MultiIndex columns.")

    loaded_techs: set[str] = set()

    for node, tech, input_name in timeseries_table.columns:
        loaded_techs.add(str(tech))

        if input_name not in model.inputs:
            raise RuntimeError(
                f"Calliope dropped input {input_name!r} supplied for "
                f"{node!r}/{tech!r}."
            )

        loaded = model.inputs[input_name].sel(
            nodes=node,
            techs=tech,
        ).squeeze(drop=True)

        if set(loaded.dims) != {"timesteps"}:
            raise RuntimeError(
                f"Calliope input {input_name!r} for {node!r}/{tech!r} "
                f"has unexpected dimensions {loaded.dims}."
            )

        actual = loaded.sel(timesteps=actual_index).to_numpy().astype(float)
        expected = timeseries.loc[actual_index, tech].to_numpy(dtype=float)

        if actual.shape != expected.shape or not np.allclose(
            actual,
            expected,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"Calliope changed timeseries values for {node!r}/{tech!r}."
            )

    if loaded_techs != set(EXPECTED_FEATURE_COLUMNS):
        raise RuntimeError(
            "Calliope did not retain exactly the expected timeseries "
            f"technologies. Received: {sorted(loaded_techs)}."
        )


def _resolve_timeseries_path(
    config: dict[str, Any],
    *,
    timeseries_dir: str | Path,
) -> Path:
    """Resolve the Calliope input dataset used by one reference case.

    ``data_params.timeseries_path`` can explicitly select a prepared dataset,
    including one whose demand series was rebuilt with
    ``module_demand_electricity``. Otherwise the project's country naming
    convention is used.
    """
    data_params = config["data_params"]
    explicit = data_params.get("timeseries_path")

    if explicit is not None:
        path = Path(explicit)
    else:
        country = data_params["country"]
        path = Path(timeseries_dir) / f"time_varying_parameters_{country}.csv"

    if not path.is_file():
        raise FileNotFoundError(f"Timeseries source does not exist: {path}")

    return path


def _reference_model_path(
    config: dict[str, Any],
    *,
    reference_dir: str | Path,
) -> Path:
    """Return the canonical reference filename used by analysis helpers."""
    data_params = config["data_params"]
    start = pd.Timestamp(data_params["start_date"])
    end = pd.Timestamp(data_params["end_date"])
    country = str(data_params["country"])

    if end <= start:
        raise ValueError("end_date must be later than start_date.")

    if not (
        start.month == 1
        and start.day == 1
        and end.month == 1
        and end.day == 1
    ):
        raise ValueError(
            "Reference models currently require complete calendar-year "
            f"horizons; received [{start.date()}, {end.date()})."
        )

    filename = (
        f"standard_{start.year}_{end.year - 1}_reference_{country}.nc"
    )
    return Path(reference_dir) / filename


def _termination_condition(model: calliope.Model) -> str | None:
    """Retrieve the solve termination condition across Calliope 0.7 APIs."""
    runtime = getattr(model, "runtime", None)

    if runtime is not None:
        termination = getattr(runtime, "termination_condition", None)
        if termination is not None:
            return str(getattr(termination, "value", termination))

        try:
            termination = runtime.get("termination_condition")
        except AttributeError:
            termination = None

        if termination is not None:
            return str(getattr(termination, "value", termination))

    if model.results is not None:
        termination = model.results.attrs.get("termination_condition")
        if termination is not None:
            return str(getattr(termination, "value", termination))

    return None


def _select_experiments(
    configs: dict[str, dict[str, Any]],
    *,
    names: list[str],
    run_all: bool,
) -> dict[str, dict[str, Any]]:
    """Return requested resolved experiment definitions."""
    if run_all:
        if names:
            raise ValueError("Do not combine explicit experiment names with --all.")
        return configs

    if not names:
        raise ValueError("Provide at least one experiment name, or use --all.")

    missing = [name for name in names if name not in configs]
    if missing:
        raise KeyError(
            "Unknown experiment(s): "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join(configs)
        )

    return {name: configs[name] for name in names}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-chronology Calliope reference cases.",
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        help="Experiment names from the resolved experiment YAML.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every experiment in the configuration.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Experiment YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Calliope model YAML (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--timeseries-dir",
        type=Path,
        default=DEFAULT_TIMESERIES_DIR,
        help=(
            "Directory used for time_varying_parameters_<country>.csv when "
            "data_params.timeseries_path is not supplied."
        ),
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help=f"Reference output directory (default: {DEFAULT_REFERENCE_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing reference NetCDF files.",
    )
    parser.add_argument(
        "--solver-output",
        action="store_true",
        help="Include solver output in the Calliope log.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error"),
        help="Calliope log level (default: info).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
