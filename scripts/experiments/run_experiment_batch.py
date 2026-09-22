"""Run experiment configurations as independent fragment-writing jobs.

This runner is intended for local execution and SLURM job arrays. Each selected
experiment is solved independently and persisted through the existing
``results/_fragments`` mechanism. Consolidation is deliberately NOT performed
here, so multiple jobs may safely execute different case IDs in parallel.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from time import perf_counter

import calliope

from scripts.helpers.config import load_experiment_config
from scripts.helpers.reference_models import (
    load_reference_model,
    resolve_reference_model_path,
)
from scripts.helpers.results import (
    generate_case_id,
    record_case_results,
)
from scripts.pipeline import run_case


DEFAULT_MODEL_PATH = Path("config/calliope/model.yaml")
DEFAULT_TIMESERIES_DIR = Path("resources/raw_timeseries")
DEFAULT_RESULTS_DIR = Path("results")

REQUIRED_FRAGMENTS = (
    "parameters",
    "capacities",
    "costs",
    "investment_metrics",
    "signal_metrics",
)


def format_duration(seconds: float) -> str:
    """Format elapsed wall-clock time."""
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def fragment_path(
    results_dir: Path,
    fragment: str,
    case_id: str,
) -> Path:
    """Return the path of one per-case result fragment."""
    return (
        results_dir
        / "_fragments"
        / fragment
        / f"{case_id}.parquet"
    )


def case_fragment_status(
    results_dir: Path,
    case_id: str,
) -> tuple[bool, list[str]]:
    """Return whether every required result fragment exists."""
    missing = [
        fragment
        for fragment in REQUIRED_FRAGMENTS
        if not fragment_path(
            results_dir,
            fragment,
            case_id,
        ).is_file()
    ]

    return not missing, missing


def validate_unique_case_ids(
    configs: dict[str, dict],
) -> None:
    """Reject duplicate scientific cases within one configuration file."""
    names_by_case_id: dict[str, list[str]] = {}

    for name, config in configs.items():
        case_id = generate_case_id(config)
        names_by_case_id.setdefault(case_id, []).append(name)

    duplicates = {
        case_id: names
        for case_id, names in names_by_case_id.items()
        if len(names) > 1
    }

    if duplicates:
        details = "; ".join(
            f"{case_id}: {', '.join(names)}"
            for case_id, names in duplicates.items()
        )
        raise RuntimeError(
            "Configuration contains duplicate scientific cases "
            f"(identical case IDs): {details}"
        )


def select_cases(
    configs: dict[str, dict],
    *,
    index: int | None,
    experiment: str | None,
    run_all: bool,
) -> list[tuple[str, dict]]:
    """Select cases by zero-based index, name, or the complete config."""
    items = list(configs.items())

    if run_all:
        return items

    if experiment is not None:
        if experiment not in configs:
            raise KeyError(
                f"Unknown experiment {experiment!r}. "
                f"Available experiments: {list(configs)}"
            )
        return [(experiment, configs[experiment])]

    if index is None:
        raise RuntimeError(
            "Internal error: no experiment selection was supplied."
        )

    if index < 0 or index >= len(items):
        raise IndexError(
            f"Experiment index {index} is outside the valid range "
            f"0..{len(items) - 1}."
        )

    return [items[index]]


def run_one_case(
    *,
    experiment_name: str,
    config: dict,
    config_path: Path,
    model_path: Path,
    timeseries_dir: Path,
    results_dir: Path,
    dry_run: bool,
    force: bool,
) -> str:
    """Run or reuse one experiment and return its case ID."""
    case_id = generate_case_id(config)
    country = str(config["data_params"]["country"])

    complete, missing = case_fragment_status(
        results_dir,
        case_id,
    )

    print()
    print("=" * 88)
    print(experiment_name)
    print("=" * 88)
    print(f"config:     {config_path}")
    print(f"case_id:    {case_id}")
    print(f"country:    {country}")
    print(
        "horizon:    "
        f"{config['data_params']['start_date']} -> "
        f"{config['data_params']['end_date']}"
    )
    print(f"k:          {config['tsa_params']['k_periods']}")
    print(
        "WP:         "
        f"{config['tsa_params']['soc_proxy']['lambda_soc']}"
    )
    print(
        "cluster:    "
        f"{config['tsa_params']['cluster']['method']}"
    )
    print(
        "represent:  "
        f"{config['tsa_params']['representation']['method']}"
    )

    scope = config["tsa_params"]["representation"].get("scope")
    if scope is not None:
        print(f"scope:      {scope}")

    if complete and not force:
        print("status:     complete fragments already exist; SKIPPING")
        return case_id

    if dry_run:
        if missing:
            print(
                "status:     would run; missing fragments: "
                + ", ".join(missing)
            )
        else:
            print("status:     would rerun (--force)")
        return case_id

    if missing:
        existing = [
            fragment
            for fragment in REQUIRED_FRAGMENTS
            if fragment not in missing
        ]
        if existing:
            print("status:     PARTIAL existing case; rerunning")
            print("existing:   " + ", ".join(existing))
            print("missing:    " + ", ".join(missing))

    timeseries_path = (
        timeseries_dir
        / f"time_varying_parameters_{country}.csv"
    )

    if not timeseries_path.is_file():
        raise FileNotFoundError(
            f"Input timeseries does not exist: {timeseries_path}"
        )

    # Cheap preflight before starting the optimisation.
    reference_path = resolve_reference_model_path(config)
    print(f"reference:  {reference_path}")

    start = perf_counter()

    result = run_case(
        config,
        timeseries_path,
        model_path=model_path,
    )

    reference_model = load_reference_model(config)

    recorded_case_id = record_case_results(
        config,
        result,
        reference_model,
        results_dir=results_dir,
    )

    if recorded_case_id != case_id:
        raise RuntimeError(
            "Generated and recorded case IDs differ: "
            f"{case_id} != {recorded_case_id}"
        )

    complete_after, missing_after = case_fragment_status(
        results_dir,
        case_id,
    )

    if not complete_after:
        raise RuntimeError(
            "Case completed but required result fragments are missing: "
            f"{missing_after}"
        )

    elapsed = perf_counter() - start

    print("status:     recorded")
    print(
        "margin:     "
        f"{result.tsa.original_proxy.margin:.1%}"
    )
    print(f"wall time:  {format_duration(elapsed)}")

    del result
    del reference_model
    gc.collect()

    return case_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent experiment cases and write per-case "
            "result fragments without consolidating them."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Experiment YAML file.",
    )

    selection = parser.add_mutually_exclusive_group(required=True)

    selection.add_argument(
        "--index",
        type=int,
        help=(
            "Zero-based experiment index in YAML declaration order. "
            "Designed for SLURM_ARRAY_TASK_ID."
        ),
    )

    selection.add_argument(
        "--experiment",
        help="Run one named experiment.",
    )

    selection.add_argument(
        "--all",
        action="store_true",
        help="Run every experiment sequentially.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would run without solving.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rerun selected cases even if all required fragments "
            "already exist."
        ),
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--timeseries-dir",
        type=Path,
        default=DEFAULT_TIMESERIES_DIR,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )

    args = parser.parse_args()

    calliope.set_log_verbosity(
        "warning",
        include_solver_output=False,
    )

    configs = load_experiment_config(args.config)
    validate_unique_case_ids(configs)

    selected = select_cases(
        configs,
        index=args.index,
        experiment=args.experiment,
        run_all=args.all,
    )

    print()
    print("=" * 88)
    print("Experiment fragment runner")
    print("=" * 88)
    print(f"Config:      {args.config}")
    print(f"Cases:       {len(configs)}")
    print(f"Selected:    {len(selected)}")
    print(f"Results:     {args.results_dir}")
    print("Consolidate: disabled in this runner")

    for experiment_name, config in selected:
        run_one_case(
            experiment_name=experiment_name,
            config=config,
            config_path=args.config,
            model_path=args.model_path,
            timeseries_dir=args.timeseries_dir,
            results_dir=args.results_dir,
            dry_run=args.dry_run,
            force=args.force,
        )


if __name__ == "__main__":
    main()
