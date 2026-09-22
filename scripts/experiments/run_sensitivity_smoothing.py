"""Run the full SoC Proxy smoothing sensitivity sweep.

The sweep is generated programmatically from available 5- and 10-year
reference models. Existing case fragments are reused by case_id.
"""

from __future__ import annotations

import argparse
import gc
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import calliope
import pandas as pd

from scripts.helpers.config import load_experiment_config
from scripts.helpers.reference_models import load_reference_model
from scripts.helpers.results import (
    consolidate_results,
    generate_case_id,
    record_case_results,
)
from scripts.pipeline import run_case


# =============================================================================
# Sweep definition
# =============================================================================

BASE_CONFIG = Path("config/experiment_config.yaml")

MODEL_PATH = Path("config/calliope/model.yaml")

REFERENCE_DIR = Path(
    "resources/calliope_models/reference"
)

TIMESERIES_DIR = Path(
    "resources/raw_timeseries"
)

RESULTS_DIR = Path("results")

MANIFEST_PATH = (
    RESULTS_DIR
    / "sensitivity_smoothing_manifest.parquet"
)


COUNTRIES = (
    "NL",
    "BE",
)

HORIZON_YEARS = {
    5,
    10,
}

METHODS = (
    "fft_lowpass",
    "gaussian",
    "moving_average",
)

TIME_HORIZONS_HOURS = (
    1,
    12,
    24,
    48,
    # 72,
    96,
    168,
    336,
    720,
)

K_PERIODS = 45


# These are the result fragments expected from a fully recorded case.
#
# Requiring all of them prevents a partially written case from being
# incorrectly treated as complete.
REQUIRED_FRAGMENTS = (
    "parameters",
    "capacities",
    "costs",
    "investment_metrics",
    "signal_metrics",
)


# Prefer these as country templates if they exist in experiment_config.yaml.
PREFERRED_TEMPLATE_EXPERIMENTS = {
    "NL": "NL_2010",
    "BE": "BE_2010",
}


_REFERENCE_PATTERN = re.compile(
    r"^standard_"
    r"(?P<start>\d{4})_"
    r"(?P<end>\d{4})_"
    r"reference_"
    r"(?P<country>[A-Za-z]+)"
    r"\.nc$"
)


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class ReferenceCase:
    """One available full-chronology reference case."""

    country: str
    start_year: int
    end_year: int
    duration_years: int
    path: Path

    @property
    def start_date(self) -> str:
        return f"{self.start_year}-01-01"

    @property
    def end_date(self) -> str:
        return f"{self.end_year + 1}-01-01"


# =============================================================================
# Helpers
# =============================================================================


def format_duration(
    seconds: float,
) -> str:
    """Format elapsed time."""

    if seconds < 60:
        return f"{seconds:.1f} s"

    minutes, seconds = divmod(
        seconds,
        60,
    )

    if minutes < 60:
        return (
            f"{int(minutes)}m "
            f"{seconds:.1f}s"
        )

    hours, minutes = divmod(
        minutes,
        60,
    )

    return (
        f"{int(hours)}h "
        f"{int(minutes)}m "
        f"{seconds:.1f}s"
    )


def discover_reference_cases() -> list[ReferenceCase]:
    """Find all available NL/BE 5- and 10-year reference models."""

    cases: list[ReferenceCase] = []

    for path in sorted(
        REFERENCE_DIR.glob(
            "standard_*_reference_*.nc"
        )
    ):
        match = _REFERENCE_PATTERN.match(
            path.name
        )

        if match is None:
            continue

        country = match.group(
            "country"
        )

        if country not in COUNTRIES:
            continue

        start_year = int(
            match.group("start")
        )

        end_year = int(
            match.group("end")
        )

        duration_years = (
            end_year
            - start_year
            + 1
        )

        if duration_years not in HORIZON_YEARS:
            continue

        cases.append(
            ReferenceCase(
                country=country,
                start_year=start_year,
                end_year=end_year,
                duration_years=duration_years,
                path=path,
            )
        )

    if not cases:
        raise RuntimeError(
            "No 5- or 10-year NL/BE reference models "
            f"found in {REFERENCE_DIR}."
        )

    return cases


def country_template(
    configs: dict[str, dict],
    country: str,
) -> dict:
    """Return a resolved experiment config to use as a country template."""

    preferred = (
        PREFERRED_TEMPLATE_EXPERIMENTS[
            country
        ]
    )

    if preferred in configs:
        return configs[preferred]

    matches = [
        config
        for config in configs.values()
        if (
            config["data_params"]["country"]
            == country
        )
    ]

    if not matches:
        raise KeyError(
            f"No experiment template found for "
            f"country {country!r}."
        )

    return matches[0]


def method_label(
    method: str,
) -> str:
    """Return concise experiment-name label."""

    return {
        "fft_lowpass": "fft",
        "gaussian": "gaussian",
        "moving_average": "moving_average",
    }[method]


def experiment_name(
    reference: ReferenceCase,
    method: str,
    time_horizon_hours: int,
) -> str:
    """Build a stable human-readable experiment name.

    The 10-year convention deliberately matches the current manually
    generated cases, e.g. NL_2010_fft_24h, so their case IDs can be reused
    if experiment_name participates in case hashing.
    """

    label = method_label(
        method
    )

    if reference.duration_years == 10:
        return (
            f"{reference.country}_"
            f"{reference.start_year}_"
            f"{label}_"
            f"{time_horizon_hours}h"
        )

    return (
        f"{reference.country}_"
        f"{reference.start_year}_"
        f"5y_"
        f"{label}_"
        f"{time_horizon_hours}h"
    )


def build_config(
    template: dict,
    reference: ReferenceCase,
    method: str,
    time_horizon_hours: int,
) -> dict:
    """Build one fully resolved sensitivity configuration."""

    config = deepcopy(
        template
    )

    config["experiment_name"] = (
        experiment_name(
            reference,
            method,
            time_horizon_hours,
        )
    )

    config["data_params"][
        "country"
    ] = reference.country

    config["data_params"][
        "start_date"
    ] = reference.start_date

    config["data_params"][
        "end_date"
    ] = reference.end_date

    config["tsa_params"][
        "k_periods"
    ] = K_PERIODS

    decomposition = (
        config[
            "soc_proxy_params"
        ].setdefault(
            "soc_decomposition",
            {},
        )
    )

    decomposition["method"] = method

    decomposition[
        "time_horizon_hours"
    ] = time_horizon_hours

    return config


def fragment_path(
    fragment: str,
    case_id: str,
) -> Path:
    """Return one per-case result-fragment path."""

    return (
        RESULTS_DIR
        / "_fragments"
        / fragment
        / f"{case_id}.parquet"
    )


def case_fragment_status(
    case_id: str,
) -> tuple[bool, list[str]]:
    """Return whether all required case fragments exist."""

    missing = [
        fragment
        for fragment in REQUIRED_FRAGMENTS
        if not fragment_path(
            fragment,
            case_id,
        ).exists()
    ]

    return (
        len(missing) == 0,
        missing,
    )


# =============================================================================
# Main
# =============================================================================


def main(
    *,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """Run or reuse all smoothing-sensitivity cases."""

    calliope.set_log_verbosity(
        "warning",
        include_solver_output=False,
    )

    base_configs = (
        load_experiment_config(
            BASE_CONFIG
        )
    )

    references = (
        discover_reference_cases()
    )

    print()
    print("=" * 88)
    print("SoC Proxy smoothing sensitivity")
    print("=" * 88)

    print(
        f"Reference cases: {len(references)}"
    )

    for reference in references:
        print(
            "  "
            f"{reference.country} "
            f"{reference.start_year}-"
            f"{reference.end_year} "
            f"({reference.duration_years}y)"
        )

    total_cases = (
        len(references)
        * len(METHODS)
        * len(TIME_HORIZONS_HOURS)
    )

    print()
    print(
        f"Methods:         {len(METHODS)}"
    )

    print(
        f"Time horizons:   "
        f"{len(TIME_HORIZONS_HOURS)}"
    )

    print(
        f"k:               {K_PERIODS}"
    )

    print(
        f"Total variants:  {total_cases}"
    )

    print()

    manifest_rows: list[
        dict[str, object]
    ] = []

    model_run_times: list[float] = []

    reused_count = 0
    executed_count = 0

    overall_start = perf_counter()

    case_number = 0

    for reference in references:

        template = country_template(
            base_configs,
            reference.country,
        )

        for method in METHODS:

            for time_horizon_hours in (
                TIME_HORIZONS_HOURS
            ):

                case_number += 1

                config = build_config(
                    template,
                    reference,
                    method,
                    time_horizon_hours,
                )

                name = config[
                    "experiment_name"
                ]

                case_id = generate_case_id(
                    config
                )

                complete, missing = (
                    case_fragment_status(
                        case_id
                    )
                )

                print()
                print("-" * 88)

                print(
                    f"[{case_number}/{total_cases}] "
                    f"{name}"
                )

                print(
                    f"case_id: {case_id}"
                )

                status = "pending"
                run_seconds = 0.0

                # ---------------------------------------------------------
                # Existing complete case
                # ---------------------------------------------------------

                if complete and not force:

                    reused_count += 1
                    status = "reused"

                    print(
                        "status:  complete fragments "
                        "already exist; SKIPPING"
                    )

                # ---------------------------------------------------------
                # Dry-run only
                # ---------------------------------------------------------

                elif dry_run:

                    status = "would_run"

                    if missing:
                        print(
                            "status:  would run; "
                            "missing fragments: "
                            + ", ".join(missing)
                        )
                    else:
                        print(
                            "status:  would rerun "
                            "(--force)"
                        )

                # ---------------------------------------------------------
                # Execute case
                # ---------------------------------------------------------

                else:

                    if missing:
                        existing = [
                            fragment
                            for fragment
                            in REQUIRED_FRAGMENTS
                            if fragment
                            not in missing
                        ]

                        if existing:
                            print(
                                "status:  PARTIAL existing "
                                "case; rerunning"
                            )

                            print(
                                "existing: "
                                + ", ".join(existing)
                            )

                            print(
                                "missing:  "
                                + ", ".join(missing)
                            )

                    timeseries_path = (
                        TIMESERIES_DIR
                        / (
                            "time_varying_parameters_"
                            f"{reference.country}.csv"
                        )
                    )

                    run_start = (
                        perf_counter()
                    )

                    result = run_case(
                        config,
                        timeseries_path,
                        model_path=MODEL_PATH,
                    )

                    reference_model = (
                        load_reference_model(
                            config
                        )
                    )

                    recorded_case_id = (
                        record_case_results(
                            config,
                            result,
                            reference_model,
                        )
                    )

                    run_seconds = (
                        perf_counter()
                        - run_start
                    )

                    if (
                        recorded_case_id
                        != case_id
                    ):
                        raise RuntimeError(
                            "Generated and recorded "
                            "case IDs differ: "
                            f"{case_id} != "
                            f"{recorded_case_id}"
                        )

                    # Confirm the result writer actually
                    # produced a complete case.
                    complete_after, (
                        missing_after
                    ) = case_fragment_status(
                        case_id
                    )

                    if not complete_after:
                        raise RuntimeError(
                            "Case completed but required "
                            "result fragments are missing: "
                            f"{missing_after}"
                        )

                    executed_count += 1
                    status = "run"

                    model_run_times.append(
                        run_seconds
                    )

                    print(
                        "status:  recorded"
                    )

                    print(
                        "margin:  "
                        f"{result.tsa.original_proxy.margin:.1%}"
                    )

                    print(
                        "time:    "
                        f"{format_duration(run_seconds)}"
                    )

                    del result
                    del reference_model
                    gc.collect()

                manifest_rows.append(
                    {
                        "case_id": case_id,
                        "experiment_name": name,
                        "country": reference.country,
                        "start_year": reference.start_year,
                        "end_year": reference.end_year,
                        "duration_years": (
                            reference.duration_years
                        ),
                        "decomposition_method": method,
                        "time_horizon_hours": (
                            time_horizon_hours
                        ),
                        "k_periods": K_PERIODS,
                        "status": status,
                        "run_seconds": run_seconds,
                    }
                )

    # ---------------------------------------------------------------------
    # Save manifest
    # ---------------------------------------------------------------------

    manifest = pd.DataFrame(
        manifest_rows
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_parquet(
        MANIFEST_PATH,
        index=False,
    )

    print()
    print("=" * 88)
    print("Sweep manifest")
    print("=" * 88)

    print(
        f"Saved: {MANIFEST_PATH}"
    )

    if dry_run:
        return

    # ---------------------------------------------------------------------
    # Consolidate all old + new fragments
    # ---------------------------------------------------------------------

    print()
    print("=" * 88)
    print("Consolidating results")
    print("=" * 88)

    consolidate_results()

    overall_seconds = (
        perf_counter()
        - overall_start
    )

    print()
    print("=" * 88)
    print("Sensitivity complete")
    print("=" * 88)

    print(
        f"Reused:       {reused_count}"
    )

    print(
        f"Executed:     {executed_count}"
    )

    if model_run_times:
        print(
            "Model time:   "
            f"{format_duration(sum(model_run_times))}"
        )

        print(
            "Average/run:  "
            f"{format_duration(
                sum(model_run_times)
                / len(model_run_times)
            )}"
        )

    print(
        "Total time:   "
        f"{format_duration(overall_seconds)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate the sweep and report which cases "
            "would run without solving anything."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rerun cases even if all required result "
            "fragments already exist."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()

    main(
        dry_run=args.dry_run,
        force=args.force,
    )