"""Run all SoC Proxy smoothing-sensitivity experiments and plot outcomes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter

import calliope
import matplotlib.pyplot as plt

from scripts.helpers.config import load_experiment_config
from scripts.helpers.reference_models import load_reference_model
from scripts.helpers.results import (
    consolidate_results,
    generate_case_id,
    record_case_results,
)
from scripts.pipeline import run_case
from scripts.plots.experiment_outcome_comparison import (
    plot_experiment_outcome_comparison,
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

CONFIG_PATH = "config/config_sensitivity_smoothing.yaml"

MODEL_PATH = "config/calliope/model.yaml"

RESULTS_DIR = Path("results")

K_PERIODS = 45

OUTPUT_PATH = (
    RESULTS_DIR
    / "figures"
    / "sensitivity_smoothing_outcomes.png"
)


calliope.set_log_verbosity(
    "warning",
    include_solver_output=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Return a compact human-readable duration."""

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


def case_results_exist(
    case_id: str,
) -> bool:
    """Return whether all results needed by the comparison plot exist."""

    required_fragments = (
        "parameters",
        "investment_metrics",
        "signal_metrics",
    )

    return all(
        (
            RESULTS_DIR
            / "_fragments"
            / fragment
            / f"{case_id}.parquet"
        ).exists()
        for fragment in required_fragments
    )


# ---------------------------------------------------------------------------
# Load sensitivity experiments
# ---------------------------------------------------------------------------

configs = load_experiment_config(
    CONFIG_PATH
)

if not configs:
    raise RuntimeError(
        f"No experiments were found in {CONFIG_PATH!r}."
    )


print()
print("=" * 88)
print("SoC Proxy smoothing sensitivity")
print("=" * 88)

print(f"Config:       {CONFIG_PATH}")
print(f"Experiments:  {len(configs)}")
print(f"k:            {K_PERIODS}")


# ---------------------------------------------------------------------------
# Run or reuse every experiment
# ---------------------------------------------------------------------------

case_ids: list[str] = []

run_times: dict[str, float] = {}
skipped: set[str] = set()

overall_start = perf_counter()


for number, (
    experiment_name,
    base_config,
) in enumerate(
    configs.items(),
    start=1,
):

    print()
    print("-" * 88)

    print(
        f"[{number}/{len(configs)}] "
        f"{experiment_name}"
    )

    print("-" * 88)

    # Work on an independent nested copy so no sensitivity experiment can
    # mutate another experiment's resolved configuration.
    config = deepcopy(
        base_config
    )

    # Hold k fixed across the complete smoothing sensitivity.
    config["tsa_params"][
        "k_periods"
    ] = K_PERIODS

    country = (
        config["data_params"]["country"]
    )

    method = (
        config
        .get("soc_proxy_params", {})
        .get("soc_decomposition", {})
        .get("method", "<default>")
    )

    horizon = (
        config
        .get("soc_proxy_params", {})
        .get("soc_decomposition", {})
        .get(
            "time_horizon_hours",
            "<default>",
        )
    )

    print(f"Country:      {country}")
    print(f"Method:       {method}")
    print(f"Horizon:      {horizon} h")
    print(f"k:            {K_PERIODS}")

    # -----------------------------------------------------------------------
    # Resolve case ID before doing any expensive work
    # -----------------------------------------------------------------------

    case_id = generate_case_id(
        config
    )

    case_ids.append(
        case_id
    )

    print(f"Case ID:      {case_id}")

    # -----------------------------------------------------------------------
    # Reuse an already-complete result
    # -----------------------------------------------------------------------

    if case_results_exist(case_id):

        skipped.add(
            experiment_name
        )

        print(
            "Status:       existing results found; "
            "skipping model run"
        )

        continue

    # -----------------------------------------------------------------------
    # Run clustered CEM
    # -----------------------------------------------------------------------

    timeseries_path = (
        "resources/raw_timeseries/"
        f"time_varying_parameters_{country}.csv"
    )

    run_start = perf_counter()

    result = run_case(
        config,
        timeseries_path,
        model_path=MODEL_PATH,
    )

    run_elapsed = (
        perf_counter()
        - run_start
    )

    run_times[
        experiment_name
    ] = run_elapsed

    # -----------------------------------------------------------------------
    # Load matching reference and record outputs
    # -----------------------------------------------------------------------

    reference_model = load_reference_model(
        config
    )

    recorded_case_id = record_case_results(
        config,
        result,
        reference_model,
    )

    # The pre-computed case ID and result writer should always agree.
    if recorded_case_id != case_id:
        raise RuntimeError(
            "Generated case ID changed during execution: "
            f"expected {case_id}, recorded {recorded_case_id}."
        )

    print(
        f"Margin:       "
        f"{result.tsa.original_proxy.margin:.1%}"
    )

    print(
        f"Run time:     "
        f"{format_duration(run_elapsed)}"
    )


# ---------------------------------------------------------------------------
# Consolidate all fragments, including reused cases
# ---------------------------------------------------------------------------

print()
print("=" * 88)
print("Consolidating results")
print("=" * 88)

consolidate_start = perf_counter()

consolidate_results()

consolidate_elapsed = (
    perf_counter()
    - consolidate_start
)

print(
    "Consolidation time: "
    f"{format_duration(consolidate_elapsed)}"
)


# ---------------------------------------------------------------------------
# Plot all sensitivity experiments in config-file order
# ---------------------------------------------------------------------------

print()
print("=" * 88)
print("Plotting sensitivity outcomes")
print("=" * 88)

fig, axes = plot_experiment_outcome_comparison(
    case_ids,
    results_dir=RESULTS_DIR,
    output_path=OUTPUT_PATH,
)

print(
    f"Figure:       {OUTPUT_PATH}"
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

overall_elapsed = (
    perf_counter()
    - overall_start
)

print()
print("=" * 88)
print("Sensitivity complete")
print("=" * 88)

for experiment_name in configs:

    if experiment_name in skipped:
        duration = "reused"
    else:
        duration = format_duration(
            run_times[experiment_name]
        )

    print(
        f"{experiment_name:<36}"
        f"{duration}"
    )


print()

print(
    f"Cases reused:   "
    f"{len(skipped)}"
)

print(
    f"Cases run:      "
    f"{len(run_times)}"
)

if run_times:
    print(
        f"Model run time: "
        f"{format_duration(sum(run_times.values()))}"
    )

print(
    f"Consolidation:  "
    f"{format_duration(consolidate_elapsed)}"
)

print(
    f"Total:          "
    f"{format_duration(overall_elapsed)}"
)


plt.show()