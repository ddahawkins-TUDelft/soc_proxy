"""Estimate SoC Proxy renewable weights using staged low-k training."""

from __future__ import annotations

import gc
import sys
from copy import deepcopy
from time import perf_counter

import calliope

from scripts.helpers.config import load_experiment_config
from scripts.helpers.results_capacities import extract_capacities
from scripts.pipeline import run_case


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

EXPERIMENT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "NL_2010"
)

K_PERIODS = 20

# Training stops once:
#
#   abs(C_n - C_(n-1)) / abs(C_(n-1)) < tolerance
#
# for every renewable technology.
CONVERGENCE_TOLERANCE = 0.05

# Always run at least this many training iterations before allowing
# convergence to stop the loop.
#
# Runs 1 and 2 use auto margin.
# Run 2 determines the margin frozen for runs 3+.
MIN_RUNS = 3

# Safety limit for the iterative training stage.
# The final auto-margin pass is additional to this.
MAX_RUNS = 10

# Fraction of the latest Calliope capacity result used to update weights.
RELAXATION = 0.75

RENEWABLE_TECHS = (
    "solar",
    "onshore_wind",
    "offshore_wind",
)


calliope.set_log_verbosity(
    "warning",
    include_solver_output=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_renewable_capacities(
    model,
) -> dict[str, float]:
    """Extract renewable output capacities from a solved Calliope model."""

    capacities = extract_capacities(
        model,
        case_id="renewable_weight_training",
        model_type="clustered",
    )

    renewable = capacities.loc[
        (capacities["capacity_type"] == "flow_cap")
        & capacities["tech"].isin(RENEWABLE_TECHS)
    ]

    values = (
        renewable
        .groupby("tech", observed=True)["value"]
        .sum()
        .to_dict()
    )

    missing = set(RENEWABLE_TECHS) - set(values)

    if missing:
        raise RuntimeError(
            "Solved model is missing renewable capacities for: "
            f"{sorted(missing)}"
        )

    return {
        tech: float(values[tech])
        for tech in RENEWABLE_TECHS
    }


def normalise(
    values: dict[str, float],
) -> dict[str, float]:
    """Return portfolio shares for diagnostic display only."""

    total = sum(values.values())

    if total <= 0:
        raise ValueError(
            "Renewable capacities/weights must sum to > 0."
        )

    return {
        tech: values[tech] / total
        for tech in RENEWABLE_TECHS
    }


def relative_changes(
    old: dict[str, float],
    new: dict[str, float],
) -> dict[str, float]:
    """Return absolute relative changes in model capacities."""

    changes: dict[str, float] = {}

    for tech in RENEWABLE_TECHS:

        old_value = float(old[tech])
        new_value = float(new[tech])

        if old_value == 0:
            changes[tech] = (
                0.0
                if new_value == 0
                else float("inf")
            )
        else:
            changes[tech] = (
                abs(new_value - old_value)
                / abs(old_value)
            )

    return changes


def relaxed_update(
    old_weights: dict[str, float],
    capacities: dict[str, float],
) -> dict[str, float]:
    """Blend current raw weights with the latest Calliope capacities."""

    return {
        tech: (
            (1.0 - RELAXATION) * old_weights[tech]
            + RELAXATION * capacities[tech]
        )
        for tech in RENEWABLE_TECHS
    }


def format_capacities(
    values: dict[str, float],
) -> str:
    """Format renewable values in GW."""

    return " | ".join(
        f"{tech}={values[tech] / 1000:7.2f} GW"
        for tech in RENEWABLE_TECHS
    )


def format_shares(
    values: dict[str, float],
) -> str:
    """Format renewable portfolio shares."""

    shares = normalise(values)

    return " | ".join(
        f"{tech}={shares[tech]:6.1%}"
        for tech in RENEWABLE_TECHS
    )


def format_duration(
    seconds: float,
) -> str:
    """Format elapsed time."""

    if seconds < 60:
        return f"{seconds:.1f} s"

    minutes, seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"

    hours, minutes = divmod(minutes, 60)

    return (
        f"{int(hours)}h "
        f"{int(minutes)}m "
        f"{seconds:.1f}s"
    )


def configure_weights(
    config: dict,
    weights: dict[str, float],
) -> None:
    """Insert raw renewable weights into a resolved experiment config."""

    for tech in RENEWABLE_TECHS:
        config["soc_proxy_params"]["renewables"][tech][
            "weight"
        ] = weights[tech]


# ---------------------------------------------------------------------------
# Load experiment
# ---------------------------------------------------------------------------

configs = load_experiment_config(
    "config/experiment_config.yaml"
)

if EXPERIMENT not in configs:
    raise KeyError(
        f"Unknown experiment {EXPERIMENT!r}. "
        f"Available experiments: {list(configs)}"
    )

base_config = configs[EXPERIMENT]

country = base_config["data_params"]["country"]

timeseries_path = (
    "resources/raw_timeseries/"
    f"time_varying_parameters_{country}.csv"
)


# ---------------------------------------------------------------------------
# Initialise training
# ---------------------------------------------------------------------------

# Deliberately uninformative initial renewable mix.
weights = {
    tech: 1.0
    for tech in RENEWABLE_TECHS
}

previous_capacities: dict[str, float] | None = None
training_capacities: dict[str, float] | None = None

frozen_margin: float | None = None
converged = False

training_run_times: list[float] = []
capacity_history: list[dict[str, float]] = []


print()
print("=" * 88)
print(
    f"Training renewable weights: "
    f"{EXPERIMENT} / {country}"
)
print(
    f"k={K_PERIODS}, "
    f"tolerance={CONVERGENCE_TOLERANCE:.1%}, "
    f"relaxation={RELAXATION:.2f}, "
    f"min runs={MIN_RUNS}, "
    f"max runs={MAX_RUNS}"
)
print("=" * 88)


overall_start = perf_counter()
training_start = perf_counter()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

for iteration in range(1, MAX_RUNS + 1):

    config = deepcopy(base_config)

    config["tsa_params"]["k_periods"] = K_PERIODS

    # Runs 1 and 2 use endogenous margin selection.
    #
    # Run 2's selected margin is then frozen for the remaining training
    # iterations, so weight convergence is evaluated under a stable proxy
    # formulation.
    if iteration <= 2:

        config["soc_proxy_params"]["margin_mode"] = "auto"

        config["soc_proxy_params"].pop(
            "margin_value",
            None,
        )

        margin_description = "auto"

    else:

        if frozen_margin is None:
            raise RuntimeError(
                "Training margin was not frozen after run 2."
            )

        config["soc_proxy_params"]["margin_mode"] = "fixed"
        config["soc_proxy_params"]["margin_value"] = frozen_margin

        margin_description = (
            f"fixed {frozen_margin:.1%}"
        )

    configure_weights(
        config,
        weights,
    )

    print()
    print("-" * 88)
    print(f"Run {iteration:02d}")
    print("-" * 88)

    print(
        "  Input weights:  "
        + format_capacities(weights)
    )

    print(
        "  Input shares:   "
        + format_shares(weights)
    )

    print(
        f"  Margin mode:    "
        f"{margin_description}"
    )

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------

    run_start = perf_counter()

    result = run_case(
        config,
        timeseries_path,
        model_path="config/calliope/model.yaml",
    )

    run_elapsed = (
        perf_counter()
        - run_start
    )

    training_run_times.append(
        run_elapsed
    )

    # -----------------------------------------------------------------------
    # Extract capacity result
    # -----------------------------------------------------------------------

    capacities = extract_renewable_capacities(
        result.calliope_model
    )

    capacity_history.append(
        capacities.copy()
    )

    training_capacities = capacities.copy()

    selected_margin = (
        result.tsa.original_proxy.margin
    )

    print()
    print(
        "  Capacities:     "
        + format_capacities(capacities)
    )

    print(
        "  Output shares:  "
        + format_shares(capacities)
    )

    print(
        f"  Proxy margin:   "
        f"{selected_margin:.1%}"
    )

    # -----------------------------------------------------------------------
    # Freeze run-2 margin
    # -----------------------------------------------------------------------

    if iteration == 2:

        frozen_margin = selected_margin

        print(
            f"  Frozen margin:  "
            f"{frozen_margin:.1%}"
        )

    # -----------------------------------------------------------------------
    # Capacity convergence
    # -----------------------------------------------------------------------

    max_change: float | None = None

    if previous_capacities is None:

        print(
            "  Capacity change: "
            "n/a (first model solution)"
        )

    else:

        changes = relative_changes(
            previous_capacities,
            capacities,
        )

        max_change = max(
            changes.values()
        )

        print(
            "  Capacity change:"
            + " | ".join(
                f" {tech}={changes[tech]:6.1%}"
                for tech in RENEWABLE_TECHS
            )
        )

        print(
            f"  Max change:      "
            f"{max_change:.1%}"
        )

    # -----------------------------------------------------------------------
    # Determine whether training is complete
    # -----------------------------------------------------------------------

    if (
        iteration >= MIN_RUNS
        and max_change is not None
        and max_change < CONVERGENCE_TOLERANCE
    ):

        converged = True

        print(
            f"  Converged:       "
            f"yes (< {CONVERGENCE_TOLERANCE:.1%})"
        )

        print(
            f"  Run time:        "
            f"{format_duration(run_elapsed)}"
        )

        del result
        gc.collect()

        break

    # -----------------------------------------------------------------------
    # Prepare next training weights
    # -----------------------------------------------------------------------

    if iteration == 1:

        # The initial 1:1:1 weights contain no useful capacity information.
        # Use the first model output directly for run 2.
        next_weights = capacities.copy()

    else:

        next_weights = relaxed_update(
            weights,
            capacities,
        )

    print(
        "  Next weights:   "
        + format_capacities(next_weights)
    )

    print(
        "  Next shares:    "
        + format_shares(next_weights)
    )

    print(
        f"  Run time:        "
        f"{format_duration(run_elapsed)}"
    )

    previous_capacities = (
        capacities.copy()
    )

    weights = next_weights

    del result
    gc.collect()


training_elapsed = (
    perf_counter()
    - training_start
)


# ---------------------------------------------------------------------------
# Final auto-margin pass
# ---------------------------------------------------------------------------

if training_capacities is None:
    raise RuntimeError(
        "No training run completed successfully."
    )


print()
print("=" * 88)
print("Final auto-margin pass")
print("=" * 88)

if converged:
    print(
        "Training converged. "
        "Re-running once with endogenous margin selection."
    )
else:
    print(
        f"Training reached the {MAX_RUNS}-run safety limit. "
        "Running the final auto-margin pass using the latest capacities."
    )


# Use the latest actual Calliope capacities directly as the proxy weights.
#
# We deliberately do not apply another relaxed update here.
final_input_weights = (
    training_capacities.copy()
)

final_config = deepcopy(
    base_config
)

final_config["tsa_params"]["k_periods"] = (
    K_PERIODS
)

final_config["soc_proxy_params"][
    "margin_mode"
] = "auto"

final_config["soc_proxy_params"].pop(
    "margin_value",
    None,
)

configure_weights(
    final_config,
    final_input_weights,
)


print()
print(
    "  Input weights:  "
    + format_capacities(final_input_weights)
)

print(
    "  Input shares:   "
    + format_shares(final_input_weights)
)

print(
    "  Margin mode:    auto"
)


final_start = perf_counter()

final_result = run_case(
    final_config,
    timeseries_path,
    model_path="config/calliope/model.yaml",
)

final_elapsed = (
    perf_counter()
    - final_start
)


final_capacities = (
    extract_renewable_capacities(
        final_result.calliope_model
    )
)

final_margin = (
    final_result.tsa.original_proxy.margin
)


print()
print(
    "  Capacities:     "
    + format_capacities(final_capacities)
)

print(
    "  Output shares:  "
    + format_shares(final_capacities)
)

print(
    f"  Final margin:   "
    f"{final_margin:.1%}"
)

print(
    f"  Run time:       "
    f"{format_duration(final_elapsed)}"
)


# Compare the final auto-margin solve with the last training solution.
final_changes = relative_changes(
    training_capacities,
    final_capacities,
)

print(
    "  Change from training:"
    + " | ".join(
        f" {tech}={final_changes[tech]:6.1%}"
        for tech in RENEWABLE_TECHS
    )
)

print(
    f"  Max final change: "
    f"{max(final_changes.values()):.1%}"
)


del final_result
gc.collect()


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

overall_elapsed = (
    perf_counter()
    - overall_start
)


print()
print("=" * 88)
print("Renewable-weight estimate")
print("=" * 88)

print(f"Experiment:       {EXPERIMENT}")
print(f"Country:          {country}")
print(f"k:                {K_PERIODS}")

print(
    f"Training status:  "
    + (
        f"converged after {len(training_run_times)} runs"
        if converged
        else f"not converged after {len(training_run_times)} runs"
    )
)

print(
    f"Frozen margin:    "
    f"{frozen_margin:.1%}"
    if frozen_margin is not None
    else "Frozen margin:    n/a"
)

print(
    f"Final margin:     "
    f"{final_margin:.1%}"
)

print(
    f"Training time:    "
    f"{format_duration(training_elapsed)}"
)

print(
    f"Final-pass time:  "
    f"{format_duration(final_elapsed)}"
)

print(
    f"Total time:       "
    f"{format_duration(overall_elapsed)}"
)

if training_run_times:

    print(
        f"Average training "
        f"run: {format_duration(sum(training_run_times) / len(training_run_times))}"
    )


print()
print("Training capacity history:")

for i, capacities in enumerate(
    capacity_history,
    start=1,
):

    print(
        f"  Run {i:02d}: "
        + format_capacities(capacities)
    )


print()
print("Final recommended weights:")

for tech in RENEWABLE_TECHS:

    print(
        f"  {tech:<15}"
        f"{final_capacities[tech]:10.1f} MW "
        f"({final_capacities[tech] / 1000:6.2f} GW)"
    )


print()
print("Normalised shares:")

final_shares = normalise(
    final_capacities
)

for tech in RENEWABLE_TECHS:

    print(
        f"  {tech:<15}"
        f"{final_shares[tech]:7.2%}"
    )
