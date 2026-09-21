"""Run one experiment and plot the SoC Proxy diagnostic."""

import calliope
import matplotlib.pyplot as plt

from scripts.helpers.config import load_experiment_config
from scripts.helpers.reference_models import load_reference_model
from scripts.helpers.results import (
    consolidate_results,
    record_case_results,
)
from scripts.helpers.results_signals import (
    extract_proxy_signal,
    extract_storage_soc,
)
from scripts.pipeline import run_case
from scripts.plots.soc_proxy_comparison import (
    plot_soc_proxy_comparison,
)


calliope.set_log_verbosity(
    "info",
    include_solver_output=True,
)


# ---------------------------------------------------------------------------
# 1. Load experiment
# ---------------------------------------------------------------------------

configs = load_experiment_config(
    "config/experiment_config.yaml"
)

config = configs["NL_2010"]


# ---------------------------------------------------------------------------
# 2. Run clustered experiment
# ---------------------------------------------------------------------------

timeseries_path = (
    "resources/raw_timeseries/"
    f"time_varying_parameters_{config['data_params']['country']}.csv"
)

result = run_case(
    config,
    timeseries_path,
    model_path="config/calliope/model.yaml",
)


# ---------------------------------------------------------------------------
# 3. Report endogenous margin selection
# ---------------------------------------------------------------------------

print()
print("=" * 72)
print("SoC Proxy margin selection")
print("=" * 72)

print(
    f"Selected margin: "
    f"{result.tsa.original_proxy.margin:.1%}"
)

diagnostics = result.tsa.original_proxy.margin_diagnostics

if diagnostics is not None:
    print(
        f"Economic m80:   "
        f"{diagnostics.economic_margin_80:.1%}"
    )
    print(
        f"Event margin:   "
        f"{diagnostics.terminal_event_margin:.1%}"
    )
    print(
        "Terminal event: "
        f"{diagnostics.terminal_event_timestamp}"
    )
    print(
        "Tail support:   "
        f"{diagnostics.terminal_event_support_fraction:.0%}"
    )
    print(
        "Lower-tail exposure at selected margin: "
        f"1%={diagnostics.selected_near_zero_1pct_delta_fraction:.1%}, "
        f"5%={diagnostics.selected_near_zero_5pct_delta_fraction:.1%}"
    )


# ---------------------------------------------------------------------------
# 4. Load matching full-chronology reference
# ---------------------------------------------------------------------------

reference_model = load_reference_model(
    config
)


# ---------------------------------------------------------------------------
# 5. Extract the four diagnostic signals
# ---------------------------------------------------------------------------

reference_proxy = extract_proxy_signal(
    result.tsa.original_proxy.data
)

reference_soc = extract_storage_soc(
    reference_model,
    target_index=reference_proxy.index,
)

alignment = reference_soc.attrs.get(
    "chronology_alignment",
    "exact",
)

print(f"Reference chronology: {alignment}")

if alignment != "exact":
    print(
        "Expanded reference timesteps: "
        f"{reference_soc.attrs['inserted_timesteps']}"
    )

clustered_soc = extract_storage_soc(
    result.calliope_model,
    cluster_map=result.tsa.cluster_map,
)

clustered_proxy = extract_proxy_signal(
    result.tsa.reconstructed_proxy.data,
)


# ---------------------------------------------------------------------------
# 6. Basic sanity checks
# ---------------------------------------------------------------------------

print()
print("=" * 72)
print("Signal sanity checks")
print("=" * 72)

print(
    f"Reference SoC:   {reference_soc.shape}"
)
print(
    f"Reference proxy: {reference_proxy.shape}"
)
print(
    f"Clustered SoC:   {clustered_soc.shape}"
)
print(
    f"Clustered proxy: {clustered_proxy.shape}"
)

assert reference_soc.index.equals(
    reference_proxy.index
)
assert reference_soc.index.equals(
    clustered_soc.index
)
assert reference_soc.index.equals(
    clustered_proxy.index
)

print()
print(
    "Reference SoC range: "
    f"{reference_soc.min():.2f} to "
    f"{reference_soc.max():.2f} MWh"
)
print(
    "Clustered SoC range: "
    f"{clustered_soc.min():.2f} to "
    f"{clustered_soc.max():.2f} MWh"
)

print(
    "Reference proxy raw range: "
    f"{reference_proxy.min():.2f} to "
    f"{reference_proxy.max():.2f} MWh"
)
print(
    "Clustered proxy raw range: "
    f"{clustered_proxy.min():.2f} to "
    f"{clustered_proxy.max():.2f} MWh"
)


# ---------------------------------------------------------------------------
# 7. Results recording
# ---------------------------------------------------------------------------

case_id = record_case_results(
    config,
    result,
    reference_model,
)

print()
print(
    f"Recorded case: {case_id}"
)


# ---------------------------------------------------------------------------
# 8. Plot
# ---------------------------------------------------------------------------

# The SoC proxy is only defined up to an additive constant. The public proxy
# deliberately no longer shifts its minimum to zero because TSA uses the
# delta signal. For visual comparison only, align proxy copies to zero.
reference_proxy_plot = (
    reference_proxy
    - reference_proxy.min()
)

clustered_proxy_plot = (
    clustered_proxy
    - clustered_proxy.min()
)

fig, ax = plot_soc_proxy_comparison(
    reference_soc,
    reference_proxy_plot,
    clustered_soc=clustered_soc,
    clustered_proxy=clustered_proxy_plot,
    title=(
        "SoC Proxy diagnostic "
        f"(m={result.tsa.original_proxy.margin:.1%})"
    ),
    ylabel="Stored energy (MWh)",
    output_path=(
        "results/figures/"
        "soc_proxy_diagnostic.png"
    ),
)

plt.show()


# ---------------------------------------------------------------------------
# 9. Consolidate results
# ---------------------------------------------------------------------------

consolidate_results()