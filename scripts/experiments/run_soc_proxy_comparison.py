"""Run one experiment and plot the SoC Proxy diagnostic."""

import matplotlib.pyplot as plt

from scripts.helpers.config import load_experiment_config
from scripts.helpers.reference_models import load_reference_model
from scripts.helpers.results_signals import (
    extract_proxy_signal,
    extract_storage_soc,
)
from scripts.pipeline import run_case
from scripts.plots.soc_proxy_comparison import (
    plot_soc_proxy_comparison,
)


# ---------------------------------------------------------------------------
# 1. Load experiment
# ---------------------------------------------------------------------------

configs = load_experiment_config(
    "config/experiment_config.yaml"
)

config = configs["baseline"]


# ---------------------------------------------------------------------------
# 2. Run clustered experiment
# ---------------------------------------------------------------------------

result = run_case(
    config,
    "resources/raw_timeseries/time_varying_parameters_NL.csv",
    model_path="config/calliope/model.yaml",
)


# ---------------------------------------------------------------------------
# 3. Load matching full-chronology reference
# ---------------------------------------------------------------------------

reference_model = load_reference_model(
    config
)


# ---------------------------------------------------------------------------
# 4. Extract the four diagnostic signals
# ---------------------------------------------------------------------------

reference_soc = extract_storage_soc(
    reference_model,
)

reference_proxy = extract_proxy_signal(
    result.tsa.original_proxy,
)

clustered_soc = extract_storage_soc(
    result.calliope_model,
    cluster_map=result.tsa.cluster_map,
)

clustered_proxy = extract_proxy_signal(
    result.tsa.reconstructed_proxy,
)


# ---------------------------------------------------------------------------
# 5. Basic sanity checks
# ---------------------------------------------------------------------------

print(f"Reference SoC:   {reference_soc.shape}")
print(f"Reference proxy: {reference_proxy.shape}")
print(f"Clustered SoC:   {clustered_soc.shape}")
print(f"Clustered proxy: {clustered_proxy.shape}")

assert reference_soc.index.equals(reference_proxy.index)
assert reference_soc.index.equals(clustered_soc.index)
assert reference_soc.index.equals(clustered_proxy.index)

print()
print(
    f"Reference SoC range: "
    f"{reference_soc.min():.2f} to {reference_soc.max():.2f} MWh"
)
print(
    f"Clustered SoC range: "
    f"{clustered_soc.min():.2f} to {clustered_soc.max():.2f} MWh"
)


# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------

fig, ax = plot_soc_proxy_comparison(
    reference_soc,
    reference_proxy,
    clustered_soc=clustered_soc,
    clustered_proxy=clustered_proxy,
    title="SoC Proxy diagnostic",
    ylabel="Stored energy (MWh)",
    output_path="results/figures/soc_proxy_diagnostic.png",
)

plt.show()
