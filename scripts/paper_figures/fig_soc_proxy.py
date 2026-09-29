"""Plot reference CEM SoC against the endogenous SoC Proxy for NL and BE.

This script does *not* solve a clustered or reference model. For each configured
experiment it:

1. loads the configured chronology;
2. constructs the SoC Proxy (including endogenous margin selection);
3. loads the already-solved matching full-resolution reference model;
4. extracts the reference storage SoC;
5. reports the selected margin, Pearson correlation, and RMSE; and
6. plots NL and BE as vertically stacked panels.

The proxy level is shifted to zero for level-based comparison only, consistent
with the signal-metric pipeline: the cumulative proxy is defined only up to an
additive constant, while TSA itself uses the proxy-delta signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soc_proxy import generate_soc_proxy
from scripts.helpers.config import load_experiment_config
from scripts.helpers.reference_models import load_reference_model
from scripts.helpers.results_signals import (
    extract_proxy_signal,
    extract_storage_soc,
)
from scripts.pipeline import load_case_timeseries


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("config/experiment_config.yaml")
EXPERIMENTS = (
    ("NL_2010", "Netherlands"),
    ("BE_2010", "Belgium"),
)

TIMESERIES_DIR = Path("resources/raw_timeseries")
OUTPUT_DIR = Path("results/figures")
OUTPUT_STEM = "fig_soc_proxy_comparison_NL_BE"

MWH_PER_TWH = 1e6

# Keep the paper palette restrained while giving each country its own part of
# the plasma scale. Within each country, CEM and proxy use nearby colours; line
# style provides an additional distinction for the proxy.
PLASMA_POSITIONS = {
    "NL": {"cem": 0.15, "proxy": 0.35},
    "BE": {"cem": 0.62, "proxy": 0.84},
}


@dataclass(frozen=True)
class ProxyComparison:
    experiment: str
    country_code: str
    country_label: str
    reference_soc_twh: pd.Series
    proxy_soc_twh: pd.Series
    margin: float
    pearson_r: float
    rmse_twh: float
    chronology_alignment: str


def _load_comparison(
    experiment_name: str,
    country_label: str,
    configs: dict,
) -> ProxyComparison:
    """Load one chronology, construct its proxy, and compare to its reference."""
    if experiment_name not in configs:
        raise KeyError(
            f"Experiment {experiment_name!r} not found in {CONFIG_PATH}. "
            f"Available experiments: {sorted(configs)}"
        )

    config = configs[experiment_name]
    data_params = config["data_params"]
    country = str(data_params["country"])

    timeseries_path = (
        TIMESERIES_DIR / f"time_varying_parameters_{country}.csv"
    )
    if not timeseries_path.is_file():
        raise FileNotFoundError(
            f"Missing source timeseries for {country}: {timeseries_path}"
        )

    # Load exactly the experiment horizon; no TSA or CEM solve is performed.
    timeseries = load_case_timeseries(config, timeseries_path)

    # Construct the original-chronology proxy. With margin_mode='auto', this
    # performs only the endogenous proxy-margin sweep, not a Calliope solve.
    proxy_result = generate_soc_proxy(
        df=timeseries.copy(),
        **config["soc_proxy_params"],
    )
    reference_proxy = extract_proxy_signal(proxy_result.data)

    # Load the already-solved matching full-resolution reference CEM.
    reference_model = load_reference_model(config)
    reference_soc = extract_storage_soc(
        reference_model,
        target_index=reference_proxy.index,
    )

    if not reference_soc.index.equals(reference_proxy.index):
        raise RuntimeError(
            f"{experiment_name}: reference SoC and proxy chronologies do not match."
        )

    # Proxy levels have an arbitrary additive origin. Align the plotted and
    # level-metric copy to zero, matching results_signal_metrics.py.
    proxy_aligned = reference_proxy.astype(float) - float(reference_proxy.min())
    reference_soc = reference_soc.astype(float)

    if reference_soc.isna().any() or proxy_aligned.isna().any():
        raise RuntimeError(f"{experiment_name}: NaN values found in comparison signals.")

    reference_values = reference_soc.to_numpy(dtype=float)
    proxy_values = proxy_aligned.to_numpy(dtype=float)

    pearson_r = float(np.corrcoef(reference_values, proxy_values)[0, 1])
    rmse_twh = float(
        np.sqrt(np.mean((proxy_values - reference_values) ** 2)) / MWH_PER_TWH
    )

    alignment = str(reference_soc.attrs.get("chronology_alignment", "exact"))

    return ProxyComparison(
        experiment=experiment_name,
        country_code=country,
        country_label=country_label,
        reference_soc_twh=reference_soc / MWH_PER_TWH,
        proxy_soc_twh=proxy_aligned / MWH_PER_TWH,
        margin=float(proxy_result.margin),
        pearson_r=pearson_r,
        rmse_twh=rmse_twh,
        chronology_alignment=alignment,
    )


def _print_summary(comparisons: list[ProxyComparison]) -> None:
    print()
    print("=" * 72)
    print("Reference SoC vs endogenous SoC Proxy")
    print("=" * 72)

    for item in comparisons:
        print()
        print(f"{item.country_label} ({item.experiment})")
        print(f"  selected margin: {item.margin:.1%}")
        print(f"  Pearson r:       {item.pearson_r:.3f}")
        print(f"  RMSE:            {item.rmse_twh:.3f} TWh")
        print(f"  chronology:      {item.chronology_alignment}")


def _plot(comparisons: list[ProxyComparison]) -> tuple[plt.Figure, np.ndarray]:
    cmap = plt.get_cmap("plasma")

    fig, axes = plt.subplots(
        nrows=len(comparisons),
        ncols=1,
        figsize=(9, 5),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes)

    # global_max = max(
    #     max(
    #         float(item.reference_soc_twh.max()),
    #         float(item.proxy_soc_twh.max()),
    #     )
    #     for item in comparisons
    # )
    # y_top = 1.06 * global_max

    for panel_index, (ax, item) in enumerate(zip(axes, comparisons)):
        positions = PLASMA_POSITIONS[item.country_code]
        cem_colour = cmap(positions["cem"])
        proxy_colour = cmap(positions["proxy"])

        ax.plot(
            item.reference_soc_twh.index,
            item.reference_soc_twh,
            color=cem_colour,
            linewidth=1.05,
            linestyle="-",
            label="SoC (CEM)",
            zorder=2,
        )
        ax.plot(
            item.proxy_soc_twh.index,
            item.proxy_soc_twh,
            color=proxy_colour,
            linewidth=1.05,
            linestyle="--",
            dashes=(5, 2.5),
            label="SoC Proxy",
            zorder=3,
        )

        panel_max = max(
            float(item.reference_soc_twh.max()),
            float(item.proxy_soc_twh.max()),
        )
        ax.set_ylim(0.0, 1.06 * panel_max)
        ax.grid(axis="both", linestyle=":", linewidth=0.6, alpha=0.45)

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(
            f"({panel_letter}) {item.country_label}",
            loc="left",
            fontsize=10.5,
        )

        # Compact, non-invasive summary rather than event callouts.
        metrics_text = (
            f"m = {item.margin:.1%}"
            "\n"
            rf"Pearson $r$ = {item.pearson_r:.3f}"
            "\n"
            f"RMSE = {item.rmse_twh:.3f} TWh"
        )
        ax.text(
            0.985,
            0.955,
            metrics_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "0.7",
                "linewidth": 0.6,
                "alpha": 0.90,
            },
        )

        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=8.5,
            ncol=2,
        )

    fig.supylabel("State of charge (TWh)", fontsize=10)
    axes[-1].set_xlabel("Time", fontsize=10)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print()
    print("Saved:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")

    return fig, axes


def main() -> None:
    configs = load_experiment_config(CONFIG_PATH)

    comparisons = [
        _load_comparison(experiment_name, country_label, configs)
        for experiment_name, country_label in EXPERIMENTS
    ]

    _print_summary(comparisons)
    _plot(comparisons)
    # plt.show()


if __name__ == "__main__":
    main()
