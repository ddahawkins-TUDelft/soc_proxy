"""Plot TSA-method sensitivity experiment results.

This is an inspection plot for assessing whether the SoC proxy improves
outcomes consistently across TSA methods, and for identifying a sensible
base TSA method.

Panels:
    (a) Signed LDES capacity error
    (b) Annualised-CAPEX-weighted MACME
    (c) Clustered-v-reference CEM SoC-delta nRMSE
    (d) TSA runtime

Encoding:
    x position : TSA method
    sub-position : proxy weight W_P = 0, 0.5, 1
    colour : number of representative periods k
    small horizontal jitter : weather horizon
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


METHOD_ORDER = [
    "hierarchical_medoid",
    "kmeans_medoid",
    "hierarchical_distribution_minmax",
]

METHOD_LABELS = {
    "hierarchical_medoid": "Hierarchical\n+ medoid",
    "kmeans_medoid": "k-means\n+ medoid",
    "hierarchical_distribution_minmax": "Hierarchical\n+ distribution-minmax",
}

WP_ORDER = [0.0, 0.5, 1.0]

# Large enough to separate W_P clearly, but still keep each method together.
WP_OFFSETS = {
    0.0: -0.22,
    0.5: 0.0,
    1.0: 0.22,
}

METRICS = [
    (
        "ldes_capacity_error_pct",
        "(a) LDES capacity error",
        "Capacity error (%)",
    ),
    (
        "macme_capex_weighted_pct",
        "(b) CAPEX-weighted MACME",
        "Weighted MACME (%)",
    ),
    (
        "soc_delta_nrmse_pct",
        "(c) CEM SoC-delta error",
        "SoC delta nRMSE (%)",
    ),
    (
        "runtime_tsa_seconds",
        "(d) TSA runtime",
        "Runtime (s)",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TSA-method sensitivity results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/sensitivity_tsa_methods"),
        help="Directory containing consolidated result parquet files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. Defaults to "
            "<results-dir>/tsa_methods_sensitivity.png."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it.",
    )
    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    table_name: str,
) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}\n"
            f"Available columns:\n{sorted(frame.columns)}"
        )


def load_results(results_dir: Path) -> pd.DataFrame:
    parameters_path = results_dir / "parameters.parquet"
    investment_path = results_dir / "investment_metrics.parquet"
    signal_path = results_dir / "signal_metrics.parquet"

    parameters = pd.read_parquet(parameters_path)
    investment = pd.read_parquet(investment_path)
    signals = pd.read_parquet(signal_path)

    require_columns(
        parameters,
        {
            "case_id",
            "experiment_name",
            "country",
            "start_date",
            "end_date",
            "k_periods",
            "lambda_soc",
            "cluster_method",
            "representation_method",
            "runtime_tsa_seconds",
        },
        table_name="parameters.parquet",
    )

    require_columns(
        investment,
        {
            "case_id",
            "metric",
            "value",
        },
        table_name="investment_metrics.parquet",
    )

    investment = (
        investment.pivot(
            index="case_id",
            columns="metric",
            values="value",
        )
        .reset_index()
    )

    investment.columns.name = None

    require_columns(
        investment,
        {
            "case_id",
            "ldes_capacity_error_signed",
            "macme_capex_weighted_annualised",
        },
        table_name="pivoted investment_metrics.parquet",
    )

    require_columns(
        signals,
        {
            "case_id",
            "error_family",
            "signal_type",
            "metric",
            "normalisation_basis",
            "window_half_width_days",
            "value",
        },
        table_name="signal_metrics.parquet",
    )

    # Full-horizon clustered-v-reference CEM error on SoC deltas.
    soc_delta = signals.loc[
        signals["error_family"].eq("cem")
        & signals["signal_type"].eq("delta")
        & signals["metric"].eq("nrmse")
        & signals["normalisation_basis"].eq("reference_proxy_full_range")
        & signals["window_half_width_days"].isna()
    ].copy()

    # If the schema explicitly records full-horizon coverage, enforce it.
    if "covers_full_horizon" in soc_delta.columns:
        soc_delta = soc_delta.loc[
            soc_delta["covers_full_horizon"].fillna(False)
        ]

    duplicate_soc = soc_delta["case_id"].duplicated(keep=False)
    if duplicate_soc.any():
        duplicate_ids = sorted(
            soc_delta.loc[duplicate_soc, "case_id"].astype(str).unique()
        )
        raise ValueError(
            "Expected one full-horizon CEM delta nRMSE row per case, "
            f"but found duplicates for: {duplicate_ids[:10]}"
        )

    soc_delta = soc_delta[["case_id", "value"]].rename(
        columns={"value": "soc_delta_nrmse"}
    )

    data = (
        parameters.merge(
            investment[
                [
                    "case_id",
                    "ldes_capacity_error_signed",
                    "macme_capex_weighted_annualised",
                ]
            ],
            on="case_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            soc_delta,
            on="case_id",
            how="left",
            validate="one_to_one",
        )
        .copy()
    )

    # ------------------------------------------------------------------
    # Build plotting dimensions.
    # ------------------------------------------------------------------
    data["start_date"] = pd.to_datetime(data["start_date"])
    data["end_date"] = pd.to_datetime(data["end_date"])
    data["horizon_start_year"] = data["start_date"].dt.year

    data["k_periods"] = data["k_periods"].astype(int)
    data["lambda_soc"] = data["lambda_soc"].astype(float)

    method_conditions = [
        data["cluster_method"].eq("hierarchical")
        & data["representation_method"].eq("medoid"),

        data["cluster_method"].eq("kmeans")
        & data["representation_method"].eq("medoid"),

        data["cluster_method"].eq("hierarchical")
        & data["representation_method"].eq("distribution_minmax"),
    ]

    data["method"] = np.select(
        method_conditions,
        METHOD_ORDER,
        default="unsupported",
    )

    unsupported = data.loc[
        data["method"].eq("unsupported"),
        [
            "experiment_name",
            "cluster_method",
            "representation_method",
        ],
    ]

    if not unsupported.empty:
        raise ValueError(
            "Found TSA combinations not represented in METHOD_ORDER:\n"
            f"{unsupported.drop_duplicates().to_string(index=False)}"
        )

    # ------------------------------------------------------------------
    # Derived plotting metrics.
    #
    # These three error quantities are stored as fractions, so convert
    # them to percentages for plotting.
    # ------------------------------------------------------------------
    data["ldes_capacity_error_pct"] = (
        100.0 * data["ldes_capacity_error_signed"]
    )

    data["macme_capex_weighted_pct"] = (
        100.0 * data["macme_capex_weighted_annualised"]
    )

    data["soc_delta_nrmse_pct"] = (
        100.0 * data["soc_delta_nrmse"]
    )

    return data


def validate_experiment(data: pd.DataFrame) -> None:
    """Print and validate the experiment dimensions."""

    countries = sorted(data["country"].dropna().unique())
    horizons = sorted(data["horizon_start_year"].dropna().unique())
    ks = sorted(data["k_periods"].dropna().unique())
    weights = sorted(data["lambda_soc"].dropna().unique())
    methods = list(dict.fromkeys(data["method"]))

    print("TSA sensitivity results")
    print("=======================")
    print(f"Cases:     {len(data)}")
    print(f"Countries: {countries}")
    print(f"Horizons:  {horizons}")
    print(f"k:         {ks}")
    print(f"W_P:       {weights}")
    print(f"Methods:   {methods}")

    expected_methods = set(METHOD_ORDER)
    if set(methods) != expected_methods:
        raise ValueError(
            "Unexpected TSA method set.\n"
            f"Expected: {sorted(expected_methods)}\n"
            f"Found:    {sorted(methods)}"
        )

    if weights != WP_ORDER:
        raise ValueError(
            f"Expected proxy weights {WP_ORDER}, found {weights}."
        )

    # Check whether any of the four plotted quantities are absent.
    plot_columns = [metric[0] for metric in METRICS]
    missing_counts = data[plot_columns].isna().sum()

    if missing_counts.any():
        print("\nMissing plotting values:")
        print(missing_counts[missing_counts.gt(0)].to_string())


def build_k_colours(k_values: list[int]) -> dict[int, tuple]:
    """Sample ordered discrete colours from plasma."""

    cmap = plt.get_cmap("plasma")

    # Avoid the very darkest and brightest ends for readability.
    positions = np.linspace(0.12, 0.88, len(k_values))

    return {
        k: cmap(position)
        for k, position in zip(k_values, positions, strict=True)
    }


def build_replicate_jitter(
    data: pd.DataFrame,
    *,
    width: float = 0.075,
) -> dict[tuple[str, int], float]:
    """Give country-horizon replicates small deterministic x offsets.

    Country is deliberately not represented as a visual dimension; this
    merely prevents observations from different countries and horizons from
    being drawn directly on top of one another.
    """

    replicates = sorted(
        {
            (str(country), int(year))
            for country, year in zip(
                data["country"],
                data["horizon_start_year"],
                strict=True,
            )
        }
    )

    if len(replicates) == 1:
        return {replicates[0]: 0.0}

    offsets = np.linspace(-width / 2, width / 2, len(replicates))

    return {
        replicate: offset
        for replicate, offset in zip(replicates, offsets, strict=True)
    }


def plot_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    ylabel: str,
    k_colours: dict[int, tuple],
    replicate_jitter: dict[tuple[str, int], float],
) -> None:
    """Plot one sensitivity metric."""

    method_positions = {
        method: index
        for index, method in enumerate(METHOD_ORDER)
    }

    # Raw weather-horizon observations.
    for method in METHOD_ORDER:
        method_data = data.loc[data["method"].eq(method)]
        method_x = method_positions[method]

        for wp in WP_ORDER:
            wp_data = method_data.loc[
                np.isclose(method_data["lambda_soc"], wp)
            ]

            centre_x = method_x + WP_OFFSETS[wp]

            for k, colour in k_colours.items():
                subset = wp_data.loc[
                    wp_data["k_periods"].eq(k)
                ].copy()

                subset = subset.dropna(subset=[value_column])

                if subset.empty:
                    continue

                x = np.array(
                    [
                        centre_x
                        + replicate_jitter[(str(country), int(year))]
                        for country, year in zip(
                            subset["country"],
                            subset["horizon_start_year"],
                            strict=True,
                        )
                    ]
                )
                y = subset[value_column].to_numpy(dtype=float)

                ax.scatter(
                    x,
                    y,
                    s=28,
                    color=colour,
                    alpha=0.62,
                    linewidths=0,
                    zorder=2,
                )

                # Median across weather horizons for this (method, W_P, k).
                median = float(np.median(y))

                ax.hlines(
                    median,
                    centre_x - 0.030,
                    centre_x + 0.030,
                    color=colour,
                    linewidth=2.2,
                    zorder=3,
                )

    # Separate the three TSA-method groups visually.
    for boundary in [0.5, 1.5]:
        ax.axvline(
            boundary,
            color="0.85",
            linewidth=0.8,
            zorder=0,
        )

    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(
        [METHOD_LABELS[method] for method in METHOD_ORDER]
    )

    ax.set_xlim(-0.48, len(METHOD_ORDER) - 0.52)

    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)

    ax.grid(
        axis="y",
        linewidth=0.6,
        alpha=0.25,
    )

    # Signed capacity error needs a clear zero-reference line.
    if value_column == "ldes_capacity_error_pct":
        ax.axhline(
            0.0,
            color="0.25",
            linewidth=0.9,
            linestyle="--",
            zorder=1,
        )


def make_figure(data: pd.DataFrame) -> plt.Figure:
    k_values = sorted(data["k_periods"].unique())
    k_colours = build_k_colours(k_values)
    replicate_jitter = build_replicate_jitter(data)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.2),
        constrained_layout=False,
    )

    axes = axes.ravel()

    for ax, (column, title, ylabel) in zip(
        axes,
        METRICS,
        strict=True,
    ):
        plot_metric(
            ax,
            data,
            value_column=column,
            title=title,
            ylabel=ylabel,
            k_colours=k_colours,
            replicate_jitter=replicate_jitter,
        )

    fig.suptitle(
        "TSA-method sensitivity",
        y=0.985,
        fontsize=14,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=k_colours[k],
            markeredgecolor="none",
            markersize=7,
            label=f"k = {k}",
        )
        for k in k_values
    ]

    fig.legend(
        handles=legend_handles,
        title="Representative periods",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(k_values),
        frameon=False,
    )

    fig.text(
        0.5,
        0.018,
        r"Within each TSA method: $W_P$ = 0 (left), 0.5 (centre), 1 (right)",
        ha="center",
        va="bottom",
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.13,
        top=0.875,
        wspace=0.25,
        hspace=0.34,
    )

    return fig


def main() -> None:
    args = parse_args()

    data = load_results(args.results_dir)
    validate_experiment(data)

    fig = make_figure(data)

    output = (
        args.output
        if args.output is not None
        else args.results_dir / "tsa_methods_sensitivity.png"
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    # Also save a vector copy beside the PNG.
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(
        pdf_output,
        bbox_inches="tight",
    )

    print(f"\nSaved: {output}")
    print(f"Saved: {pdf_output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()