"""Plot TSA-method sensitivity experiment results.

This inspection figure assesses whether the SoC proxy improves outcomes
consistently across TSA methods and helps identify a sensible base TSA method.

Panels
------
(a) Signed LDES capacity error
(b) Annualised-CAPEX-weighted MACME
(c) Clustered-v-reference CEM SoC-delta nRMSE
(d) TSA runtime

Visual encoding
---------------
- Main x position: TSA method
- Broad within-method offset: proxy weight W_P = 0, 0.5, 1
- Narrow within-weight offset + colour: representative-period count k
- Tiny jitter: country-weather-horizon replicate
- Short horizontal bar: median across country-weather-horizon replicates

Countries are deliberately pooled rather than visually differentiated.
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
    "hierarchical_distribution_local",
    "hierarchical_distribution_global",
    "hierarchical_distribution_minmax_local",
    "hierarchical_distribution_minmax_global",
]

METHOD_LABELS = {
    "hierarchical_medoid": "Hierarchical\n+ medoid",
    "kmeans_medoid": "k-means\n+ medoid",
    "hierarchical_distribution_local": ("Hierarchical\n+ distribution\n(local)"),
    "hierarchical_distribution_global": ("Hierarchical\n+ distribution\n(global)"),
    "hierarchical_distribution_minmax_local": (
        "Hierarchical\n+ distribution\nminmax\n(local)"
    ),
    "hierarchical_distribution_minmax_global": (
        "Hierarchical\n+ distribution\nminmax\n(global)"
    ),
}

WP_ORDER = [0.0, 0.5, 1.0]

# Broad sub-columns inside each TSA-method block.
WP_OFFSETS = {
    0.0: -0.285,
    0.5: 0.0,
    1.0: 0.285,
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
    parser = argparse.ArgumentParser(description="Plot TSA-method sensitivity results.")
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
            "Output image path. Defaults to <results-dir>/tsa_methods_sensitivity.png."
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


def pivot_investment_metrics(investment: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form investment metrics to one row per case."""

    require_columns(
        investment,
        {"case_id", "metric", "value"},
        table_name="investment_metrics.parquet",
    )

    duplicated = investment.duplicated(
        subset=["case_id", "metric"],
        keep=False,
    )
    if duplicated.any():
        duplicates = investment.loc[
            duplicated,
            ["case_id", "metric"],
        ].sort_values(["case_id", "metric"])

        raise ValueError(
            "investment_metrics.parquet contains duplicate "
            "(case_id, metric) rows:\n"
            f"{duplicates.head(30).to_string(index=False)}"
        )

    wide = investment.pivot(
        index="case_id",
        columns="metric",
        values="value",
    ).reset_index()
    wide.columns.name = None

    require_columns(
        wide,
        {
            "case_id",
            "ldes_capacity_error_signed",
            "macme_capex_weighted_annualised",
        },
        table_name="pivoted investment_metrics.parquet",
    )

    return wide


def extract_soc_delta_nrmse(signals: pd.DataFrame) -> pd.DataFrame:
    """Extract one full-horizon CEM SoC-delta nRMSE value per case."""

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

    selected = signals.loc[
        signals["error_family"].eq("cem")
        & signals["signal_type"].eq("delta")
        & signals["metric"].eq("nrmse")
        & signals["normalisation_basis"].eq("reference_proxy_full_range")
        & signals["window_half_width_days"].isna()
    ].copy()

    # Use this additional flag when it exists in the schema.
    if "covers_full_horizon" in selected.columns:
        selected = selected.loc[selected["covers_full_horizon"].fillna(False)]

    duplicated = selected["case_id"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(selected.loc[duplicated, "case_id"].astype(str).unique())
        raise ValueError(
            "Expected one full-horizon CEM delta nRMSE row per case, "
            "but found duplicates for:\n" + "\n".join(duplicate_ids[:20])
        )

    return selected[["case_id", "value"]].rename(columns={"value": "soc_delta_nrmse"})


def infer_representation_scope(data: pd.DataFrame) -> pd.Series:
    """Return local/global representation scope.

    Prefer an explicit recorded scope column. For older results that do not
    record scope, infer it from experiment names. Existing historical
    distribution-minmax cases without an explicit ``_local_`` token are
    treated as global, matching the original experiment configuration.
    """

    for column in ("representation_scope", "scope"):
        if column in data.columns:
            return data[column].fillna("").astype(str).str.lower()

    representation = data["representation_method"].astype(str)
    experiment = data["experiment_name"].astype(str)

    scope = pd.Series("", index=data.index, dtype="object")
    distribution_mask = representation.isin(["distribution", "distribution_minmax"])

    scope.loc[distribution_mask & experiment.str.contains("_local_", regex=False)] = (
        "local"
    )

    scope.loc[distribution_mask & ~experiment.str.contains("_local_", regex=False)] = (
        "global"
    )

    return scope


def build_method_column(data: pd.DataFrame) -> pd.Series:
    """Build plot method labels from clustering and representation settings."""

    cluster = data["cluster_method"].astype(str).str.lower()
    representation = data["representation_method"].astype(str).str.lower()
    scope = infer_representation_scope(data)

    method = pd.Series("unsupported", index=data.index, dtype="object")

    method.loc[cluster.eq("hierarchical") & representation.eq("medoid")] = (
        "hierarchical_medoid"
    )

    method.loc[cluster.eq("kmeans") & representation.eq("medoid")] = "kmeans_medoid"

    method.loc[
        cluster.eq("hierarchical")
        & representation.eq("distribution")
        & scope.eq("local")
    ] = "hierarchical_distribution_local"

    method.loc[
        cluster.eq("hierarchical")
        & representation.eq("distribution")
        & scope.eq("global")
    ] = "hierarchical_distribution_global"

    method.loc[
        cluster.eq("hierarchical")
        & representation.eq("distribution_minmax")
        & scope.eq("local")
    ] = "hierarchical_distribution_minmax_local"

    method.loc[
        cluster.eq("hierarchical")
        & representation.eq("distribution_minmax")
        & scope.eq("global")
    ] = "hierarchical_distribution_minmax_global"

    return method


def load_results(results_dir: Path) -> pd.DataFrame:
    parameters_path = results_dir / "parameters.parquet"
    investment_path = results_dir / "investment_metrics.parquet"
    signal_path = results_dir / "signal_metrics.parquet"

    parameters = pd.read_parquet(parameters_path)
    investment = pivot_investment_metrics(pd.read_parquet(investment_path))
    soc_delta = extract_soc_delta_nrmse(pd.read_parquet(signal_path))

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

    data["start_date"] = pd.to_datetime(data["start_date"])
    data["end_date"] = pd.to_datetime(data["end_date"])
    data["horizon_start_year"] = data["start_date"].dt.year.astype(int)

    data["k_periods"] = data["k_periods"].astype(int)
    data["lambda_soc"] = data["lambda_soc"].astype(float)

    data["method"] = build_method_column(data)

    unsupported_columns = [
        column
        for column in [
            "experiment_name",
            "cluster_method",
            "representation_method",
            "representation_scope",
            "scope",
        ]
        if column in data.columns
    ]
    unsupported = data.loc[
        data["method"].eq("unsupported"),
        unsupported_columns,
    ].drop_duplicates()

    if not unsupported.empty:
        raise ValueError(
            "Found unsupported TSA method combinations:\n"
            f"{unsupported.to_string(index=False)}"
        )

    # Stored as fractions; convert to percentage-point values for display.
    data["ldes_capacity_error_pct"] = 100.0 * data["ldes_capacity_error_signed"]
    data["macme_capex_weighted_pct"] = 100.0 * data["macme_capex_weighted_annualised"]
    data["soc_delta_nrmse_pct"] = 100.0 * data["soc_delta_nrmse"]

    return data


def validate_experiment(data: pd.DataFrame) -> None:
    """Print experiment dimensions and validate expected plotting inputs."""

    countries = sorted(str(x) for x in data["country"].dropna().unique())
    horizons = sorted(int(x) for x in data["horizon_start_year"].dropna().unique())
    ks = sorted(int(x) for x in data["k_periods"].dropna().unique())
    weights = sorted(float(x) for x in data["lambda_soc"].dropna().unique())
    methods = [method for method in METHOD_ORDER if method in set(data["method"])]

    print("TSA sensitivity results")
    print("=======================")
    print(f"Cases:     {len(data)}")
    print(f"Countries: {countries}")
    print(f"Horizons:  {horizons}")
    print(f"k:         {ks}")
    print(f"W_P:       {weights}")
    print(f"Methods:   {methods}")

    unknown_methods = set(data["method"]) - set(METHOD_ORDER)
    if unknown_methods:
        raise ValueError(f"Unexpected TSA methods: {sorted(unknown_methods)}")

    if weights != WP_ORDER:
        raise ValueError(f"Expected proxy weights {WP_ORDER}, found {weights}.")

    plot_columns = [metric[0] for metric in METRICS]
    missing_counts = data[plot_columns].isna().sum()

    if missing_counts.any():
        print("\nMissing plotting values:")
        print(missing_counts[missing_counts.gt(0)].to_string())


def build_k_colours(k_values: list[int]) -> dict[int, tuple]:
    """Sample ordered discrete colours from plasma."""

    cmap = plt.get_cmap("plasma")
    positions = np.linspace(0.12, 0.88, len(k_values))

    return {k: cmap(position) for k, position in zip(k_values, positions, strict=True)}


def build_k_offsets(
    k_values: list[int],
    *,
    width: float = 0.125,
) -> dict[int, float]:
    """Give each k value its own narrow sub-column."""

    if len(k_values) == 1:
        return {k_values[0]: 0.0}

    offsets = np.linspace(-width / 2, width / 2, len(k_values))
    return dict(zip(k_values, offsets, strict=True))


def build_replicate_jitter(
    data: pd.DataFrame,
    *,
    width: float = 0.015,
) -> dict[tuple[str, int], float]:
    """Give country-horizon replicates a tiny deterministic x jitter."""

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

    offsets = np.linspace(
        -width / 2,
        width / 2,
        len(replicates),
    )

    return dict(zip(replicates, offsets, strict=True))


def build_method_positions(
    active_methods: list[str],
) -> dict[str, float]:
    """Position methods, with an extra gap after the two medoid methods."""

    positions: dict[str, float] = {}
    x = 0.0

    for index, method in enumerate(active_methods):
        positions[method] = x

        # Visually separate clustering-method comparison from the
        # distribution-representation family when both are present.
        if index == 1 and len(active_methods) > 2:
            x += 1.20
        else:
            x += 1.0

    return positions


def plot_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    ylabel: str,
    k_colours: dict[int, tuple],
    k_offsets: dict[int, float],
    replicate_jitter: dict[tuple[str, int], float],
    active_methods: list[str],
) -> None:
    """Plot one TSA sensitivity metric."""

    method_positions = build_method_positions(active_methods)

    for method in active_methods:
        method_data = data.loc[data["method"].eq(method)]
        method_x = method_positions[method]

        for wp in WP_ORDER:
            wp_data = method_data.loc[np.isclose(method_data["lambda_soc"], wp)]
            wp_centre = method_x + WP_OFFSETS[wp]

            for k, colour in k_colours.items():
                subset = wp_data.loc[wp_data["k_periods"].eq(k)].dropna(
                    subset=[value_column]
                )

                if subset.empty:
                    continue

                k_centre = wp_centre + k_offsets[k]

                xs = np.array(
                    [
                        k_centre + replicate_jitter[(str(country), int(year))]
                        for country, year in zip(
                            subset["country"],
                            subset["horizon_start_year"],
                            strict=True,
                        )
                    ]
                )
                ys = subset[value_column].to_numpy(dtype=float)

                ax.scatter(
                    xs,
                    ys,
                    s=24,
                    color=colour,
                    alpha=0.58,
                    linewidths=0,
                    zorder=2,
                )

                median = float(np.median(ys))
                median_half_width = 0.027

                # White halo separates the summary statistic from the raw observations.
                ax.hlines(
                    median,
                    k_centre - median_half_width,
                    k_centre + median_half_width,
                    color="white",
                    linewidth=3,
                    zorder=4,
                )

                # Neutral dark bar: position identifies k; colour remains reserved
                # for the individual k observations.
                ax.hlines(
                    median,
                    k_centre - median_half_width,
                    k_centre + median_half_width,
                    color="0.12",
                    linewidth=2,
                    zorder=5,
                )

    positions = [method_positions[m] for m in active_methods]

    # Subtle method boundaries.
    for left, right in zip(positions[:-1], positions[1:], strict=True):
        ax.axvline(
            (left + right) / 2,
            color="0.88",
            linewidth=0.7,
            zorder=0,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [METHOD_LABELS[method] for method in active_methods],
        fontsize=8.5,
    )

    ax.set_xlim(
        min(positions) - 0.47,
        max(positions) + 0.47,
    )

    ax.set_title(
        title,
        loc="left",
        fontsize=11,
        pad=8,
    )
    ax.set_ylabel(ylabel)

    ax.grid(
        axis="y",
        linewidth=0.6,
        alpha=0.25,
        zorder=0,
    )

    if value_column == "ldes_capacity_error_pct":
        ax.axhline(
            0.0,
            color="0.25",
            linewidth=0.9,
            linestyle="--",
            zorder=1,
        )


def make_figure(data: pd.DataFrame) -> plt.Figure:
    k_values = sorted(int(k) for k in data["k_periods"].unique())
    active_methods = [
        method for method in METHOD_ORDER if method in set(data["method"])
    ]

    if not active_methods:
        raise ValueError("No supported TSA methods were found to plot.")

    k_colours = build_k_colours(k_values)
    k_offsets = build_k_offsets(k_values)
    replicate_jitter = build_replicate_jitter(data)

    # Wide enough for all six method blocks; still works with the current
    # partial dataset because only methods present in the data are displayed.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17.5, 9.7),
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
            k_offsets=k_offsets,
            replicate_jitter=replicate_jitter,
            active_methods=active_methods,
        )

    # ------------------------------------------------------------------
    # Header area: title, legend, and explanation all sit above plots.
    # ------------------------------------------------------------------
    fig.suptitle(
        "TSA-method sensitivity",
        y=0.987,
        fontsize=15,
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

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="0.12",
            linewidth=2.5,
            label="Median",
        )
    )

    fig.legend(
        handles=legend_handles,
        title="Representative periods",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.946),
        ncol=len(k_values) + 1,
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.6,
    )

    fig.text(
        0.5,
        0.878,
        (
            r"Within each method: $W_P = 0,\ 0.5,\ 1$ from left to right; "
            r"within each $W_P$: $k = 20,\ 40,\ 60,\ 90$ from left to right. "
            "Points are country–weather-horizon cases."
        ),
        ha="center",
        va="center",
        fontsize=9.3,
        color="0.30",
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.992,
        bottom=0.145,
        top=0.805,
        wspace=0.20,
        hspace=0.42,
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
