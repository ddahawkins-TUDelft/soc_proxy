"""Plot 10-year k-means + medoid sensitivity results.

Filters the consolidated parquet outputs to k-means clustering with medoid
representation, then plots representative-period count k on the x-axis.

Panels
------
(a) Signed LDES capacity error
(b) Annualised-CAPEX-weighted MACME
(c) Clustered-v-reference CEM SoC-delta nRMSE
(d) Workflow runtime (proxy+tsa+calliope)

Visual encoding
---------------
- Main x position: representative-period count k
- Within-k offset + colour: proxy weight W_P
- W_P = 0: light grey (no-proxy reference)
- W_P > 0: ordered colours sampled from plasma
- Tiny deterministic jitter: weather-horizon replicate
- Short horizontal bar: median across displayed replicates

Country filtering
-----------------
--country NL    Netherlands only
--country BE    Belgium only
--country both  Pool NL and BE without visually distinguishing them
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


WP_ORDER = [0.0, 0.25, 0.5, 0.75, 1.0]

METRICS = [
    ("ldes_capacity_error_pct", "(a) LDES capacity error", "Capacity error (%)"),
    ("macme_capex_weighted_pct", "(b) CAPEX-weighted MACME", "Weighted MACME (%)"),
    # ("soc_delta_nrmse_pct", "(c) CEM SoC-delta error", "SoC delta nRMSE (%)"),
    ("ldes_abs_error_improvement_pp","(c) Improvement in LDES capacity error","Reduction in absolute capacity error (pp)"),
    ("runtime_method_seconds", "(d) Workflow runtime", "Runtime (s)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 10-year k-means + medoid sensitivity results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/10_year"),
        help="Directory containing consolidated result parquet files.",
    )
    parser.add_argument(
        "--country",
        choices=["NL", "BE", "both"],
        default="both",
        help=(
            "Country subset to plot. 'both' pools NL and BE without "
            "visually distinguishing them."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. Defaults to "
            "<results-dir>/kmeans_medoid_sensitivity_<country>.png."
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
            duplicated, ["case_id", "metric"]
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

    if "covers_full_horizon" in selected.columns:
        selected = selected.loc[selected["covers_full_horizon"].fillna(False)]

    duplicated = selected["case_id"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(
            selected.loc[duplicated, "case_id"].astype(str).unique()
        )
        raise ValueError(
            "Expected one full-horizon CEM delta nRMSE row per case, "
            "but found duplicates for:\n" + "\n".join(duplicate_ids[:20])
        )

    return selected[["case_id", "value"]].rename(
        columns={"value": "soc_delta_nrmse"}
    )


def load_results(results_dir: Path, country: str) -> pd.DataFrame:
    """Load, merge, filter, and prepare the 10-year experiment results."""
    parameters_path = results_dir / "parameters.parquet"
    investment_path = results_dir / "investment_metrics.parquet"
    signal_path = results_dir / "signal_metrics.parquet"

    parameters = pd.read_parquet(parameters_path)

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
            "runtime_method_seconds",
        },
        table_name="parameters.parquet",
    )

    # Results/10_year may contain historical or side-experiment methods.
    # This figure deliberately isolates the selected paper method.
    method_mask = (
        parameters["cluster_method"].astype(str).str.lower().eq("kmeans")
        & parameters["representation_method"].astype(str).str.lower().eq("medoid")
    )
    parameters = parameters.loc[method_mask].copy()

    if parameters.empty:
        raise ValueError(
            "No k-means + medoid cases were found in parameters.parquet."
        )

    if country != "both":
        parameters = parameters.loc[
            parameters["country"].astype(str).eq(country)
        ].copy()

    if parameters.empty:
        raise ValueError(
            f"No k-means + medoid cases remain after filtering country={country!r}."
        )

    case_ids = set(parameters["case_id"].astype(str))

    investment_raw = pd.read_parquet(investment_path)
    investment_raw = investment_raw.loc[
        investment_raw["case_id"].astype(str).isin(case_ids)
    ].copy()
    investment = pivot_investment_metrics(investment_raw)

    signals_raw = pd.read_parquet(signal_path)
    signals_raw = signals_raw.loc[
        signals_raw["case_id"].astype(str).isin(case_ids)
    ].copy()
    soc_delta = extract_soc_delta_nrmse(signals_raw)

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

    data["ldes_capacity_error_pct"] = (
        100.0 * data["ldes_capacity_error_signed"]
    )
    data["macme_capex_weighted_pct"] = (
        100.0 * data["macme_capex_weighted_annualised"]
    )
    data["soc_delta_nrmse_pct"] = 100.0 * data["soc_delta_nrmse"]

    # Pair each proxy-weight case with its otherwise-identical W_P = 0 case.
    pair_columns = [
        "country",
        "start_date",
        "end_date",
        "k_periods",
    ]

    baseline = (
        data.loc[
            np.isclose(data["lambda_soc"], 0.0),
            pair_columns + ["ldes_capacity_error_pct"],
        ]
        .rename(
            columns={
                "ldes_capacity_error_pct":
                    "ldes_capacity_error_baseline_pct"
            }
        )
    )

    data = data.merge(
        baseline,
        on=pair_columns,
        how="left",
        validate="many_to_one",
    )

    # Positive = closer to zero than W_P = 0.
    # Negative = further from zero than W_P = 0.
    data["ldes_abs_error_improvement_pp"] = (
        data["ldes_capacity_error_baseline_pct"].abs()
        - data["ldes_capacity_error_pct"].abs()
    )

    # Optional diagnostic: did the investment error cross zero?
    data["ldes_crossed_reference"] = (
        ~np.isclose(data["lambda_soc"], 0.0)
        & ~np.isclose(data["ldes_capacity_error_baseline_pct"], 0.0)
        & (
            np.sign(data["ldes_capacity_error_baseline_pct"])
            != np.sign(data["ldes_capacity_error_pct"])
        )
    )

    return data


def validate_experiment(data: pd.DataFrame) -> None:
    """Print dimensions and validate plotting inputs."""
    countries = sorted(str(x) for x in data["country"].dropna().unique())
    horizons = sorted(
        int(x) for x in data["horizon_start_year"].dropna().unique()
    )
    ks = sorted(int(x) for x in data["k_periods"].dropna().unique())
    weights = sorted(float(x) for x in data["lambda_soc"].dropna().unique())

    print("10-year k-means + medoid sensitivity results")
    print("===========================================")
    print(f"Cases:     {len(data)}")
    print(f"Countries: {countries}")
    print(f"Horizons:  {horizons}")
    print(f"k:         {ks}")
    print(f"W_P:       {weights}")

    unknown_weights = [
        value
        for value in weights
        if not any(np.isclose(value, expected) for expected in WP_ORDER)
    ]
    if unknown_weights:
        raise ValueError(
            f"Unexpected proxy weights {unknown_weights}; "
            f"expected subset of {WP_ORDER}."
        )

    plot_columns = [metric[0] for metric in METRICS]
    missing_counts = data[plot_columns].isna().sum()
    if missing_counts.any():
        print("\nMissing plotting values:")
        print(missing_counts[missing_counts.gt(0)].to_string())


def build_wp_colours(weights: list[float]) -> dict[float, object]:
    """Use light grey for W_P=0 and ordered plasma colours otherwise."""
    colours: dict[float, object] = {}

    if any(np.isclose(weight, 0.0) for weight in weights):
        colours[0.0] = "0.72"

    positive = sorted(
        weight for weight in weights if not np.isclose(weight, 0.0)
    )
    if positive:
        cmap = plt.get_cmap("plasma")
        positions = np.linspace(0.18, 0.88, len(positive))
        for weight, position in zip(positive, positions, strict=True):
            colours[weight] = cmap(position)

    return colours


def build_wp_offsets(
    weights: list[float],
    *,
    width: float = 0.46,
) -> dict[float, float]:
    """Give each proxy weight a stable sub-position within one k category."""
    if len(weights) == 1:
        return {weights[0]: 0.0}

    offsets = np.linspace(-width / 2, width / 2, len(weights))
    return dict(zip(weights, offsets, strict=True))


def build_replicate_jitter(
    data: pd.DataFrame,
    *,
    width: float = 0.055,
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

    offsets = np.linspace(-width / 2, width / 2, len(replicates))
    return dict(zip(replicates, offsets, strict=True))


def plot_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    ylabel: str,
    k_values: list[int],
    wp_values: list[float],
    wp_colours: dict[float, object],
    wp_offsets: dict[float, float],
    replicate_jitter: dict[tuple[str, int], float],
) -> None:
    """Plot one metric against representative-period count k."""
    k_positions = {k: float(index) for index, k in enumerate(k_values)}

    for k in k_values:
        k_data = data.loc[data["k_periods"].eq(k)]
        k_x = k_positions[k]

        for wp in wp_values:
            wp_data = k_data.loc[
                np.isclose(k_data["lambda_soc"], wp)
            ].dropna(subset=[value_column])

            if wp_data.empty:
                continue

            centre = k_x + wp_offsets[wp]

            xs = np.array(
                [
                    centre
                    + replicate_jitter[(str(country), int(year))]
                    for country, year in zip(
                        wp_data["country"],
                        wp_data["horizon_start_year"],
                        strict=True,
                    )
                ]
            )
            ys = wp_data[value_column].to_numpy(dtype=float)

            ax.scatter(
                xs,
                ys,
                s=28,
                color=wp_colours[wp],
                alpha=0.78 if np.isclose(wp, 0.0) else 0.62,
                linewidths=0,
                zorder=2,
            )

            median = float(np.median(ys))
            median_half_width = 0.045

            ax.hlines(
                median,
                centre - median_half_width,
                centre + median_half_width,
                color="white",
                linewidth=3.2,
                zorder=4,
            )
            ax.hlines(
                median,
                centre - median_half_width,
                centre + median_half_width,
                color="0.12",
                linewidth=2.0,
                zorder=5,
            )

    positions = [k_positions[k] for k in k_values]

    for left, right in zip(positions[:-1], positions[1:], strict=True):
        ax.axvline(
            (left + right) / 2,
            color="0.90",
            linewidth=0.7,
            zorder=0,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlim(min(positions) - 0.52, max(positions) + 0.52)
    ax.set_xlabel("Representative periods, k")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, pad=8)
    ax.grid(axis="y", linewidth=0.6, alpha=0.25, zorder=0)

    if value_column == "ldes_capacity_error_pct":
        ax.axhline(
            0.0,
            color="0.25",
            linewidth=0.9,
            linestyle="--",
            zorder=1,
        )
        ax.axhline(
            10.0,
            color="0.30",
            linewidth=0.8,
            linestyle=":",
            zorder=1,
        )
        ax.axhline(
            -10.0,
            color="0.30",
            linewidth=0.8,
            linestyle=":",
            zorder=1,
        )


def make_figure(data: pd.DataFrame, *, country_selection: str) -> plt.Figure:
    k_values = sorted(int(k) for k in data["k_periods"].unique())
    wp_values = [
        weight
        for weight in WP_ORDER
        if any(np.isclose(data["lambda_soc"], weight))
    ]

    if not k_values:
        raise ValueError("No representative-period counts were found to plot.")
    if not wp_values:
        raise ValueError("No supported proxy weights were found to plot.")

    wp_colours = build_wp_colours(wp_values)
    wp_offsets = build_wp_offsets(wp_values)
    replicate_jitter = build_replicate_jitter(data)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 9.5),
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
            k_values=k_values,
            wp_values=wp_values,
            wp_colours=wp_colours,
            wp_offsets=wp_offsets,
            replicate_jitter=replicate_jitter,
        )

    selection_label = {
        "NL": "Netherlands",
        "BE": "Belgium",
        "both": "Netherlands + Belgium (pooled)",
    }[country_selection]

    fig.suptitle(
        f"10-year sensitivity — k-means + medoid\n{selection_label}",
        y=0.985,
        fontsize=15,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=wp_colours[wp],
            markeredgecolor="none",
            markersize=7,
            label=rf"$W_P$ = {wp:g}",
        )
        for wp in wp_values
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
        title="Proxy weight",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(wp_values) + 1,
        frameon=False,
        columnspacing=1.7,
        handletextpad=0.6,
    )

    if country_selection == "both":
        replicate_text = (
            "Points are country–weather-horizon cases; NL and BE are pooled "
            "without separate visual encoding."
        )
    else:
        replicate_text = "Points are weather-horizon cases."

    fig.text(
        0.5,
        0.855,
        (
            r"Within each $k$, points are offset by proxy weight $W_P$ "
            "from left to right. "
            + replicate_text
            + r" Dotted lines in (a) mark $\pm 10\%$."
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color="0.30",
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.09,
        top=0.81,
        wspace=0.20,
        hspace=0.34,
    )

    return fig


def main() -> None:
    args = parse_args()

    data = load_results(
        args.results_dir,
        country=args.country,
    )
    validate_experiment(data)

    fig = make_figure(
        data,
        country_selection=args.country,
    )

    country_tag = args.country.lower()
    output = (
        args.output
        if args.output is not None
        else args.results_dir / f"kmeans_medoid_sensitivity_{country_tag}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=300, bbox_inches="tight")

    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight")

    print(f"\nSaved: {output}")
    print(f"Saved: {pdf_output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
