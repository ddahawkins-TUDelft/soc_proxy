"""Create Figure 5: improvement in LDES capacity error with the SoC proxy.

For each selected TSAM method, the figure compares the W_P = 0.5 case with
its otherwise-identical W_P = 0 (original TSA) case.

Improvement is defined as:

    |LDES capacity error at W_P = 0|
    - |LDES capacity error at W_P = 0.5|

and is reported in percentage points.

Therefore:
- positive values mean the SoC proxy moves LDES investment closer to the
  full-resolution reference;
- negative values mean the SoC proxy moves it further away;
- zero means no change in absolute LDES capacity error.

The figure deliberately pools k rather than giving it separate visual
channels. To keep the comparison fair across TSAM methods, only the
representative-period counts tested for every method are retained:
    k = 20, 40, 60, 90

The selected method channels are:
- hierarchical + medoid
- k-means + medoid
- hierarchical + local distribution
- hierarchical + local distribution + min/max

Source:
    results/2_5_10_year/*.parquet

Outputs:
    results/figures/fig_LDES_cap_improvement/
        fig_LDES_cap_improvement.png
        fig_LDES_cap_improvement.pdf
        fig_LDES_cap_improvement_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


# ---------------------------------------------------------------------------
# Figure-specific configuration
# ---------------------------------------------------------------------------

METRIC_NAME = "ldes_capacity_error_signed"

DEFAULT_SOURCE_DIR = Path("results/2_5_10_year")
DEFAULT_OUTPUT_DIR = Path("results/figures/fig_LDES_cap_improvement")
OUTPUT_STEM = "fig_LDES_cap_improvement"

DEFAULT_WIDTH_PX = 1800
DEFAULT_HEIGHT_PX = 1200
DEFAULT_DPI = 300

TARGET_HORIZON_YEARS = 10.0
HORIZON_TOLERANCE_YEARS = 0.1

TARGET_PROXY_WEIGHT = 0.5
BASELINE_PROXY_WEIGHT = 0.0
COMMON_K_VALUES = (20, 40, 60, 90)

# Keep the same colour semantics as the other paper figures:
# W_P is mapped onto the lower 90% of plasma.
PLASMA_MIN = 0.0
PLASMA_MAX = 0.90

POINT_SIZE = 26
POINT_ALPHA = 0.68
JITTER_WIDTH = 0.22
JITTER_SEED = 42
MEDIAN_HALF_WIDTH = 0.12

Y_TICK_INTERVAL_PP = 10


METHOD_ORDER = [
    "hierarchical_medoid",
    "kmeans_medoid",
    "hierarchical_distribution_local",
    "hierarchical_distribution_minmax_local",
]

METHOD_LABELS = {
    "hierarchical_medoid": "Hierarchical\n+ medoid",
    "kmeans_medoid": "K-means\n+ medoid",
    "hierarchical_distribution_local": "Hierarchical\n+ distribution",
    "hierarchical_distribution_minmax_local": (
        "Hierarchical\n+ distribution\n w/ min/max"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the improvement in absolute LDES capacity error for "
            "W_P=0.5 relative to the original W_P=0 TSA formulation."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=(
            "Directory containing parameters.parquet and investment_metrics.parquet."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which figure and summary files are written.",
    )
    parser.add_argument(
        "--width-px",
        type=int,
        default=DEFAULT_WIDTH_PX,
        help=f"PNG canvas width in pixels (default: {DEFAULT_WIDTH_PX}).",
    )
    parser.add_argument(
        "--height-px",
        type=int,
        default=DEFAULT_HEIGHT_PX,
        help=f"PNG canvas height in pixels (default: {DEFAULT_HEIGHT_PX}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=(
            f"PNG resolution in dots per inch (default: {DEFAULT_DPI}). "
            "Together with the pixel dimensions this also sets the physical "
            "size of the PDF figure."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    table_name: str,
) -> None:
    """Raise a clear error if a required input column is absent."""
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}\n"
            f"Available columns:\n{sorted(frame.columns)}"
        )


def filter_ten_year_runs(parameters: pd.DataFrame) -> pd.DataFrame:
    """Retain cases whose start/end dates span approximately ten years."""
    parameters = parameters.copy()
    parameters["start_date"] = pd.to_datetime(parameters["start_date"])
    parameters["end_date"] = pd.to_datetime(parameters["end_date"])

    parameters["horizon_years"] = (
        parameters["end_date"] - parameters["start_date"]
    ).dt.total_seconds() / (365.2425 * 24 * 60 * 60)

    mask = np.isclose(
        parameters["horizon_years"],
        TARGET_HORIZON_YEARS,
        atol=HORIZON_TOLERANCE_YEARS,
        rtol=0.0,
    )
    filtered = parameters.loc[mask].copy()

    if filtered.empty:
        available = sorted(
            {
                round(float(value), 3)
                for value in parameters["horizon_years"].dropna().unique()
            }
        )
        raise ValueError(
            "No approximately 10-year runs were found after filtering "
            f"start_date/end_date. Available horizon lengths (years): {available}"
        )

    return filtered


def infer_representation_scope(parameters: pd.DataFrame) -> pd.Series:
    """Infer local/global representation scope.

    Prefer an explicit scope column if one exists. The current consolidated
    results encode local/global scope in experiment_name, so that is used as
    the fallback.
    """
    for column in ("representation_scope", "scope"):
        if column in parameters.columns:
            return parameters[column].astype("string").str.lower().fillna("")

    names = parameters["experiment_name"].astype("string").str.lower().fillna("")

    scope = pd.Series("", index=parameters.index, dtype="string")
    scope.loc[names.str.contains("local", regex=False)] = "local"
    scope.loc[names.str.contains("global", regex=False)] = "global"
    return scope


def assign_method_key(parameters: pd.DataFrame) -> pd.Series:
    """Map selected parameter combinations onto the four plotted channels."""
    cluster = parameters["cluster_method"].astype(str).str.lower()
    representation = parameters["representation_method"].astype(str).str.lower()
    scope = infer_representation_scope(parameters)

    method_key = pd.Series(pd.NA, index=parameters.index, dtype="string")

    method_key.loc[cluster.eq("hierarchical") & representation.eq("medoid")] = (
        "hierarchical_medoid"
    )

    method_key.loc[cluster.eq("kmeans") & representation.eq("medoid")] = "kmeans_medoid"

    method_key.loc[
        cluster.eq("hierarchical")
        & representation.eq("distribution")
        & scope.eq("local")
    ] = "hierarchical_distribution_local"

    method_key.loc[
        cluster.eq("hierarchical")
        & representation.eq("distribution_minmax")
        & scope.eq("local")
    ] = "hierarchical_distribution_minmax_local"

    return method_key


def describe_available_methods(parameters: pd.DataFrame) -> str:
    """Return a compact diagnostic description of available method metadata."""
    columns = [
        column
        for column in (
            "cluster_method",
            "representation_method",
            "experiment_name",
        )
        if column in parameters.columns
    ]

    if not columns:
        return "<no method metadata available>"

    return (
        parameters[columns]
        .drop_duplicates()
        .sort_values(columns)
        .head(80)
        .to_string(index=False)
    )


def load_results(source_dir: Path) -> pd.DataFrame:
    """Load, filter, pair, and calculate LDES capacity improvement."""
    parameters_path = source_dir / "parameters.parquet"
    investment_path = source_dir / "investment_metrics.parquet"

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
        },
        table_name="parameters.parquet",
    )

    parameters = filter_ten_year_runs(parameters)

    parameters["case_id"] = parameters["case_id"].astype(str)
    parameters["k_periods"] = parameters["k_periods"].astype(int)
    parameters["lambda_soc"] = parameters["lambda_soc"].astype(float)
    parameters["method_key"] = assign_method_key(parameters)

    # Keep only the four method definitions intended for this comparison.
    selected = parameters.loc[
        parameters["method_key"].isin(METHOD_ORDER)
        & parameters["k_periods"].isin(COMMON_K_VALUES)
        & (
            np.isclose(parameters["lambda_soc"], BASELINE_PROXY_WEIGHT)
            | np.isclose(parameters["lambda_soc"], TARGET_PROXY_WEIGHT)
        )
    ].copy()

    missing_methods = [
        method
        for method in METHOD_ORDER
        if method not in set(selected["method_key"].dropna())
    ]
    if missing_methods:
        raise ValueError(
            "The following required method channels were not found after "
            f"filtering: {missing_methods}\n\n"
            "Available method metadata in the 10-year results:\n"
            f"{describe_available_methods(parameters)}"
        )

    # Validate that all four common k values are present for each method and
    # for both the original-TSA and W_P=0.5 cases.
    validation_rows: list[str] = []
    for method in METHOD_ORDER:
        method_data = selected.loc[selected["method_key"].eq(method)]

        for weight in (BASELINE_PROXY_WEIGHT, TARGET_PROXY_WEIGHT):
            weight_data = method_data.loc[np.isclose(method_data["lambda_soc"], weight)]
            available_k = set(weight_data["k_periods"].unique())
            missing_k = sorted(set(COMMON_K_VALUES) - available_k)

            if missing_k:
                validation_rows.append(
                    f"{method}, W_P={weight:g}: missing k={missing_k}"
                )

    if validation_rows:
        raise ValueError(
            "Not all selected methods contain the common k set for both "
            "W_P=0 and W_P=0.5:\n" + "\n".join(validation_rows)
        )

    case_ids = set(selected["case_id"])

    investment = pd.read_parquet(investment_path)
    require_columns(
        investment,
        {"case_id", "metric", "value"},
        table_name="investment_metrics.parquet",
    )

    investment["case_id"] = investment["case_id"].astype(str)
    investment = investment.loc[
        investment["case_id"].isin(case_ids) & investment["metric"].eq(METRIC_NAME),
        ["case_id", "value"],
    ].copy()

    duplicated = investment["case_id"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(investment.loc[duplicated, "case_id"].unique())
        raise ValueError(
            f"Expected one {METRIC_NAME} value per case, but duplicates "
            "were found for:\n" + "\n".join(duplicate_ids[:30])
        )

    investment = investment.rename(columns={"value": "ldes_capacity_error_signed"})

    data = selected.merge(
        investment,
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    data["ldes_capacity_error_pct"] = 100.0 * data["ldes_capacity_error_signed"]

    missing_metric = data["ldes_capacity_error_pct"].isna()
    if missing_metric.any():
        missing_ids = data.loc[missing_metric, "case_id"].tolist()
        raise ValueError(
            "Missing LDES capacity error values for "
            f"{len(missing_ids)} selected cases. First cases:\n"
            + "\n".join(missing_ids[:20])
        )

    # The development analysis paired cases by country, horizon, and k because
    # it contained only one TSAM method. Here method_key must also be included
    # so that every W_P=0.5 case is compared with its own method's W_P=0 case.
    pair_columns = [
        "method_key",
        "country",
        "start_date",
        "end_date",
        "k_periods",
    ]

    baseline = data.loc[
        np.isclose(data["lambda_soc"], BASELINE_PROXY_WEIGHT),
        pair_columns + ["ldes_capacity_error_pct"],
    ].copy()

    duplicate_baseline = baseline.duplicated(pair_columns, keep=False)
    if duplicate_baseline.any():
        raise ValueError(
            "More than one W_P=0 baseline exists for one or more comparison "
            "keys. These cases need an additional pairing column:\n"
            + baseline.loc[duplicate_baseline, pair_columns]
            .sort_values(pair_columns)
            .head(30)
            .to_string(index=False)
        )

    baseline = baseline.rename(
        columns={"ldes_capacity_error_pct": "ldes_capacity_error_original_pct"}
    )

    proxy = data.loc[np.isclose(data["lambda_soc"], TARGET_PROXY_WEIGHT)].copy()

    duplicate_proxy = proxy.duplicated(pair_columns, keep=False)
    if duplicate_proxy.any():
        raise ValueError(
            "More than one W_P=0.5 case exists for one or more comparison "
            "keys. These cases need an additional pairing column:\n"
            + proxy.loc[duplicate_proxy, pair_columns]
            .sort_values(pair_columns)
            .head(30)
            .to_string(index=False)
        )

    proxy = proxy.merge(
        baseline,
        on=pair_columns,
        how="left",
        validate="one_to_one",
    )

    missing_baseline = proxy["ldes_capacity_error_original_pct"].isna()
    if missing_baseline.any():
        raise ValueError(
            "Some W_P=0.5 cases do not have a matching W_P=0 baseline:\n"
            + proxy.loc[missing_baseline, pair_columns].head(30).to_string(index=False)
        )

    # Positive = W_P=0.5 is closer to zero than original TSA.
    # Negative = W_P=0.5 is further from zero than original TSA.
    proxy["ldes_abs_error_improvement_pp"] = (
        proxy["ldes_capacity_error_original_pct"].abs()
        - proxy["ldes_capacity_error_pct"].abs()
    )

    proxy["improvement_class"] = np.select(
        [
            proxy["ldes_abs_error_improvement_pp"] > 0.0,
            proxy["ldes_abs_error_improvement_pp"] < 0.0,
        ],
        ["improved", "worsened"],
        default="unchanged",
    )

    proxy["method_label"] = proxy["method_key"].map(METHOD_LABELS)

    return proxy


def calculate_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the proportion of cases improved overall and by method."""

    def summarise(group: pd.DataFrame) -> pd.Series:
        values = group["ldes_abs_error_improvement_pp"]

        n_total = len(group)
        n_improved = int((values > 0.0).sum())
        n_worsened = int((values < 0.0).sum())
        n_unchanged = n_total - n_improved - n_worsened

        return pd.Series(
            {
                "n_total": n_total,
                "n_improved": n_improved,
                "n_worsened": n_worsened,
                "n_unchanged": n_unchanged,
                "proportion_improved": (n_improved / n_total if n_total else np.nan),
                "median_improvement_pp": values.median(),
                "mean_improvement_pp": values.mean(),
            }
        )

    method_rows = []
    for method in METHOD_ORDER:
        group = data.loc[data["method_key"].eq(method)]
        stats = summarise(group).to_dict()
        stats["method_key"] = method
        stats["method_label"] = METHOD_LABELS[method]
        method_rows.append(stats)

    by_method = pd.DataFrame(method_rows)

    # Preserve the visual method order.
    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    by_method["_order"] = by_method["method_key"].map(method_order)
    by_method = (
        by_method.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    )

    overall_values = data["ldes_abs_error_improvement_pp"]
    n_total = len(data)
    n_improved = int((overall_values > 0.0).sum())
    n_worsened = int((overall_values < 0.0).sum())

    overall = pd.DataFrame(
        [
            {
                "method_key": "overall",
                "method_label": "Overall",
                "n_total": n_total,
                "n_improved": n_improved,
                "n_worsened": n_worsened,
                "n_unchanged": n_total - n_improved - n_worsened,
                "proportion_improved": (n_improved / n_total if n_total else np.nan),
                "median_improvement_pp": overall_values.median(),
                "mean_improvement_pp": overall_values.mean(),
            }
        ]
    )

    return pd.concat([by_method, overall], ignore_index=True)


def print_summary(data: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Print selected dimensions and improvement statistics."""
    countries = sorted(str(x) for x in data["country"].dropna().unique())
    horizons = sorted(
        round(float(x), 3) for x in data["horizon_years"].dropna().unique()
    )
    ks = sorted(int(x) for x in data["k_periods"].dropna().unique())

    print("Figure 5 — Improvement in LDES capacity error")
    print("==============================================")
    print(f"Cases plotted:   {len(data)}")
    print(f"Countries:       {countries}")
    print(f"Horizon years:   {horizons}")
    print(f"k:               {ks}")
    print(f"Proxy weight:    W_P = {TARGET_PROXY_WEIGHT:g}")

    print("\nImprovement summary")
    print("-------------------")

    for row in summary.itertuples(index=False):
        label = str(row.method_label).replace("\n", " ")
        pct = 100.0 * float(row.proportion_improved)
        print(
            f"{label}: "
            f"{int(row.n_improved)}/{int(row.n_total)} improved "
            f"({pct:.1f}%); "
            f"median = {float(row.median_improvement_pp):+.2f} pp"
        )


def proxy_colour() -> object:
    """Return the fixed plasma colour corresponding to W_P=0.5."""
    cmap = plt.get_cmap("plasma")
    position = PLASMA_MIN + TARGET_PROXY_WEIGHT * (PLASMA_MAX - PLASMA_MIN)
    return cmap(position)


def make_figure(
    data: pd.DataFrame,
    *,
    width_px: int,
    height_px: int,
    dpi: int,
) -> plt.Figure:
    """Build the publication figure."""
    if width_px <= 0 or height_px <= 0 or dpi <= 0:
        raise ValueError("width-px, height-px, and dpi must all be positive.")

    colour = proxy_colour()
    figsize = (width_px / dpi, height_px / dpi)

    rc = {
        "font.size": 10.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,
            constrained_layout=True,
        )

        method_positions = {
            method: float(index) for index, method in enumerate(METHOD_ORDER)
        }

        rng = np.random.default_rng(JITTER_SEED)

        # Sort first so the deterministic jitter is reproducible even if the
        # parquet row order changes.
        plot_data = data.sort_values(
            [
                "method_key",
                "country",
                "start_date",
                "end_date",
                "k_periods",
            ]
        )

        for method in METHOD_ORDER:
            method_data = plot_data.loc[plot_data["method_key"].eq(method)]
            centre = method_positions[method]

            jitter = rng.uniform(
                -JITTER_WIDTH / 2,
                JITTER_WIDTH / 2,
                size=len(method_data),
            )
            xs = centre + jitter
            ys = method_data["ldes_abs_error_improvement_pp"].to_numpy(dtype=float)

            ax.scatter(
                xs,
                ys,
                s=POINT_SIZE,
                color=colour,
                alpha=POINT_ALPHA,
                linewidths=0,
                zorder=3,
            )

            median = float(np.median(ys))
            ax.hlines(
                median,
                centre - MEDIAN_HALF_WIDTH,
                centre + MEDIAN_HALF_WIDTH,
                color="white",
                linewidth=2.7,
                zorder=4,
            )
            ax.hlines(
                median,
                centre - MEDIAN_HALF_WIDTH,
                centre + MEDIAN_HALF_WIDTH,
                color="0.12",
                linewidth=1.6,
                zorder=5,
            )

        ax.axhline(
            0.0,
            color="0.3",
            linewidth=0.7,
            linestyle="-",
            zorder=1,
        )

        positions = [method_positions[method] for method in METHOD_ORDER]
        ax.set_xticks(positions)
        ax.set_xticklabels([METHOD_LABELS[method] for method in METHOD_ORDER])
        ax.set_xlim(
            min(positions) - 0.45,
            max(positions) + 0.45,
        )

        ax.set_xlabel("TSA clustering + representation methods")
        ax.set_ylabel("Reduction in absolute LDES capacity error\n(percentage points)")

        # Use the full data range here rather than q95 clipping because this
        # figure is intended to show every improved/worsened case.
        values = data["ldes_abs_error_improvement_pp"].to_numpy(dtype=float)
        ymin = min(0.0, float(np.min(values)))
        ymax = max(0.0, float(np.max(values)))

        ymin = Y_TICK_INTERVAL_PP * np.floor(ymin / Y_TICK_INTERVAL_PP)
        ymax = Y_TICK_INTERVAL_PP * np.ceil(ymax / Y_TICK_INTERVAL_PP)

        # Ensure a non-zero plotting range in the unlikely event that all
        # values fall on zero.
        if np.isclose(ymin, ymax):
            ymin -= Y_TICK_INTERVAL_PP
            ymax += Y_TICK_INTERVAL_PP

        ypad = max(0.5, 0.025 * (ymax - ymin))
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_INTERVAL_PP))

        ax.grid(
            axis="y",
            linewidth=0.6,
            alpha=0.3,
            zorder=0,
        )

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=colour,
                markeredgecolor="none",
                markersize=6.5,
                label=rf"$W_P = {TARGET_PROXY_WEIGHT:g}$",
            ),
            Line2D(
                [0],
                [0],
                color="0.12",
                linewidth=1.6,
                label="Median",
            ),
        ]

        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            frameon=False,
            columnspacing=1.4,
            handletextpad=0.55,
        )

        return fig


def save_outputs(
    fig: plt.Figure,
    summary: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path, Path]:
    """Save PNG, vector PDF, and the numerical improvement summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    summary_path = output_dir / f"{OUTPUT_STEM}_summary.csv"

    # Avoid bbox_inches="tight" so the requested PNG pixel dimensions are
    # preserved exactly.
    fig.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
    )

    summary.to_csv(summary_path, index=False)

    return png_path, pdf_path, summary_path


def main() -> None:
    args = parse_args()

    data = load_results(args.source_dir)
    summary = calculate_summary(data)
    print_summary(data, summary)

    fig = make_figure(
        data,
        width_px=args.width_px,
        height_px=args.height_px,
        dpi=args.dpi,
    )

    png_path, pdf_path, summary_path = save_outputs(
        fig,
        summary,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {summary_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
