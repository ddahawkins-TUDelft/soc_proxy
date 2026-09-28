"""Create an appendix TSA-method sensitivity figure using median + IQR.

The figure evaluates how the SoC proxy behaves across different TSA methods
for 10-year cases only, using two vertically stacked panels:

    (a) signed LDES capacity error
    (b) TSA runtime

Only local-distribution variants are retained; global-distribution variants
are excluded.

Visual encoding
---------------
- x-axis main groups: TSA methods
- within each method: W_P = 0, 0.5, 1 from left to right
- within each W_P: selected k values from left to right
- colour: representative-period count k
- marker: median across pooled country/weather cases
- vertical whisker: interquartile range (Q25-Q75)

Source:
    results/2_5_10_year/
        parameters.parquet
        investment_metrics.parquet

Outputs:
    results/figures/fig_appendix_tsa_methods_iqr/
        fig_appendix_tsa_methods_iqr.png
        fig_appendix_tsa_methods_iqr.pdf
        fig_appendix_tsa_methods_iqr_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


DEFAULT_SOURCE_DIR = Path("results/2_5_10_year")
DEFAULT_OUTPUT_DIR = Path("results/figures/fig_appendix_tsa_methods_iqr")
OUTPUT_STEM = "fig_appendix_tsa_methods_iqr"

DEFAULT_WIDTH_PX = 1600
DEFAULT_HEIGHT_PX = 2400
DEFAULT_DPI = 300

DEFAULT_K_VALUES = (20, 40, 60)
WP_ORDER = (0.0, 0.5, 1.0)

TARGET_HORIZON_YEARS = 10.0
HORIZON_TOLERANCE_YEARS = 0.10

PLASMA_MIN = 0.0
PLASMA_MAX = 0.90

MEDIAN_MARKER_SIZE = 34
IQR_LINEWIDTH = 1.5
IQR_CAPSIZE = 3.0
IQR_ALPHA = 0.82

WP_GROUP_WIDTH = 0.72
K_GROUP_WIDTH = 0.14

LDES_TICK_INTERVAL = 10

LDES_METRIC = "ldes_capacity_error_signed"

TSA_RUNTIME_CANDIDATES = (
    "runtime_tsa_seconds",
    "runtime_tsam_seconds",
    "tsa_runtime_seconds",
    "tsam_runtime_seconds",
)

METHOD_ORDER = [
    "hierarchical_medoid",
    "kmeans_medoid",
    "hierarchical_distribution_local",
    "hierarchical_distribution_minmax_local",
]

METHOD_LABELS = {
    "hierarchical_medoid": "Hierarchical\n+ medoid",
    "kmeans_medoid": "k-means\n+ medoid",
    "hierarchical_distribution_local": "Hierarchical\n+ distribution",
    "hierarchical_distribution_minmax_local": (
        "Hierarchical\n+ distribution\nminmax"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the appendix TSA-method sensitivity figure using "
            "median markers and interquartile-range whiskers."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
    )
    parser.add_argument(
        "--width-px",
        type=int,
        default=DEFAULT_WIDTH_PX,
    )
    parser.add_argument(
        "--height-px",
        type=int,
        default=DEFAULT_HEIGHT_PX,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
    )
    parser.add_argument(
        "--show",
        action="store_true",
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


def resolve_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    purpose: str,
) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    runtime_columns = sorted(
        c for c in frame.columns if "runtime" in c.lower()
    )
    raise ValueError(
        f"Could not find the {purpose} column. "
        f"Tried: {list(candidates)}\n"
        f"Runtime-like columns available:\n{runtime_columns}"
    )


def filter_ten_year_runs(parameters: pd.DataFrame) -> pd.DataFrame:
    parameters = parameters.copy()
    parameters["start_date"] = pd.to_datetime(parameters["start_date"])
    parameters["end_date"] = pd.to_datetime(parameters["end_date"])

    parameters["horizon_years"] = (
        (parameters["end_date"] - parameters["start_date"]).dt.total_seconds()
        / (365.2425 * 24 * 60 * 60)
    )

    parameters = parameters.loc[
        np.isclose(
            parameters["horizon_years"],
            TARGET_HORIZON_YEARS,
            atol=HORIZON_TOLERANCE_YEARS,
            rtol=0.0,
        )
    ].copy()

    if parameters.empty:
        raise ValueError("No approximately 10-year cases were found.")

    return parameters


def infer_representation_scope(parameters: pd.DataFrame) -> pd.Series:
    for column in ("representation_scope", "scope"):
        if column in parameters.columns:
            return parameters[column].astype("string").str.lower().fillna("")

    names = (
        parameters["experiment_name"]
        .astype("string")
        .str.lower()
        .fillna("")
    )

    scope = pd.Series("", index=parameters.index, dtype="string")
    scope.loc[names.str.contains("local", regex=False)] = "local"
    scope.loc[names.str.contains("global", regex=False)] = "global"
    return scope


def assign_method_key(parameters: pd.DataFrame) -> pd.Series:
    cluster = parameters["cluster_method"].astype(str).str.lower()
    representation = parameters["representation_method"].astype(str).str.lower()
    scope = infer_representation_scope(parameters)

    method_key = pd.Series(pd.NA, index=parameters.index, dtype="string")

    method_key.loc[
        cluster.eq("hierarchical") & representation.eq("medoid")
    ] = "hierarchical_medoid"

    method_key.loc[
        cluster.eq("kmeans") & representation.eq("medoid")
    ] = "kmeans_medoid"

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


def load_results(
    source_dir: Path,
    *,
    k_values: list[int],
) -> tuple[pd.DataFrame, str]:
    parameters = pd.read_parquet(source_dir / "parameters.parquet")

    require_columns(
        parameters,
        {
            "case_id",
            "country",
            "start_date",
            "end_date",
            "k_periods",
            "lambda_soc",
            "cluster_method",
            "representation_method",
            "experiment_name",
        },
        table_name="parameters.parquet",
    )

    parameters = filter_ten_year_runs(parameters)

    runtime_column = resolve_column(
        parameters,
        TSA_RUNTIME_CANDIDATES,
        purpose="TSA runtime",
    )

    parameters["case_id"] = parameters["case_id"].astype(str)
    parameters["k_periods"] = parameters["k_periods"].astype(int)
    parameters["lambda_soc"] = parameters["lambda_soc"].astype(float)
    parameters["method_key"] = assign_method_key(parameters)

    k_values = sorted(set(int(k) for k in k_values))

    weight_mask = np.zeros(len(parameters), dtype=bool)
    for wp in WP_ORDER:
        weight_mask |= np.isclose(parameters["lambda_soc"], wp)

    selected = parameters.loc[
        parameters["method_key"].isin(METHOD_ORDER)
        & parameters["k_periods"].isin(k_values)
        & weight_mask
    ].copy()

    missing_methods = [
        method for method in METHOD_ORDER
        if method not in set(selected["method_key"].dropna())
    ]
    if missing_methods:
        raise ValueError(f"Missing methods after filtering: {missing_methods}")

    case_ids = set(selected["case_id"])

    investment = pd.read_parquet(source_dir / "investment_metrics.parquet")
    require_columns(
        investment,
        {"case_id", "metric", "value"},
        table_name="investment_metrics.parquet",
    )

    investment["case_id"] = investment["case_id"].astype(str)
    investment = investment.loc[
        investment["case_id"].isin(case_ids)
        & investment["metric"].eq(LDES_METRIC),
        ["case_id", "value"],
    ].copy()

    investment = investment.rename(
        columns={"value": "ldes_capacity_error_signed"}
    )

    data = selected.merge(
        investment,
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    data["ldes_capacity_error_pct"] = (
        100.0 * data["ldes_capacity_error_signed"]
    )
    data["tsa_runtime_seconds"] = data[runtime_column].astype(float)

    return data, runtime_column


def summarise_metric(
    data: pd.DataFrame,
    *,
    value_column: str,
    metric_name: str,
) -> pd.DataFrame:
    summary = (
        data.groupby(
            ["method_key", "lambda_soc", "k_periods"],
            as_index=False,
        )
        .agg(
            n=(value_column, "size"),
            q25=(value_column, lambda x: x.quantile(0.25)),
            median=(value_column, "median"),
            q75=(value_column, lambda x: x.quantile(0.75)),
        )
    )
    summary["metric"] = metric_name
    return summary


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            summarise_metric(
                data,
                value_column="ldes_capacity_error_pct",
                metric_name="ldes_capacity_error_pct",
            ),
            summarise_metric(
                data,
                value_column="tsa_runtime_seconds",
                metric_name="tsa_runtime_seconds",
            ),
        ],
        ignore_index=True,
    )


def build_k_colours(k_values: list[int]) -> dict[int, object]:
    cmap = plt.get_cmap("plasma")
    positions = np.linspace(PLASMA_MIN, PLASMA_MAX, len(k_values))
    return {
        k: cmap(position)
        for k, position in zip(k_values, positions, strict=True)
    }


def build_wp_offsets() -> dict[float, float]:
    offsets = np.linspace(
        -WP_GROUP_WIDTH / 2,
        WP_GROUP_WIDTH / 2,
        len(WP_ORDER),
    )
    return dict(zip(WP_ORDER, offsets, strict=True))


def build_k_offsets(k_values: list[int]) -> dict[int, float]:
    offsets = np.linspace(
        -K_GROUP_WIDTH / 2,
        K_GROUP_WIDTH / 2,
        len(k_values),
    )
    return dict(zip(k_values, offsets, strict=True))


def plot_summary_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    metric_name: str,
    ylabel: str,
    title: str,
    method_positions: dict[str, float],
    wp_offsets: dict[float, float],
    k_offsets: dict[int, float],
    k_colours: dict[int, object],
) -> None:
    panel = summary.loc[summary["metric"].eq(metric_name)]

    for method in METHOD_ORDER:
        method_data = panel.loc[panel["method_key"].eq(method)]
        method_x = method_positions[method]

        for wp in WP_ORDER:
            wp_data = method_data.loc[
                np.isclose(method_data["lambda_soc"], wp)
            ]

            for k in k_offsets:
                row = wp_data.loc[wp_data["k_periods"].eq(k)]
                if row.empty:
                    continue

                row = row.iloc[0]

                centre = (
                    method_x
                    + wp_offsets[wp]
                    + k_offsets[k]
                )

                median = float(row["median"])
                q25 = float(row["q25"])
                q75 = float(row["q75"])

                ax.errorbar(
                    centre,
                    median,
                    yerr=np.array(
                        [[median - q25], [q75 - median]]
                    ),
                    fmt="o",
                    markersize=np.sqrt(MEDIAN_MARKER_SIZE),
                    color=k_colours[k],
                    ecolor=k_colours[k],
                    elinewidth=IQR_LINEWIDTH,
                    capsize=IQR_CAPSIZE,
                    capthick=IQR_LINEWIDTH,
                    alpha=IQR_ALPHA,
                    markeredgewidth=0,
                    zorder=3,
                )

    positions = [method_positions[m] for m in METHOD_ORDER]

    for left, right in zip(positions[:-1], positions[1:], strict=True):
        ax.axvline(
            (left + right) / 2,
            color="0.90",
            linewidth=0.7,
            zorder=0,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_xlim(min(positions) - 0.50, max(positions) + 0.50)

    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left",pad=5)
    ax.grid(axis="y", linewidth=0.6, alpha=0.28, zorder=0)


def set_ldes_axis(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel = summary.loc[
        summary["metric"].eq("ldes_capacity_error_pct")
    ]

    ymin = min(0.0, float(panel["q25"].min()))
    ymax = max(0.0, float(panel["q75"].max()))

    rounded_min = LDES_TICK_INTERVAL * np.floor(
        ymin / LDES_TICK_INTERVAL
    )
    rounded_max = LDES_TICK_INTERVAL * np.ceil(
        ymax / LDES_TICK_INTERVAL
    )

    pad = max(
        0.6,
        0.025 * (rounded_max - rounded_min),
    )

    ax.set_ylim(rounded_min - pad, rounded_max + pad)
    ax.yaxis.set_major_locator(MultipleLocator(LDES_TICK_INTERVAL))

    ax.axhline(
        0.0,
        color="0.35",
        linewidth=0.8,
        linestyle="--",
        zorder=1,
    )


def set_runtime_axis(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel = summary.loc[
        summary["metric"].eq("tsa_runtime_seconds")
    ]

    ymax = float(panel["q75"].max())
    pad = max(0.4, 0.05 * ymax)
    ax.set_ylim(0.0, ymax + pad)


def make_figure(
    summary: pd.DataFrame,
    *,
    width_px: int,
    height_px: int,
    dpi: int,
) -> plt.Figure:
    figsize = (width_px / dpi, height_px / dpi)

    k_values = sorted(int(x) for x in summary["k_periods"].unique())

    k_colours = build_k_colours(k_values)
    wp_offsets = build_wp_offsets()
    k_offsets = build_k_offsets(k_values)

    method_positions = {
        method: float(index)
        for index, method in enumerate(METHOD_ORDER)
    }

    rc = {
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figsize,
            dpi=dpi,
            sharex=True,
            constrained_layout=False,
        )

        plot_summary_panel(
            axes[0],
            summary,
            metric_name="ldes_capacity_error_pct",
            ylabel="LDES capacity error (%)",
            title="(a) LDES capacity error",
            method_positions=method_positions,
            wp_offsets=wp_offsets,
            k_offsets=k_offsets,
            k_colours=k_colours,
        )
        set_ldes_axis(axes[0], summary)

        plot_summary_panel(
            axes[1],
            summary,
            metric_name="tsa_runtime_seconds",
            ylabel="TSA runtime (s)",
            title="(b) TSA runtime",
            method_positions=method_positions,
            wp_offsets=wp_offsets,
            k_offsets=k_offsets,
            k_colours=k_colours,
        )
        set_runtime_axis(axes[1], summary)

        axes[0].tick_params(axis="x", labelbottom=True)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=k_colours[k],
                markeredgecolor="none",
                markersize=6.2,
                label=rf"$k={k}$",
            )
            for k in k_values
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.565, 0.985),
            ncol=len(legend_handles),
            frameon=False,
            columnspacing=1.35,
            handletextpad=0.55,
        )

        fig.text(
            0.565,
            0.952,
            (
                "Markers show medians; whiskers show Q25-Q75.\n"
                r"Within each method: $W_P = 0,\ 0.5,\ 1$ "
                r"from left to right."
            ),
            ha="center",
            va="top",
            fontsize=7.5,
            color="0.4",
        )

        fig.subplots_adjust(
            left=0.15,
            right=0.98,
            top=0.88,
            bottom=0.08,
            hspace=0.25,
        )

        return fig


def save_outputs(
    fig: plt.Figure,
    summary: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    summary_path = output_dir / f"{OUTPUT_STEM}_summary.csv"

    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    summary.to_csv(summary_path, index=False)

    return png_path, pdf_path, summary_path


def main() -> None:
    args = parse_args()

    data, runtime_column = load_results(
        args.source_dir,
        k_values=args.k_values,
    )
    summary = build_summary(data)

    print("Appendix TSA-method sensitivity — median + IQR")
    print("================================================")
    print(f"Cases:          {len(data)}")
    print(f"Horizon:        ~{TARGET_HORIZON_YEARS:g} years")
    print(f"k:              {sorted(data['k_periods'].unique())}")
    print(f"W_P:            {sorted(data['lambda_soc'].unique())}")
    print(f"Runtime column: {runtime_column}")

    fig = make_figure(
        summary,
        width_px=args.width_px,
        height_px=args.height_px,
        dpi=args.dpi,
    )

    outputs = save_outputs(
        fig,
        summary,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    print("\nSaved:")
    for output in outputs:
        print(f"  {output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
