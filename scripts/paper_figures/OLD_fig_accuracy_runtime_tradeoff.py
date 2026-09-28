"""Create Figure 8: accuracy-runtime trade-off.

The figure evaluates the trade-off between model accuracy and computational
speedup for the selected paper formulation:

- clustering method: k-means
- representation method: medoid
- SoC proxy weight: W_P = 0.5
- horizons: configurable subset of 2, 5, and 10 years
- representative-period counts:
      k = 10, 20, 30, 40, 50, 60, 90, 180

Two vertically stacked panels are produced:
    (a) absolute LDES capacity error vs speedup
    (b) annualised-CAPEX-weighted MACME vs speedup

Speedup is calculated case-by-case as:

    reference_runtime_calliope_seconds / runtime_method_seconds

where runtime_method_seconds (or runtime_methods_seconds) is the combined
clustered workflow runtime including SoC proxy construction, TSA, and
Calliope execution.

Visual encoding:
- colour: representative-period count k, using plasma from 0-90%
- marker: modelling horizon
- dashed black line: Pareto front

The Pareto front contains points for which no other point has both:
- equal or lower error, and
- equal or higher speedup,
with at least one strict improvement.

Source:
    results/2_5_10_year/
        parameters.parquet
        investment_metrics.parquet

Outputs:
    results/figures/OLD_fig_accuracy_runtime_tradeoff/
        OLD_fig_accuracy_runtime_tradeoff.png
        OLD_fig_accuracy_runtime_tradeoff.pdf
        OLD_fig_accuracy_runtime_tradeoff_points.csv
        OLD_fig_accuracy_runtime_tradeoff_pareto_ldes.csv
        OLD_fig_accuracy_runtime_tradeoff_pareto_macme.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


# ---------------------------------------------------------------------------
# Figure-specific configuration
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_DIR = Path("results/2_5_10_year")
DEFAULT_OUTPUT_DIR = Path("results/figures/OLD_fig_accuracy_runtime_tradeoff")
OUTPUT_STEM = "OLD_fig_accuracy_runtime_tradeoff"

DEFAULT_WIDTH_PX = 2000
DEFAULT_HEIGHT_PX = 4000
DEFAULT_DPI = 300

CLUSTER_METHOD = "kmeans"
REPRESENTATION_METHOD = "medoid"
TARGET_PROXY_WEIGHT = 0.5

HORIZON_ORDER = (2, 5, 10)
DEFAULT_HORIZONS = HORIZON_ORDER
HORIZON_TOLERANCE_YEARS = 0.10

DEFAULT_K_VALUES = (10, 20, 30, 40, 50, 60, 90, 180)

PLASMA_MIN = 0.0
PLASMA_MAX = 0.90

POINT_SIZE = 25
POINT_ALPHA = 0.78

PARETO_LINEWIDTH = 1.3
PARETO_MARKER_SIZE = 3.8

HORIZON_MARKERS = {
    2: "^",
    5: "s",
    10: "o",
}

REFERENCE_RUNTIME_COLUMN = "reference_runtime_calliope_seconds"

TOTAL_RUNTIME_CANDIDATES = (
    "runtime_method_seconds",
    "runtime_methods_seconds",
)

LDES_METRIC = "ldes_capacity_error_signed"
MACME_METRIC = "macme_capex_weighted_annualised"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot accuracy-runtime trade-offs for W_P=0.5 k-means + medoid cases."
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
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help=(
            "Representative-period counts to include "
            f"(default: {' '.join(map(str, DEFAULT_K_VALUES))})."
        ),
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        choices=list(HORIZON_ORDER),
        help=(
            "Modelling horizons to include, chosen from 2, 5, and 10 years "
            "(default: 2 5 10). For example, use '--horizons 10' to plot "
            "only the 10-year cases."
        ),
    )
    parser.add_argument(
        "--wp",
        type=float,
        default=TARGET_PROXY_WEIGHT,
        help=(f"SoC proxy weight to plot (default: {TARGET_PROXY_WEIGHT:g})."),
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
            "Together with the pixel dimensions this sets the PDF size."
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


def resolve_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    purpose: str,
) -> str:
    """Resolve one column while allowing known naming variants."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    runtime_columns = sorted(
        column for column in frame.columns if "runtime" in column.lower()
    )
    raise ValueError(
        f"Could not find the {purpose} column. "
        f"Tried: {list(candidates)}\n"
        f"Runtime-like columns available:\n{runtime_columns}"
    )


def assign_horizon_duration(
    parameters: pd.DataFrame,
    *,
    horizons: list[int],
) -> pd.DataFrame:
    """Assign each case to its nearest selected modelling horizon."""
    parameters = parameters.copy()

    parameters["start_date"] = pd.to_datetime(parameters["start_date"])
    parameters["end_date"] = pd.to_datetime(parameters["end_date"])

    parameters["horizon_years_exact"] = (
        parameters["end_date"] - parameters["start_date"]
    ).dt.total_seconds() / (365.2425 * 24 * 60 * 60)

    targets = np.asarray(horizons, dtype=float)
    exact = parameters["horizon_years_exact"].to_numpy(dtype=float)

    nearest_index = np.abs(exact[:, None] - targets[None, :]).argmin(axis=1)
    nearest_target = targets[nearest_index]
    nearest_distance = np.abs(exact - nearest_target)

    parameters["horizon_duration"] = nearest_target.astype(int)
    parameters["horizon_distance"] = nearest_distance

    parameters = parameters.loc[
        parameters["horizon_distance"].le(HORIZON_TOLERANCE_YEARS)
    ].copy()

    if parameters.empty:
        raise ValueError(
            "No runs were found for the selected horizons "
            f"{horizons} within the configured tolerance of "
            f"{HORIZON_TOLERANCE_YEARS:g} years."
        )

    return parameters


def load_results(
    source_dir: Path,
    *,
    k_values: list[int],
    horizons: list[int],
    proxy_weight: float,
) -> tuple[pd.DataFrame, str]:
    """Load and prepare the Figure 8 cases."""
    parameters_path = source_dir / "parameters.parquet"
    investment_path = source_dir / "investment_metrics.parquet"

    parameters = pd.read_parquet(parameters_path)

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
            REFERENCE_RUNTIME_COLUMN,
        },
        table_name="parameters.parquet",
    )

    total_runtime_column = resolve_column(
        parameters,
        TOTAL_RUNTIME_CANDIDATES,
        purpose="combined clustered-workflow runtime",
    )

    parameters = assign_horizon_duration(
        parameters,
        horizons=horizons,
    )

    parameters["case_id"] = parameters["case_id"].astype(str)
    parameters["k_periods"] = parameters["k_periods"].astype(int)
    parameters["lambda_soc"] = parameters["lambda_soc"].astype(float)

    method_mask = parameters["cluster_method"].astype(str).str.lower().eq(
        CLUSTER_METHOD
    ) & parameters["representation_method"].astype(str).str.lower().eq(
        REPRESENTATION_METHOD
    )

    k_values = sorted(set(int(k) for k in k_values))
    if not k_values:
        raise ValueError("At least one k value must be supplied.")

    horizons = [
        horizon for horizon in HORIZON_ORDER if horizon in set(int(h) for h in horizons)
    ]
    if not horizons:
        raise ValueError("At least one modelling horizon must be supplied.")

    parameters = parameters.loc[
        method_mask
        & parameters["horizon_duration"].isin(horizons)
        & parameters["k_periods"].isin(k_values)
        & np.isclose(parameters["lambda_soc"], proxy_weight)
    ].copy()

    if parameters.empty:
        raise ValueError(
            "No cases remain after filtering for "
            f"{CLUSTER_METHOD} + {REPRESENTATION_METHOD}, "
            f"W_P={proxy_weight:g}, horizons={horizons}, "
            f"and k={k_values}."
        )

    parameters = parameters.rename(
        columns={total_runtime_column: "runtime_method_seconds"}
    )

    runtime_columns = [
        "runtime_method_seconds",
        REFERENCE_RUNTIME_COLUMN,
    ]

    missing_runtime = parameters[runtime_columns].isna().any(axis=1)
    if missing_runtime.any():
        missing_ids = parameters.loc[missing_runtime, "case_id"].tolist()
        raise ValueError(
            "Missing runtime values for selected cases. First cases:\n"
            + "\n".join(missing_ids[:20])
        )

    for column in runtime_columns:
        invalid = parameters[column].le(0)
        if invalid.any():
            raise ValueError(
                f"{column} contains {int(invalid.sum())} non-positive "
                "values; speedup cannot be calculated."
            )

    case_ids = set(parameters["case_id"])

    investment = pd.read_parquet(investment_path)
    require_columns(
        investment,
        {"case_id", "metric", "value"},
        table_name="investment_metrics.parquet",
    )

    investment["case_id"] = investment["case_id"].astype(str)

    investment = investment.loc[
        investment["case_id"].isin(case_ids)
        & investment["metric"].isin({LDES_METRIC, MACME_METRIC}),
        ["case_id", "metric", "value"],
    ].copy()

    duplicated = investment.duplicated(
        subset=["case_id", "metric"],
        keep=False,
    )
    if duplicated.any():
        duplicate_rows = investment.loc[
            duplicated,
            ["case_id", "metric"],
        ].sort_values(["case_id", "metric"])

        raise ValueError(
            "investment_metrics.parquet contains duplicate "
            "(case_id, metric) rows:\n" + duplicate_rows.head(30).to_string(index=False)
        )

    investment = investment.pivot(
        index="case_id",
        columns="metric",
        values="value",
    ).reset_index()
    investment.columns.name = None

    require_columns(
        investment,
        {"case_id", LDES_METRIC, MACME_METRIC},
        table_name="pivoted investment_metrics.parquet",
    )

    data = parameters.merge(
        investment,
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    missing_metrics = data[[LDES_METRIC, MACME_METRIC]].isna().any(axis=1)

    if missing_metrics.any():
        missing_ids = data.loc[missing_metrics, "case_id"].tolist()
        raise ValueError(
            "Missing required error metrics for selected cases. "
            "First cases:\n" + "\n".join(missing_ids[:20])
        )

    data["ldes_abs_error_pct"] = 100.0 * data[LDES_METRIC].abs()
    data["capex_weighted_macme_pct"] = 100.0 * data[MACME_METRIC]
    data["speedup"] = data[REFERENCE_RUNTIME_COLUMN] / data["runtime_method_seconds"]

    invalid_speedup = ~np.isfinite(data["speedup"]) | data["speedup"].le(0)
    if invalid_speedup.any():
        raise ValueError("Calculated speedup contains invalid or non-positive values.")

    return data, total_runtime_column


def pareto_front(
    data: pd.DataFrame,
    *,
    error_column: str,
) -> pd.DataFrame:
    """Return the non-dominated accuracy-speedup frontier.

    Lower error is better and higher speedup is better.

    The implementation first collapses identical error values to the
    highest-speedup point and then walks from lowest to highest error,
    retaining every point that establishes a new maximum speedup.
    """
    candidates = (
        data.sort_values(
            [error_column, "speedup"],
            ascending=[True, False],
        )
        .drop_duplicates(
            subset=[error_column],
            keep="first",
        )
        .copy()
    )

    keep = []
    best_speedup = -np.inf

    for row in candidates.itertuples():
        speedup = float(row.speedup)
        if speedup > best_speedup:
            keep.append(row.Index)
            best_speedup = speedup

    return candidates.loc[keep].sort_values(error_column).reset_index(drop=True)


def build_k_colours(
    k_values: list[int],
) -> dict[int, object]:
    """Distribute k values across the lower 90% of plasma."""
    cmap = plt.get_cmap("plasma")

    if len(k_values) == 1:
        return {k_values[0]: cmap((PLASMA_MIN + PLASMA_MAX) / 2)}

    positions = np.linspace(
        PLASMA_MIN,
        PLASMA_MAX,
        len(k_values),
    )

    return {
        k: cmap(position)
        for k, position in zip(
            k_values,
            positions,
            strict=True,
        )
    }


def percentage_formatter(
    value: float,
    _: float,
) -> str:
    """Format an axis value in percentage points."""
    return f"{value:g}%"


def add_tradeoff_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    error_column: str,
    xlabel: str,
    panel_label: str,
    k_colours: dict[int, object],
    horizons: list[int],
) -> pd.DataFrame:
    """Plot one accuracy-speedup panel and return its Pareto front."""
    k_values = sorted(k_colours)

    for horizon in horizons:
        horizon_data = data.loc[data["horizon_duration"].eq(horizon)]

        for k in k_values:
            subset = horizon_data.loc[horizon_data["k_periods"].eq(k)]

            if subset.empty:
                continue

            ax.scatter(
                subset[error_column],
                subset["speedup"],
                s=POINT_SIZE,
                marker=HORIZON_MARKERS[horizon],
                color=k_colours[k],
                alpha=POINT_ALPHA,
                linewidths=0,
                zorder=3,
            )

    pareto = pareto_front(
        data,
        error_column=error_column,
    )

    ax.plot(
        pareto[error_column],
        pareto["speedup"],
        color="0.18",
        linestyle="--",
        linewidth=PARETO_LINEWIDTH,
        marker="o",
        markersize=PARETO_MARKER_SIZE,
        markerfacecolor="0.18",
        markeredgewidth=0,
        zorder=5,
    )

    # Speedup = 1 is the point at which clustering ceases to provide any
    # computational advantage over the corresponding full-resolution run.
    ax.axhline(
        1.0,
        color="0.55",
        linewidth=0.8,
        zorder=1,
    )

    ax.set_yscale("log")
    ax.set_ylabel(r"Speedup ($\times$)")
    ax.set_xlabel(xlabel)

    ax.xaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Only major-order horizontal grid lines, matching the runtime figure.
    ax.grid(
        axis="y",
        which="major",
        linewidth=0.65,
        alpha=0.32,
        linestyle=(0, (1.5, 2.5)),
        zorder=0,
    )
    ax.grid(axis="x", visible=False)

    ax.text(
        0.0,
        1.02,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
    )

    return pareto


def validate_plot_inputs(
    data: pd.DataFrame,
    *,
    proxy_weight: float,
    runtime_source_column: str,
) -> None:
    """Print selected dimensions and basic speedup diagnostics."""
    countries = sorted(str(x) for x in data["country"].dropna().unique())
    horizons = sorted(int(x) for x in data["horizon_duration"].unique())
    ks = sorted(int(x) for x in data["k_periods"].unique())

    print("Figure 8 — Accuracy-runtime trade-off")
    print("=====================================")
    print(f"Cases:           {len(data)}")
    print(f"Countries:       {countries}")
    print(f"Horizons:        {horizons}")
    print(f"k:               {ks}")
    print(f"W_P:             {proxy_weight:g}")
    print(f"Runtime column:  {runtime_source_column}")
    print(
        f"Speedup range:   {data['speedup'].min():.2f}x to {data['speedup'].max():.2f}x"
    )


def make_figure(
    data: pd.DataFrame,
    *,
    horizons: list[int],
    width_px: int,
    height_px: int,
    dpi: int,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """Build the two-panel Figure 8 trade-off plot."""
    if width_px <= 0 or height_px <= 0 or dpi <= 0:
        raise ValueError("width-px, height-px, and dpi must all be positive.")

    figsize = (
        width_px / dpi,
        height_px / dpi,
    )

    k_values = sorted(int(x) for x in data["k_periods"].unique())
    k_colours = build_k_colours(k_values)

    rc = {
        "font.size": 10.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.2,
        "legend.title_fontsize": 9.8,
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
            constrained_layout=False,
        )

        pareto_ldes = add_tradeoff_panel(
            axes[0],
            data,
            error_column="ldes_abs_error_pct",
            xlabel="Absolute LDES capacity error",
            panel_label="(a) LDES capacity error",
            k_colours=k_colours,
            horizons=horizons,
        )

        pareto_macme = add_tradeoff_panel(
            axes[1],
            data,
            error_column="capex_weighted_macme_pct",
            xlabel="CAPEX-weighted MACME",
            panel_label="(b) CAPEX-weighted MACME",
            k_colours=k_colours,
            horizons=horizons,
        )

        # Match y limits so vertical speedup comparisons between the two panels
        # are immediate.
        speedup_min = float(data["speedup"].min())
        speedup_max = float(data["speedup"].max())

        ymin = speedup_min / 1.35
        ymax = speedup_max * 1.35

        for ax in axes:
            ax.set_ylim(ymin, ymax)

        # Colour legend: representative-period count k.
        k_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=k_colours[k],
                markeredgecolor="none",
                markersize=6.3,
                label=rf"$k={k}$",
            )
            for k in reversed(k_values)
        ]

        # Marker legend: horizon duration.
        # horizon_handles = [
        #     Line2D(
        #         [0],
        #         [0],
        #         marker=HORIZON_MARKERS[horizon],
        #         linestyle="none",
        #         markerfacecolor="0.45",
        #         markeredgecolor="none",
        #         markersize=6.3,
        #         label=f"{horizon}-year",
        #     )
        #     for horizon in horizons
        # ]

        pareto_handle = Line2D(
            [0],
            [0],
            color="0.18",
            linestyle="--",
            linewidth=PARETO_LINEWIDTH,
            marker="o",
            markersize=3.8,
            label="Pareto front",
        )

        # Two small figure-level legends on the right keep the visual roles
        # explicit without mixing k and horizon into one ambiguous list.
        legend_k = fig.legend(
            handles=k_handles + [pareto_handle],
            # title=r"Representative periods, $k$",
            loc="upper center",
            bbox_to_anchor=(0.50, 0.99),
            ncol=5,
            frameon=False,
            handletextpad=0.55,
        )
        fig.add_artist(legend_k)

        fig.subplots_adjust(
            left=0.13,
            right=0.9,
            top=0.84,
            bottom=0.08,
            hspace=0.30,
        )

        return fig, pareto_ldes, pareto_macme


def save_outputs(
    fig: plt.Figure,
    data: pd.DataFrame,
    pareto_ldes: pd.DataFrame,
    pareto_macme: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path, Path, Path, Path]:
    """Save figure and underlying point/frontier data."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    points_path = output_dir / f"{OUTPUT_STEM}_points.csv"
    pareto_ldes_path = output_dir / f"{OUTPUT_STEM}_pareto_ldes.csv"
    pareto_macme_path = output_dir / f"{OUTPUT_STEM}_pareto_macme.csv"

    fig.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
    )

    output_columns = [
        "case_id",
        "country",
        "start_date",
        "end_date",
        "horizon_duration",
        "k_periods",
        "lambda_soc",
        "ldes_abs_error_pct",
        "capex_weighted_macme_pct",
        "runtime_method_seconds",
        REFERENCE_RUNTIME_COLUMN,
        "speedup",
    ]

    data[output_columns].to_csv(
        points_path,
        index=False,
    )

    pareto_ldes[output_columns].to_csv(
        pareto_ldes_path,
        index=False,
    )

    pareto_macme[output_columns].to_csv(
        pareto_macme_path,
        index=False,
    )

    return (
        png_path,
        pdf_path,
        points_path,
        pareto_ldes_path,
        pareto_macme_path,
    )


def main() -> None:
    args = parse_args()

    horizons = [horizon for horizon in HORIZON_ORDER if horizon in set(args.horizons)]

    data, runtime_source_column = load_results(
        args.source_dir,
        k_values=args.k_values,
        horizons=horizons,
        proxy_weight=args.wp,
    )

    validate_plot_inputs(
        data,
        proxy_weight=args.wp,
        runtime_source_column=runtime_source_column,
    )

    fig, pareto_ldes, pareto_macme = make_figure(
        data,
        horizons=horizons,
        width_px=args.width_px,
        height_px=args.height_px,
        dpi=args.dpi,
    )

    outputs = save_outputs(
        fig,
        data,
        pareto_ldes,
        pareto_macme,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    print("\nPareto points")
    print("-------------")
    print(f"LDES capacity: {len(pareto_ldes)} / {len(data)} cases")
    print(f"CAPEX-weighted MACME: {len(pareto_macme)} / {len(data)} cases")

    print("\nSaved:")
    for output in outputs:
        print(f"  {output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
