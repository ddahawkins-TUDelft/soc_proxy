"""Create Figure 6 (median + IQR version): horizon sensitivity results.

This variant of Figure 6 keeps the same overall layout and dimensions as the
main horizon-sensitivity plot, but replaces raw point clouds with summary
markers:

    - coloured marker = median
    - vertical whisker = interquartile range (Q25-Q75)

The figure compares 2-, 5-, and 10-year modelling horizons for the selected
paper method (k-means clustering + medoid representation).

Two vertically stacked panels are produced:
    (a) signed LDES capacity error
    (b) annualised-CAPEX-weighted MACME

Visual encoding
---------------
- Main x grouping: horizon duration (2, 5, 10 years)
- Within-horizon channels: SoC proxy weight W_P = 0 and 0.5
- Marker colour: SoC proxy weight
- Marker: median across all selected country / weather-horizon / k cases
- Vertical whisker: interquartile range (Q25-Q75)

The k values are deliberately pooled within each horizon / W_P channel, but
the subset is configurable from the command line.

Source:
    results/2_5_10_year/*.parquet

Outputs:
    results/figures/fig_6_horizon_results_iqr/
        fig_6_horizon_results_iqr.png
        fig_6_horizon_results_iqr.pdf
        fig_6_horizon_results_iqr_summary.csv
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

DEFAULT_SOURCE_DIR = Path("results/2_5_10_year")
DEFAULT_OUTPUT_DIR = Path("results/figures/fig_6_horizon_results_iqr")
OUTPUT_STEM = "fig_6_horizon_results_iqr"

DEFAULT_WIDTH_PX = 1500
DEFAULT_HEIGHT_PX = 3000
DEFAULT_DPI = 300

CLUSTER_METHOD = "kmeans"
REPRESENTATION_METHOD = "medoid"

HORIZON_ORDER = (2, 5, 10)
HORIZON_TOLERANCE_YEARS = 0.10

WP_ORDER = (0.0, 0.5)
DEFAULT_K_VALUES = (10, 20, 30, 40, 50, 60, 90, 180)

PLASMA_MIN = 0.0
PLASMA_MAX = 0.90

CHANNEL_WIDTH = 0.30

MEDIAN_MARKER_SIZE = 60
IQR_LINEWIDTH = 1.8
IQR_CAPSIZE = 3.8
IQR_ALPHA = 0.84

LDES_TICK_INTERVAL = 10
MACME_TICK_INTERVAL = 5

AXES_LEFT = 0.15
AXES_RIGHT = 0.985
AXES_CENTER = (AXES_LEFT + AXES_RIGHT) / 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 2-, 5-, and 10-year horizon results for k-means + medoid "
            "using median markers and IQR whiskers."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=(
            "Directory containing parameters.parquet and "
            "investment_metrics.parquet."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which the PNG, PDF, and summary CSV are written.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help=(
            "Representative-period counts to pool within each horizon / W_P "
            f"channel (default: {' '.join(map(str, DEFAULT_K_VALUES))})."
        ),
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


def assign_horizon_duration(parameters: pd.DataFrame) -> pd.DataFrame:
    """Assign each case to the nearest requested 2-, 5-, or 10-year horizon."""
    parameters = parameters.copy()
    parameters["start_date"] = pd.to_datetime(parameters["start_date"])
    parameters["end_date"] = pd.to_datetime(parameters["end_date"])

    parameters["horizon_years_exact"] = (
        (parameters["end_date"] - parameters["start_date"]).dt.total_seconds()
        / (365.2425 * 24 * 60 * 60)
    )

    targets = np.asarray(HORIZON_ORDER, dtype=float)
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
            "No 2-, 5-, or 10-year runs were found within the configured "
            f"tolerance of {HORIZON_TOLERANCE_YEARS:g} years."
        )

    return parameters


def load_results(
    source_dir: Path,
    *,
    k_values: list[int],
) -> pd.DataFrame:
    """Load and prepare the data required for Figure 6."""
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
        },
        table_name="parameters.parquet",
    )

    method_mask = (
        parameters["cluster_method"]
        .astype(str)
        .str.lower()
        .eq(CLUSTER_METHOD)
        & parameters["representation_method"]
        .astype(str)
        .str.lower()
        .eq(REPRESENTATION_METHOD)
    )
    parameters = parameters.loc[method_mask].copy()

    if parameters.empty:
        raise ValueError(
            "No k-means + medoid cases were found in parameters.parquet."
        )

    parameters = assign_horizon_duration(parameters)

    parameters["case_id"] = parameters["case_id"].astype(str)
    parameters["k_periods"] = parameters["k_periods"].astype(int)
    parameters["lambda_soc"] = parameters["lambda_soc"].astype(float)

    k_values = sorted(set(int(k) for k in k_values))
    if not k_values:
        raise ValueError("At least one k value must be supplied.")

    weight_mask = np.zeros(len(parameters), dtype=bool)
    for wp in WP_ORDER:
        weight_mask |= np.isclose(parameters["lambda_soc"], wp)

    parameters = parameters.loc[
        parameters["k_periods"].isin(k_values)
        & parameters["horizon_duration"].isin(HORIZON_ORDER)
        & weight_mask
    ].copy()

    if parameters.empty:
        raise ValueError(
            "No cases remain after filtering the requested horizons, proxy "
            f"weights, and k values {k_values}."
        )

    missing_combinations: list[str] = []
    requested_k = set(k_values)

    for horizon in HORIZON_ORDER:
        for wp in WP_ORDER:
            subset = parameters.loc[
                parameters["horizon_duration"].eq(horizon)
                & np.isclose(parameters["lambda_soc"], wp)
            ]
            available_k = set(subset["k_periods"].unique())
            missing_k = sorted(requested_k - available_k)

            if missing_k:
                missing_combinations.append(
                    f"{horizon}-year, W_P={wp:g}: missing k={missing_k}"
                )

    if missing_combinations:
        raise ValueError(
            "The selected horizon / W_P channels do not all contain the same "
            "requested k subset:\n"
            + "\n".join(missing_combinations)
        )

    case_ids = set(parameters["case_id"])

    investment = pd.read_parquet(investment_path)
    require_columns(
        investment,
        {"case_id", "metric", "value"},
        table_name="investment_metrics.parquet",
    )

    investment["case_id"] = investment["case_id"].astype(str)

    metrics = {
        "ldes_capacity_error_signed",
        "macme_capex_weighted_annualised",
    }

    investment = investment.loc[
        investment["case_id"].isin(case_ids)
        & investment["metric"].isin(metrics),
        ["case_id", "metric", "value"],
    ].copy()

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
            + duplicates.head(30).to_string(index=False)
        )

    investment = investment.pivot(
        index="case_id",
        columns="metric",
        values="value",
    ).reset_index()
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

    data = parameters.merge(
        investment,
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    data["ldes_capacity_error_pct"] = (
        100.0 * data["ldes_capacity_error_signed"]
    )
    data["capex_weighted_macme_pct"] = (
        100.0 * data["macme_capex_weighted_annualised"]
    )

    missing = data[
        [
            "ldes_capacity_error_pct",
            "capex_weighted_macme_pct",
        ]
    ].isna().any(axis=1)

    if missing.any():
        missing_ids = data.loc[missing, "case_id"].tolist()
        raise ValueError(
            "Missing required investment metrics for "
            f"{len(missing_ids)} selected cases. First cases:\n"
            + "\n".join(missing_ids[:20])
        )

    return data


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise each horizon / W_P channel as median and IQR."""
    frames = []

    for metric_name, value_column in [
        ("ldes_capacity_error_pct", "ldes_capacity_error_pct"),
        ("capex_weighted_macme_pct", "capex_weighted_macme_pct"),
    ]:
        summary = (
            data.groupby(["horizon_duration", "lambda_soc"], as_index=False)
            .agg(
                n=(value_column, "size"),
                q25=(value_column, lambda x: x.quantile(0.25)),
                median=(value_column, "median"),
                q75=(value_column, lambda x: x.quantile(0.75)),
            )
        )
        summary["metric"] = metric_name
        frames.append(summary)

    return pd.concat(frames, ignore_index=True)


def validate_plot_inputs(
    data: pd.DataFrame,
    *,
    k_values: list[int],
) -> None:
    """Print the selected experiment dimensions."""
    countries = sorted(str(x) for x in data["country"].dropna().unique())
    horizons = sorted(
        int(x) for x in data["horizon_duration"].dropna().unique()
    )
    weights = sorted(
        float(x) for x in data["lambda_soc"].dropna().unique()
    )
    actual_k = sorted(int(x) for x in data["k_periods"].dropna().unique())

    print("Figure 6 — Horizon sensitivity (median + IQR)")
    print("============================================")
    print(f"Cases:       {len(data)}")
    print(f"Countries:   {countries}")
    print(f"Horizons:    {horizons}")
    print(f"k selected:  {sorted(set(k_values))}")
    print(f"k present:   {actual_k}")
    print(f"W_P:         {weights}")


def build_wp_colours() -> dict[float, object]:
    """Map W_P=0..1 onto the lower 90% of the plasma colour map."""
    cmap = plt.get_cmap("plasma")
    return {
        wp: cmap(PLASMA_MIN + wp * (PLASMA_MAX - PLASMA_MIN))
        for wp in WP_ORDER
    }


def build_wp_offsets(
    *,
    width: float = CHANNEL_WIDTH,
) -> dict[float, float]:
    """Assign W_P values stable sub-positions within each horizon group."""
    offsets = np.linspace(-width / 2, width / 2, len(WP_ORDER))
    return dict(zip(WP_ORDER, offsets, strict=True))


def legend_label(weight: float) -> str:
    """Use an explicit label for the no-proxy/original TSA cases."""
    if np.isclose(weight, 0.0):
        return "$W_P = 0$ (Original TSA)"
    return rf"$W_P = {weight:g}$"


def add_metric_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    metric_name: str,
    ylabel: str,
    panel_label: str,
    wp_colours: dict[float, object],
    wp_offsets: dict[float, float],
) -> None:
    """Plot one horizon-sensitivity metric using median markers and IQR."""
    horizon_positions = {
        horizon: float(index)
        for index, horizon in enumerate(HORIZON_ORDER)
    }

    panel = summary.loc[summary["metric"].eq(metric_name)].copy()

    for horizon in HORIZON_ORDER:
        horizon_data = panel.loc[
            panel["horizon_duration"].eq(horizon)
        ]
        horizon_x = horizon_positions[horizon]

        for wp in WP_ORDER:
            row = horizon_data.loc[np.isclose(horizon_data["lambda_soc"], wp)]

            if row.empty:
                continue

            row = row.iloc[0]
            centre = horizon_x + wp_offsets[wp]

            median = float(row["median"])
            q25 = float(row["q25"])
            q75 = float(row["q75"])

            ax.errorbar(
                centre,
                median,
                yerr=np.array([[median - q25], [q75 - median]]),
                fmt="o",
                markersize=np.sqrt(MEDIAN_MARKER_SIZE),
                color=wp_colours[wp],
                ecolor=wp_colours[wp],
                elinewidth=IQR_LINEWIDTH,
                capsize=IQR_CAPSIZE,
                capthick=IQR_LINEWIDTH,
                alpha=IQR_ALPHA,
                markeredgewidth=0,
                zorder=3,
            )

    positions = [horizon_positions[h] for h in HORIZON_ORDER]

    for left, right in zip(positions[:-1], positions[1:], strict=True):
        ax.axvline(
            (left + right) / 2,
            color="0.90",
            linewidth=0.7,
            zorder=0,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(h) for h in HORIZON_ORDER])
    ax.set_xlim(min(positions) - 0.44, max(positions) + 0.44)

    ax.set_ylabel(ylabel)
    ax.text(
        0.0,
        1.02,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
    )

    ax.grid(
        axis="y",
        linewidth=0.6,
        alpha=0.30,
        zorder=0,
    )


def set_padded_axis_limits(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    metric_name: str,
    tick_interval: float,
    include_zero: bool,
) -> None:
    """Round limits outward and add slight padding around the IQR range."""
    panel = summary.loc[summary["metric"].eq(metric_name)]

    ymin = float(panel["q25"].min())
    ymax = float(panel["q75"].max())

    if include_zero:
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)

    rounded_min = tick_interval * np.floor(ymin / tick_interval)
    rounded_max = tick_interval * np.ceil(ymax / tick_interval)

    if np.isclose(rounded_min, rounded_max):
        rounded_min -= tick_interval
        rounded_max += tick_interval

    pad = max(
        0.4,
        0.03 * (rounded_max - rounded_min),
    )

    ax.set_ylim(
        rounded_min - pad,
        rounded_max + pad,
    )


def make_figure(
    summary: pd.DataFrame,
    *,
    width_px: int,
    height_px: int,
    dpi: int,
) -> plt.Figure:
    """Build the two-panel horizon figure."""
    if width_px <= 0 or height_px <= 0 or dpi <= 0:
        raise ValueError(
            "width-px, height-px, and dpi must all be positive."
        )

    figsize = (width_px / dpi, height_px / dpi)
    wp_colours = build_wp_colours()
    wp_offsets = build_wp_offsets()

    rc = {
        "font.size": 10.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
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

        add_metric_panel(
            axes[0],
            summary,
            metric_name="ldes_capacity_error_pct",
            ylabel="LDES capacity error (%)",
            panel_label="(a) LDES capacity error",
            wp_colours=wp_colours,
            wp_offsets=wp_offsets,
        )

        axes[0].axhline(
            0.0,
            color="0.30",
            linewidth=0.6,
            zorder=1,
        )

        set_padded_axis_limits(
            axes[0],
            summary,
            metric_name="ldes_capacity_error_pct",
            tick_interval=LDES_TICK_INTERVAL,
            include_zero=True,
        )

        add_metric_panel(
            axes[1],
            summary,
            metric_name="capex_weighted_macme_pct",
            ylabel="CAPEX-weighted MACME (%)",
            panel_label="(b) CAPEX-weighted MACME",
            wp_colours=wp_colours,
            wp_offsets=wp_offsets,
        )

        set_padded_axis_limits(
            axes[1],
            summary,
            metric_name="capex_weighted_macme_pct",
            tick_interval=MACME_TICK_INTERVAL,
            include_zero=True,
        )

        axes[1].set_xlabel("Modelling horizon (years)")

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=wp_colours[wp],
                markeredgecolor="none",
                markersize=6.5,
                label=legend_label(wp),
            )
            for wp in WP_ORDER
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(AXES_CENTER, 0.992),
            ncol=len(legend_handles),
            frameon=False,
            columnspacing=1.45,
            handletextpad=0.55,
        )

        fig.text(
            AXES_CENTER,
            0.957,
            (
                r"Markers show medians; whiskers show Q25-Q75."
            ),
            ha="center",
            va="top",
            fontsize=9.2,
            color="0.38",
        )

        fig.subplots_adjust(
            left=AXES_LEFT,
            right=AXES_RIGHT,
            top=0.885,
            bottom=0.05,
            hspace=0.22,
        )

        return fig


def save_outputs(
    fig: plt.Figure,
    summary: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path, Path]:
    """Save the figure as exact-size PNG, vector PDF, and summary CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    summary_path = output_dir / f"{OUTPUT_STEM}_summary.csv"

    fig.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    return png_path, pdf_path, summary_path


def main() -> None:
    args = parse_args()

    data = load_results(
        args.source_dir,
        k_values=args.k_values,
    )
    validate_plot_inputs(
        data,
        k_values=args.k_values,
    )

    summary = build_summary(data)

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
