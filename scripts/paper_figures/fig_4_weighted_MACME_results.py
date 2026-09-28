"""Create Figure 3: Capex-weighted MACME against representative-period count.

The figure is intentionally narrow in scope:
- source: results/2_5_10_year/*.parquet
- horizon: 10-year runs only
- clustering method: k-means
- representation method: medoid
- metric: Capex-weighted MACME capacity error
- x grouping: representative-period count k
- within-k channels: SoC proxy weight W_P

Outputs are written as both PNG and PDF to:
    results/figures/fig_4_weighted_MACME_results/

The PNG canvas size can be specified directly in pixels. The PDF uses the
same physical figure dimensions and remains vector-based.
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

METRIC_NAME = 'macme_capex_weighted_annualised'

DEFAULT_SOURCE_DIR = Path("results/2_5_10_year")
DEFAULT_OUTPUT_DIR = Path("results/figures/fig_4_weighted_MACME_results")
OUTPUT_STEM = "fig_4_weighted_MACME_results"

DEFAULT_WIDTH_PX = 2400
DEFAULT_HEIGHT_PX = 1200
DEFAULT_DPI = 300

CLUSTER_METHOD = "kmeans"
REPRESENTATION_METHOD = "medoid"
TARGET_HORIZON_YEARS = 10.0
HORIZON_TOLERANCE_YEARS = 0.1

PLASMA_MIN = 0.0
PLASMA_MAX = 0.90

CHANNEL_WIDTH = 0.68
REPLICATE_JITTER_WIDTH = 0.0
POINT_SIZE = 24
POINT_ALPHA = 0.72
MEDIAN_HALF_WIDTH = 0.06

REFERENCE_ERROR_PCT = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 10-year k-means + medoid error results "
            "against representative-period count k."
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
        help="Directory in which the PNG and PDF figure files are written.",
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

    horizon_years = (
        (parameters["end_date"] - parameters["start_date"]).dt.total_seconds()
        / (365.2425 * 24 * 60 * 60)
    )
    parameters["horizon_years"] = horizon_years

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


def load_results(source_dir: Path) -> pd.DataFrame:
    """Load and prepare only the data required for Figure 3."""
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

    parameters = filter_ten_year_runs(parameters)

    parameters["k_periods"] = parameters["k_periods"].astype(int)
    parameters["lambda_soc"] = parameters["lambda_soc"].astype(float)

    case_ids = set(parameters["case_id"].astype(str))

    investment = pd.read_parquet(investment_path)
    require_columns(
        investment,
        {"case_id", "metric", "value"},
        table_name="investment_metrics.parquet",
    )

    investment = investment.loc[
        investment["case_id"].astype(str).isin(case_ids)
        & investment["metric"].eq(METRIC_NAME),
        ["case_id", "value"],
    ].copy()

    duplicated = investment["case_id"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(
            investment.loc[duplicated, "case_id"].astype(str).unique()
        )
        raise ValueError(
            "Expected one value per case, but "
            "duplicates were found for:\n"
            + "\n".join(duplicate_ids[:30])
        )

    investment = investment.rename(
        columns={"value": METRIC_NAME}
    )

    data = parameters.merge(
        investment,
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    data["capex_weighted_macme_pct"] = (
        100.0 * data[METRIC_NAME]
    )

    missing = data["capex_weighted_macme_pct"].isna()
    if missing.any():
        missing_ids = data.loc[missing, "case_id"].astype(str).tolist()
        raise ValueError(
            "Missing error values for "
            f"{len(missing_ids)} selected cases. First cases:\n"
            + "\n".join(missing_ids[:20])
        )

    return data


def validate_plot_inputs(data: pd.DataFrame) -> None:
    """Print the selected experiment dimensions and check proxy weights."""
    countries = sorted(str(x) for x in data["country"].dropna().unique())
    ks = sorted(int(x) for x in data["k_periods"].dropna().unique())
    weights = sorted(float(x) for x in data["lambda_soc"].dropna().unique())
    horizon_lengths = sorted(
        round(float(x), 3) for x in data["horizon_years"].dropna().unique()
    )

    print("Figure 4 — Capex-weighted MACME")
    print("==============================")
    print(f"Cases:          {len(data)}")
    print(f"Countries:      {countries}")
    print(f"Horizon years:  {horizon_lengths}")
    print(f"k:              {ks}")
    print(f"W_P:            {weights}")

    outside_range = [
        weight for weight in weights if weight < 0.0 or weight > 1.0
    ]
    if outside_range:
        raise ValueError(
            "Proxy weights must lie within [0, 1] for the fixed plasma "
            f"mapping. Found: {outside_range}"
        )

    if not any(np.isclose(weight, 0.0) for weight in weights):
        print(
            "\nWarning: no W_P = 0 cases were found, so the original TSA "
            "reference method will not appear in the figure."
        )


def build_wp_colours(weights: list[float]) -> dict[float, object]:
    """Map W_P=0..1 onto the lower 90% of the plasma colour map."""
    cmap = plt.get_cmap("plasma")
    return {
        weight: cmap(PLASMA_MIN + weight * (PLASMA_MAX - PLASMA_MIN))
        for weight in weights
    }


def build_wp_offsets(
    weights: list[float],
    *,
    width: float = CHANNEL_WIDTH,
) -> dict[float, float]:
    """Assign each W_P value a vertical channel within each k group."""
    if len(weights) == 1:
        return {weights[0]: 0.0}

    offsets = np.linspace(-width / 2, width / 2, len(weights))
    return dict(zip(weights, offsets, strict=True))


def build_replicate_jitter(
    data: pd.DataFrame,
    *,
    width: float = REPLICATE_JITTER_WIDTH,
) -> dict[tuple[str, pd.Timestamp, pd.Timestamp], float]:
    """Give country/horizon replicates a small deterministic x jitter."""
    replicates = sorted(
        {
            (
                str(row.country),
                pd.Timestamp(row.start_date),
                pd.Timestamp(row.end_date),
            )
            for row in data.itertuples()
        }
    )

    if len(replicates) == 1:
        return {replicates[0]: 0.0}

    offsets = np.linspace(-width / 2, width / 2, len(replicates))
    return dict(zip(replicates, offsets, strict=True))


def legend_label(weight: float) -> str:
    """Use an explicit label for the no-proxy/original TSA cases."""
    if np.isclose(weight, 0.0):
        return "$W_P = 0$\n(Original TSA)"
    return rf"$W_P = {weight:g}$"


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

    k_values = sorted(int(k) for k in data["k_periods"].unique())
    wp_values = sorted(float(w) for w in data["lambda_soc"].unique())

    if not k_values:
        raise ValueError("No representative-period counts were found to plot.")
    if not wp_values:
        raise ValueError("No proxy weights were found to plot.")

    wp_colours = build_wp_colours(wp_values)
    wp_offsets = build_wp_offsets(wp_values)
    replicate_jitter = build_replicate_jitter(data)

    figsize = (width_px / dpi, height_px / dpi)

    rc = {
        "font.size": 10.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "legend.title_fontsize": 10,
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

        k_positions = {
            k: float(index) for index, k in enumerate(k_values)
        }

        for k in k_values:
            k_data = data.loc[data["k_periods"].eq(k)]
            k_x = k_positions[k]

            for wp in wp_values:
                wp_data = k_data.loc[
                    np.isclose(k_data["lambda_soc"], wp)
                ].copy()

                if wp_data.empty:
                    continue

                centre = k_x + wp_offsets[wp]

                xs = np.array(
                    [
                        centre
                        + replicate_jitter[
                            (
                                str(row.country),
                                pd.Timestamp(row.start_date),
                                pd.Timestamp(row.end_date),
                            )
                        ]
                        for row in wp_data.itertuples()
                    ],
                    dtype=float,
                )
                ys = wp_data["capex_weighted_macme_pct"].to_numpy(
                    dtype=float
                )

                ax.scatter(
                    xs,
                    ys,
                    s=POINT_SIZE,
                    color=wp_colours[wp],
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
                    linewidth=2.5,
                    zorder=4,
                )
                ax.hlines(
                    median,
                    centre - MEDIAN_HALF_WIDTH,
                    centre + MEDIAN_HALF_WIDTH,
                    color="0.12",
                    linewidth=1.5,
                    zorder=5,
                )

        # Visually separate the main k groups while retaining the W_P
        # sub-channels within each group.
        positions = [k_positions[k] for k in k_values]
        for left, right in zip(
            positions[:-1],
            positions[1:],
            strict=True,
        ):
            ax.axvline(
                (left + right) / 2,
                color="0.90",
                linewidth=0.7,
                zorder=0,
            )

        # Zero is the actual reference model capacity; +/-10% are retained as
        # a useful visual tolerance guide from the earlier analysis.
        ax.axhline(
            0.0,
            color="0.3",
            linewidth=0.5,
            linestyle="-",
            zorder=1,
        )
        # for error in (-REFERENCE_ERROR_PCT, REFERENCE_ERROR_PCT):
        #     ax.axhline(
        #         error,
        #         color="0.45",
        #         linewidth=0.8,
        #         linestyle=":",
        #         zorder=1,
        #     )
        # ax.axhspan(
        #     -REFERENCE_ERROR_PCT,
        #     REFERENCE_ERROR_PCT,
        #     facecolor="0.95",   # very light grey
        #     edgecolor="none",
        #     zorder=0,
        # )


        ax.set_xticks(positions)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlim(
            min(positions) - 0.52,
            max(positions) + 0.52,
        )
        ax.set_xlabel(r"Representative periods, $k$")
        ax.set_ylabel("Capex-weighted MACME (%)")

        # ax.yaxis.set_major_locator(MultipleLocator(10))
        # ax.set_ylim(top=50)

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
                markerfacecolor=wp_colours[wp],
                markeredgecolor="none",
                markersize=6.5,
                label=legend_label(wp),
            )
            for wp in wp_values
        ]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="0.12",
                linewidth=2.0,
                label="Median",
            )
        )

        legend = ax.legend(
            handles=legend_handles,
            # title="SoC proxy weight",
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(legend_handles), 6),
            frameon=False,
            columnspacing=1.25,
            handletextpad=0.55,
        )
        legend.get_texts()[0].set_multialignment("center")

        return fig


def save_figure(
    fig: plt.Figure,
    *,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    """Save the same figure as raster PNG and vector PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"

    # Do not use bbox_inches="tight": that would alter the requested PNG
    # canvas dimensions. constrained_layout handles the spacing instead.
    fig.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
    )

    return png_path, pdf_path


def main() -> None:
    args = parse_args()

    data = load_results(args.source_dir)
    validate_plot_inputs(data)

    fig = make_figure(
        data,
        width_px=args.width_px,
        height_px=args.height_px,
        dpi=args.dpi,
    )

    png_path, pdf_path = save_figure(
        fig,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
