"""Create Figure 7: runtime results.

The figure compares the runtime of the selected clustered paper method
(k-means + medoid, default W_P = 0.5) against the corresponding
full-resolution Calliope reference runs.

Visual encoding
---------------
- x-axis: modelling horizon (2, 5, 10 years)
- colour: representative-period count k
- grey: full-resolution reference runtime
- y-axis: runtime in minutes on a logarithmic scale
- plotted statistic: median by default (mean available by CLI option)

The clustered workflow runtime uses the consolidated runtime_method_seconds
(or runtime_methods_seconds) parameter, which combines SoC proxy, TSA, and
Calliope runtimes.

Reference runtimes are read from:
    reference_runtime_calliope_seconds

Repeated reference runtimes attached to multiple clustered case rows are
deduplicated before aggregation.

The script also reports mean and median TSA and SoC-proxy runtimes,
overall and by horizon, and saves the numerical summaries to CSV.

Source:
    results/2_5_10_year/parameters.parquet

Outputs:
    results/figures/fig_7_runtime_results/
        fig_7_runtime_results.png
        fig_7_runtime_results.pdf
        fig_7_runtime_results_clustered_summary.csv
        fig_7_runtime_results_component_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter


# ---------------------------------------------------------------------------
# Figure-specific configuration
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_DIR = Path("results/2_5_10_year")
DEFAULT_OUTPUT_DIR = Path("results/figures/fig_7_runtime_results")
OUTPUT_STEM = "fig_7_runtime_results"

DEFAULT_WIDTH_PX = 1600
DEFAULT_HEIGHT_PX = 1200
DEFAULT_DPI = 300

CLUSTER_METHOD = "kmeans"
REPRESENTATION_METHOD = "medoid"
TARGET_PROXY_WEIGHT = 0.5

HORIZON_ORDER = (2, 5, 10)
HORIZON_TOLERANCE_YEARS = 0.10
DEFAULT_K_VALUES = (10, 20, 30, 40, 50, 60, 90, 180)

DEFAULT_SUMMARY_STAT = "median"

PLASMA_MIN = 0.0
PLASMA_MAX = 0.90

POINT_SIZE = 28
REFERENCE_POINT_SIZE = 34

TOTAL_RUNTIME_CANDIDATES = (
    "runtime_method_seconds",
    "runtime_methods_seconds",
)

TSA_RUNTIME_CANDIDATES = (
    "runtime_tsa_seconds",
    "runtime_tsam_seconds",
    "tsa_runtime_seconds",
    "tsam_runtime_seconds",
)

PROXY_RUNTIME_CANDIDATES = (
    "runtime_soc_proxy_seconds",
    "runtime_proxy_seconds",
    "runtime_socproxy_seconds",
    "soc_proxy_runtime_seconds",
    "proxy_runtime_seconds",
)

REFERENCE_RUNTIME_COLUMN = "reference_runtime_calliope_seconds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot clustered workflow and full-resolution reference runtimes "
            "for 2-, 5-, and 10-year horizons."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing parameters.parquet.",
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
        "--wp",
        type=float,
        default=TARGET_PROXY_WEIGHT,
        help=(
            "SoC proxy weight used for clustered runtime results "
            f"(default: {TARGET_PROXY_WEIGHT:g})."
        ),
    )
    parser.add_argument(
        "--summary-stat",
        choices=["median", "mean"],
        default=DEFAULT_SUMMARY_STAT,
        help=(
            "Statistic used for the plotted clustered and reference runtime "
            f"summaries (default: {DEFAULT_SUMMARY_STAT})."
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
    """Resolve one runtime column while tolerating known naming variants."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    runtime_columns = sorted(
        column for column in frame.columns if "runtime" in column.lower()
    )
    raise ValueError(
        f"Could not find the {purpose} runtime column. "
        f"Tried: {list(candidates)}\n"
        f"Runtime-like columns available in parameters.parquet:\n"
        f"{runtime_columns}"
    )


def assign_horizon_duration(parameters: pd.DataFrame) -> pd.DataFrame:
    """Assign each row to the nearest requested 2-, 5-, or 10-year horizon."""
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
    proxy_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load clustered runtimes and deduplicated full-resolution references."""
    parameters_path = source_dir / "parameters.parquet"
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
        purpose="combined clustered-workflow",
    )
    tsa_runtime_column = resolve_column(
        parameters,
        TSA_RUNTIME_CANDIDATES,
        purpose="TSA",
    )
    proxy_runtime_column = resolve_column(
        parameters,
        PROXY_RUNTIME_CANDIDATES,
        purpose="SoC proxy",
    )

    resolved_columns = {
        "total": total_runtime_column,
        "tsa": tsa_runtime_column,
        "proxy": proxy_runtime_column,
        "reference": REFERENCE_RUNTIME_COLUMN,
    }

    parameters = assign_horizon_duration(parameters)

    parameters["case_id"] = parameters["case_id"].astype(str)
    parameters["k_periods"] = parameters["k_periods"].astype(int)
    parameters["lambda_soc"] = parameters["lambda_soc"].astype(float)

    method_mask = (
        parameters["cluster_method"].astype(str).str.lower().eq(CLUSTER_METHOD)
        & parameters["representation_method"]
        .astype(str)
        .str.lower()
        .eq(REPRESENTATION_METHOD)
    )

    k_values = sorted(set(int(k) for k in k_values))
    if not k_values:
        raise ValueError("At least one k value must be supplied.")

    clustered = parameters.loc[
        method_mask
        & parameters["horizon_duration"].isin(HORIZON_ORDER)
        & parameters["k_periods"].isin(k_values)
        & np.isclose(parameters["lambda_soc"], proxy_weight)
    ].copy()

    if clustered.empty:
        raise ValueError(
            "No clustered cases remain after filtering for "
            f"{CLUSTER_METHOD} + {REPRESENTATION_METHOD}, "
            f"W_P={proxy_weight:g}, horizons={HORIZON_ORDER}, "
            f"and k={k_values}."
        )

    clustered = clustered.rename(
        columns={
            total_runtime_column: "runtime_method_seconds",
            tsa_runtime_column: "runtime_tsa_seconds",
            proxy_runtime_column: "runtime_soc_proxy_seconds",
        }
    )

    required_runtime_columns = [
        "runtime_method_seconds",
        "runtime_tsa_seconds",
        "runtime_soc_proxy_seconds",
        REFERENCE_RUNTIME_COLUMN,
    ]

    missing_runtime = clustered[required_runtime_columns].isna()
    if missing_runtime.any().any():
        missing_counts = missing_runtime.sum()
        missing_counts = missing_counts[missing_counts.gt(0)]
        raise ValueError(
            "Selected runtime data contain missing values:\n"
            + missing_counts.to_string()
        )

    for column in required_runtime_columns:
        non_positive = clustered[column].le(0)
        if non_positive.any():
            raise ValueError(
                f"{column} contains {int(non_positive.sum())} non-positive "
                "values. Log-scale runtime plots require strictly positive "
                "runtimes."
            )

    reference_identity = [
        "country",
        "start_date",
        "end_date",
        "horizon_duration",
        REFERENCE_RUNTIME_COLUMN,
    ]

    references = (
        parameters.loc[
            method_mask & parameters["horizon_duration"].isin(HORIZON_ORDER),
            reference_identity,
        ]
        .dropna(subset=[REFERENCE_RUNTIME_COLUMN])
        .drop_duplicates()
        .copy()
    )

    if references.empty:
        raise ValueError(
            f"No values were found in {REFERENCE_RUNTIME_COLUMN}."
        )

    if references[REFERENCE_RUNTIME_COLUMN].le(0).any():
        raise ValueError(
            f"{REFERENCE_RUNTIME_COLUMN} contains non-positive values; "
            "these cannot be shown on a logarithmic axis."
        )

    return clustered, references, resolved_columns


def aggregate_runtime(
    clustered: pd.DataFrame,
    references: pd.DataFrame,
    *,
    summary_stat: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate clustered runtimes by horizon/k and references by horizon."""
    aggregation = summary_stat

    clustered_summary = (
        clustered.groupby(
            ["horizon_duration", "k_periods"],
            as_index=False,
        )
        .agg(
            runtime_seconds=("runtime_method_seconds", aggregation),
            n_cases=("case_id", "nunique"),
        )
        .sort_values(["horizon_duration", "k_periods"])
        .reset_index(drop=True)
    )

    reference_summary = (
        references.groupby(
            "horizon_duration",
            as_index=False,
        )
        .agg(
            reference_runtime_seconds=(REFERENCE_RUNTIME_COLUMN, aggregation),
            n_reference_runs=(REFERENCE_RUNTIME_COLUMN, "size"),
        )
        .sort_values("horizon_duration")
        .reset_index(drop=True)
    )

    clustered_summary["runtime_minutes"] = (
        clustered_summary["runtime_seconds"] / 60.0
    )
    reference_summary["reference_runtime_minutes"] = (
        reference_summary["reference_runtime_seconds"] / 60.0
    )

    return clustered_summary, reference_summary


def calculate_component_summary(
    clustered: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate TSA and SoC-proxy runtime statistics for paper reporting."""

    def one_summary(group: pd.DataFrame, label: str) -> dict[str, object]:
        tsa = group["runtime_tsa_seconds"]
        proxy = group["runtime_soc_proxy_seconds"]

        return {
            "horizon": label,
            "n_cases": len(group),
            "tsa_mean_seconds": tsa.mean(),
            "tsa_median_seconds": tsa.median(),
            "tsa_mean_minutes": tsa.mean() / 60.0,
            "tsa_median_minutes": tsa.median() / 60.0,
            "soc_proxy_mean_seconds": proxy.mean(),
            "soc_proxy_median_seconds": proxy.median(),
            "soc_proxy_mean_minutes": proxy.mean() / 60.0,
            "soc_proxy_median_minutes": proxy.median() / 60.0,
        }

    rows = [
        one_summary(
            clustered.loc[clustered["horizon_duration"].eq(horizon)],
            f"{horizon}-year",
        )
        for horizon in HORIZON_ORDER
    ]
    rows.append(one_summary(clustered, "overall"))

    return pd.DataFrame(rows)


def print_summary(
    clustered: pd.DataFrame,
    references: pd.DataFrame,
    clustered_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    component_summary: pd.DataFrame,
    *,
    summary_stat: str,
    proxy_weight: float,
    resolved_columns: dict[str, str],
) -> None:
    """Print experiment dimensions and reporting statistics."""
    ks = sorted(int(x) for x in clustered["k_periods"].unique())
    countries = sorted(str(x) for x in clustered["country"].unique())

    print("Figure 7 — Runtime results")
    print("==========================")
    print(f"Plot statistic:   {summary_stat}")
    print(f"Method:           {CLUSTER_METHOD} + {REPRESENTATION_METHOD}")
    print(f"W_P:              {proxy_weight:g}")
    print(f"Countries:        {countries}")
    print(f"k:                {ks}")
    print(f"Clustered cases:  {len(clustered)}")
    print(f"Reference runs:   {len(references)}")

    print("\nResolved runtime columns")
    print("------------------------")
    for key, value in resolved_columns.items():
        print(f"{key:10s}: {value}")

    print("\nPlotted runtime summary")
    print("-----------------------")
    for horizon in HORIZON_ORDER:
        horizon_data = clustered_summary.loc[
            clustered_summary["horizon_duration"].eq(horizon)
        ].sort_values("k_periods")
        reference_data = reference_summary.loc[
            reference_summary["horizon_duration"].eq(horizon)
        ]

        print(f"{horizon}-year:")
        for row in horizon_data.itertuples(index=False):
            print(
                f"  k={int(row.k_periods):>3}: "
                f"{float(row.runtime_minutes):.2f} min "
                f"(n={int(row.n_cases)})"
            )

        if not reference_data.empty:
            row = reference_data.iloc[0]
            print(
                "  reference: "
                f"{float(row['reference_runtime_minutes']):.2f} min "
                f"(n={int(row['n_reference_runs'])})"
            )

    overall = component_summary.loc[
        component_summary["horizon"].eq("overall")
    ].iloc[0]

    print("\nRuntime components for paper reporting")
    print("--------------------------------------")
    print(
        "TSA runtime: "
        f"mean {overall['tsa_mean_seconds']:.2f} s "
        f"({overall['tsa_mean_minutes']:.3f} min); "
        f"median {overall['tsa_median_seconds']:.2f} s"
    )
    print(
        "SoC proxy runtime: "
        f"mean {overall['soc_proxy_mean_seconds']:.2f} s "
        f"({overall['soc_proxy_mean_minutes']:.3f} min); "
        f"median {overall['soc_proxy_median_seconds']:.2f} s"
    )


def build_k_colours(k_values: list[int]) -> dict[int, object]:
    """Distribute the selected k values across the lower 90% of plasma."""
    cmap = plt.get_cmap("plasma")
    if len(k_values) == 1:
        return {k_values[0]: cmap((PLASMA_MIN + PLASMA_MAX) / 2)}

    positions = np.linspace(PLASMA_MIN, PLASMA_MAX, len(k_values))
    return {
        k: cmap(position)
        for k, position in zip(k_values, positions, strict=True)
    }


def make_figure(
    clustered_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    *,
    width_px: int,
    height_px: int,
    dpi: int,
) -> plt.Figure:
    """Build the publication runtime figure."""
    if width_px <= 0 or height_px <= 0 or dpi <= 0:
        raise ValueError(
            "width-px, height-px, and dpi must all be positive."
        )

    figsize = (width_px / dpi, height_px / dpi)

    k_values = sorted(int(x) for x in clustered_summary["k_periods"].unique())
    k_colours = build_k_colours(k_values)

    horizon_positions = {
        horizon: float(index) for index, horizon in enumerate(HORIZON_ORDER)
    }

    rc = {
        "font.size": 10.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,
            constrained_layout=False,
        )

        # Clustered-runtime points grouped by horizon, coloured by k.
        for k in k_values:
            k_data = clustered_summary.loc[
                clustered_summary["k_periods"].eq(k)
            ].sort_values("horizon_duration")

            if k_data.empty:
                continue

            xs = [
                horizon_positions[int(h)]
                for h in k_data["horizon_duration"]
            ]
            ys = k_data["runtime_minutes"].to_numpy(dtype=float)

            ax.scatter(
                xs,
                ys,
                s=POINT_SIZE,
                color=k_colours[k],
                alpha=0.95,
                # linewidths=0.35,
                # edgecolors="0.15",
                zorder=3,
                label=f"k={k}",
            )

        # Full-resolution reference points in grey.
        if not reference_summary.empty:
            xs = [
                horizon_positions[int(h)]
                for h in reference_summary["horizon_duration"]
            ]
            ys = reference_summary["reference_runtime_minutes"].to_numpy(
                dtype=float
            )

            ax.scatter(
                xs,
                ys,
                s=REFERENCE_POINT_SIZE,
                color="0.55",
                alpha=1.0,
                # linewidths=0.45,
                # edgecolors="0.20",
                zorder=4,
                label="ref",
            )

        ax.set_xticks(
            [horizon_positions[h] for h in HORIZON_ORDER]
        )
        ax.set_xticklabels([str(h) for h in HORIZON_ORDER])
        ax.set_xlim(
            min(horizon_positions.values()) - 0.35,
            max(horizon_positions.values()) + 0.35,
        )

        ax.set_xlabel("Horizon (years)")
        ax.set_ylabel("Runtime (minutes, log scale)")
        ax.set_yscale("log")

        # Show only the major 1, 10, and 100 min markers and grid lines.
        major_ticks = [1.0, 10.0, 100.0]
        ax.yaxis.set_major_locator(FixedLocator(major_ticks))
        ax.yaxis.set_major_formatter(
            FixedFormatter(["1.0 min", "10 min", "100 min"])
        )
        ax.yaxis.set_minor_formatter(NullFormatter())

        # Expand limits slightly while respecting positive log scale.
        all_minutes = list(clustered_summary["runtime_minutes"].to_numpy(dtype=float))
        all_minutes.extend(
            reference_summary["reference_runtime_minutes"].to_numpy(dtype=float)
        )
        ymin = min(all_minutes)
        ymax = max(all_minutes)
        ax.set_ylim(ymin * 0.75, ymax * 1.35)

        ax.grid(
            axis="y",
            which="major",
            linewidth=0.7,
            alpha=0.35,
            linestyle=(0, (1.5, 2.5)),
            zorder=0,
        )
        ax.grid(axis="x", visible=False)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=k_colours[k],
                markeredgecolor="1",
                markersize=6.5,
                label=f"k={k}",
            )
            for k in k_values
        ]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="0.55",
                markeredgecolor="1",
                markersize=6.8,
                label="ref",
            )
        )

        legend_handles = legend_handles[::-1]

        legend = fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.82, 0.5),
            ncol=1,
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.55,
        )

        fig.subplots_adjust(
            left=0.18,
            right=0.82,
            top=0.96,
            bottom=0.12,
        )

        return fig


def save_outputs(
    fig: plt.Figure,
    clustered_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    component_summary: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path, Path, Path]:
    """Save figure and numerical summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    clustered_csv_path = output_dir / f"{OUTPUT_STEM}_clustered_summary.csv"
    component_csv_path = output_dir / f"{OUTPUT_STEM}_component_summary.csv"

    fig.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
    )

    reference_for_merge = reference_summary[
        [
            "horizon_duration",
            "reference_runtime_seconds",
            "reference_runtime_minutes",
            "n_reference_runs",
        ]
    ]

    clustered_output = clustered_summary.merge(
        reference_for_merge,
        on="horizon_duration",
        how="left",
        validate="many_to_one",
    )

    clustered_output.to_csv(clustered_csv_path, index=False)
    component_summary.to_csv(component_csv_path, index=False)

    return png_path, pdf_path, clustered_csv_path, component_csv_path


def main() -> None:
    args = parse_args()

    clustered, references, resolved_columns = load_results(
        args.source_dir,
        k_values=args.k_values,
        proxy_weight=args.wp,
    )

    clustered_summary, reference_summary = aggregate_runtime(
        clustered,
        references,
        summary_stat=args.summary_stat,
    )

    component_summary = calculate_component_summary(clustered)

    print_summary(
        clustered,
        references,
        clustered_summary,
        reference_summary,
        component_summary,
        summary_stat=args.summary_stat,
        proxy_weight=args.wp,
        resolved_columns=resolved_columns,
    )

    fig = make_figure(
        clustered_summary,
        reference_summary,
        width_px=args.width_px,
        height_px=args.height_px,
        dpi=args.dpi,
    )

    outputs = save_outputs(
        fig,
        clustered_summary,
        reference_summary,
        component_summary,
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
