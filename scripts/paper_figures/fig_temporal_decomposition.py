"""Create the temporal-decomposition sensitivity figure.

The figure compares reference SoC Proxy delta error across decomposition
methods and smoothing timescales for 10-year sensitivity cases.

Visual encoding
---------------
- x-axis: smoothing timescale (hours; logarithmic scale)
- colour: decomposition method
- marker: median across pooled country/weather cases
- vertical whisker: interquartile range (Q25--Q75)
- black outline: selected Gaussian 24-hour formulation
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
# Paths / output
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST_PATH = Path(
    "results/sensitivity_smoothing/sensitivity_smoothing_manifest.parquet"
)
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_DIR = Path("results/figures/fig_temp_decomp_sensitivity")
OUTPUT_STEM = "fig_temp_decomp_sensitivity"

DEFAULT_WIDTH_PX = 1600
DEFAULT_HEIGHT_PX = 1600
DEFAULT_DPI = 300


# ---------------------------------------------------------------------------
# Plot conventions
# ---------------------------------------------------------------------------

TARGET_HORIZON_YEARS = 10

METHOD_ORDER = (
    "fft_lowpass",
    "gaussian",
    "moving_average",
)

METHOD_LABELS = {
    "fft_lowpass": "FFT low-pass",
    "gaussian": "Gaussian",
    "moving_average": "Moving average",
}

# Multiplicative offsets separate methods while preserving the log-scaled
# smoothing-timescale axis. Gaussian is centred on the nominal timescale.
METHOD_LOG10_OFFSETS = {
    "fft_lowpass": -0.055,
    "gaussian": 0.0,
    "moving_average": 0.055,
}

PLASMA_MIN = 0.08
PLASMA_MAX = 0.88

MEDIAN_MARKER_SIZE = 34
IQR_LINEWIDTH = 1.45
IQR_CAPSIZE = 3.0
IQR_ALPHA = 0.86
CONNECT_LINEWIDTH = 1.15
CONNECT_ALPHA = 0.82

SELECTED_METHOD = "gaussian"
SELECTED_TIMESCALE_HOURS = 48.0
SELECTED_MARKER_SIZE = 64
SELECTED_EDGEWIDTH = 1.15

LDES_TICK_INTERVAL = 10
PROXY_NRMSE_TICK_INTERVAL = 5

LDES_METRIC = "ldes_capacity_error_signed"

REFERENCE_PROXY_DELTA_SPEC = {
    "error_family": "reference_approximation",
    "signal_type": "delta",
    "metric": "nrmse",
    "normalisation_basis": "comparison_reference_full_range",
    "window_half_width_days": None,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the temporal-decomposition sensitivity figure using "
            "median markers and interquartile-range whiskers."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
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


# ---------------------------------------------------------------------------
# Validation / extraction
# ---------------------------------------------------------------------------


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


def _to_case_values(
    selected: pd.DataFrame,
    case_ids: list[str],
    *,
    description: str,
) -> dict[str, float]:
    """Validate and return exactly one metric value per requested case."""
    counts = selected.groupby("case_id").size()
    duplicates = counts[counts > 1]

    if not duplicates.empty:
        raise RuntimeError(
            f"Metric {description!r} produced multiple rows for cases: "
            f"{duplicates.index.tolist()}"
        )

    values = selected.set_index("case_id")["value"].to_dict()
    missing = [case_id for case_id in case_ids if case_id not in values]

    if missing:
        raise KeyError(
            f"Metric {description!r} is unavailable for case IDs: {missing}"
        )

    return values


def _investment_metric(
    metrics: pd.DataFrame,
    *,
    case_ids: list[str],
    metric: str,
) -> dict[str, float]:
    selected = metrics.loc[
        metrics["case_id"].isin(case_ids) & metrics["metric"].eq(metric),
        ["case_id", "value"],
    ].copy()

    return _to_case_values(
        selected,
        case_ids,
        description=metric,
    )


def _signal_metric(
    metrics: pd.DataFrame,
    *,
    case_ids: list[str],
    error_family: str,
    signal_type: str,
    metric: str,
    normalisation_basis: str | None,
    window_half_width_days: int | None,
) -> dict[str, float]:
    mask = (
        metrics["case_id"].isin(case_ids)
        & metrics["error_family"].eq(error_family)
        & metrics["signal_type"].eq(signal_type)
        & metrics["metric"].eq(metric)
    )

    if normalisation_basis is None:
        mask &= metrics["normalisation_basis"].isna()
    else:
        mask &= metrics["normalisation_basis"].eq(normalisation_basis)

    if window_half_width_days is None:
        mask &= metrics["window_half_width_days"].isna()
    else:
        mask &= metrics["window_half_width_days"].eq(window_half_width_days)

    selected = metrics.loc[mask, ["case_id", "value"]].copy()

    description = (
        f"{error_family} / {signal_type} / {metric} / "
        f"{normalisation_basis} / window={window_half_width_days}"
    )

    return _to_case_values(
        selected,
        case_ids,
        description=description,
    )


def _resolve_metric_table(
    filename: str,
    *,
    manifest_path: Path,
    results_dir: Path,
) -> Path:
    """Resolve a consolidated metric table used by the sensitivity figure.

    Sensitivity runs may be consolidated either into their own experiment
    directory or into the repository-level ``results`` tables. Prefer the
    experiment-local table when present and fall back to ``results_dir``.
    """
    candidates = (
        manifest_path.parent / filename,
        results_dir / filename,
    )

    for path in candidates:
        if path.exists():
            return path

    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find {filename}. Tried:\n{attempted}"
    )


def load_plot_data(
    *,
    manifest_path: Path,
    results_dir: Path,
) -> pd.DataFrame:
    """Load 10-year smoothing cases and reference proxy-delta error."""
    manifest = pd.read_parquet(manifest_path)

    signal_path = _resolve_metric_table(
        "signal_metrics.parquet",
        manifest_path=manifest_path,
        results_dir=results_dir,
    )
    signal_metrics = pd.read_parquet(signal_path)

    require_columns(
        manifest,
        {
            "case_id",
            "country",
            "start_year",
            "end_year",
            "duration_years",
            "decomposition_method",
            "time_horizon_hours",
        },
        table_name=str(manifest_path),
    )
    require_columns(
        signal_metrics,
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

    data = manifest.loc[
        manifest["duration_years"].eq(TARGET_HORIZON_YEARS),
        [
            "case_id",
            "country",
            "start_year",
            "end_year",
            "duration_years",
            "decomposition_method",
            "time_horizon_hours",
        ],
    ].copy()

    if data.empty:
        raise ValueError(
            f"No {TARGET_HORIZON_YEARS}-year cases were found in {manifest_path}."
        )

    data["case_id"] = data["case_id"].astype(str)
    signal_metrics["case_id"] = signal_metrics["case_id"].astype(str)

    recognised = set(METHOD_ORDER)
    unknown = sorted(set(data["decomposition_method"].dropna()) - recognised)
    if unknown:
        print(
            "Warning: ignoring unrecognised decomposition methods: "
            f"{unknown}"
        )
        data = data.loc[
            data["decomposition_method"].isin(recognised)
        ].copy()

    if data.empty:
        raise ValueError("No recognised decomposition methods remain after filtering.")

    proxy_delta_values = _signal_metric(
        signal_metrics,
        case_ids=data["case_id"].tolist(),
        **REFERENCE_PROXY_DELTA_SPEC,
    )

    data["soc_proxy_delta_nrmse_pct"] = (
        data["case_id"].map(proxy_delta_values) * 100.0
    )

    return data
# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarise_metric(
    data: pd.DataFrame,
    *,
    value_column: str,
    metric_name: str,
) -> pd.DataFrame:
    summary = (
        data.groupby(
            ["decomposition_method", "time_horizon_hours"],
            as_index=False,
        )
        .agg(
            n=(value_column, "size"),
            q25=(value_column, lambda x: x.quantile(0.25)),
            median=(value_column, "median"),
            q75=(value_column, lambda x: x.quantile(0.75)),
        )
        .sort_values(["decomposition_method", "time_horizon_hours"])
    )
    summary["metric"] = metric_name
    return summary


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    return summarise_metric(
        data,
        value_column="soc_proxy_delta_nrmse_pct",
        metric_name="soc_proxy_delta_nrmse_pct",
    )


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def build_method_colours(methods: list[str]) -> dict[str, object]:
    cmap = plt.get_cmap("plasma")
    positions = np.linspace(PLASMA_MIN, PLASMA_MAX, len(methods))
    return {
        method: cmap(position)
        for method, position in zip(methods, positions, strict=True)
    }


def _format_horizon(hours: float) -> str:
    hours = int(round(hours))

    if hours >= 24 * 30:
        return f"{hours}h\n({hours / (24 * 30):.0f}mo)"
    if hours >= 24 * 7:
        return f"{hours}h\n({hours / (24 * 7):.0f}wk)"
    if hours >= 24:
        return f"{hours}h\n({hours / 24:.0f}d)"
    return f"{hours}h"


def _method_x(horizon: float, method: str) -> float:
    factor = 10 ** METHOD_LOG10_OFFSETS.get(method, 0.0)
    return float(horizon) * factor


def plot_summary_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    metric_name: str,
    methods: list[str],
    method_colours: dict[str, object],
) -> None:
    """Plot method medians with Q25-Q75 vertical whiskers."""
    panel = summary.loc[summary["metric"].eq(metric_name)]

    for method in methods:
        method_data = panel.loc[
            panel["decomposition_method"].eq(method)
        ].sort_values("time_horizon_hours")

        if method_data.empty:
            continue

        colour = method_colours[method]
        xs = [
            _method_x(float(horizon), method)
            for horizon in method_data["time_horizon_hours"]
        ]
        medians = method_data["median"].to_numpy(dtype=float)
        q25 = method_data["q25"].to_numpy(dtype=float)
        q75 = method_data["q75"].to_numpy(dtype=float)

        # Connecting median lines retain the response-shape information from
        # the original sensitivity figure without plotting every observation.
        ax.plot(
            xs,
            medians,
            color=colour,
            linewidth=CONNECT_LINEWIDTH,
            alpha=CONNECT_ALPHA,
            zorder=2,
        )

        ax.errorbar(
            xs,
            medians,
            yerr=np.vstack([medians - q25, q75 - medians]),
            fmt="o",
            markersize=np.sqrt(MEDIAN_MARKER_SIZE),
            color=colour,
            ecolor=colour,
            elinewidth=IQR_LINEWIDTH,
            capsize=IQR_CAPSIZE,
            capthick=IQR_LINEWIDTH,
            alpha=IQR_ALPHA,
            markeredgewidth=0,
            zorder=3,
        )

        # Highlight the formulation retained for the paper.
        selected = method_data.loc[
            np.isclose(
                method_data["time_horizon_hours"].astype(float),
                SELECTED_TIMESCALE_HOURS,
            )
        ]
        if method == SELECTED_METHOD and not selected.empty:
            row = selected.iloc[0]
            # ax.scatter(
            #     [_method_x(float(row["time_horizon_hours"]), method)],
            #     [float(row["median"])],
            #     s=SELECTED_MARKER_SIZE,
            #     facecolors="none",
            #     edgecolors="black",
            #     linewidths=SELECTED_EDGEWIDTH,
            #     zorder=5,
            # )


def set_ldes_axis(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel = summary.loc[summary["metric"].eq("ldes_capacity_error_pct")]

    ymin = min(0.0, float(panel["q25"].min()))
    ymax = max(0.0, float(panel["q75"].max()))

    rounded_min = LDES_TICK_INTERVAL * np.floor(ymin / LDES_TICK_INTERVAL)
    rounded_max = LDES_TICK_INTERVAL * np.ceil(ymax / LDES_TICK_INTERVAL)
    span = max(LDES_TICK_INTERVAL, rounded_max - rounded_min)
    pad = max(0.8, 0.035 * span)

    ax.set_ylim(rounded_min - pad, rounded_max + pad)
    ax.yaxis.set_major_locator(MultipleLocator(LDES_TICK_INTERVAL))
    ax.axhline(
        0.0,
        color="0.35",
        linewidth=0.8,
        linestyle="--",
        zorder=1,
    )


def set_proxy_error_axis(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel = summary.loc[summary["metric"].eq("soc_proxy_delta_nrmse_pct")]

    ymin = max(0.0, float(panel["q25"].min()))
    ymax = float(panel["q75"].max())
    pad = max(0.5, 0.05 * max(ymax - ymin, 1.0))

    ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax.yaxis.set_major_locator(MultipleLocator(PROXY_NRMSE_TICK_INTERVAL))


def make_figure(
    summary: pd.DataFrame,
    *,
    width_px: int,
    height_px: int,
    dpi: int,
) -> plt.Figure:
    figsize = (width_px / dpi, height_px / dpi)

    methods = [
        method
        for method in METHOD_ORDER
        if method in set(summary["decomposition_method"])
    ]
    if not methods:
        raise ValueError("No recognised decomposition methods were found.")

    method_colours = build_method_colours(methods)
    horizons = sorted(float(x) for x in summary["time_horizon_hours"].unique())

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
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            dpi=dpi,
            constrained_layout=False,
        )

        plot_summary_panel(
            ax,
            summary,
            metric_name="soc_proxy_delta_nrmse_pct",
            methods=methods,
            method_colours=method_colours,
        )

        ax.set_ylabel("SoC Proxy delta nRMSE (%)")
        ax.set_xlabel("Smoothing timescale (hours)")
        ax.set_xscale("log")
        ax.set_xticks(horizons)
        ax.set_xticklabels([_format_horizon(horizon) for horizon in horizons])
        ax.minorticks_off()
        ax.grid(axis="y", linewidth=0.6, alpha=0.28, zorder=0)

        set_proxy_error_axis(ax, summary)

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=method_colours[method],
                linewidth=CONNECT_LINEWIDTH,
                marker="o",
                markerfacecolor=method_colours[method],
                markeredgecolor="none",
                markersize=5.5,
                label=METHOD_LABELS[method],
            )
            for method in methods
        ]

        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=SELECTED_EDGEWIDTH,
                markersize=7.2,
                label="Selected: Gaussian, 24 h",
            )
        )

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.56, 0.98),
            ncol=len(legend_handles),
            frameon=False,
            columnspacing=1.3,
            handletextpad=0.5,
        )

        fig.text(
            0.56,
            0.91,
            "Markers show medians; whiskers show Q25--Q75.",
            ha="center",
            va="top",
            fontsize=7.5,
            color="0.4",
        )

        fig.subplots_adjust(
            left=0.14,
            right=0.98,
            top=0.84,
            bottom=0.18,
        )

        return fig
# ---------------------------------------------------------------------------
# Save / main
# ---------------------------------------------------------------------------


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

    data = load_plot_data(
        manifest_path=args.manifest_path,
        results_dir=args.results_dir,
    )
    summary = build_summary(data)

    print("Temporal-decomposition sensitivity — median + IQR")
    print("=================================================")
    print(f"Cases:      {len(data)}")
    print(f"Horizon:    {TARGET_HORIZON_YEARS} years")
    print(
        "Methods:    "
        + ", ".join(
            METHOD_LABELS[m]
            for m in METHOD_ORDER
            if m in set(data["decomposition_method"])
        )
    )
    print(
        "Timescales: "
        f"{sorted(float(x) for x in data['time_horizon_hours'].unique())} h"
    )
    print(
        "Selected:   "
        f"{METHOD_LABELS[SELECTED_METHOD]}, {SELECTED_TIMESCALE_HOURS:g} h"
    )

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
