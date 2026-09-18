"""Compare key proxy-design metrics across experiment cases."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_proxy_experiment_comparison(
    case_ids: Sequence[str],
    *,
    results_dir: str | Path = "results",
    output_path: str | Path | None = None,
):
    """Plot four proxy-design metrics for selected experiment cases.

    Parameters
    ----------
    case_ids
        Case IDs to compare, in the desired x-axis order.
    results_dir
        Directory containing consolidated results Parquet files.
    output_path
        Optional path at which to save the figure.

    Returns
    -------
    tuple
        ``(fig, axes)``
    """
    if not case_ids:
        raise ValueError("At least one case_id must be supplied.")

    results_dir = Path(results_dir)

    parameters = pd.read_parquet(
        results_dir / "parameters.parquet"
    )
    investment_metrics = pd.read_parquet(
        results_dir / "investment_metrics.parquet"
    )
    signal_metrics = pd.read_parquet(
        results_dir / "signal_metrics.parquet"
    )

    case_ids = list(case_ids)

    labels = _experiment_labels(
        parameters,
        case_ids,
    )

    panels = [
        (
            "Reference level nRMSE window=90d",
            _signal_metric(
                signal_metrics,
                case_ids,
                error_family="reference_approximation",
                signal_type="level",
                metric="nrmse",
                normalisation_basis="reference_proxy_full_range",
                window_half_width_days=90,
            ),
            "nRMSE (%)",
            100.0,
            False,
        ),
        (
            "Reference delta nRMSE window=90d",
            _signal_metric(
                signal_metrics,
                case_ids,
                error_family="reference_approximation",
                signal_type="delta",
                metric="nrmse",
                normalisation_basis="reference_proxy_full_range",
                window_half_width_days=90,
            ),
            "nRMSE (%)",
            100.0,
            False,
        ),
        (
            "LDES capacity error",
            _investment_metric(
                investment_metrics,
                case_ids,
                "ldes_capacity_error_signed",
            ),
            "Capacity error (%)",
            100.0,
            True,
        ),
        # (
        #     "MACME",
        #     _investment_metric(
        #         investment_metrics,
        #         case_ids,
        #         "macme",
        #     ),
        #     "MACME (%)",
        #     100.0,
        #     False,
        # ),
        (
            "CAPEX-weighted MACME",
            _investment_metric(
                investment_metrics,
                case_ids,
                "macme_capex_weighted_annualised",
            ),
            "Weighted MACME (%)",
            100.0,
            False,
        ),
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 4.5),
        layout="constrained",
    )

    x = range(len(case_ids))

    for ax, (
        title,
        values,
        ylabel,
        scale,
        zero_line,
    ) in zip(
        axes,
        panels,
        strict=True,
    ):
        y = [
            values[case_id] * scale
            for case_id in case_ids
        ]

        ax.plot(
            x,
            y,
            marker="o",
            linestyle="none",
            markersize=7,
        )

        if zero_line:
            ax.axhline(
                0,
                linewidth=1,
                linestyle="--",
            )

        ax.set_title(title)
        ax.set_ylabel(ylabel)

        ax.set_xticks(
            list(x),
            [labels[case_id] for case_id in case_ids],
            rotation=35,
            ha="right",
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

    return fig, axes


def _experiment_labels(
    parameters: pd.DataFrame,
    case_ids: list[str],
) -> dict[str, str]:
    """Return experiment-name labels for the requested cases."""
    selected = parameters.loc[
        parameters["case_id"].isin(case_ids),
        [
            "case_id",
            "experiment_name",
        ],
    ].drop_duplicates()

    duplicates = selected["case_id"].duplicated(
        keep=False
    )

    if duplicates.any():
        raise RuntimeError(
            "parameters.parquet contains multiple experiment names "
            "for the same case_id."
        )

    labels = selected.set_index(
        "case_id"
    )["experiment_name"].to_dict()

    missing = [
        case_id
        for case_id in case_ids
        if case_id not in labels
    ]

    if missing:
        raise KeyError(
            "Case IDs not found in parameters.parquet: "
            f"{missing}"
        )

    return labels


def _investment_metric(
    metrics: pd.DataFrame,
    case_ids: list[str],
    metric: str,
) -> dict[str, float]:
    """Extract one investment metric for each requested case."""
    selected = metrics.loc[
        metrics["case_id"].isin(case_ids)
        & (metrics["metric"] == metric),
        [
            "case_id",
            "value",
        ],
    ]

    return _to_case_values(
        selected,
        case_ids,
        description=metric,
    )


def _signal_metric(
    metrics: pd.DataFrame,
    case_ids: list[str],
    *,
    error_family: str,
    signal_type: str,
    metric: str,
    normalisation_basis: str | None,
    window_half_width_days: int | None,
) -> dict[str, float]:
    """Extract one precisely defined signal metric per requested case."""
    mask = (
        metrics["case_id"].isin(case_ids)
        & (metrics["error_family"] == error_family)
        & (metrics["signal_type"] == signal_type)
        & (metrics["metric"] == metric)
    )

    if normalisation_basis is None:
        mask &= metrics[
            "normalisation_basis"
        ].isna()
    else:
        mask &= (
            metrics["normalisation_basis"]
            == normalisation_basis
        )

    if window_half_width_days is None:
        mask &= metrics[
            "window_half_width_days"
        ].isna()
    else:
        mask &= (
            metrics["window_half_width_days"]
            == window_half_width_days
        )

    selected = metrics.loc[
        mask,
        [
            "case_id",
            "value",
        ],
    ]

    description = (
        f"{error_family} / {signal_type} / {metric} / "
        f"{normalisation_basis} / window={window_half_width_days}"
    )

    return _to_case_values(
        selected,
        case_ids,
        description=description,
    )


def _to_case_values(
    selected: pd.DataFrame,
    case_ids: list[str],
    *,
    description: str,
) -> dict[str, float]:
    """Validate and map one metric value to each case."""
    counts = selected.groupby(
        "case_id"
    ).size()

    duplicates = counts[
        counts > 1
    ]

    if not duplicates.empty:
        raise RuntimeError(
            f"Metric {description!r} produced multiple rows for cases: "
            f"{duplicates.index.tolist()}"
        )

    values = selected.set_index(
        "case_id"
    )["value"].to_dict()

    missing = [
        case_id
        for case_id in case_ids
        if case_id not in values
    ]

    if missing:
        raise KeyError(
            f"Metric {description!r} is unavailable for case IDs: "
            f"{missing}"
        )

    return values
