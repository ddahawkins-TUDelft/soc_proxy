"""Plot storage SoC and SoC Proxy signals for one experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scripts.helpers._plots import (
    format_datetime_axis,
    plasma_colors,
    save_figure,
)


def plot_soc_proxy_comparison(
    reference_soc: pd.Series,
    reference_proxy: pd.Series,
    *,
    clustered_soc: pd.Series | None = None,
    clustered_proxy: pd.Series | None = None,
    title: str | None = None,
    ylabel: str = "Energy",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot complete SoC and SoC Proxy chronologies for one experiment.

    An unclustered comparison contains:

    * reference storage SoC;
    * reference SoC Proxy.

    When clustered signals are supplied, the figure additionally contains:

    * reconstructed clustered storage SoC;
    * clustered SoC Proxy.

    No normalisation, resampling, smoothing, or other transformation is
    performed. The supplied signals are plotted exactly as provided.

    Parameters
    ----------
    reference_soc
        Chronological storage state of charge from the reference model.
    reference_proxy
        SoC Proxy generated from the original chronological timeseries.
    clustered_soc
        Reconstructed chronological storage state of charge from a clustered
        model.
    clustered_proxy
        SoC Proxy generated from the TSAM-reconstructed timeseries.
    title
        Optional figure title.
    ylabel
        Y-axis label.
    output_path
        Optional destination for the completed figure.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    matplotlib.axes.Axes
        Axes containing the comparison.
    """
    _validate_signal(
        reference_soc,
        name="reference_soc",
    )
    _validate_signal(
        reference_proxy,
        name="reference_proxy",
    )

    _require_matching_index(
        reference_soc,
        reference_proxy,
        left_name="reference_soc",
        right_name="reference_proxy",
    )

    if (clustered_soc is None and clustered_proxy is not None) or (
        clustered_soc is not None and clustered_proxy is None
    ):
        raise ValueError(
            "clustered_soc and clustered_proxy must either both be "
            "supplied or both be omitted."
        )

    clustered = clustered_soc is not None

    if clustered:
        _validate_signal(
            clustered_soc,
            name="clustered_soc",
        )
        _validate_signal(
            clustered_proxy,
            name="clustered_proxy",
        )

        _require_matching_index(
            reference_soc,
            clustered_soc,
            left_name="reference_soc",
            right_name="clustered_soc",
        )
        _require_matching_index(
            reference_soc,
            clustered_proxy,
            left_name="reference_soc",
            right_name="clustered_proxy",
        )

    colors = plasma_colors((0.12, 0.36, 0.64, 0.88))

    fig, ax = plt.subplots(
        figsize=(12, 5),
    )

    # Signal type is also encoded by line style:
    # solid = physical SoC
    # dashed = SoC Proxy
    ax.plot(
        reference_soc.index,
        reference_soc,
        color=colors[0],
        linewidth=1.4,
        linestyle="-",
        label="Reference SoC",
    )

    ax.plot(
        reference_proxy.index,
        reference_proxy,
        color=colors[1],
        linewidth=1.2,
        linestyle="--",
        label="Reference SoC Proxy",
    )

    if clustered:
        ax.plot(
            clustered_soc.index,
            clustered_soc,
            color=colors[2],
            linewidth=1.4,
            linestyle="-",
            label="Clustered SoC",
        )

        ax.plot(
            clustered_proxy.index,
            clustered_proxy,
            color=colors[3],
            linewidth=1.2,
            linestyle="--",
            label="Clustered SoC Proxy",
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

    if title is not None:
        ax.set_title(title)

    format_datetime_axis(ax)

    ax.legend(
        frameon=False,
        ncol=2 if clustered else 1,
    )

    ax.margins(
        x=0,
    )

    fig.tight_layout()

    if output_path is not None:
        save_figure(
            fig,
            output_path,
        )

    return fig, ax


def _validate_signal(
    signal: pd.Series,
    *,
    name: str,
) -> None:
    """Validate one chronological signal."""
    if not isinstance(
        signal,
        pd.Series,
    ):
        raise TypeError(f"{name} must be a pandas Series.")

    if signal.empty:
        raise ValueError(f"{name} cannot be empty.")

    if not isinstance(
        signal.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(f"{name} must use a DatetimeIndex.")

    if signal.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate timestamps.")

    if not signal.index.is_monotonic_increasing:
        raise ValueError(f"{name} timestamps must be monotonically increasing.")

    if signal.isna().any():
        raise ValueError(f"{name} contains missing values.")


def _require_matching_index(
    left: pd.Series,
    right: pd.Series,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Require two signals to describe exactly the same chronology."""
    if left.index.equals(right.index):
        return

    raise ValueError(
        f"{left_name} and {right_name} must use exactly the same chronological index."
    )
