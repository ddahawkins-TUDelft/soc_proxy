"""Shared plotting helpers for SoC Proxy figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure


def plasma_colors(
    positions: tuple[float, ...],
) -> list:
    """Return colours sampled from Matplotlib's plasma colourmap."""
    cmap = plt.get_cmap("plasma")
    return [cmap(position) for position in positions]


def format_datetime_axis(
    ax: Axes,
) -> None:
    """Apply compact date formatting to a chronological x-axis."""
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
) -> None:
    """Save a figure, creating its parent directory if required."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )
