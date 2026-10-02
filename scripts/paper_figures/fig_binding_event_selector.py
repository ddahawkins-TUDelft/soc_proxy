"""Illustrate the endogenous binding-event margin informant from scratch.



This script deliberately does *not* depend on previously generated event-persistence

or binding-event parquet files. It:



1. loads one resolved experiment from ``config/experiment_config.yaml``;

2. loads the corresponding raw chronology;

3. runs the current SoC Proxy once with ``margin_mode="auto"``;

4. takes the freshly generated margin sweep and terminal-event result directly

   from ``result.margin_diagnostics``;

5. re-runs only the fixed margins needed for the illustrative SoC Proxy overlay;

6. produces:

      - a conventional dominant-event-date vs margin figure;

      - an experimental SoC Proxy overlay coloured by successive dominant-peak

        regimes.



The event-regime colours are a *visual aid*. The actual terminal-event margin is

always taken from the current automatic selector diagnostics rather than being

reimplemented here.



Run from the repository root, for example::



    pixi run --as-is python -u -m scripts.paper_figures.fig_binding_event_selector



or::



    pixi run --as-is python -u -m scripts.paper_figures.fig_binding_event_selector \\

        --experiment NL_2006



A Belgian case can be substituted in exactly the same way, e.g. ``BE_2010``.

"""



from __future__ import annotations



import argparse

import inspect

from copy import deepcopy

from pathlib import Path



import matplotlib.dates as mdates

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.lines import Line2D



from scripts.helpers.config import load_experiment_config

from scripts.helpers.timeseries import read_calliope_timeseries

from src.soc_proxy.soc_proxy import generate_soc_proxy





# =============================================================================

# Defaults

# =============================================================================



CONFIG_PATH = Path("config/experiment_config.yaml")

OUTPUT_DIR = Path("results/figures/fig_binding_event_selector")



DEFAULT_EXPERIMENT = "NL_2008"



# This is only used to group successive dominant-peak locations into visually

# distinct regimes. It does not determine the selector result. Using 90 days

# matches the current terminal-event matching tolerance.

DEFAULT_EVENT_SHIFT_TOLERANCE_DAYS = 90



# Once the selector's persistent terminal event first appears, retain only this

# many additional margin samples in the proxy-overlay figure.

DEFAULT_POST_TERMINAL_POINTS = 4



# Plot every nth candidate before the terminal event. Transition points and the

# terminal-event onset are retained even when thinning is enabled.

DEFAULT_PLOT_EVERY = 1



# Plasma range deliberately stops before the yellow end of the colour map.

PLASMA_MIN = 0.08

PLASMA_MAX = 0.72



# SoC Proxy is in MWh for MW input data. Shift by the arbitrary additive

# constant and report the storage excursion in TWh.

MWH_PER_TWH = 1e6



PNG_DPI = 300





# =============================================================================

# CLI

# =============================================================================



def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description=(

            "Regenerate and illustrate the binding-event margin selector for "

            "one configured chronology."

        )

    )

    parser.add_argument(

        "--experiment",

        default=DEFAULT_EXPERIMENT,

        help="Resolved experiment key in config/experiment_config.yaml.",

    )

    parser.add_argument(

        "--config-path",

        type=Path,

        default=CONFIG_PATH,

    )

    parser.add_argument(

        "--source-path",

        type=Path,

        default=None,

        help=(

            "Optional raw-timeseries CSV override. By default the script uses "

            "resources/raw_timeseries/time_varying_parameters_<COUNTRY>.csv."

        ),

    )

    parser.add_argument(

        "--output-dir",

        type=Path,

        default=OUTPUT_DIR,

    )

    parser.add_argument(

        "--dispatchable-capacity",

        type=float,

        default=None,

        help=(

            "Optional explicit MW override. Normally this should already be "

            "resolved into soc_proxy_params by the project config loader."

        ),

    )

    parser.add_argument(

        "--event-shift-tolerance-days",

        type=int,

        default=DEFAULT_EVENT_SHIFT_TOLERANCE_DAYS,

        help=(

            "Visual grouping tolerance for successive dominant-peak regimes. "

            "This does not alter the selector result."

        ),

    )

    parser.add_argument(

        "--post-terminal-points",

        type=int,

        default=DEFAULT_POST_TERMINAL_POINTS,

        help=(

            "Number of additional fixed-margin proxy traces to plot after the "

            "persistent terminal event first appears."

        ),

    )

    parser.add_argument(

        "--plot-every",

        type=int,

        default=DEFAULT_PLOT_EVERY,

        help=(

            "Plot every nth pre-terminal margin candidate in the proxy overlay. "

            "Regime changes and terminal onset are always retained."

        ),

    )

    parser.add_argument(

        "--show",

        action="store_true",

    )

    return parser.parse_args()





# =============================================================================

# Experiment loading

# =============================================================================



def load_case(

    *,

    experiment: str,

    config_path: Path,

    source_path: Path | None,

) -> tuple[dict, pd.DataFrame, str, pd.Timestamp, pd.Timestamp]:

    """Load one resolved experiment and its exact hourly chronology."""

    configs = load_experiment_config(config_path)



    if experiment not in configs:

        available = ", ".join(sorted(configs))

        raise KeyError(

            f"Experiment {experiment!r} was not found in {config_path}. "

            f"Available experiments: {available}"

        )



    config = deepcopy(configs[experiment])

    data_params = config["data_params"]



    country = str(data_params["country"])

    start = pd.Timestamp(data_params["start_date"])

    end = pd.Timestamp(data_params["end_date"])



    if source_path is None:

        source_path = (

            Path("resources/raw_timeseries")

            / f"time_varying_parameters_{country}.csv"

        )



    source = read_calliope_timeseries(source_path)

    frame = source.loc[(source.index >= start) & (source.index < end)].copy()



    _validate_horizon(frame, start=start, end=end)



    return config, frame, country, start, end





def _validate_horizon(

    frame: pd.DataFrame,

    *,

    start: pd.Timestamp,

    end: pd.Timestamp,

) -> None:

    if frame.empty:

        raise RuntimeError(f"No timeseries data for {start} -> {end}.")



    expected_end = end - pd.Timedelta(hours=1)

    if frame.index[0] != start or frame.index[-1] != expected_end:

        raise RuntimeError(

            f"Incomplete horizon {start} -> {end}. "

            f"Observed {frame.index[0]} -> {frame.index[-1]}."

        )



    if frame.index.has_duplicates:

        raise RuntimeError("Timeseries horizon contains duplicate timestamps.")



    if not frame.index.is_monotonic_increasing:

        raise RuntimeError("Timeseries horizon is not monotonically increasing.")





# =============================================================================

# SoC Proxy calls

# =============================================================================



def build_proxy_kwargs(

    config: dict,

    *,

    dispatchable_capacity_override: float | None,

) -> dict:

    """Resolve only arguments accepted by the installed generate_soc_proxy API.



    ``load_experiment_config`` is expected to return a fully resolved

    ``soc_proxy_params`` mapping. Filtering against the live function signature

    makes the figure script tolerant of harmless config additions.

    """

    if "soc_proxy_params" not in config:

        raise KeyError("Resolved experiment has no 'soc_proxy_params' section.")



    params = deepcopy(config["soc_proxy_params"])

    signature = inspect.signature(generate_soc_proxy)



    kwargs: dict[str, object] = {}

    reserved = {"df", "margin_mode", "margin_value"}



    for name, parameter in signature.parameters.items():

        if name in reserved:

            continue



        if name == "verbosity":

            kwargs[name] = "off"

            continue



        if name in params:

            kwargs[name] = params[name]



    if dispatchable_capacity_override is not None:

        if "dispatchable_capacity" not in signature.parameters:

            raise TypeError(

                "The installed generate_soc_proxy API has no "

                "'dispatchable_capacity' argument."

            )

        kwargs["dispatchable_capacity"] = float(dispatchable_capacity_override)



    # Do not silently fall back to zero if the study expects a resolved

    # country-specific must-run/background capacity.

    if (

        "dispatchable_capacity" in signature.parameters

        and "dispatchable_capacity" not in kwargs

    ):

        raise KeyError(

            "The resolved experiment does not contain "

            "soc_proxy_params.dispatchable_capacity. The study configuration "

            "indicates that this quantity is resolved per country, so the "

            "figure script will not silently use the API default of zero. "

            "Either update the config resolver or supply "

            "--dispatchable-capacity explicitly."

        )



    missing_required: list[str] = []

    for name, parameter in signature.parameters.items():

        if name in reserved:

            continue

        if name == "df":

            continue

        if parameter.default is inspect.Parameter.empty and name not in kwargs:

            missing_required.append(name)



    if missing_required:

        raise KeyError(

            "The resolved soc_proxy_params do not supply required current "

            f"generate_soc_proxy arguments: {missing_required}"

        )



    return kwargs





def run_auto_proxy(

    frame: pd.DataFrame,

    proxy_kwargs: dict,

):

    """Run the current endogenous selector and return its fresh diagnostics."""

    result = generate_soc_proxy(

        df=frame,

        **proxy_kwargs,

        margin_mode="auto",

    )



    diagnostics = getattr(result, "margin_diagnostics", None)

    if diagnostics is None:

        raise RuntimeError(

            "generate_soc_proxy did not return automatic margin diagnostics. "

            "This script expects the current SocProxyResult API with "

            "result.margin_diagnostics."

        )



    sweep = diagnostics.sweep.copy().sort_values("margin").reset_index(drop=True)

    sweep["dominant_peak_timestamp"] = pd.to_datetime(

        sweep["dominant_peak_timestamp"]

    )



    required = {

        "margin",

        "dominant_peak_timestamp",

        "terminal_event_match",

    }

    missing = sorted(required - set(sweep.columns))

    if missing:

        raise RuntimeError(

            f"Current margin diagnostics are missing expected columns: {missing}"

        )



    return result, diagnostics, sweep





def run_fixed_proxy(

    frame: pd.DataFrame,

    proxy_kwargs: dict,

    *,

    margin: float,

) -> pd.Series:

    """Regenerate one fixed-margin LDES SoC Proxy."""

    result = generate_soc_proxy(

        df=frame,

        **proxy_kwargs,

        margin_mode="fixed",

        margin_value=float(margin),

    )



    if hasattr(result, "data"):

        proxy_frame = result.data

    elif isinstance(result, tuple) and result and isinstance(result[0], pd.DataFrame):

        # Compatibility with an older development API.

        proxy_frame = result[0]

    else:

        raise TypeError(

            "Could not extract the proxy dataframe from generate_soc_proxy result."

        )



    if "soc_proxy_LDES" not in proxy_frame.columns:

        raise KeyError(

            "Generated proxy dataframe has no 'soc_proxy_LDES' column."

        )



    proxy = proxy_frame["soc_proxy_LDES"].astype(float).copy()



    if not isinstance(proxy.index, pd.DatetimeIndex):

        raise TypeError("soc_proxy_LDES must use a DatetimeIndex.")



    return proxy





# =============================================================================

# Fresh event-regime annotation

# =============================================================================



def annotate_dominant_peak_regimes(

    sweep: pd.DataFrame,

    *,

    tolerance_days: int,

) -> pd.DataFrame:

    """Assign sequential IDs to large shifts in the dominant peak timestamp.



    This annotation is only for figure colouring. The actual terminal-event

    onset used by the selector comes directly from ``margin_diagnostics``.

    """

    out = sweep.sort_values("margin").copy().reset_index(drop=True)



    event_ids: list[int] = []

    transitions: list[bool] = []

    anchors: list[pd.Timestamp] = []



    event_id = 0

    anchor: pd.Timestamp | None = None

    tolerance = pd.Timedelta(days=tolerance_days)



    for timestamp in pd.to_datetime(out["dominant_peak_timestamp"]):

        timestamp = pd.Timestamp(timestamp)



        if anchor is None:

            anchor = timestamp

            event_ids.append(event_id)

            transitions.append(False)

            anchors.append(anchor)

            continue



        changed = abs(timestamp - anchor) > tolerance

        if changed:

            event_id += 1

            anchor = timestamp



        event_ids.append(event_id)

        transitions.append(changed)

        anchors.append(anchor)



    out["visual_event_id"] = event_ids

    out["visual_event_transition"] = transitions

    out["visual_event_anchor"] = anchors

    return out


def annotate_selector_colour_families(
    sweep: pd.DataFrame,
    *,
    terminal_event_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Create plotting families that explicitly reflect terminal-event matching.

    Non-terminal rows retain the exploratory dominant-peak regime assignment.
    Rows whose dominant peak matches the persistent terminal event are assigned
    to one dedicated terminal family. The plotted colour therefore changes at
    the selector's actual terminal-event onset even when that onset lies within
    a broader visual peak regime.

    This affects plotting only; it does not alter the selector result.
    """
    out = sweep.sort_values("margin").copy().reset_index(drop=True)

    raw_keys: list[tuple[str, int | None]] = []
    for row in out.itertuples(index=False):
        if bool(row.terminal_event_match):
            raw_keys.append(("terminal", None))
        else:
            raw_keys.append(("visual", int(row.visual_event_id)))

    family_order = list(dict.fromkeys(raw_keys))
    family_id_by_key = {
        key: family_id
        for family_id, key in enumerate(family_order)
    }

    labels_by_key: dict[tuple[str, int | None], str] = {}
    for key in family_order:
        family_type, visual_event_id = key

        if family_type == "terminal":
            labels_by_key[key] = (
                f"Persistent terminal event "
                f"{pd.Timestamp(terminal_event_timestamp):%Y-%m-%d}"
            )
            continue

        group = out.loc[out["visual_event_id"].eq(int(visual_event_id))]
        anchor_timestamp = pd.Timestamp(group["visual_event_anchor"].iloc[0])
        labels_by_key[key] = f"Dominant event {anchor_timestamp:%Y-%m-%d}"

    out["plot_family_id"] = [
        family_id_by_key[key] for key in raw_keys
    ]
    out["plot_family_label"] = [
        labels_by_key[key] for key in raw_keys
    ]
    out["plot_family_transition"] = out["plot_family_id"].ne(
        out["plot_family_id"].shift()
    )

    return out





def rows_for_overlay(

    sweep: pd.DataFrame,

    *,

    terminal_event_margin: float,

    post_terminal_points: int,

    plot_every: int,

) -> pd.DataFrame:

    """Keep the sweep through a few points after terminal-event onset."""

    ordered = sweep.sort_values("margin").reset_index(drop=True)



    terminal_positions = np.flatnonzero(

        np.isclose(

            ordered["margin"].to_numpy(dtype=float),

            float(terminal_event_margin),

        )

    )

    if len(terminal_positions) != 1:

        raise RuntimeError(

            "Could not uniquely locate terminal_event_margin "

            f"{terminal_event_margin:.3f} in the fresh sweep."

        )



    terminal_idx = int(terminal_positions[0])

    stop_idx = min(

        len(ordered) - 1,

        terminal_idx + max(0, int(post_terminal_points)),

    )



    candidate = ordered.iloc[: stop_idx + 1].copy().reset_index(drop=True)



    plot_every = max(1, int(plot_every))

    if plot_every == 1:

        return candidate



    keep: set[int] = set(range(0, len(candidate), plot_every))

    keep.add(0)

    keep.add(len(candidate) - 1)

    keep.add(terminal_idx)



    transition_positions = np.flatnonzero(

        candidate["visual_event_transition"].to_numpy(dtype=bool)

    )

    for idx in transition_positions:

        keep.add(int(idx))

        if idx > 0:

            keep.add(int(idx - 1))



    return candidate.iloc[sorted(keep)].reset_index(drop=True)





# =============================================================================

# Plot helpers

# =============================================================================



def event_colours(event_ids: list[int]) -> dict[int, object]:

    unique = list(dict.fromkeys(int(value) for value in event_ids))

    cmap = plt.get_cmap("plasma")

    positions = np.linspace(PLASMA_MIN, PLASMA_MAX, len(unique))

    return {

        event_id: cmap(position)

        for event_id, position in zip(unique, positions, strict=True)

    }





def event_labels(sweep: pd.DataFrame) -> dict[int, str]:
    """Return one legend label for each selector-aware plotting family."""
    labels: dict[int, str] = {}
    for family_id, group in sweep.groupby("plot_family_id", sort=False):
        labels[int(family_id)] = str(group["plot_family_label"].iloc[0])
    return labels

def alpha_values(n: int) -> np.ndarray:

    if n <= 1:

        return np.asarray([0.95])



    return np.asarray([0.95] + [0.15] * (n - 1))





def shifted_proxy_twh(proxy: pd.Series) -> pd.Series:

    """Remove the arbitrary additive SoC offset and convert MWh to TWh."""

    shifted = proxy - float(proxy.min())

    return shifted / MWH_PER_TWH





def nearest_value(series: pd.Series, timestamp: pd.Timestamp) -> float:

    position = int(series.index.get_indexer([timestamp], method="nearest")[0])

    if position < 0:

        raise RuntimeError(f"Could not locate {timestamp} in proxy chronology.")

    return float(series.iloc[position])





# =============================================================================

# Conventional figure: dominant event date vs margin

# =============================================================================



def plot_event_date_figure(
    sweep: pd.DataFrame,
    *,
    terminal_event_margin: float,
    terminal_event_timestamp: pd.Timestamp,
    country: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> plt.Figure:
    colours = event_colours(sweep["plot_family_id"].tolist())
    labels = event_labels(sweep)

    fig, ax = plt.subplots(figsize=(8.2, 4), layout="constrained")

    # Thin neutral line shows the progression without implying that calendar
    # date is a continuously varying response variable.
    ax.plot(
        100.0 * sweep["margin"],
        sweep["dominant_peak_timestamp"],
        color="0.72",
        linewidth=1.0,
        zorder=1,
    )

    for family_id, group in sweep.groupby("plot_family_id", sort=False):
        family_id = int(family_id)
        ax.scatter(
            100.0 * group["margin"],
            group["dominant_peak_timestamp"],
            s=26,
            color=colours[family_id],
            label=labels[family_id],
            zorder=3,
        )

    # Circle every plotted family transition, including the selector's actual
    # terminal-event onset.
    transitions = sweep.loc[sweep["plot_family_transition"]]
    ax.scatter(
        100.0 * transitions["margin"],
        transitions["dominant_peak_timestamp"],
        s=95,
        facecolors="none",
        edgecolors="black",
        linewidths=1.25,
        zorder=5,
    )

    ax.axvline(
        100.0 * terminal_event_margin,
        color="0.25",
        linestyle="--",
        linewidth=1.1,
        zorder=2,
    )

    terminal_rows = sweep.loc[
        np.isclose(sweep["margin"], terminal_event_margin)
    ]
    if len(terminal_rows) != 1:
        raise RuntimeError(
            "Could not uniquely locate terminal_event_margin in the plotted sweep."
        )

    terminal_row = terminal_rows.iloc[0]
    ax.annotate(
        (
            "Persistent terminal event\n"
            f"first matched at $m={100*terminal_event_margin:.1f}\\%$"
        ),
        xy=(
            100.0 * terminal_event_margin,
            pd.Timestamp(terminal_row["dominant_peak_timestamp"]),
        ),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=10,
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="none",
            alpha=0.92,
        ),
        zorder=8,
    )

    ax.set_xlabel("Margin $m$ (%)")
    ax.set_ylabel("Dominant SoC Proxy event")
    ax.yaxis.set_major_locator(mdates.YearLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", linewidth=0.6, alpha=0.25)

    end_year = end.year - 1
    ax.set_title(
        f"{country} {start.year}--{end_year}: dominant event across margin sweep"
    )

    legend = ax.legend(
        frameon=True,
        fontsize=10,
        loc="upper right",
        borderpad=0.65,
        labelspacing=0.55,
        handlelength=2.2,
        fancybox=True,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.92)

    return fig

# =============================================================================

# Experimental figure: overlaid SoC Proxy trajectories

# =============================================================================



def plot_proxy_overlay_figure(
    plot_rows: pd.DataFrame,
    proxy_by_margin: dict[float, pd.Series],
    *,
    terminal_event_margin: float,
    country: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> plt.Figure:
    colours = event_colours(plot_rows["plot_family_id"].tolist())
    labels = event_labels(plot_rows)

    fig, ax = plt.subplots(figsize=(11.0, 4.5), layout="constrained")

    # The first trace in each family is prominent. Later traces immediately
    # recede to a fixed low opacity, preserving the envelope without clutter.
    for family_id, group in plot_rows.groupby("plot_family_id", sort=False):
        family_id = int(family_id)
        group = group.sort_values("margin")
        alphas = alpha_values(len(group))

        for alpha, row in zip(alphas, group.itertuples(index=False), strict=True):
            margin = float(row.margin)
            proxy = shifted_proxy_twh(proxy_by_margin[margin])

            ax.plot(
                proxy.index,
                proxy.to_numpy(dtype=float),
                color=colours[family_id],
                alpha=float(alpha),
                linewidth=1.35,
                zorder=2,
            )

    # Mark the first trace of every selector-aware colour family. Since the
    # persistent terminal event has its own family, the colour change and marker
    # occur exactly at terminal_event_margin.
    family_starts = plot_rows.loc[
        plot_rows["plot_family_transition"]
    ]

    for row in family_starts.itertuples(index=False):
        family_id = int(row.plot_family_id)
        margin = float(row.margin)
        peak_timestamp = pd.Timestamp(row.dominant_peak_timestamp)
        proxy = shifted_proxy_twh(proxy_by_margin[margin])
        peak_value = nearest_value(proxy, peak_timestamp)
        is_terminal = bool(row.terminal_event_match)

        ax.scatter(
            [peak_timestamp],
            [peak_value],
            s=115,
            facecolors="none",
            edgecolors=colours[family_id],
            linewidths=2.0,
            zorder=6,
        )

        label = (
            f"terminal onset: $m={100*margin:.1f}\\%$"
            if is_terminal
            else f"$m={100*margin:.1f}\\%$"
        )

        # Keep the label centreline aligned horizontally with the peak marker.
        ax.annotate(
            label,
            xy=(peak_timestamp, peak_value),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=10.5,
            color=colours[family_id],
            ha="left",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="white",
                edgecolor="none",
                alpha=0.92,
            ),
            zorder=8,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colours[family_id],
            linewidth=2.2,
            label=labels[family_id],
        )
        for family_id in labels
    ]

    legend = ax.legend(
        handles=legend_handles,
        frameon=True,
        fontsize=10,
        loc="upper right",
        borderpad=0.65,
        labelspacing=0.55,
        handlelength=2.2,
        fancybox=True,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.92)

    ax.set_xlabel("Time")
    ax.set_ylabel("SoC Proxy (TWh)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", linewidth=0.6, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    end_year = end.year - 1
    # ax.set_title(
    #     f"{country} {start.year}--{end_year}: SoC Proxy response to increasing margin"
    # )

    return fig

# =============================================================================

# Output / reporting

# =============================================================================



def save_figure(fig: plt.Figure, stem: Path) -> tuple[Path, Path]:

    stem.parent.mkdir(parents=True, exist_ok=True)

    png = stem.with_suffix(".png")

    pdf = stem.with_suffix(".pdf")



    fig.savefig(png, dpi=PNG_DPI, facecolor="white")

    fig.savefig(pdf, facecolor="white")

    return png, pdf





def print_summary(

    sweep: pd.DataFrame,

    diagnostics,

    *,

    experiment: str,

) -> None:

    print()

    print("=" * 72)

    print(f"Binding-event selector illustration | {experiment}")

    print("=" * 72)

    print(f"Evaluated margins:       {len(sweep)}")

    print(f"Maximum evaluated m:     {float(sweep['margin'].max()):.1%}")

    print(f"Economic m80:            {float(diagnostics.economic_margin_80):.1%}")

    print(f"Terminal-event margin:   {float(diagnostics.terminal_event_margin):.1%}")

    print(f"Selected margin:         {float(diagnostics.selected_margin):.1%}")

    print(

        "Terminal event:          "

        f"{pd.Timestamp(diagnostics.terminal_event_timestamp)}"

    )

    print()



    print("Dominant-peak regime changes:")

    starts = sweep.loc[

        sweep["visual_event_id"].ne(sweep["visual_event_id"].shift())

    ]

    for row in starts.itertuples(index=False):

        print(

            f"  event {int(row.visual_event_id):2d} | "

            f"m={float(row.margin):5.1%} | "

            f"peak={pd.Timestamp(row.dominant_peak_timestamp)}"

        )





# =============================================================================

# Main

# =============================================================================



def main() -> None:

    args = parse_args()



    config, frame, country, start, end = load_case(

        experiment=args.experiment,

        config_path=args.config_path,

        source_path=args.source_path,

    )



    proxy_kwargs = build_proxy_kwargs(

        config,

        dispatchable_capacity_override=args.dispatchable_capacity,

    )



    _, diagnostics, sweep = run_auto_proxy(

        frame,

        proxy_kwargs,

    )



    sweep = annotate_dominant_peak_regimes(

        sweep,

        tolerance_days=args.event_shift_tolerance_days,

    )

    sweep = annotate_selector_colour_families(
        sweep,
        terminal_event_timestamp=pd.Timestamp(
            diagnostics.terminal_event_timestamp
        ),
    )



    print_summary(

        sweep,

        diagnostics,

        experiment=args.experiment,

    )



    plot_rows = rows_for_overlay(

        sweep,

        terminal_event_margin=float(diagnostics.terminal_event_margin),

        post_terminal_points=args.post_terminal_points,

        plot_every=args.plot_every,

    )



    proxy_by_margin: dict[float, pd.Series] = {}

    print()

    print("Regenerating fixed-margin trajectories used in overlay:")

    for margin in plot_rows["margin"].to_numpy(dtype=float):

        print(f"  m={margin:.1%}")

        proxy_by_margin[float(margin)] = run_fixed_proxy(

            frame,

            proxy_kwargs,

            margin=float(margin),

        )



    case_tag = f"{country}_{start.year}_{end.year - 1}"

    output_dir = args.output_dir / case_tag

    output_dir.mkdir(parents=True, exist_ok=True)



    # Save the freshly developed sweep as a transparent audit trail.

    sweep_path = output_dir / "binding_event_sweep.csv"

    sweep.to_csv(sweep_path, index=False)



    event_date_fig = plot_event_date_figure(

        sweep,

        terminal_event_margin=float(diagnostics.terminal_event_margin),

        terminal_event_timestamp=pd.Timestamp(

            diagnostics.terminal_event_timestamp

        ),

        country=country,

        start=start,

        end=end,

    )

    event_outputs = save_figure(

        event_date_fig,

        output_dir / "fig_binding_event_date",

    )



    overlay_fig = plot_proxy_overlay_figure(

        plot_rows,

        proxy_by_margin,

        terminal_event_margin=float(diagnostics.terminal_event_margin),

        country=country,

        start=start,

        end=end,

    )

    overlay_outputs = save_figure(

        overlay_fig,

        output_dir / "fig_binding_event_proxy_overlay",

    )



    print()

    print("Saved:")

    for path in (*event_outputs, *overlay_outputs, sweep_path):

        print(f"  {path}")



    if args.show:

        plt.show()

    else:

        plt.close(event_date_fig)

        plt.close(overlay_fig)





if __name__ == "__main__":

    main()
