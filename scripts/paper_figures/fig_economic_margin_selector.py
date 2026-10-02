"""Illustrate the endogenous economic margin selector with labelled chronologies.



Updates relative to the earlier draft:

- each chronology receives its own colour from the plasma colormap;

- chronology labels are shown explicitly in a legend;

- q80 and q90 crossings remain highlighted but the full trajectories are drawn

  without point markers for a cleaner visual hierarchy;

- the figure uses a square aspect ratio and slightly larger text.



Run from the repository root with:



    pixi run --as-is python -u -m scripts.paper_figures.fig_economic_margin_selector

"""



from __future__ import annotations



import argparse

import inspect

from copy import deepcopy

from pathlib import Path



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.lines import Line2D



from scripts.helpers.config import load_experiment_config

from scripts.helpers.timeseries import read_calliope_timeseries

from src.soc_proxy.soc_proxy import generate_soc_proxy





CONFIG_PATH = Path("config/experiment_config.yaml")

OUTPUT_DIR = Path("results/figures/fig_economic_margin_selector")



COUNTRIES = ("NL", "BE")

HORIZON_YEARS = 10



CAPTURE_80 = 0.80

CAPTURE_90 = 0.90



PNG_DPI = 300



# visual settings

FIGSIZE = (7.6, 7.6)

TITLE_FONTSIZE = 15

LABEL_FONTSIZE = 13

TICK_FONTSIZE = 11.5

LEGEND_FONTSIZE = 10.5

ANNOTATION_FONTSIZE = 11

LINEWIDTH = 1.9

PROFILE_ALPHA = 0.82

MARKER_SIZE_Q80 = 52

MARKER_SIZE_Q90 = 46

CMAP_MIN = 0.08

CMAP_MAX = 0.90
X_AXIS_MAX_PERCENT = 15.0





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--config-path", type=Path, default=CONFIG_PATH)

    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    parser.add_argument(

        "--countries",

        nargs="+",

        default=list(COUNTRIES),

        help="Country codes to include. Default: NL BE.",

    )

    parser.add_argument("--show", action="store_true")

    return parser.parse_args()





def discover_unique_cases(

    configs: dict,

    *,

    countries: tuple[str, ...],

    horizon_years: int,

) -> list[tuple[str, dict]]:

    candidates: dict[

        tuple[str, pd.Timestamp, pd.Timestamp],

        list[tuple[str, dict]],

    ] = {}



    for experiment_name, config in configs.items():

        data_params = config.get("data_params", {})

        country = str(data_params.get("country", ""))



        if country not in countries:

            continue

        if "start_date" not in data_params or "end_date" not in data_params:

            continue



        start = pd.Timestamp(data_params["start_date"])

        end = pd.Timestamp(data_params["end_date"])



        years = end.year - start.year

        if years != horizon_years:

            continue

        if start + pd.DateOffset(years=years) != end:

            continue



        key = (country, start, end)

        candidates.setdefault(key, []).append((experiment_name, config))



    selected: list[tuple[str, dict]] = []

    for (country, start, _end), options in sorted(

        candidates.items(),

        key=lambda item: (item[0][0], item[0][1]),

    ):

        canonical = f"{country}_{start.year}"

        preferred = [item for item in options if item[0] == canonical]

        if preferred:

            selected.append(preferred[0])

        else:

            selected.append(sorted(options, key=lambda item: item[0])[0])



    if not selected:

        raise RuntimeError(

            f"No unique {horizon_years}-year cases found for {countries}."

        )



    return selected





def load_case_frame(

    config: dict,

) -> tuple[pd.DataFrame, str, pd.Timestamp, pd.Timestamp]:

    data_params = config["data_params"]



    country = str(data_params["country"])

    start = pd.Timestamp(data_params["start_date"])

    end = pd.Timestamp(data_params["end_date"])



    source_path = (

        Path("resources/raw_timeseries")

        / f"time_varying_parameters_{country}.csv"

    )

    source = read_calliope_timeseries(source_path)



    frame = source.loc[

        (source.index >= start)

        & (source.index < end)

    ].copy()



    _validate_horizon(frame, start=start, end=end)

    return frame, country, start, end





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

        raise RuntimeError("Timeseries contains duplicate timestamps.")



    if not frame.index.is_monotonic_increasing:

        raise RuntimeError("Timeseries is not monotonically increasing.")





def build_proxy_kwargs(config: dict) -> dict:

    if "soc_proxy_params" not in config:

        raise KeyError("Resolved experiment has no soc_proxy_params section.")



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



    if (

        "dispatchable_capacity" in signature.parameters

        and "dispatchable_capacity" not in kwargs

    ):

        raise KeyError(

            "Resolved soc_proxy_params has no dispatchable_capacity. "

            "This script will not silently use the API default."

        )



    missing_required = [

        name

        for name, parameter in signature.parameters.items()

        if (

            name not in reserved

            and name != "df"

            and parameter.default is inspect.Parameter.empty

            and name not in kwargs

        )

    ]

    if missing_required:

        raise KeyError(

            "Missing required generate_soc_proxy arguments: "

            f"{missing_required}"

        )



    return kwargs





def run_economic_sweep(

    frame: pd.DataFrame,

    proxy_kwargs: dict,

) -> tuple[object, pd.DataFrame]:

    result = generate_soc_proxy(

        df=frame,

        **proxy_kwargs,

        margin_mode="auto",

    )



    diagnostics = getattr(result, "margin_diagnostics", None)

    if diagnostics is None:

        raise RuntimeError(

            "generate_soc_proxy returned no automatic margin diagnostics."

        )



    sweep = diagnostics.sweep.copy().sort_values("margin").reset_index(drop=True)



    required = {"margin", "annual_cost", "economic_saving_capture"}

    missing = sorted(required - set(sweep.columns))

    if missing:

        raise RuntimeError(

            f"Automatic sweep is missing expected columns: {missing}"

        )



    return diagnostics, sweep





def first_margin_reaching_capture(

    sweep: pd.DataFrame,

    *,

    threshold: float,

) -> float:

    eligible = sweep.loc[

        sweep["economic_saving_capture"] >= threshold - 1e-12,

        "margin",

    ]

    if eligible.empty:

        raise RuntimeError(

            f"No sampled margin reaches {threshold:.0%} capture."

        )

    return float(eligible.iloc[0])





def row_at_margin(

    sweep: pd.DataFrame,

    margin: float,

) -> pd.Series:

    rows = sweep.loc[

        np.isclose(

            sweep["margin"].to_numpy(dtype=float),

            float(margin),

        )

    ]

    if len(rows) != 1:

        raise RuntimeError(

            f"Expected one row at margin {margin:.3f}; found {len(rows)}."

        )

    return rows.iloc[0]





def chronology_label(country: str, start: pd.Timestamp, end: pd.Timestamp) -> str:

    return f"{country} {start.year}–{end.year - 1}"





def build_results(

    cases: list[tuple[str, dict]],

) -> tuple[pd.DataFrame, pd.DataFrame]:

    sweep_frames: list[pd.DataFrame] = []

    summary_rows: list[dict[str, object]] = []



    for experiment_name, config in cases:

        frame, country, start, end = load_case_frame(config)

        proxy_kwargs = build_proxy_kwargs(config)



        print()

        print("=" * 72)

        print(

            f"{experiment_name} | {country} "

            f"{start.year}-{end.year - 1}"

        )

        print("=" * 72)



        diagnostics, sweep = run_economic_sweep(frame, proxy_kwargs)



        m80 = first_margin_reaching_capture(

            sweep,

            threshold=CAPTURE_80,

        )

        m90 = first_margin_reaching_capture(

            sweep,

            threshold=CAPTURE_90,

        )



        row80 = row_at_margin(sweep, m80)

        row90 = row_at_margin(sweep, m90)



        optimum_idx = sweep["annual_cost"].idxmin()

        optimum_margin = float(sweep.loc[optimum_idx, "margin"])



        labelled = sweep.copy()

        labelled.insert(0, "experiment", experiment_name)

        labelled.insert(1, "country", country)

        labelled.insert(2, "start_date", start)

        labelled.insert(3, "end_date", end)

        labelled.insert(4, "chronology_label", chronology_label(country, start, end))

        sweep_frames.append(labelled)



        summary_rows.append(

            {

                "experiment": experiment_name,

                "country": country,

                "start_date": start,

                "end_date": end,

                "start_year": start.year,

                "end_year": end.year - 1,

                "chronology_label": chronology_label(country, start, end),

                "m_80": m80,

                "m_90": m90,

                "m_80_capture": float(row80["economic_saving_capture"]),

                "m_90_capture": float(row90["economic_saving_capture"]),

                "economic_optimal_margin": optimum_margin,

                "economic_margin_80_from_selector": float(

                    diagnostics.economic_margin_80

                ),

                "terminal_event_margin": float(

                    diagnostics.terminal_event_margin

                ),

                "selected_margin": float(

                    diagnostics.selected_margin

                ),

                "evaluated_margin_max": float(

                    diagnostics.evaluated_margin_max

                ),

            }

        )



        if not np.isclose(

            m80,

            float(diagnostics.economic_margin_80),

        ):

            raise RuntimeError(

                f"{experiment_name}: calculated m80={m80:.3f} "

                "does not match selector "

                f"m80={float(diagnostics.economic_margin_80):.3f}."

            )



        print(

            f"m80={m80:.1%} | "

            f"m90={m90:.1%} | "

            f"cost minimum={optimum_margin:.1%} | "

            f"selected={float(diagnostics.selected_margin):.1%}"

        )



    return (

        pd.concat(sweep_frames, ignore_index=True),

        pd.DataFrame(summary_rows),

    )





def plot_economic_saturation(

    results: pd.DataFrame,

    summary: pd.DataFrame,

) -> plt.Figure:

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")



    summary_ordered = summary.sort_values(["country", "start_year"]).reset_index(drop=True)

    cmap = plt.get_cmap("plasma")

    color_positions = np.linspace(CMAP_MIN, CMAP_MAX, len(summary_ordered))

    color_map = {

        row.experiment: cmap(pos)

        for row, pos in zip(summary_ordered.itertuples(index=False), color_positions)

    }



    chronology_handles: list[Line2D] = []



    for row in summary_ordered.itertuples(index=False):

        group = results.loc[results["experiment"].eq(row.experiment)].sort_values("margin")

        color = color_map[row.experiment]



        x = 100.0 * group["margin"].to_numpy(dtype=float)

        y = 100.0 * group["economic_saving_capture"].to_numpy(dtype=float)



        ax.plot(

            x,

            y,

            color=color,

            alpha=PROFILE_ALPHA,

            linewidth=LINEWIDTH,

            zorder=2,

        )



        row80 = row_at_margin(group, float(row.m_80))

        row90 = row_at_margin(group, float(row.m_90))



        ax.scatter(

            [100.0 * float(row.m_80)],

            [100.0 * float(row80["economic_saving_capture"])],

            marker="o",

            s=MARKER_SIZE_Q80,

            facecolors="white",

            edgecolors=[color],

            linewidths=1.4,

            zorder=5,

        )

        ax.scatter(

            [100.0 * float(row.m_90)],

            [100.0 * float(row90["economic_saving_capture"])],

            marker="D",

            s=MARKER_SIZE_Q90,

            facecolors="white",

            edgecolors=[color],

            linewidths=1.4,

            zorder=5,

        )



        chronology_handles.append(

            Line2D(

                [0],

                [0],

                color=color,

                linewidth=2.2,

                label=row.chronology_label,

            )

        )



    # threshold region and lines

    ax.axhspan(

        100.0 * CAPTURE_80,

        100.0 * CAPTURE_90,

        color="0.4",

        alpha=0.06,

        zorder=0,

    )

    ax.axhline(

        100.0 * CAPTURE_80,

        color="0.30",

        linestyle=":",

        linewidth=1.25,

        zorder=1,

    )

    ax.axhline(

        100.0 * CAPTURE_90,

        color="0.30",

        linestyle="--",

        linewidth=1.25,

        zorder=1,

    )



    xmax = X_AXIS_MAX_PERCENT

    ax.text(

        X_AXIS_MAX_PERCENT - 0.25,

        100.0 * CAPTURE_80 + 0.4,

        "80%",

        ha="right",

        va="bottom",

        fontsize=ANNOTATION_FONTSIZE,

        color="0.25",

        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.8),

    )

    ax.text(

        X_AXIS_MAX_PERCENT - 0.25,

        100.0 * CAPTURE_90 + 0.4,

        "90%",

        ha="right",

        va="bottom",

        fontsize=ANNOTATION_FONTSIZE,

        color="0.25",

        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.8),

    )



    crossing_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7.2,
            markerfacecolor="white",
            markeredgecolor="0.2",
            label=r"$q_{80}$",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markersize=6.6,
            markerfacecolor="white",
            markeredgecolor="0.2",
            label=r"$q_{90}$",
        ),
    ]

    legend = ax.legend(
        handles=chronology_handles + crossing_handles,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        loc="lower right",
        borderpad=0.65,
        labelspacing=0.45,
        columnspacing=1.2,
        handlelength=2.5,
        fancybox=True,
        ncol=1,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.86)

    ax.set_xlabel("Margin $m$ (%)", fontsize=LABEL_FONTSIZE)

    ax.set_ylabel("Attainable saving captured (%)", fontsize=LABEL_FONTSIZE)



    ax.grid(axis="y", linewidth=0.7, alpha=0.24)

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)



    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)



    ax.set_xlim(left=0.0, right=X_AXIS_MAX_PERCENT)

    ax.set_ylim(bottom=0.0, top=104.0)



    return fig





def save_figure(

    fig: plt.Figure,

    stem: Path,

) -> tuple[Path, Path]:

    stem.parent.mkdir(parents=True, exist_ok=True)



    png = stem.with_suffix(".png")

    pdf = stem.with_suffix(".pdf")



    fig.savefig(png, dpi=PNG_DPI, facecolor="white", bbox_inches="tight")

    fig.savefig(pdf, facecolor="white", bbox_inches="tight")



    return png, pdf





def main() -> None:

    args = parse_args()



    configs = load_experiment_config(args.config_path)



    cases = discover_unique_cases(

        configs,

        countries=tuple(args.countries),

        horizon_years=HORIZON_YEARS,

    )



    print("Discovered chronology cases:")

    for experiment_name, config in cases:

        data = config["data_params"]

        print(

            f"  {experiment_name}: "

            f"{data['country']} "

            f"{pd.Timestamp(data['start_date']).date()} -> "

            f"{pd.Timestamp(data['end_date']).date()}"

        )



    results, summary = build_results(cases)



    args.output_dir.mkdir(parents=True, exist_ok=True)



    results_path = args.output_dir / "economic_margin_sweeps.csv"

    summary_path = args.output_dir / "economic_margin_summary.csv"



    results.to_csv(results_path, index=False)

    summary.to_csv(summary_path, index=False)



    fig = plot_economic_saturation(results, summary)



    figure_outputs = save_figure(

        fig,

        args.output_dir / "fig_economic_margin_selector",

    )



    print()

    print("=" * 72)

    print("Economic selector summary")

    print("=" * 72)

    print(

        summary[

            [

                "experiment",

                "country",

                "start_year",

                "end_year",

                "m_80",

                "m_90",

                "economic_optimal_margin",

                "selected_margin",

            ]

        ].to_string(index=False)

    )



    print()

    print("Saved:")

    for path in (*figure_outputs, results_path, summary_path):

        print(f"  {path}")



    if args.show:

        plt.show()

    else:

        plt.close(fig)





if __name__ == "__main__":

    main()
