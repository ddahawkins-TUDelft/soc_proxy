"""Summarise endogenous SoC-proxy calibration and approximation quality.

The script extracts, for each unique 10-year country/weather-year-set case:

* the resolved endogenous proxy margin stored as ``margin_value`` in
  ``parameters.parquet``; and
* unwindowed reference-proxy approximation metrics from
  ``signal_metrics.parquet``.

Metrics retained from ``signal_metrics.parquet``::

    error_family = "reference_approximation"
    signal_type = "level"
    metric in {"pearson", "rmse", "mbe"}
    window_half_width_days is empty / null

Multiple clustered model configurations can share the same country and weather
window. Because the original SoC Proxy (and therefore its selected margin and
reference-approximation metrics) is a property of that original chronology,
these values should be identical across duplicate case IDs. The script checks
that assumption before collapsing to one row per unique weather set.

Source::

    results/2_5_10_year/
        parameters.parquet
        signal_metrics.parquet

Outputs::

    results/tables/table_ref_proxy_quality/
        table_ref_proxy_quality_summary.csv
        table_ref_proxy_quality_cases.csv
        table_ref_proxy_quality.tex

The LaTeX table is transposed for a single-column paper layout: measures are
rows and countries are columns. Each cell reports median (full range).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE_DIR = Path("results/2_5_10_year")
DEFAULT_OUTPUT_DIR = Path("results/tables/table_ref_proxy_quality")

TARGET_HORIZON_YEARS = 10.0
HORIZON_TOLERANCE_YEARS = 0.10

ERROR_FAMILY = "reference_approximation"
SIGNAL_TYPE = "level"
METRICS = ("pearson", "rmse")

DUPLICATE_VALUE_ATOL = 1e-10

# Preferred display order; any additional countries are appended alphabetically.
PREFERRED_COUNTRY_ORDER = ("BE", "NL")

MEASURE_ORDER = ("selected_margin", "pearson", "rmse")
MEASURE_LABELS = {
    "selected_margin": r"Selected margin $m$",
    "pearson": r"Pearson $r$",
    "rmse": r"RMSE (TWh)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise 10-year endogenous proxy margins and reference "
            "SoC-proxy approximation metrics by country."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing parameters.parquet and signal_metrics.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which CSV and LaTeX outputs are written.",
    )
    return parser.parse_args()


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


def load_ten_year_cases(source_dir: Path) -> pd.DataFrame:
    """Load 10-year cases including the resolved case-specific proxy margin."""
    parameters = pd.read_parquet(source_dir / "parameters.parquet")

    require_columns(
        parameters,
        {"case_id", "country", "start_date", "end_date", "margin_value"},
        table_name="parameters.parquet",
    )

    parameters = parameters.copy()
    parameters["case_id"] = parameters["case_id"].astype(str)
    parameters["start_date"] = pd.to_datetime(parameters["start_date"])
    parameters["end_date"] = pd.to_datetime(parameters["end_date"])
    parameters["margin_value"] = pd.to_numeric(
        parameters["margin_value"], errors="coerce"
    )

    parameters["horizon_years"] = (
        parameters["end_date"] - parameters["start_date"]
    ).dt.total_seconds() / (365.2425 * 24 * 60 * 60)

    parameters = parameters.loc[
        np.isclose(
            parameters["horizon_years"],
            TARGET_HORIZON_YEARS,
            atol=HORIZON_TOLERANCE_YEARS,
            rtol=0.0,
        )
    ].copy()

    if parameters.empty:
        raise ValueError(
            "No approximately 10-year cases were found in parameters.parquet."
        )

    missing_margin = parameters["margin_value"].isna()
    if missing_margin.any():
        examples = parameters.loc[
            missing_margin,
            ["case_id", "country", "start_date", "end_date"],
        ].head(20)
        raise ValueError(
            "Some 10-year cases do not contain a resolved margin_value. "
            "The paper table expects the selected endogenous margin to have "
            "been persisted in parameters.parquet. Example rows:\n"
            + examples.to_string(index=False)
        )

    keep = [
        "case_id",
        "country",
        "start_date",
        "end_date",
        "horizon_years",
        "margin_value",
    ]
    if "margin_mode" in parameters.columns:
        keep.append("margin_mode")

    return parameters[keep].drop_duplicates()


def empty_window_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()

    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(
        series.dtype
    ):
        mask |= series.astype("string").str.strip().fillna("").eq("")

    return mask


def load_reference_proxy_metrics(
    source_dir: Path,
    ten_year_cases: pd.DataFrame,
) -> pd.DataFrame:
    metrics = pd.read_parquet(source_dir / "signal_metrics.parquet")

    require_columns(
        metrics,
        {
            "case_id",
            "error_family",
            "signal_type",
            "metric",
            "window_half_width_days",
            "value",
        },
        table_name="signal_metrics.parquet",
    )

    metrics = metrics.copy()
    metrics["case_id"] = metrics["case_id"].astype(str)

    error_family = metrics["error_family"].astype("string").str.lower()
    signal_type = metrics["signal_type"].astype("string").str.lower()
    metric_name = metrics["metric"].astype("string").str.lower()

    metrics = metrics.loc[
        error_family.eq(ERROR_FAMILY)
        & signal_type.eq(SIGNAL_TYPE)
        & metric_name.isin(METRICS)
        & empty_window_mask(metrics["window_half_width_days"])
    ].copy()

    metrics["metric"] = metric_name.loc[metrics.index]

    if metrics.empty:
        raise ValueError(
            "No matching reference-approximation level metrics were found in "
            "signal_metrics.parquet."
        )

    case_ids = set(ten_year_cases["case_id"])
    metrics = metrics.loc[metrics["case_id"].isin(case_ids)].copy()

    if metrics.empty:
        raise ValueError(
            "Reference-approximation metrics exist, but none correspond to "
            "the approximately 10-year case_ids."
        )

    duplicated = metrics.duplicated(
        subset=["case_id", "metric"],
        keep=False,
    )
    if duplicated.any():
        dup = metrics.loc[
            duplicated,
            ["case_id", "metric", "value"],
        ].sort_values(["case_id", "metric"])

        raise ValueError(
            "Expected one unwindowed reference metric per case_id / metric, "
            "but duplicates were found:\n" + dup.head(40).to_string(index=False)
        )

    return metrics[["case_id", "metric", "value"]]


def collapse_margins_to_unique_weather_sets(
    ten_year_cases: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse repeated model configurations to one selected margin per horizon."""
    key = ["country", "start_date", "end_date"]

    consistency = ten_year_cases.groupby(key, as_index=False).agg(
        n_case_ids=("case_id", "nunique"),
        margin_min=("margin_value", "min"),
        margin_max=("margin_value", "max"),
    )

    inconsistent = consistency.loc[
        ~np.isclose(
            consistency["margin_min"],
            consistency["margin_max"],
            atol=DUPLICATE_VALUE_ATOL,
            rtol=0.0,
        )
    ]

    if not inconsistent.empty:
        raise ValueError(
            "Resolved proxy margins differ across case_ids that represent the "
            "same country/weather-year set. The selected margin should be a "
            "property of the original chronology. Inspect these rows first:\n"
            + inconsistent.head(40).to_string(index=False)
        )

    margins = (
        ten_year_cases.groupby(key, as_index=False)
        .agg(
            selected_margin=("margin_value", "mean"),
            n_duplicate_case_ids=("case_id", "nunique"),
        )
        .sort_values(["country", "start_date"])
        .reset_index(drop=True)
    )

    return margins


def collapse_metrics_to_unique_weather_sets(
    ten_year_cases: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    data = ten_year_cases[
        ["case_id", "country", "start_date", "end_date"]
    ].merge(
        metrics,
        on="case_id",
        how="inner",
        validate="one_to_many",
    )

    if data.empty:
        raise ValueError(
            "Joining 10-year parameters to signal metrics produced no rows."
        )

    key = ["country", "start_date", "end_date", "metric"]

    consistency = data.groupby(key, as_index=False).agg(
        n_case_ids=("case_id", "nunique"),
        value_min=("value", "min"),
        value_max=("value", "max"),
    )

    inconsistent = consistency.loc[
        ~np.isclose(
            consistency["value_min"],
            consistency["value_max"],
            atol=DUPLICATE_VALUE_ATOL,
            rtol=0.0,
        )
    ]

    if not inconsistent.empty:
        raise ValueError(
            "Reference proxy metrics differ across case_ids that represent "
            "the same country/weather-year set. Inspect these rows first:\n"
            + inconsistent.head(40).to_string(index=False)
        )

    unique_metrics = (
        data.groupby(key, as_index=False)
        .agg(
            value=("value", "mean"),
            n_duplicate_case_ids=("case_id", "nunique"),
        )
        .sort_values(["country", "start_date", "metric"])
        .reset_index(drop=True)
    )

    return unique_metrics


def build_case_table(
    margins: pd.DataFrame,
    unique_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Return one wide row per country/weather set for convenient inspection."""
    metric_wide = unique_metrics.pivot(
        index=["country", "start_date", "end_date"],
        columns="metric",
        values="value",
    ).reset_index()
    metric_wide.columns.name = None

    cases = margins.merge(
        metric_wide,
        on=["country", "start_date", "end_date"],
        how="inner",
        validate="one_to_one",
    )

    missing_metrics = [metric for metric in METRICS if metric not in cases.columns]
    if missing_metrics:
        raise ValueError(
            "The case table is missing expected reference-proxy metrics: "
            f"{missing_metrics}"
        )

    return cases.sort_values(["country", "start_date"]).reset_index(drop=True)


def build_summary(case_table: pd.DataFrame) -> pd.DataFrame:
    """Return tidy country/measure summaries used by CSV and LaTeX outputs."""
    rows: list[dict[str, object]] = []

    for country, country_data in case_table.groupby("country", sort=True):
        for measure in MEASURE_ORDER:
            values = pd.to_numeric(country_data[measure], errors="coerce").dropna()
            if values.empty:
                continue

            rows.append(
                {
                    "country": country,
                    "measure": measure,
                    "n": int(values.size),
                    "median": float(values.median()),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )

    return pd.DataFrame(rows)


def ordered_countries(summary: pd.DataFrame) -> list[str]:
    available = list(summary["country"].drop_duplicates())
    preferred = [c for c in PREFERRED_COUNTRY_ORDER if c in available]
    extras = sorted(c for c in available if c not in preferred)
    return preferred + extras


def format_stat(measure: str, value: float) -> str:
    if measure == "selected_margin":
        return f"{100.0 * value:.1f}\\%"
    return f"{value:.2f}"


def build_latex_table(summary: pd.DataFrame) -> str:
    """Build a narrow transposed table suitable for one paper column."""
    countries = ordered_countries(summary)
    lookup = {
        (row.country, row.measure): row
        for row in summary.itertuples(index=False)
    }

    column_spec = "l" + "c" * len(countries)
    header = "Measure & " + " & ".join(countries) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\begin{{tabular}}{{@{{}}{column_spec}@{{}}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for measure in MEASURE_ORDER:
        cells = [MEASURE_LABELS[measure]]
        for country in countries:
            row = lookup.get((country, measure))
            if row is None:
                cells.append("--")
                continue

            median = format_stat(measure, float(row.median))
            minimum = format_stat(measure, float(row.minimum))
            maximum = format_stat(measure, float(row.maximum))
            cells.append(f"{median} ({minimum}--{maximum})")

        lines.append(" & ".join(cells) + r" \\")

    counts = summary.groupby("country")["n"].min().astype(int)
    unique_counts = sorted(counts.unique())

    if len(unique_counts) == 1:
        n_text = f"$n={unique_counts[0]}$ per country"
    else:
        n_text = ", ".join(
            f"{country}: $n={count}$" for country, count in counts.items()
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Endogenous SoC Proxy calibration and approximation "
                r"quality across unique 10-year reference weather-year sets ("
                + n_text
                + r"). Cells report median (full range). The selected margin "
                r"$m$ is the case-specific endogenous margin; approximation "
                r"metrics compare the resulting proxy with the CEM SoC trajectory.}"
            ),
            r"\label{tab:ref-proxy-qual}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def print_summary(
    case_table: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    print("Endogenous SoC-proxy calibration and approximation quality")
    print("==========================================================")

    weather_sets = (
        case_table[["country", "start_date", "end_date"]]
        .drop_duplicates()
        .groupby("country")
        .size()
    )

    print("\nUnique 10-year weather sets:")
    for country, count in weather_sets.items():
        print(f"  {country}: {count}")

    print("\nMedian and range:")
    for country in ordered_countries(summary):
        print(f"\n{country}")
        country_data = summary.loc[summary["country"].eq(country)]

        for measure in MEASURE_ORDER:
            match = country_data.loc[country_data["measure"].eq(measure)]
            if match.empty:
                continue
            row = match.iloc[0]

            if measure == "selected_margin":
                print(
                    f"  {measure:16s}: "
                    f"median={100 * row['median']:.1f}%, "
                    f"range={100 * row['minimum']:.1f}% to "
                    f"{100 * row['maximum']:.1f}%, "
                    f"n={int(row['n'])}"
                )
            else:
                print(
                    f"  {measure:16s}: "
                    f"median={row['median']:.4g}, "
                    f"range={row['minimum']:.4g} to {row['maximum']:.4g}, "
                    f"n={int(row['n'])}"
                )


def save_outputs(
    case_table: pd.DataFrame,
    summary: pd.DataFrame,
    latex: str,
    *,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_path = output_dir / "table_ref_proxy_quality_cases.csv"
    summary_path = output_dir / "table_ref_proxy_quality_summary.csv"
    latex_path = output_dir / "table_ref_proxy_quality.tex"

    case_table.to_csv(cases_path, index=False)
    summary.to_csv(summary_path, index=False)
    latex_path.write_text(latex, encoding="utf-8")

    return cases_path, summary_path, latex_path


def main() -> None:
    args = parse_args()

    ten_year_cases = load_ten_year_cases(args.source_dir)
    metrics = load_reference_proxy_metrics(
        args.source_dir,
        ten_year_cases,
    )

    margins = collapse_margins_to_unique_weather_sets(ten_year_cases)
    unique_metrics = collapse_metrics_to_unique_weather_sets(
        ten_year_cases,
        metrics,
    )
    case_table = build_case_table(margins, unique_metrics)
    summary = build_summary(case_table)
    latex = build_latex_table(summary)

    print_summary(case_table, summary)

    outputs = save_outputs(
        case_table,
        summary,
        latex,
        output_dir=args.output_dir,
    )

    print("\nSaved:")
    for output in outputs:
        print(f"  {output}")

    print("\nLaTeX table")
    print("-----------")
    print(latex)


if __name__ == "__main__":
    main()
