"""Summarise reference SoC-proxy approximation quality for the paper table.

The script extracts proxy-approximation metrics for unique 10-year
country/weather-year-set cases and reports the median and full range by country.

Metrics retained from signal_metrics.parquet:
    error_family = "reference_approximation"
    signal_type = "level"
    metric in {"pearson", "rmse", "mbe"}
    window_half_width_days is empty / null

Multiple model configurations can share the same country and weather-year set.
Because these are reference-proxy statistics, they should be identical across
such duplicate case_ids. The script therefore collapses them to one result per:

    country + start_date + end_date + metric

and raises an error if duplicate case_ids disagree materially.

Source:
    results/2_5_10_year/
        parameters.parquet
        signal_metrics.parquet

Outputs:
    results/tables/table_ref_proxy_quality/
        table_ref_proxy_quality_summary.csv
        table_ref_proxy_quality_cases.csv
        table_ref_proxy_quality.tex
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
METRICS = ("pearson", "rmse", "mbe")

DUPLICATE_VALUE_ATOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise 10-year reference SoC-proxy approximation metrics "
            "by country."
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
    parameters = pd.read_parquet(source_dir / "parameters.parquet")

    require_columns(
        parameters,
        {"case_id", "country", "start_date", "end_date"},
        table_name="parameters.parquet",
    )

    parameters = parameters.copy()
    parameters["case_id"] = parameters["case_id"].astype(str)
    parameters["start_date"] = pd.to_datetime(parameters["start_date"])
    parameters["end_date"] = pd.to_datetime(parameters["end_date"])

    parameters["horizon_years"] = (
        (parameters["end_date"] - parameters["start_date"]).dt.total_seconds()
        / (365.2425 * 24 * 60 * 60)
    )

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

    return parameters[
        ["case_id", "country", "start_date", "end_date", "horizon_years"]
    ].drop_duplicates()


def empty_window_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()

    if (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
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
            "but duplicates were found:\n"
            + dup.head(40).to_string(index=False)
        )

    return metrics[["case_id", "metric", "value"]]


def collapse_to_unique_weather_sets(
    ten_year_cases: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    data = ten_year_cases.merge(
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

    consistency = (
        data.groupby(key, as_index=False)
        .agg(
            n_case_ids=("case_id", "nunique"),
            value_min=("value", "min"),
            value_max=("value", "max"),
        )
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

    unique_cases = (
        data.groupby(key, as_index=False)
        .agg(
            value=("value", "mean"),
            n_duplicate_case_ids=("case_id", "nunique"),
        )
        .sort_values(["country", "start_date", "metric"])
        .reset_index(drop=True)
    )

    return unique_cases


def build_summary(unique_cases: pd.DataFrame) -> pd.DataFrame:
    return (
        unique_cases.groupby(["country", "metric"], as_index=False)
        .agg(
            n=("value", "size"),
            median=("value", "median"),
            minimum=("value", "min"),
            maximum=("value", "max"),
        )
        .sort_values(["country", "metric"])
        .reset_index(drop=True)
    )


def format_value(metric: str, value: float) -> str:
    if metric == "pearson":
        return f"{value:.2f}"
    return f"{value:.2f}"


def build_latex_table(summary: pd.DataFrame) -> str:
    countries = list(summary["country"].drop_duplicates())

    lookup = {
        (row.country, row.metric): row
        for row in summary.itertuples(index=False)
    }

    lines = [
        r"\begin{table}[]",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        (
            r"\multicolumn{1}{c}{} & "
            r"\multicolumn{2}{c}{Pearson $R$} & "
            r"\multicolumn{2}{c}{RMSE} & "
            r"\multicolumn{2}{c}{MBE} \\ \midrule"
        ),
        r"Country & Median & Range & Median & Range & Median & Range \\",
    ]

    for country in countries:
        cells = [str(country)]

        for metric in ("pearson", "rmse", "mbe"):
            row = lookup.get((country, metric))

            if row is None:
                cells.extend(["--", "--"])
                continue

            median = format_value(metric, float(row.median))
            minimum = format_value(metric, float(row.minimum))
            maximum = format_value(metric, float(row.maximum))

            cells.extend([median, f"{minimum}--{maximum}"])

        lines.append(" & ".join(cells) + r" \\")

    counts = summary.groupby("country")["n"].min().astype(int)
    unique_counts = sorted(counts.unique())

    if len(unique_counts) == 1:
        n_text = f"$n={unique_counts[0]}$ per country"
    else:
        n_text = ", ".join(
            f"{country}: $n={count}$"
            for country, count in counts.items()
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{SoC proxy approximation quality across unique "
                r"10-year reference weather-year sets ("
                + n_text
                + r"). Reported are the median and full range of Pearson "
                r"correlation, RMSE, and mean bias error (MBE) between the "
                r"proxy and the CEM SoC trajectory.}"
            ),
            r"\label{tab:ref-proxy-qual}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def print_summary(
    unique_cases: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    print("Reference SoC-proxy approximation quality")
    print("=========================================")

    weather_sets = (
        unique_cases[["country", "start_date", "end_date"]]
        .drop_duplicates()
        .groupby("country")
        .size()
    )

    print("\nUnique 10-year weather sets:")
    for country, count in weather_sets.items():
        print(f"  {country}: {count}")

    print("\nMedian and range:")
    for country in summary["country"].drop_duplicates():
        print(f"\n{country}")
        country_data = summary.loc[summary["country"].eq(country)]

        for row in country_data.itertuples(index=False):
            print(
                f"  {row.metric:8s}: "
                f"median={row.median:.4g}, "
                f"range={row.minimum:.4g} to {row.maximum:.4g}, "
                f"n={int(row.n)}"
            )


def save_outputs(
    unique_cases: pd.DataFrame,
    summary: pd.DataFrame,
    latex: str,
    *,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_path = output_dir / "table_ref_proxy_quality_cases.csv"
    summary_path = output_dir / "table_ref_proxy_quality_summary.csv"
    latex_path = output_dir / "table_ref_proxy_quality.tex"

    unique_cases.to_csv(cases_path, index=False)
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

    unique_cases = collapse_to_unique_weather_sets(
        ten_year_cases,
        metrics,
    )

    summary = build_summary(unique_cases)
    latex = build_latex_table(summary)

    print_summary(unique_cases, summary)

    outputs = save_outputs(
        unique_cases,
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
