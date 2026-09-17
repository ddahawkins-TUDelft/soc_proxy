"""Adapters for the fixed Calliope timeseries CSV format used in this project.

Calliope input files contain several metadata rows before the hourly data. The
rest of the SoC Proxy workflow should not need to understand that format, so
this module converts between the Calliope CSV representation and a simple
``DatetimeIndex`` dataframe.

Model horizons use a half-open interval: ``[start, end)``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_FEATURE_COLUMNS = (
    "demand_power",
    "solar",
    "offshore_wind",
    "onshore_wind",
)


def read_calliope_timeseries(
    path: str | Path,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read a project Calliope timeseries file into a simple dataframe.

    Parameters
    ----------
    path
        Calliope-formatted CSV file.
    start, end
        Optional model horizon bounds. If supplied, data are returned over the
        half-open interval ``[start, end)``.

    Returns
    -------
    pandas.DataFrame
        Hourly numeric features indexed by ``timesteps``.
    """
    rows = _read_csv_rows(path)
    tech_row = _find_marker_row(rows, "techs")
    timestep_row = _find_marker_row(rows, "timesteps", start=tech_row + 1)

    columns = tuple(value.strip() for value in rows[tech_row][1:] if value.strip())
    _validate_feature_columns(columns)

    n_fields = len(columns) + 1
    data_rows = [row[:n_fields] for row in rows[timestep_row + 1 :] if row and row[0].strip()]

    if not data_rows:
        raise ValueError(f"No timeseries data found in {Path(path)}.")

    frame = pd.DataFrame(data_rows, columns=["timesteps", *columns])
    frame["timesteps"] = pd.to_datetime(frame["timesteps"], errors="raise")

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame = frame.set_index("timesteps")
    frame.index = pd.DatetimeIndex(frame.index, name="timesteps")

    _validate_regular_timeseries(frame)

    if start is not None or end is not None:
        frame = slice_timeseries(frame, start=start, end=end)

    return frame


def slice_timeseries(
    data: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a complete half-open model horizon ``[start, end)``.

    Bounds must align to timestamps in the source data. The end bound may also
    be exactly one timestep after the final source timestamp.
    """
    _validate_regular_timeseries(data)

    start_ts = pd.Timestamp(start) if start is not None else data.index[0]
    step = _timestep(data.index)
    end_ts = pd.Timestamp(end) if end is not None else data.index[-1] + step

    if start_ts >= end_ts:
        raise ValueError("Timeseries start must be earlier than end.")

    if start_ts not in data.index:
        raise ValueError(f"Requested start {start_ts} is not present in the source data.")

    valid_end = end_ts in data.index or end_ts == data.index[-1] + step
    if not valid_end:
        raise ValueError(
            f"Requested end {end_ts} is neither present in the source data nor "
            "one timestep after its final timestamp."
        )

    sliced = data.loc[(data.index >= start_ts) & (data.index < end_ts)].copy()
    expected_index = pd.date_range(start=start_ts, end=end_ts, freq=step, inclusive="left")

    if not sliced.index.equals(expected_index):
        missing = expected_index.difference(sliced.index)
        raise ValueError(
            "Requested model horizon is incomplete. "
            f"Missing {len(missing)} timestep(s); first missing values: "
            f"{missing[:5].tolist()}"
        )

    return sliced


def write_calliope_timeseries(
    data: pd.DataFrame,
    path: str | Path,
    *,
    template_path: str | Path,
) -> Path:
    """Write a simple dataframe back to the project Calliope CSV format.

    Metadata rows (comments, nodes, technologies, and parameters) are copied
    from ``template_path``. Only the hourly data section is replaced. This
    keeps the idiosyncratic Calliope table structure at the edge of the
    workflow while allowing TSAM and the SoC Proxy to work with ordinary
    dataframes internally.
    """
    _validate_regular_timeseries(data)

    rows = _read_csv_rows(template_path)
    tech_row = _find_marker_row(rows, "techs")
    timestep_row = _find_marker_row(rows, "timesteps", start=tech_row + 1)
    template_columns = tuple(
        value.strip() for value in rows[tech_row][1:] if value.strip()
    )

    _validate_feature_columns(template_columns)

    missing = set(template_columns) - set(data.columns)
    extra = set(data.columns) - set(template_columns)
    if missing or extra:
        raise ValueError(
            "Timeseries columns do not match the Calliope template. "
            f"Missing: {sorted(missing)}; extra: {sorted(extra)}"
        )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(rows[: timestep_row + 1])

        ordered = data.loc[:, template_columns]
        for timestamp, values in ordered.iterrows():
            writer.writerow(
                [timestamp.strftime("%Y/%m/%d %H:%M")]
                + [_format_numeric(value) for value in values]
            )

    return output


def _read_csv_rows(path: str | Path) -> list[list[str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Timeseries file does not exist: {source}")

    with source.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.reader(file))


def _find_marker_row(
    rows: list[list[str]],
    marker: str,
    *,
    start: int = 0,
) -> int:
    marker = marker.lower()
    for index, row in enumerate(rows[start:], start=start):
        if row and row[0].strip().lower() == marker:
            return index
    raise ValueError(f"Could not find {marker!r} row in Calliope timeseries file.")


def _validate_feature_columns(columns: Iterable[str]) -> None:
    columns = tuple(columns)
    if columns != EXPECTED_FEATURE_COLUMNS:
        raise ValueError(
            "Unexpected Calliope timeseries feature columns. "
            f"Expected {EXPECTED_FEATURE_COLUMNS}, received {columns}."
        )


def _validate_regular_timeseries(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Timeseries data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Timeseries data cannot be empty.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Timeseries data must use a DatetimeIndex.")
    if data.index.has_duplicates:
        raise ValueError("Timeseries timestamps must be unique.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("Timeseries timestamps must be monotonically increasing.")
    if data.isna().any().any():
        raise ValueError("Timeseries data contain missing values.")

    _timestep(data.index)


def _timestep(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        raise ValueError("At least two timestamps are required.")

    differences = index.to_series().diff().dropna().unique()
    if len(differences) != 1:
        raise ValueError("Timeseries data must have a regular timestep.")

    step = pd.Timedelta(differences[0])
    if step <= pd.Timedelta(0):
        raise ValueError("Timeseries timestep must be positive.")
    return step


def _format_numeric(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Cannot write missing values to Calliope timeseries.")
    return format(float(value), ".15g")
