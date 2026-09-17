"""Orchestrate extraction and persistence of experiment results."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from scripts.helpers.results_capacities import (
    extract_capacities,
)
from scripts.helpers.results_costs import (
    extract_costs,
)
from scripts.helpers.results_parameters import (
    extract_parameters,
)
from scripts.helpers.results_investment_metrics import (
    calculate_investment_metrics,
)
from scripts.helpers.results_signal_metrics import (
    extract_signal_metrics,
)


_CASE_CONFIG_KEYS = (
    "dispatch_mode",
    "data_params",
    "soc_proxy_params",
    "tsa_params",
    "calliope_params",
)


@dataclass(frozen=True)
class CaseResults:
    """Compact persistent results from one experiment case."""

    case_id: str
    parameters: pd.DataFrame
    capacities: pd.DataFrame
    costs: pd.DataFrame
    investment_metrics: pd.DataFrame
    signal_metrics: pd.DataFrame


def generate_case_id(
    config: dict,
    *,
    length: int = 16,
) -> str:
    """Generate a deterministic ID from the scientific case definition."""
    definition = {
        key: config[key]
        for key in _CASE_CONFIG_KEYS
        if key in config
    }

    canonical = _canonicalise(definition)

    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()[:length]


def extract_case_results(
    config: dict,
    case,
    reference_model,
) -> CaseResults:
    """Extract all persistent results for one solved case."""
    case_id = generate_case_id(config)

    model = case.calliope_model

    parameters = extract_parameters(
        config,
        model,
        case_id=case_id,
    )

    capacities = pd.concat(
        [
            extract_capacities(
                reference_model,
                case_id=case_id,
                model_type="reference",
            ),
            extract_capacities(
                model,
                case_id=case_id,
                model_type="clustered",
            ),
        ],
        ignore_index=True,
    )

    costs = pd.concat(
        [
            extract_costs(
                reference_model,
                case_id=case_id,
                model_type="reference",
            ),
            extract_costs(
                model,
                case_id=case_id,
                model_type="clustered",
            ),
        ],
        ignore_index=True,
    )

    investment_metrics = calculate_investment_metrics(
        capacities,
        costs,
    )

    signal_metrics = extract_signal_metrics(
        case_id=case_id,
        reference_model=reference_model,
        clustered_model=model,
        original_proxy=case.tsa.original_proxy,
        reconstructed_proxy=case.tsa.reconstructed_proxy,
        cluster_map=case.tsa.cluster_map,
    )

    return CaseResults(
        case_id=case_id,
        parameters=parameters,
        capacities=capacities,
        costs=costs,
        investment_metrics=investment_metrics,
        signal_metrics=signal_metrics,
    )

def record_case_results(
    config: dict,
    case,
    reference_model,
    *,
    results_dir: str | Path = "results",
) -> str:
    """Extract and persist all results from one solved case.

    Returns
    -------
    str
        Deterministic case ID.
    """
    results = extract_case_results(
        config,
        case,
        reference_model,
    )

    root = Path(results_dir) / "_fragments"

    _write_fragment(
        results.parameters,
        root / "parameters" / f"{results.case_id}.parquet",
    )

    _write_fragment(
        results.capacities,
        root / "capacities" / f"{results.case_id}.parquet",
    )

    _write_fragment(
        results.costs,
        root / "costs" / f"{results.case_id}.parquet",
    )

    _write_fragment(
        results.investment_metrics,
        root / "investment_metrics" / f"{results.case_id}.parquet",
    )

    _write_fragment(
        results.signal_metrics,
        root / "signal_metrics" / f"{results.case_id}.parquet",
    )

    return results.case_id


def consolidate_results(
    *,
    results_dir: str | Path = "results",
) -> None:
    """Consolidate per-case fragments into analysis-ready Parquet files."""
    root = Path(results_dir)

    for table in (
        "parameters",
        "capacities",
        "costs",
        "investment_metrics",
        "signal_metrics",
    ):
        source = root / "_fragments" / table
        paths = sorted(source.glob("*.parquet"))

        if not paths:
            continue

        frames = [
            pd.read_parquet(path)
            for path in paths
        ]

        combined = pd.concat(
            frames,
            ignore_index=True,
        )

        _atomic_to_parquet(
            combined,
            root / f"{table}.parquet",
        )


def _write_fragment(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write one case fragment atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _atomic_to_parquet(
        frame,
        path,
    )


def _atomic_to_parquet(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write a Parquet file without exposing a partially written file."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        f".{path.stem}.{uuid4().hex}.tmp.parquet"
    )

    try:
        frame.to_parquet(
            temporary,
            index=False,
        )

        os.replace(
            temporary,
            path,
        )

    finally:
        if temporary.exists():
            temporary.unlink()


def _canonicalise(value):
    """Convert configuration values into stable JSON-compatible values."""
    if isinstance(value, dict):
        return {
            str(key): _canonicalise(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _canonicalise(item)
            for item in value
        ]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    return value
