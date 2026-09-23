"""Capacity and investment metrics comparing clustered and reference models."""

from __future__ import annotations

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "case_id",
    "metric",
    "value",
]


def calculate_investment_metrics(
    capacities: pd.DataFrame,
    costs: pd.DataFrame,
    *,
    ldes_tech: str = "h2_salt_cavern",
) -> pd.DataFrame:
    """Calculate investment-related metrics for all cases.

    Metrics are returned in long format with one row per case and metric.

    Relative errors are expressed as fractions, not percentages. For example,
    ``0.1`` means +10% and ``-0.1`` means -10%.

    MACME remains an absolute-error metric by definition. Cost and dedicated
    LDES errors are signed so direction can be retained in subsequent analysis.
    """
    _validate_inputs(
        capacities,
        costs,
    )

    case_ids = sorted(set(capacities["case_id"]) | set(costs["case_id"]))

    rows: list[dict[str, object]] = []

    for case_id in case_ids:
        case_capacities = capacities.loc[capacities["case_id"] == case_id].copy()

        case_costs = costs.loc[costs["case_id"] == case_id].copy()

        rows.extend(
            _calculate_case_metrics(
                case_id,
                case_capacities,
                case_costs,
                ldes_tech=ldes_tech,
            )
        )

    return pd.DataFrame(
        rows,
        columns=METRIC_COLUMNS,
    )


def _calculate_case_metrics(
    case_id: str,
    capacities: pd.DataFrame,
    costs: pd.DataFrame,
    *,
    ldes_tech: str,
) -> list[dict[str, object]]:
    """Calculate all investment metrics for one case."""
    capacity_mix = _capacity_mix(capacities)

    metrics = {
        "ldes_capacity_error_signed": _ldes_capacity_error(
            capacity_mix,
            ldes_tech=ldes_tech,
        ),
        "macme": _macme(
            capacity_mix,
        ),
        "flow_capacity_nacd": _capacity_nacd(
            capacities,
            capacity_type="flow_cap",
        ),
        "storage_capacity_nacd": _capacity_nacd(
            capacities,
            capacity_type="storage_cap",
        ),
        "macme_capex_weighted_annualised": _capex_weighted_macme(
            capacity_mix,
            costs,
            cost_column="value",
        ),
        "macme_capex_weighted_unannualised": _capex_weighted_macme(
            capacity_mix,
            costs,
            cost_column="unannualised_value",
        ),
        "flow_capacity_mix_distance": _capacity_mix_distance(
            capacities,
            capacity_type="flow_cap",
        ),
        "storage_capacity_mix_distance": _capacity_mix_distance(
            capacities,
            capacity_type="storage_cap",
        ),
        "system_cost_error_signed": _cost_error(
            costs,
            cost_class=None,
            column="value",
        ),
        "capex_annualised_error_signed": _cost_error(
            costs,
            cost_class="capex",
            column="value",
        ),
        "capex_unannualised_error_signed": _cost_error(
            costs,
            cost_class="capex",
            column="unannualised_value",
        ),
        "opex_error_signed": _cost_error(
            costs,
            cost_class="opex",
            column="value",
        ),
    }

    return [
        {
            "case_id": case_id,
            "metric": metric,
            "value": value,
        }
        for metric, value in metrics.items()
    ]


def _capacity_mix(
    capacities: pd.DataFrame,
) -> pd.DataFrame:
    """Construct the original MACME capacity basis.

    Storage technologies are represented by storage energy capacity.
    All other technologies are represented by output flow capacity.

    This yields one capacity value per node/technology/model combination.
    """
    storage_techs = set(
        capacities.loc[
            capacities["capacity_type"] == "storage_cap",
            ["node", "tech"],
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    keep = []

    for row in capacities.itertuples():
        key = (
            row.node,
            row.tech,
        )

        if key in storage_techs:
            keep.append(row.capacity_type == "storage_cap")
        else:
            keep.append(row.capacity_type == "flow_cap")

    mix = capacities.loc[
        keep,
        [
            "model_type",
            "node",
            "tech",
            "value",
        ],
    ].copy()

    duplicates = mix.duplicated(
        subset=[
            "model_type",
            "node",
            "tech",
        ],
        keep=False,
    )

    if duplicates.any():
        raise RuntimeError(
            "Capacity mix did not resolve to exactly one capacity value per "
            "model/node/technology. Duplicate rows:\n"
            f"{mix.loc[duplicates]}"
        )

    return mix


def _paired_capacities(
    capacities: pd.DataFrame,
) -> pd.DataFrame:
    """Pivot reference and clustered capacity values into paired columns."""
    paired = capacities.pivot(
        index=[
            "node",
            "tech",
        ],
        columns="model_type",
        values="value",
    )

    for model_type in (
        "reference",
        "clustered",
    ):
        if model_type not in paired.columns:
            paired[model_type] = np.nan

    return paired[
        [
            "reference",
            "clustered",
        ]
    ]


def _ldes_capacity_error(
    capacity_mix: pd.DataFrame,
    *,
    ldes_tech: str,
) -> float:
    """Signed relative LDES capacity error."""
    subset = capacity_mix.loc[capacity_mix["tech"] == ldes_tech]

    paired = _paired_capacities(subset)

    if len(paired) != 1:
        raise RuntimeError(
            f"Expected exactly one {ldes_tech!r} capacity comparison, "
            f"found {len(paired)}."
        )

    reference = float(paired["reference"].iloc[0])

    clustered = float(paired["clustered"].iloc[0])

    return _signed_relative_error(
        clustered,
        reference,
        metric="LDES capacity error",
    )


def _macme(
    capacity_mix: pd.DataFrame,
) -> float:
    """Mean Absolute Capacity Mix Error."""
    paired = _paired_capacities(capacity_mix)

    paired["clustered"] = paired["clustered"].fillna(0.0)

    eligible = paired.loc[paired["reference"].notna() & (paired["reference"].abs() > 0)]

    if eligible.empty:
        raise RuntimeError(
            "MACME cannot be calculated because all reference capacities "
            "are zero or missing."
        )

    errors = (eligible["clustered"] - eligible["reference"]).abs() / eligible[
        "reference"
    ].abs()

    return float(errors.mean())


def _capacity_nacd(
    capacities: pd.DataFrame,
    *,
    capacity_type: str,
) -> float:
    """Normalised absolute capacity deviation for one capacity type."""
    subset = capacities.loc[capacities["capacity_type"] == capacity_type]

    paired = _paired_capacities(subset).fillna(0.0)

    denominator = float(paired["reference"].abs().sum())

    if np.isclose(
        denominator,
        0.0,
    ):
        return np.nan

    numerator = float((paired["clustered"] - paired["reference"]).abs().sum())

    return numerator / denominator


def _capex_weighted_macme(
    capacity_mix: pd.DataFrame,
    costs: pd.DataFrame,
    *,
    cost_column: str,
) -> float:
    """MACME weighted by each technology's reference CAPEX."""
    paired = _paired_capacities(capacity_mix)

    paired["clustered"] = paired["clustered"].fillna(0.0)

    eligible = paired.loc[
        paired["reference"].notna() & (paired["reference"].abs() > 0)
    ].copy()

    if eligible.empty:
        return np.nan

    eligible["capacity_error"] = (
        eligible["clustered"] - eligible["reference"]
    ).abs() / eligible["reference"].abs()

    reference_capex = (
        costs.loc[
            (costs["model_type"] == "reference") & (costs["cost_class"] == "capex"),
            [
                "node",
                "tech",
                cost_column,
            ],
        ]
        .dropna(
            subset=[
                cost_column,
            ]
        )
        .groupby(
            [
                "node",
                "tech",
            ],
            as_index=True,
        )[cost_column]
        .sum()
    )

    eligible = eligible.join(
        reference_capex.rename("weight_value"),
        how="left",
    )

    eligible["weight_value"] = eligible["weight_value"].fillna(0.0)

    denominator = float(eligible["weight_value"].sum())

    if np.isclose(
        denominator,
        0.0,
    ):
        return np.nan

    weights = eligible["weight_value"] / denominator

    return float((weights * eligible["capacity_error"]).sum())


def _capacity_mix_distance(
    capacities: pd.DataFrame,
    *,
    capacity_type: str,
) -> float:
    """Total-variation distance between reference and clustered capacity mix."""
    subset = capacities.loc[capacities["capacity_type"] == capacity_type]

    paired = _paired_capacities(subset).fillna(0.0)

    reference_total = float(paired["reference"].sum())

    clustered_total = float(paired["clustered"].sum())

    if np.isclose(reference_total, 0.0) or np.isclose(clustered_total, 0.0):
        return np.nan

    reference_share = paired["reference"] / reference_total

    clustered_share = paired["clustered"] / clustered_total

    return float(0.5 * (reference_share - clustered_share).abs().sum())


def _cost_error(
    costs: pd.DataFrame,
    *,
    cost_class: str | None,
    column: str,
) -> float:
    """Signed relative cost error."""
    subset = costs

    if cost_class is not None:
        subset = subset.loc[subset["cost_class"] == cost_class]

    totals = subset.groupby("model_type")[column].sum(min_count=1)

    if "reference" not in totals or "clustered" not in totals:
        return np.nan

    reference = float(totals["reference"])

    clustered = float(totals["clustered"])

    return _signed_relative_error(
        clustered,
        reference,
        metric=f"{cost_class or 'system'} cost error",
    )


def _signed_relative_error(
    estimate: float,
    reference: float,
    *,
    metric: str,
) -> float:
    """Return (estimate - reference) / reference."""
    if np.isclose(
        reference,
        0.0,
    ):
        raise RuntimeError(
            f"{metric} is undefined because the reference value is zero."
        )

    return (estimate - reference) / reference


def _validate_inputs(
    capacities: pd.DataFrame,
    costs: pd.DataFrame,
) -> None:
    required_capacities = {
        "case_id",
        "model_type",
        "node",
        "tech",
        "capacity_type",
        "carrier",
        "value",
    }

    required_costs = {
        "case_id",
        "model_type",
        "node",
        "tech",
        "cost_class",
        "value",
        "unannualised_value",
    }

    missing_capacities = required_capacities - set(capacities.columns)

    missing_costs = required_costs - set(costs.columns)

    if missing_capacities:
        raise ValueError(
            f"capacities is missing required columns: {sorted(missing_capacities)}"
        )

    if missing_costs:
        raise ValueError(f"costs is missing required columns: {sorted(missing_costs)}")
