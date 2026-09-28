"""Quick diagnostic: median annualised CAPEX by investable technology.

Filters parameters.parquet to:
- approximately 10-year runs
- k-means clustering
- medoid representation
- W_P = 0.5

Then reads costs.parquet and:
- keeps cost_class == "capex"
- uses `value` (annualised CAPEX; not `unannualised_value`)
- sums across nodes within each case/model_type/technology
- reports median, min, max and n by technology
- additionally reports combined LDES-system CAPEX:
    h2_salt_cavern + electrolyser + h2_elec_conversion

Source:
    results/2_5_10_year/
        parameters.parquet
        costs.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results/2_5_10_year")

TARGET_HORIZON_YEARS = 10.0
HORIZON_TOLERANCE_YEARS = 0.10
TARGET_WP = 0.5

CLUSTER_METHOD = "kmeans"
REPRESENTATION_METHOD = "medoid"

LDES_STORAGE_TECH = "h2_salt_cavern"

H2_STORAGE_PATHWAY_TECHS = (
    "h2_salt_cavern",
    "electrolyser",
    "h2_elec_conversion",
)


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


def select_cases(parameters: pd.DataFrame) -> pd.DataFrame:
    """Return 10-year k-means + medoid W_P=0.5 cases."""
    require_columns(
        parameters,
        {
            "case_id",
            "country",
            "start_date",
            "end_date",
            "lambda_soc",
            "cluster_method",
            "representation_method",
        },
        table_name="parameters.parquet",
    )

    data = parameters.copy()

    data["case_id"] = data["case_id"].astype(str)
    data["start_date"] = pd.to_datetime(data["start_date"])
    data["end_date"] = pd.to_datetime(data["end_date"])
    data["lambda_soc"] = data["lambda_soc"].astype(float)

    data["horizon_years"] = (
        (data["end_date"] - data["start_date"]).dt.total_seconds()
        / (365.2425 * 24 * 60 * 60)
    )

    selected = data.loc[
        np.isclose(
            data["horizon_years"],
            TARGET_HORIZON_YEARS,
            atol=HORIZON_TOLERANCE_YEARS,
            rtol=0.0,
        )
        & data["cluster_method"].astype(str).str.lower().eq(CLUSTER_METHOD)
        & data["representation_method"]
        .astype(str)
        .str.lower()
        .eq(REPRESENTATION_METHOD)
        & np.isclose(data["lambda_soc"], TARGET_WP)
    ].copy()

    if selected.empty:
        raise ValueError(
            "No approximately 10-year k-means + medoid W_P=0.5 cases found."
        )

    return selected


def load_capex(
    costs: pd.DataFrame,
    selected_cases: pd.DataFrame,
) -> pd.DataFrame:
    """Filter annualised CAPEX rows and aggregate across nodes."""
    require_columns(
        costs,
        {
            "case_id",
            "model_type",
            "node",
            "tech",
            "cost_class",
            "value",
        },
        table_name="costs.parquet",
    )

    costs = costs.copy()
    costs["case_id"] = costs["case_id"].astype(str)

    case_ids = set(selected_cases["case_id"])

    capex = costs.loc[
        costs["case_id"].isin(case_ids)
        & costs["cost_class"].astype(str).str.lower().eq("capex")
    ].copy()

    if capex.empty:
        raise ValueError(
            "No CAPEX rows in costs.parquet match the selected case_ids."
        )

    capex["value"] = pd.to_numeric(capex["value"], errors="raise")

    # Attach experiment metadata for diagnostics.
    metadata_cols = [
        column
        for column in (
            "case_id",
            "country",
            "start_date",
            "end_date",
            "k_periods",
            "lambda_soc",
        )
        if column in selected_cases.columns
    ]

    capex = capex.merge(
        selected_cases[metadata_cols].drop_duplicates("case_id"),
        on="case_id",
        how="left",
        validate="many_to_one",
    )

    # A technology can potentially exist at more than one node.
    # Sum its annualised CAPEX contribution within each model run first.
    group_cols = [
        "case_id",
        "model_type",
        "tech",
    ]

    for column in (
        "country",
        "start_date",
        "end_date",
        "k_periods",
        "lambda_soc",
    ):
        if column in capex.columns:
            group_cols.append(column)

    per_case_tech = (
        capex.groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )
        .agg(
            annualised_capex=("value", "sum"),
            n_nodes=("node", "nunique"),
        )
    )

    return per_case_tech


def add_h2_pathway_total(per_case_tech: pd.DataFrame) -> pd.DataFrame:
    """Add a combined hydrogen-storage-pathway CAPEX row.

    This is deliberately not labelled as LDES CAPEX: the electrolyser and
    H2-to-electricity conversion unit are enabling pathway technologies,
    whereas h2_salt_cavern is the LDES storage technology itself.
    """
    pathway = per_case_tech.loc[
        per_case_tech["tech"].isin(H2_STORAGE_PATHWAY_TECHS)
    ].copy()

    if pathway.empty:
        return per_case_tech

    id_cols = [
        column
        for column in per_case_tech.columns
        if column not in {
            "tech",
            "annualised_capex",
            "n_nodes",
        }
    ]

    combined = (
        pathway.groupby(
            id_cols,
            as_index=False,
            dropna=False,
        )
        .agg(
            annualised_capex=("annualised_capex", "sum"),
            n_nodes=("n_nodes", "sum"),
        )
    )

    combined["tech"] = "H2_storage_pathway_total"
    combined = combined[per_case_tech.columns]

    return pd.concat(
        [per_case_tech, combined],
        ignore_index=True,
    )


def summarise(per_case_tech: pd.DataFrame) -> pd.DataFrame:
    """Return median and range of annualised CAPEX by model type/technology."""
    return (
        per_case_tech.groupby(
            ["model_type", "tech"],
            as_index=False,
        )
        .agg(
            n=("annualised_capex", "size"),
            median_annualised_capex=("annualised_capex", "median"),
            minimum=("annualised_capex", "min"),
            maximum=("annualised_capex", "max"),
        )
        .sort_values(
            ["model_type", "median_annualised_capex"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def calculate_capex_shares(
    per_case_tech: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate storage-tech and H2-pathway shares of total CAPEX case-by-case.

    `ldes_storage_capex_share_pct` refers only to h2_salt_cavern.

    `h2_pathway_capex_share_pct` additionally includes the electrolyser and
    H2-to-electricity conversion unit and is reported separately to avoid
    conflating the complete hydrogen-storage pathway with the storage
    technology itself.
    """
    totals = (
        per_case_tech.groupby(
            ["case_id", "model_type"],
            as_index=False,
        )
        .agg(
            total_annualised_capex=("annualised_capex", "sum"),
        )
    )

    storage = (
        per_case_tech.loc[
            per_case_tech["tech"].eq(LDES_STORAGE_TECH)
        ]
        .groupby(
            ["case_id", "model_type"],
            as_index=False,
        )
        .agg(
            ldes_storage_annualised_capex=("annualised_capex", "sum"),
        )
    )

    pathway = (
        per_case_tech.loc[
            per_case_tech["tech"].isin(H2_STORAGE_PATHWAY_TECHS)
        ]
        .groupby(
            ["case_id", "model_type"],
            as_index=False,
        )
        .agg(
            h2_pathway_annualised_capex=("annualised_capex", "sum"),
        )
    )

    shares = (
        totals
        .merge(
            storage,
            on=["case_id", "model_type"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            pathway,
            on=["case_id", "model_type"],
            how="left",
            validate="one_to_one",
        )
    )

    shares["ldes_storage_annualised_capex"] = (
        shares["ldes_storage_annualised_capex"].fillna(0.0)
    )
    shares["h2_pathway_annualised_capex"] = (
        shares["h2_pathway_annualised_capex"].fillna(0.0)
    )

    shares["ldes_storage_capex_share_pct"] = (
        100.0
        * shares["ldes_storage_annualised_capex"]
        / shares["total_annualised_capex"]
    )

    shares["h2_pathway_capex_share_pct"] = (
        100.0
        * shares["h2_pathway_annualised_capex"]
        / shares["total_annualised_capex"]
    )

    summary = (
        shares.groupby("model_type", as_index=False)
        .agg(
            n=("ldes_storage_capex_share_pct", "size"),
            median_ldes_storage_share_pct=(
                "ldes_storage_capex_share_pct",
                "median",
            ),
            ldes_storage_share_minimum=(
                "ldes_storage_capex_share_pct",
                "min",
            ),
            ldes_storage_share_maximum=(
                "ldes_storage_capex_share_pct",
                "max",
            ),
            median_h2_pathway_share_pct=(
                "h2_pathway_capex_share_pct",
                "median",
            ),
            h2_pathway_share_minimum=(
                "h2_pathway_capex_share_pct",
                "min",
            ),
            h2_pathway_share_maximum=(
                "h2_pathway_capex_share_pct",
                "max",
            ),
        )
        .sort_values("model_type")
        .reset_index(drop=True)
    )

    return shares, summary


def main() -> None:
    parameters = pd.read_parquet(
        RESULTS_DIR / "parameters.parquet"
    )
    costs = pd.read_parquet(
        RESULTS_DIR / "costs.parquet"
    )

    selected = select_cases(parameters)
    per_case_tech = load_capex(costs, selected)

    # Calculate shares before adding the synthetic pathway-total row,
    # otherwise the pathway components would be counted twice in total CAPEX.
    capex_shares, capex_share_summary = calculate_capex_shares(
        per_case_tech
    )

    per_case_tech_with_pathway = add_h2_pathway_total(per_case_tech)
    summary = summarise(per_case_tech_with_pathway)

    print("Selected experiment cases")
    print("=========================")
    print(f"Cases:     {selected['case_id'].nunique()}")
    print(f"Countries: {sorted(selected['country'].astype(str).unique())}")

    if "k_periods" in selected.columns:
        print(
            "k values:  "
            f"{sorted(selected['k_periods'].astype(int).unique())}"
        )

    print(f"W_P:       {TARGET_WP:g}")
    print("Horizon:   ~10 years")
    print("Method:    k-means + medoid")

    print("\nAnnualised CAPEX medians")
    print("========================")
    print(
        summary.to_string(
            index=False,
            formatters={
                "median_annualised_capex": lambda x: f"{x:,.2f}",
                "minimum": lambda x: f"{x:,.2f}",
                "maximum": lambda x: f"{x:,.2f}",
            },
        )
    )

    print("\nCAPEX shares")
    print("============")
    print(
        capex_share_summary.to_string(
            index=False,
            formatters={
                "median_ldes_storage_share_pct": lambda x: f"{x:.2f}%",
                "ldes_storage_share_minimum": lambda x: f"{x:.2f}%",
                "ldes_storage_share_maximum": lambda x: f"{x:.2f}%",
                "median_h2_pathway_share_pct": lambda x: f"{x:.2f}%",
                "h2_pathway_share_minimum": lambda x: f"{x:.2f}%",
                "h2_pathway_share_maximum": lambda x: f"{x:.2f}%",
            },
        )
    )

    reference_share = capex_share_summary.loc[
        capex_share_summary["model_type"]
        .astype(str)
        .str.lower()
        .eq("reference")
    ]

    if not reference_share.empty:
        row = reference_share.iloc[0]

        print(
            "\nReference-model headline: "
            "the LDES storage technology (h2_salt_cavern) accounts for a "
            f"median {row['median_ldes_storage_share_pct']:.1f}% "
            "of total annualised CAPEX "
            f"(range {row['ldes_storage_share_minimum']:.1f}%–"
            f"{row['ldes_storage_share_maximum']:.1f}%)."
        )

        print(
            "For comparison, the complete hydrogen-storage pathway "
            "(salt cavern + electrolyser + H2-to-electricity conversion) "
            f"accounts for a median {row['median_h2_pathway_share_pct']:.1f}% "
            "of total annualised CAPEX."
        )

    print(
        "\nNote: `value` from costs.parquet is used throughout; "
        "`unannualised_value` is deliberately ignored."
    )

    # A compact reference-only view is often the most useful for discussing
    # the CAPEX weights behind the reference-vs-clustered investment metric.
    reference = summary.loc[
        summary["model_type"].astype(str).str.lower().eq("reference")
    ]

    if not reference.empty:
        print("\nReference model — compact comparison")
        print("====================================")
        print(
            reference[
                [
                    "tech",
                    "median_annualised_capex",
                ]
            ].to_string(
                index=False,
                formatters={
                    "median_annualised_capex": lambda x: f"{x:,.2f}",
                },
            )
        )


if __name__ == "__main__":
    main()
