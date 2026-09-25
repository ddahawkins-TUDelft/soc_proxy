"""Single-predictor diagnostic screening for LDES capacity error.

For each combination of:

- error family:
    - clustered approximation
    - TSA
    - RPCC
- signal type:
    - level
    - delta
- metric:
    - nMBE
    - nRMSE
    - Pearson correlation
- temporal window:
    - full horizon
    - ±15, ±30, ±45, ±90, ±180, ±273, ±365,
      ±548, ±730, ±912 days

fit a univariate linear regression against LDES capacity error.

The analysis can be configured to:

1. Use all proxy weights or a single W_P value.
2. Predict signed or absolute LDES capacity error.

Inputs
------
results/10_year/signal_metrics.parquet
results/10_year/investment_metrics.parquet
results/10_year/parameters.parquet

Outputs
-------
results/10_year/single_predictor_diagnostics_<configuration>.csv
results/10_year/single_predictor_diagnostics_<configuration>.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================
# USER CONFIGURATION
# =====================================================================

# Proxy weight to analyse.
#
# Examples:
#   PROXY_WEIGHT = 0.5   -> only W_P = 0.5
#   PROXY_WEIGHT = 0.25  -> only W_P = 0.25
#   PROXY_WEIGHT = None  -> pool all proxy weights
PROXY_WEIGHT = 0.5


# Capacity-error target.
#
# Options:
#   "signed"   -> retain direction of LDES capacity error
#   "absolute" -> use magnitude of LDES capacity error
CAPACITY_ERROR = "signed"


# =====================================================================
# Paths
# =====================================================================

RESULTS_DIR = Path("results") / "10_year"

SIGNAL_METRICS_PATH = RESULTS_DIR / "signal_metrics.parquet"
INVESTMENT_METRICS_PATH = RESULTS_DIR / "investment_metrics.parquet"
PARAMETERS_PATH = RESULTS_DIR / "parameters.parquet"


# =====================================================================
# Analysis configuration
# =====================================================================

OUTCOME_METRIC = "ldes_capacity_error_signed"

ERROR_FAMILIES = [
    "clustered_approximation",
    "tsa",
    "rpcc",
]

SIGNAL_TYPES = [
    "level",
    "delta",
]

METRICS = [
    "nmbe",
    "nrmse",
    "pearson",
]

WINDOWS = [
    15,
    30,
    45,
    90,
    180,
    273,
    365,
    548,
    730,
    912,
    None,   # Full horizon
]

FAMILY_LABELS = {
    ("clustered_approximation", "level"): "Proxy Approx.",
    ("tsa", "level"): "Proxy TSA",
    ("rpcc", "level"): "Proxy RPCC",
    ("clustered_approximation", "delta"): "Proxy-Delta Approx.",
    ("tsa", "delta"): "Proxy-Delta TSA",
    ("rpcc", "delta"): "Proxy-Delta RPCC",
}

METRIC_LABELS = {
    "nmbe": "nMBE",
    "nrmse": "nRMSE",
    "pearson": "Pearson",
}


# =====================================================================
# Helpers
# =====================================================================

def fit_single_predictor(
    x: pd.Series,
    y: pd.Series,
) -> dict[str, float]:
    """Fit y = intercept + coefficient * x."""

    valid = x.notna() & y.notna()

    x = x.loc[valid].astype(float).to_numpy()
    y = y.loc[valid].astype(float).to_numpy()

    n = len(x)
    p = 1

    if n <= p + 1 or np.unique(x).size < 2:
        return {
            "n": n,
            "r2": np.nan,
            "adjusted_r2": np.nan,
            "coefficient": np.nan,
            "intercept": np.nan,
        }

    X = np.column_stack(
        [
            np.ones(n),
            x,
        ]
    )

    intercept, coefficient = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )[0]

    predicted = intercept + coefficient * x

    residual_ss = np.sum((y - predicted) ** 2)
    total_ss = np.sum((y - np.mean(y)) ** 2)

    if total_ss == 0:
        r2 = np.nan
        adjusted_r2 = np.nan
    else:
        r2 = 1 - residual_ss / total_ss

        adjusted_r2 = (
            1
            - (1 - r2)
            * (n - 1)
            / (n - p - 1)
        )

    return {
        "n": n,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "coefficient": coefficient,
        "intercept": intercept,
    }


def window_label(window: object) -> str:
    """Human-readable temporal-window label."""

    if pd.isna(window):
        return "Full horizon"

    return f"±{int(window)}"


def configuration_label() -> str:
    """Filename-safe description of the chosen configuration."""

    if PROXY_WEIGHT is None:
        weight = "wp_all"
    else:
        weight = f"wp_{PROXY_WEIGHT:g}".replace(".", "p")

    return f"{weight}_{CAPACITY_ERROR}"


def extract_proxy_weights(
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    """Extract one proxy-weight value per case_id.

    This supports either:

    1. a wide parameters table containing a proxy-weight column, or
    2. a long parameters table of parameter/value pairs.

    Add another alias below if parameters.parquet uses a different name.
    """

    weight_aliases = [
        "lambda_soc",
    ]

    # -------------------------------------------------------------
    # Wide-format parameters.parquet
    # -------------------------------------------------------------

    for column in weight_aliases:
        if column in parameters.columns:
            result = parameters[
                ["case_id", column]
            ].copy()

            result = result.rename(
                columns={
                    column: "proxy_weight",
                }
            )

            return result.drop_duplicates()


    # -------------------------------------------------------------
    # Long-format parameters.parquet
    # -------------------------------------------------------------

    parameter_name_columns = [
        "parameter",
        "parameter_name",
        "name",
        "key",
    ]

    name_column = next(
        (
            column
            for column in parameter_name_columns
            if column in parameters.columns
        ),
        None,
    )

    if (
        name_column is not None
        and "value" in parameters.columns
    ):
        mask = (
            parameters[name_column]
            .astype(str)
            .isin(weight_aliases)
        )

        result = (
            parameters.loc[
                mask,
                ["case_id", "value"],
            ]
            .rename(
                columns={
                    "value": "proxy_weight",
                }
            )
            .copy()
        )

        if not result.empty:
            return result.drop_duplicates()


    raise ValueError(
        "Could not identify W_P in parameters.parquet.\n"
        f"Available columns: {parameters.columns.tolist()}\n"
        f"Expected one of these names: {weight_aliases}"
    )


# =====================================================================
# Validate configuration
# =====================================================================

if CAPACITY_ERROR not in {
    "signed",
    "absolute",
}:
    raise ValueError(
        "CAPACITY_ERROR must be either "
        "'signed' or 'absolute'."
    )


# =====================================================================
# Output paths
# =====================================================================

CONFIG_LABEL = configuration_label()

RESULTS_TABLE_PATH = (
    RESULTS_DIR
    / f"single_predictor_diagnostics_{CONFIG_LABEL}.csv"
)

FIGURE_PATH = (
    RESULTS_DIR
    / f"single_predictor_diagnostics_{CONFIG_LABEL}.png"
)


# =====================================================================
# Load data
# =====================================================================

print(f"Reading {SIGNAL_METRICS_PATH}")
signal_metrics = pd.read_parquet(
    SIGNAL_METRICS_PATH
)

print(f"Reading {INVESTMENT_METRICS_PATH}")
investment_metrics = pd.read_parquet(
    INVESTMENT_METRICS_PATH
)

print(f"Reading {PARAMETERS_PATH}")
parameters = pd.read_parquet(
    PARAMETERS_PATH
)


# =====================================================================
# Extract proxy weight
# =====================================================================

proxy_weights = extract_proxy_weights(
    parameters
)

proxy_weights["proxy_weight"] = pd.to_numeric(
    proxy_weights["proxy_weight"],
    errors="raise",
)


# Make sure each case has only one W_P value.
weight_counts = (
    proxy_weights
    .groupby("case_id")["proxy_weight"]
    .nunique()
)

invalid_weights = weight_counts[
    weight_counts > 1
]

if not invalid_weights.empty:
    raise ValueError(
        "Some case_ids have more than one proxy weight. "
        f"Examples: {invalid_weights.index[:10].tolist()}"
    )

proxy_weights = (
    proxy_weights
    .drop_duplicates(subset=["case_id"])
)


print("\nProxy-weight distribution:")

print(
    proxy_weights["proxy_weight"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)


# =====================================================================
# Extract outcome variable
# =====================================================================

ldes_error = (
    investment_metrics.loc[
        investment_metrics["metric"]
        == OUTCOME_METRIC,
        [
            "case_id",
            "value",
        ],
    ]
    .rename(
        columns={
            "value": "ldes_capacity_error_signed",
        }
    )
    .copy()
)

if ldes_error.empty:
    raise ValueError(
        f"No investment metric found with "
        f"metric={OUTCOME_METRIC!r}"
    )

if ldes_error["case_id"].duplicated().any():
    duplicates = (
        ldes_error.loc[
            ldes_error["case_id"]
            .duplicated(keep=False),
            "case_id",
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        "Expected exactly one signed LDES capacity "
        "error per case_id. Duplicate case_ids include: "
        f"{duplicates[:10]}"
    )


# Construct the requested target.
if CAPACITY_ERROR == "signed":
    ldes_error["ldes_capacity_error"] = (
        ldes_error["ldes_capacity_error_signed"]
    )

elif CAPACITY_ERROR == "absolute":
    ldes_error["ldes_capacity_error"] = (
        ldes_error["ldes_capacity_error_signed"]
        .abs()
    )


# Attach W_P to outcome table.
ldes_error = ldes_error.merge(
    proxy_weights,
    on="case_id",
    how="left",
    validate="one_to_one",
)

if ldes_error["proxy_weight"].isna().any():
    missing = ldes_error.loc[
        ldes_error["proxy_weight"].isna(),
        "case_id",
    ]

    raise ValueError(
        "No proxy weight found for some cases. "
        f"Examples: {missing.head(10).tolist()}"
    )


# =====================================================================
# Apply W_P configuration
# =====================================================================

if PROXY_WEIGHT is not None:

    weight_mask = np.isclose(
        ldes_error["proxy_weight"].astype(float),
        float(PROXY_WEIGHT),
    )

    ldes_error = (
        ldes_error.loc[
            weight_mask
        ]
        .copy()
    )

    print(
        f"\nFiltered to W_P = {PROXY_WEIGHT:g}"
    )

else:
    print(
        "\nUsing all W_P values."
    )


print(
    f"Outcome: {CAPACITY_ERROR} LDES capacity error"
)

print(
    f"Cases retained: "
    f"{ldes_error['case_id'].nunique():,}"
)

if ldes_error.empty:
    raise ValueError(
        "No cases remain after applying "
        "the proxy-weight filter."
    )


# =====================================================================
# Select diagnostic predictors
# =====================================================================

diagnostics = signal_metrics.loc[
    signal_metrics["error_family"]
    .isin(ERROR_FAMILIES)
    & signal_metrics["signal_type"]
    .isin(SIGNAL_TYPES)
    & signal_metrics["metric"]
    .isin(METRICS)
].copy()


# nMBE / nRMSE:
# normalised against the full-horizon reference-proxy range.
#
# Pearson:
# dimensionless; therefore no normalisation basis.
normalisation_mask = (
    (
        diagnostics["metric"]
        .isin(["nmbe", "nrmse"])
        & (
            diagnostics["normalisation_basis"]
            == "reference_proxy_full_range"
        )
    )
    |
    (
        (
            diagnostics["metric"]
            == "pearson"
        )
        & diagnostics["normalisation_basis"]
        .isna()
    )
)

diagnostics = diagnostics.loc[
    normalisation_mask
].copy()


# =====================================================================
# Join predictors and outcome
# =====================================================================

data = diagnostics.merge(
    ldes_error[
        [
            "case_id",
            "ldes_capacity_error",
            "ldes_capacity_error_signed",
            "proxy_weight",
        ]
    ],
    on="case_id",
    how="inner",
    validate="many_to_one",
)


print(
    f"Signal rows retained: {len(data):,}"
)

print(
    f"Unique cases in regression dataset: "
    f"{data['case_id'].nunique():,}"
)


# =====================================================================
# Fit each predictor independently
# =====================================================================

group_columns = [
    "error_family",
    "signal_type",
    "metric",
    "window_half_width_days",
]

rows = []

for group_values, group in data.groupby(
    group_columns,
    dropna=False,
    observed=True,
):

    (
        error_family,
        signal_type,
        metric,
        window_half_width_days,
    ) = group_values

    regression = fit_single_predictor(
        x=group["value"],
        y=group["ldes_capacity_error"],
    )

    rows.append(
        {
            "error_family": error_family,
            "signal_type": signal_type,
            "metric": metric,
            "window_half_width_days":
                window_half_width_days,
            **regression,
        }
    )


results = pd.DataFrame(rows)


# =====================================================================
# Add human-readable labels
# =====================================================================

results["family_label"] = [
    FAMILY_LABELS[
        (
            error_family,
            signal_type,
        )
    ]
    for error_family, signal_type
    in zip(
        results["error_family"],
        results["signal_type"],
    )
]

results["metric_label"] = (
    results["metric"]
    .map(METRIC_LABELS)
)

results["window_label"] = (
    results["window_half_width_days"]
    .map(window_label)
)


WINDOW_LABELS = [
    window_label(window)
    for window in WINDOWS
]

FAMILY_ORDER = [
    "Proxy Approx.",
    "Proxy TSA",
    "Proxy RPCC",
    "Proxy-Delta Approx.",
    "Proxy-Delta TSA",
    "Proxy-Delta RPCC",
]

METRIC_ORDER = [
    "nMBE",
    "nRMSE",
    "Pearson",
]


results["family_label"] = pd.Categorical(
    results["family_label"],
    categories=FAMILY_ORDER,
    ordered=True,
)

results["metric_label"] = pd.Categorical(
    results["metric_label"],
    categories=METRIC_ORDER,
    ordered=True,
)

results["window_label"] = pd.Categorical(
    results["window_label"],
    categories=WINDOW_LABELS,
    ordered=True,
)


results = (
    results
    .sort_values(
        [
            "family_label",
            "metric_label",
            "window_label",
        ]
    )
    .reset_index(drop=True)
)


# =====================================================================
# Save regression results
# =====================================================================

results.to_csv(
    RESULTS_TABLE_PATH,
    index=False,
)

print(
    f"\nSaved regression results to "
    f"{RESULTS_TABLE_PATH}"
)


# =====================================================================
# Construct heatmap matrix
# =====================================================================

heatmap = results.pivot(
    index=[
        "family_label",
        "metric_label",
    ],
    columns="window_label",
    values="adjusted_r2",
)


heatmap = heatmap.reindex(
    pd.MultiIndex.from_product(
        [
            FAMILY_ORDER,
            METRIC_ORDER,
        ],
        names=[
            "family_label",
            "metric_label",
        ],
    )
)


heatmap = heatmap.reindex(
    columns=WINDOW_LABELS
)


# =====================================================================
# Plot
# =====================================================================

fig, ax = plt.subplots(
    figsize=(11, 8),
)

values = heatmap.to_numpy(
    dtype=float
)

finite_values = values[
    np.isfinite(values)
]

if finite_values.size == 0:
    raise ValueError(
        "No finite adjusted R² values "
        "were produced."
    )


# Keep zero meaningful while retaining any
# negative adjusted R² values.
vmin = min(
    0.0,
    float(np.nanmin(finite_values)),
)

vmax = max(
    0.30,
    float(np.nanmax(finite_values)),
)


image = ax.imshow(
    values,
    aspect="auto",
    interpolation="nearest",
    vmin=vmin,
    vmax=vmax,
    cmap="plasma",
)


# ---------------------------------------------------------------------
# X axis
# ---------------------------------------------------------------------

ax.set_xticks(
    np.arange(
        len(WINDOW_LABELS)
    )
)

ax.set_xticklabels(
    WINDOW_LABELS,
    rotation=45,
    ha="right",
)

ax.set_xlabel(
    "Window half-width around reference-proxy peak"
)


# ---------------------------------------------------------------------
# Y axis
# ---------------------------------------------------------------------

row_labels = [
    f"{family}   {metric}"
    for family, metric
    in heatmap.index
]

ax.set_yticks(
    np.arange(
        len(row_labels)
    )
)

ax.set_yticklabels(
    row_labels
)


# ---------------------------------------------------------------------
# Family separators
# ---------------------------------------------------------------------

for boundary in range(
    len(METRIC_ORDER),
    len(row_labels),
    len(METRIC_ORDER),
):
    ax.axhline(
        boundary - 0.5,
        linewidth=1.0,
        color="white",
    )


# ---------------------------------------------------------------------
# Colour bar
# ---------------------------------------------------------------------

colourbar = fig.colorbar(
    image,
    ax=ax,
)

colourbar.set_label(
    "Single-predictor adjusted $R^2$"
)


# ---------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------

if PROXY_WEIGHT is None:
    weight_title = "all $W_P$"
else:
    weight_title = (
        f"$W_P$ = {PROXY_WEIGHT:g}"
    )

error_title = (
    "signed"
    if CAPACITY_ERROR == "signed"
    else "absolute"
)

ax.set_title(
    "Single-predictor explanatory power for "
    f"{error_title} LDES capacity error\n"
    f"({weight_title})"
)


# ---------------------------------------------------------------------
# Cell annotations
# ---------------------------------------------------------------------

for row_idx in range(
    values.shape[0]
):
    for col_idx in range(
        values.shape[1]
    ):

        value = values[
            row_idx,
            col_idx,
        ]

        if not np.isfinite(value):
            continue

        ax.text(
            col_idx,
            row_idx,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7,
        )


fig.tight_layout()

fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"Saved figure to {FIGURE_PATH}"
)

plt.show()


# =====================================================================
# Print strongest individual predictors
# =====================================================================

print(
    "\nTop 20 individual predictors "
    "by adjusted R²:\n"
)

top_predictors = (
    results
    .sort_values(
        "adjusted_r2",
        ascending=False,
    )
    .loc[
        :,
        [
            "family_label",
            "metric_label",
            "window_label",
            "n",
            "adjusted_r2",
            "r2",
            "coefficient",
        ],
    ]
    .head(20)
)


print(
    top_predictors.to_string(
        index=False,
    )
)