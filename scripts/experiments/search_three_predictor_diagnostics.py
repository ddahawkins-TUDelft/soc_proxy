"""Search three-predictor diagnostics for signed LDES capacity bias.

This is a MODEL-DEVELOPMENT / EXPLORATORY search.

Key principles
--------------
1. Predictor candidates must not use reference-CEM outcomes.
2. Signed LDES capacity error is used as the training target.
3. Cross-validation is blocked by weather/horizon window.
4. High-VIF models are RETAINED rather than discarded.
5. Strong high-VIF models are inspected for interpretable
   same-metric / different-window reparameterisations:
       global
       local - global
6. Final generalisation performance must be assessed separately after
   model structure has been selected.

Inputs
------
results/10_year/signal_metrics.parquet
results/10_year/investment_metrics.parquet
results/10_year/parameters.parquet
"""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut


# =====================================================================
# USER CONFIGURATION
# =====================================================================

# ---------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------

# "positive" -> lambda_soc > 0
# "all"      -> all values
# float      -> one specific value, e.g. 0.5
PROXY_WEIGHT = "positive"

PROXY_WEIGHT_FIELD = "lambda_soc"

# Both countries are used for model development here.
DEVELOPMENT_COUNTRIES = [
    "NL",
    "BE",
]

# Diagnostic target.
CAPACITY_ERROR = "signed"


# ---------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------

# Hold out the same weather/horizon window in both countries together.
#
# This gives five blocked temporal folds if there are five 10-year
# windows in the current experiment.
#
# Important:
# Because the underlying 10-year windows overlap in calendar years,
# this is NOT fully independent external validation. It is a blocked
# model-selection / stability assessment.
CV_GROUP_FIELDS = [
    "start_date",
    "end_date",
]


# ---------------------------------------------------------------------
# Allowed predictor space
# ---------------------------------------------------------------------

# These are intentionally restricted to quantities available without
# reference-CEM outcomes.
ALLOWED_ERROR_FAMILIES = [
    "clustered_approximation",
    "tsa",
    "rpcc",
]

ALLOWED_SIGNAL_TYPES = [
    "level",
    "delta",
]

ALLOWED_METRICS = [
    "nmbe",
    "nrmse",
    "pearson",
]

# Use the same deployment-available normalisation basis across all
# normalised predictor families.
NORMALISATION_BASIS = "reference_proxy_full_range"


# ---------------------------------------------------------------------
# Candidate reduction
# ---------------------------------------------------------------------

# Retain strong predictors overall...
TOP_N_OVERALL = 40

# ...but also preserve several windows within every conceptual metric
# series so that useful combinations / contrasts are not eliminated
# simply because a predictor is weak in isolation.
TOP_N_PER_SERIES = 3

# Always retain full-horizon variants.
KEEP_FULL_HORIZON = True

# Require predictors in a candidate model to be available for at least
# this fraction of development cases.
MIN_CASE_FRACTION = 0.95


# ---------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------

# This is NOT a raw-search exclusion threshold.
# It is used only to flag models.
VIF_THRESHOLD = 5.0


# ---------------------------------------------------------------------
# Contrast / reparameterisation search
# ---------------------------------------------------------------------

# Only investigate high-VIF models that are reasonably competitive
# with the best low-VIF raw model.
#
# Example:
# if best acceptable model CV R² = 0.80,
# inspect high-VIF models down to 0.70.
CONTRAST_MAX_CV_R2_GAP = 0.10

# Safety cap if there are very many competitive high-VIF models.
CONTRAST_SOURCE_TOP_N = 2000


# ---------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------

PROGRESS_EVERY = 5000
CONTRAST_PROGRESS_EVERY = 500


# ---------------------------------------------------------------------
# Saved summaries
# ---------------------------------------------------------------------

TOP_MODELS_TO_SAVE = 200


# =====================================================================
# Paths
# =====================================================================

RESULTS_DIR = Path("results") / "10_year"

SIGNAL_METRICS_PATH = (
    RESULTS_DIR
    / "signal_metrics.parquet"
)

INVESTMENT_METRICS_PATH = (
    RESULTS_DIR
    / "investment_metrics.parquet"
)

PARAMETERS_PATH = (
    RESULTS_DIR
    / "parameters.parquet"
)


SINGLE_RESULTS_PATH = (
    RESULTS_DIR
    / "diagnostic_search_single_predictors.csv"
)

RAW_ALL_PATH = (
    RESULTS_DIR
    / "diagnostic_search_three_predictor_raw_all.parquet"
)

RAW_TOP_LOW_VIF_PATH = (
    RESULTS_DIR
    / "diagnostic_search_three_predictor_raw_low_vif_top.csv"
)

RAW_TOP_HIGH_VIF_PATH = (
    RESULTS_DIR
    / "diagnostic_search_three_predictor_raw_high_vif_top.csv"
)

CONTRAST_ALL_PATH = (
    RESULTS_DIR
    / "diagnostic_search_contrast_models_all.parquet"
)

CONTRAST_TOP_PATH = (
    RESULTS_DIR
    / "diagnostic_search_contrast_models_top.csv"
)


# =====================================================================
# General helpers
# =====================================================================

def window_label(
    value: object,
) -> str:
    """Stable human/machine-readable temporal-window label."""

    if pd.isna(value):
        return "full"

    return f"{int(value)}d"


def window_scope(
    value: object,
) -> float:
    """Numeric scope for ordering local vs global windows."""

    if pd.isna(value):
        return np.inf

    return float(value)


def make_feature_name(
    row: pd.Series,
) -> str:
    """Create a unique predictor name."""

    basis = (
        "none"
        if pd.isna(
            row["normalisation_basis"]
        )
        else str(
            row["normalisation_basis"]
        )
    )

    return (
        f"{row['error_family']}"
        f"__{row['signal_type']}"
        f"__{row['metric']}"
        f"__{basis}"
        f"__{window_label(row['window_half_width_days'])}"
    )


def format_duration(
    seconds: float,
) -> str:
    """Pretty duration for progress messages."""

    if not np.isfinite(seconds):
        return "unknown"

    seconds = int(
        round(seconds)
    )

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, secs = divmod(
        remainder,
        60,
    )

    if hours:
        return (
            f"{hours:d}h "
            f"{minutes:02d}m "
            f"{secs:02d}s"
        )

    if minutes:
        return (
            f"{minutes:d}m "
            f"{secs:02d}s"
        )

    return f"{secs:d}s"


def report_progress(
    *,
    done: int,
    total: int,
    start_time: float,
    label: str,
) -> None:
    """Print progress, speed, elapsed time and estimated completion."""

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if (
        done <= 0
        or elapsed <= 0
    ):
        return

    rate = (
        done
        / elapsed
    )

    remaining_items = (
        total
        - done
    )

    eta_seconds = (
        remaining_items
        / rate
        if rate > 0
        else np.inf
    )

    finish_time = (
        datetime.now()
        + timedelta(
            seconds=eta_seconds
        )
        if np.isfinite(
            eta_seconds
        )
        else None
    )

    percentage = (
        100
        * done
        / total
    )

    finish_string = (
        finish_time.strftime(
            "%H:%M:%S"
        )
        if finish_time
        is not None
        else "unknown"
    )

    print(
        f"  {label}: "
        f"{done:,} / {total:,} "
        f"({percentage:5.1f}%)"
        f" | elapsed {format_duration(elapsed)}"
        f" | {rate:,.1f} models/s"
        f" | remaining {format_duration(eta_seconds)}"
        f" | finish ~{finish_string}"
    )


# =====================================================================
# OLS helpers
# =====================================================================

def fit_ols_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """OLS with an intercept."""

    X_train_design = np.column_stack(
        [
            np.ones(
                len(X_train)
            ),
            X_train,
        ]
    )

    beta = np.linalg.lstsq(
        X_train_design,
        y_train,
        rcond=None,
    )[0]

    X_test_design = np.column_stack(
        [
            np.ones(
                len(X_test)
            ),
            X_test,
        ]
    )

    return (
        X_test_design
        @ beta
    )


def fit_ols_full(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """Fit OLS on all supplied rows."""

    n = len(y)
    p = X.shape[1]

    X_design = np.column_stack(
        [
            np.ones(n),
            X,
        ]
    )

    beta = np.linalg.lstsq(
        X_design,
        y,
        rcond=None,
    )[0]

    y_hat = (
        X_design
        @ beta
    )

    r2 = r2_score(
        y,
        y_hat,
    )

    if n <= p + 1:
        adjusted_r2 = np.nan
    else:
        adjusted_r2 = (
            1
            - (
                1
                - r2
            )
            * (
                n - 1
            )
            / (
                n - p - 1
            )
        )

    return (
        beta,
        y_hat,
        r2,
        adjusted_r2,
    )


# =====================================================================
# Metadata
# =====================================================================

def parameters_to_wide(
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per case."""

    if (
        "case_id"
        not in parameters.columns
    ):
        raise ValueError(
            "parameters.parquet must contain case_id."
        )

    # Current results schema is already wide.
    if not parameters[
        "case_id"
    ].duplicated().any():
        return parameters.copy()

    # Fallback for long-format parameter tables.
    name_candidates = [
        "parameter",
        "parameter_name",
        "name",
        "key",
    ]

    name_column = next(
        (
            column
            for column
            in name_candidates
            if column
            in parameters.columns
        ),
        None,
    )

    if (
        name_column is None
        or "value"
        not in parameters.columns
    ):
        raise ValueError(
            "Could not convert parameters.parquet "
            "to one row per case.\n"
            f"Columns: {parameters.columns.tolist()}"
        )

    wide = (
        parameters[
            [
                "case_id",
                name_column,
                "value",
            ]
        ]
        .drop_duplicates()
        .pivot_table(
            index="case_id",
            columns=name_column,
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    wide.columns.name = None

    return wide


# =====================================================================
# Predictor preparation
# =====================================================================

def prepare_signal_metrics(
    signal_metrics: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build strictly reference-CEM-free predictor matrix."""

    diagnostics = (
        signal_metrics.loc[
            signal_metrics[
                "error_family"
            ].isin(
                ALLOWED_ERROR_FAMILIES
            )
            &
            signal_metrics[
                "signal_type"
            ].isin(
                ALLOWED_SIGNAL_TYPES
            )
            &
            signal_metrics[
                "metric"
            ].isin(
                ALLOWED_METRICS
            )
        ]
        .copy()
    )


    # =============================================================
    # Reference-CEM leakage guard
    # =============================================================

    forbidden_families = {
        "reference_approximation",
        "cem",
    }

    if diagnostics[
        "error_family"
    ].isin(
        forbidden_families
    ).any():

        raise RuntimeError(
            "Reference-CEM-dependent metric entered "
            "the diagnostic predictor pool."
        )


    # =============================================================
    # Normalisation
    # =============================================================

    normalised_mask = (
        diagnostics[
            "metric"
        ].isin(
            [
                "nmbe",
                "nrmse",
            ]
        )
        &
        (
            diagnostics[
                "normalisation_basis"
            ]
            == NORMALISATION_BASIS
        )
    )

    pearson_mask = (
        (
            diagnostics[
                "metric"
            ]
            == "pearson"
        )
        &
        diagnostics[
            "normalisation_basis"
        ].isna()
    )

    diagnostics = (
        diagnostics.loc[
            normalised_mask
            | pearson_mask
        ]
        .copy()
    )


    diagnostics[
        "feature"
    ] = diagnostics.apply(
        make_feature_name,
        axis=1,
    )


    duplicates = (
        diagnostics.duplicated(
            subset=[
                "case_id",
                "feature",
            ],
            keep=False,
        )
    )

    if duplicates.any():

        example = (
            diagnostics.loc[
                duplicates,
                [
                    "case_id",
                    "feature",
                ],
            ]
            .head(20)
        )

        raise ValueError(
            "Duplicate case/feature pairs found:\n"
            f"{example.to_string(index=False)}"
        )


    catalogue = (
        diagnostics[
            [
                "feature",
                "error_family",
                "signal_type",
                "metric",
                "normalisation_basis",
                "window_half_width_days",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


    X = diagnostics.pivot(
        index="case_id",
        columns="feature",
        values="value",
    )

    X.columns.name = None

    return (
        X,
        catalogue,
    )


# =====================================================================
# Cross-validation
# =====================================================================

def make_group_splits(
    groups: np.ndarray,
) -> list[
    tuple[
        np.ndarray,
        np.ndarray,
    ]
]:
    """Leave one weather/horizon group out."""

    unique_groups = np.unique(
        groups
    )

    if (
        len(unique_groups)
        < 3
    ):
        raise ValueError(
            "Need at least three independent CV groups."
        )

    splitter = (
        LeaveOneGroupOut()
    )

    dummy = np.zeros(
        len(groups)
    )

    return list(
        splitter.split(
            dummy,
            groups=groups,
        )
    )


def cross_validated_r2(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
) -> dict[str, float]:
    """Blocked out-of-fold R²."""

    predictions = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    fold_r2 = []


    for (
        train_idx,
        test_idx,
    ) in splits:

        X_train = X[
            train_idx
        ]

        y_train = y[
            train_idx
        ]

        X_test = X[
            test_idx
        ]

        y_test = y[
            test_idx
        ]


        train_valid = (
            np.isfinite(
                y_train
            )
            &
            np.all(
                np.isfinite(
                    X_train
                ),
                axis=1,
            )
        )

        test_valid = (
            np.isfinite(
                y_test
            )
            &
            np.all(
                np.isfinite(
                    X_test
                ),
                axis=1,
            )
        )


        if (
            train_valid.sum()
            <= X.shape[1] + 1
        ):
            continue

        if (
            test_valid.sum()
            < 2
        ):
            continue


        pred = fit_ols_predict(
            X_train[
                train_valid
            ],
            y_train[
                train_valid
            ],
            X_test[
                test_valid
            ],
        )


        test_rows = (
            test_idx[
                test_valid
            ]
        )

        predictions[
            test_rows
        ] = pred


        y_test_valid = (
            y_test[
                test_valid
            ]
        )

        if (
            np.std(
                y_test_valid
            )
            > 0
        ):
            fold_r2.append(
                r2_score(
                    y_test_valid,
                    pred,
                )
            )


    valid_oof = (
        np.isfinite(y)
        &
        np.isfinite(
            predictions
        )
    )


    if (
        valid_oof.sum()
        < 3
    ):
        oof_r2 = np.nan
    else:
        oof_r2 = r2_score(
            y[
                valid_oof
            ],
            predictions[
                valid_oof
            ],
        )


    return {
        "cv_r2_oof":
            float(oof_r2),
        "cv_r2_fold_mean":
            (
                float(
                    np.mean(
                        fold_r2
                    )
                )
                if fold_r2
                else np.nan
            ),
        "cv_r2_fold_std":
            (
                float(
                    np.std(
                        fold_r2
                    )
                )
                if fold_r2
                else np.nan
            ),
        "cv_n_predicted":
            int(
                valid_oof.sum()
            ),
    }


# =====================================================================
# VIF
# =====================================================================

def calculate_vif(
    X: np.ndarray,
) -> np.ndarray:
    """Calculate VIF for all supplied predictors."""

    finite = np.all(
        np.isfinite(X),
        axis=1,
    )

    X = X[
        finite
    ]

    p = X.shape[1]

    if (
        len(X) <= p
        or p < 2
    ):
        return np.full(
            p,
            np.nan,
        )


    std = np.std(
        X,
        axis=0,
        ddof=1,
    )

    if np.any(
        std == 0
    ):
        return np.full(
            p,
            np.inf,
        )


    Z = (
        X
        - np.mean(
            X,
            axis=0,
        )
    ) / std


    corr = np.corrcoef(
        Z,
        rowvar=False,
    )


    if not np.all(
        np.isfinite(
            corr
        )
    ):
        return np.full(
            p,
            np.inf,
        )


    try:
        inv_corr = (
            np.linalg.inv(
                corr
            )
        )

    except np.linalg.LinAlgError:

        return np.full(
            p,
            np.inf,
        )


    return np.diag(
        inv_corr
    )


# =====================================================================
# Feature metadata / contrast helpers
# =====================================================================

def feature_signature(
    row: pd.Series,
) -> tuple:
    """Conceptual metric identity, excluding temporal window."""

    basis = (
        None
        if pd.isna(
            row[
                "normalisation_basis"
            ]
        )
        else row[
            "normalisation_basis"
        ]
    )

    return (
        row[
            "error_family"
        ],
        row[
            "signal_type"
        ],
        row[
            "metric"
        ],
        basis,
    )


def identify_window_pair(
    feature_a: str,
    feature_b: str,
    catalogue_lookup: pd.DataFrame,
) -> (
    tuple[
        str,
        str,
    ]
    | None
):
    """Return (local_feature, global_feature) if same metric/different window."""

    row_a = (
        catalogue_lookup.loc[
            feature_a
        ]
    )

    row_b = (
        catalogue_lookup.loc[
            feature_b
        ]
    )


    if (
        feature_signature(
            row_a
        )
        !=
        feature_signature(
            row_b
        )
    ):
        return None


    window_a = (
        row_a[
            "window_half_width_days"
        ]
    )

    window_b = (
        row_b[
            "window_half_width_days"
        ]
    )


    # Same window -> no meaningful contrast.
    if (
        pd.isna(
            window_a
        )
        and pd.isna(
            window_b
        )
    ):
        return None

    if (
        not pd.isna(
            window_a
        )
        and not pd.isna(
            window_b
        )
        and float(
            window_a
        )
        == float(
            window_b
        )
    ):
        return None


    scope_a = window_scope(
        window_a
    )

    scope_b = window_scope(
        window_b
    )


    if (
        scope_a
        < scope_b
    ):
        return (
            feature_a,
            feature_b,
        )

    return (
        feature_b,
        feature_a,
    )


# =====================================================================
# Load
# =====================================================================

print(
    f"Reading "
    f"{SIGNAL_METRICS_PATH}"
)

signal_metrics = (
    pd.read_parquet(
        SIGNAL_METRICS_PATH
    )
)


print(
    f"Reading "
    f"{INVESTMENT_METRICS_PATH}"
)

investment_metrics = (
    pd.read_parquet(
        INVESTMENT_METRICS_PATH
    )
)


print(
    f"Reading "
    f"{PARAMETERS_PATH}"
)

parameters = (
    pd.read_parquet(
        PARAMETERS_PATH
    )
)


metadata = parameters_to_wide(
    parameters
)


# =====================================================================
# Validate metadata
# =====================================================================

required_metadata = (
    [
        "case_id",
        PROXY_WEIGHT_FIELD,
        "country",
    ]
    + CV_GROUP_FIELDS
)


missing_metadata = [
    column
    for column
    in required_metadata
    if column
    not in metadata.columns
]


if missing_metadata:

    raise ValueError(
        "Missing required metadata columns: "
        f"{missing_metadata}"
    )


metadata[
    "proxy_weight"
] = pd.to_numeric(
    metadata[
        PROXY_WEIGHT_FIELD
    ],
    errors="coerce",
)


metadata[
    "cv_group"
] = (
    metadata[
        CV_GROUP_FIELDS
    ]
    .astype(str)
    .agg(
        " | ".join,
        axis=1,
    )
)


print(
    "\nCV grouping fields:",
    CV_GROUP_FIELDS,
)

print(
    "Total weather/horizon groups:",
    metadata[
        "cv_group"
    ].nunique(),
)


# =====================================================================
# Target
# =====================================================================

target = (
    investment_metrics.loc[
        investment_metrics[
            "metric"
        ]
        == "ldes_capacity_error_signed",
        [
            "case_id",
            "value",
        ],
    ]
    .rename(
        columns={
            "value":
            "ldes_capacity_error_signed",
        }
    )
    .copy()
)


if target[
    "case_id"
].duplicated().any():

    raise ValueError(
        "Expected one signed LDES capacity error per case."
    )


if (
    CAPACITY_ERROR
    == "signed"
):

    target[
        "target"
    ] = (
        target[
            "ldes_capacity_error_signed"
        ]
    )

elif (
    CAPACITY_ERROR
    == "absolute"
):

    target[
        "target"
    ] = (
        target[
            "ldes_capacity_error_signed"
        ].abs()
    )

else:

    raise ValueError(
        "CAPACITY_ERROR must be "
        "'signed' or 'absolute'."
    )


# =====================================================================
# Predictors
# =====================================================================

X_wide, catalogue = (
    prepare_signal_metrics(
        signal_metrics
    )
)


catalogue_lookup = (
    catalogue
    .set_index(
        "feature"
    )
)


# =====================================================================
# Analysis table
# =====================================================================

analysis = (
    target
    .merge(
        metadata[
            [
                "case_id",
                "country",
                "proxy_weight",
                "cv_group",
            ]
        ],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    .set_index(
        "case_id"
    )
    .join(
        X_wide,
        how="inner",
    )
)


# ---------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------

if (
    DEVELOPMENT_COUNTRIES
    is not None
):

    analysis = (
        analysis.loc[
            analysis[
                "country"
            ].isin(
                DEVELOPMENT_COUNTRIES
            )
        ]
        .copy()
    )


# ---------------------------------------------------------------------
# W_P
# ---------------------------------------------------------------------

if (
    PROXY_WEIGHT
    == "positive"
):

    analysis = (
        analysis.loc[
            analysis[
                "proxy_weight"
            ]
            > 0
        ]
        .copy()
    )

elif (
    PROXY_WEIGHT
    == "all"
):

    pass

elif isinstance(
    PROXY_WEIGHT,
    (
        float,
        int,
    ),
):

    analysis = (
        analysis.loc[
            np.isclose(
                analysis[
                    "proxy_weight"
                ],
                float(
                    PROXY_WEIGHT
                ),
            )
        ]
        .copy()
    )

else:

    raise ValueError(
        "PROXY_WEIGHT must be "
        "'positive', 'all', or numeric."
    )


if analysis.empty:

    raise ValueError(
        "No cases remain after filtering."
    )


print(
    "\nCases retained:",
    len(
        analysis
    ),
)

print(
    "Countries:",
    sorted(
        analysis[
            "country"
        ].unique()
    ),
)

print(
    "Blocked CV groups:",
    analysis[
        "cv_group"
    ].nunique(),
)


print(
    "\nCases per blocked CV group:\n"
)

print(
    analysis[
        "cv_group"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# =====================================================================
# Common CV splits
# =====================================================================

y = (
    analysis[
        "target"
    ]
    .to_numpy(
        dtype=float
    )
)


groups = (
    analysis[
        "cv_group"
    ]
    .astype(str)
    .to_numpy()
)


splits = make_group_splits(
    groups
)


print(
    f"\nUsing "
    f"{len(splits)} "
    "leave-one-weather-window-out folds."
)


# =====================================================================
# 1. Single-predictor screening
# =====================================================================

print(
    "\nScoring individual predictors..."
)


single_rows = []


for feature in X_wide.columns:

    x = (
        analysis[
            [feature]
        ]
        .to_numpy(
            dtype=float
        )
    )


    valid = (
        np.isfinite(y)
        &
        np.isfinite(
            x[:, 0]
        )
    )


    if (
        valid.sum()
        >= 3
    ):

        beta, _, r2, adj_r2 = (
            fit_ols_full(
                x[
                    valid
                ],
                y[
                    valid
                ],
            )
        )

    else:

        beta = np.array(
            [
                np.nan,
                np.nan,
            ]
        )

        r2 = np.nan
        adj_r2 = np.nan


    cv = cross_validated_r2(
        x,
        y,
        splits,
    )


    single_rows.append(
        {
            "feature":
                feature,
            "n":
                int(
                    valid.sum()
                ),
            "r2":
                r2,
            "adjusted_r2":
                adj_r2,
            "intercept":
                beta[0],
            "coefficient":
                beta[1],
            **cv,
        }
    )


single = (
    pd.DataFrame(
        single_rows
    )
    .merge(
        catalogue,
        on="feature",
        how="left",
        validate="one_to_one",
    )
    .sort_values(
        "cv_r2_oof",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


single.to_csv(
    SINGLE_RESULTS_PATH,
    index=False,
)


print(
    "Saved single-predictor results to "
    f"{SINGLE_RESULTS_PATH}"
)


# =====================================================================
# 2. Diversity-aware predictor selection
# =====================================================================

selected_features: set[str] = set()


selected_features.update(
    single.head(
        TOP_N_OVERALL
    )[
        "feature"
    ]
)


series_group = [
    "error_family",
    "signal_type",
    "metric",
]


for _, subset in (
    single.groupby(
        series_group,
        dropna=False,
    )
):

    subset = (
        subset.sort_values(
            "cv_r2_oof",
            ascending=False,
        )
    )

    selected_features.update(
        subset.head(
            TOP_N_PER_SERIES
        )[
            "feature"
        ]
    )


if KEEP_FULL_HORIZON:

    selected_features.update(
        single.loc[
            single[
                "window_half_width_days"
            ].isna(),
            "feature",
        ]
    )


selected_features = sorted(
    selected_features
)


n_features = len(
    selected_features
)


n_combinations = (
    n_features
    * (
        n_features - 1
    )
    * (
        n_features - 2
    )
    // 6
)


print(
    "\nPredictors before screening:",
    len(
        single
    ),
)

print(
    "Predictors entering raw 3-way search:",
    n_features,
)

print(
    "Raw 3-predictor combinations:",
    f"{n_combinations:,}",
)


minimum_cases = int(
    np.ceil(
        MIN_CASE_FRACTION
        * len(
            analysis
        )
    )
)


# =====================================================================
# 3. RAW three-predictor search
# =====================================================================

print(
    "\nSearching raw 3-predictor models..."
)

print(
    "High-VIF models will be retained and flagged, "
    "not discarded."
)


raw_rows = []

raw_start = (
    time.perf_counter()
)


for i, feature_tuple in enumerate(
    combinations(
        selected_features,
        3,
    ),
    start=1,
):

    features = list(
        feature_tuple
    )


    X = (
        analysis[
            features
        ]
        .to_numpy(
            dtype=float
        )
    )


    valid = (
        np.isfinite(y)
        &
        np.all(
            np.isfinite(
                X
            ),
            axis=1,
        )
    )


    n = int(
        valid.sum()
    )


    if (
        n
        >= minimum_cases
    ):

        # ---------------------------------------------------------
        # VIF
        # ---------------------------------------------------------

        vif = calculate_vif(
            X[
                valid
            ]
        )


        max_vif = float(
            np.nanmax(
                vif
            )
        )


        passes_vif = bool(
            np.isfinite(
                max_vif
            )
            and max_vif
            <= VIF_THRESHOLD
        )


        # ---------------------------------------------------------
        # Full-data fit
        # ---------------------------------------------------------

        beta, _, r2, adj_r2 = (
            fit_ols_full(
                X[
                    valid
                ],
                y[
                    valid
                ],
            )
        )


        # ---------------------------------------------------------
        # Blocked CV
        # ---------------------------------------------------------

        cv = cross_validated_r2(
            X,
            y,
            splits,
        )


        raw_rows.append(
            {
                "feature_1":
                    features[0],
                "feature_2":
                    features[1],
                "feature_3":
                    features[2],

                "n":
                    n,

                "r2":
                    r2,
                "adjusted_r2":
                    adj_r2,

                **cv,

                "intercept":
                    beta[0],

                "beta_1":
                    beta[1],
                "beta_2":
                    beta[2],
                "beta_3":
                    beta[3],

                "vif_1":
                    vif[0],
                "vif_2":
                    vif[1],
                "vif_3":
                    vif[2],

                "max_vif":
                    max_vif,

                "passes_vif":
                    passes_vif,
            }
        )


    if (
        i % PROGRESS_EVERY
        == 0
        or i
        == n_combinations
    ):

        report_progress(
            done=i,
            total=n_combinations,
            start_time=raw_start,
            label="raw search",
        )


raw_models = pd.DataFrame(
    raw_rows
)


if raw_models.empty:

    raise ValueError(
        "No raw candidate models were fitted."
    )


raw_models = (
    raw_models.sort_values(
        [
            "cv_r2_oof",
            "max_vif",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


raw_models[
    "raw_rank"
] = (
    np.arange(
        len(
            raw_models
        )
    )
    + 1
)


raw_models.to_parquet(
    RAW_ALL_PATH,
    index=False,
)


# =====================================================================
# 4. Separate low- and high-VIF raw models
# =====================================================================

low_vif_models = (
    raw_models.loc[
        raw_models[
            "passes_vif"
        ]
    ]
    .copy()
)


high_vif_models = (
    raw_models.loc[
        ~raw_models[
            "passes_vif"
        ]
    ]
    .copy()
)


low_vif_models.head(
    TOP_MODELS_TO_SAVE
).to_csv(
    RAW_TOP_LOW_VIF_PATH,
    index=False,
)


high_vif_models.head(
    TOP_MODELS_TO_SAVE
).to_csv(
    RAW_TOP_HIGH_VIF_PATH,
    index=False,
)


print(
    "\nRaw search complete."
)

print(
    f"  total models: "
    f"{len(raw_models):,}"
)

print(
    f"  VIF <= {VIF_THRESHOLD:g}: "
    f"{len(low_vif_models):,}"
)

print(
    f"  VIF > {VIF_THRESHOLD:g}: "
    f"{len(high_vif_models):,}"
)


if not low_vif_models.empty:

    print(
        "\nBest low-VIF raw model:"
    )

    print(
        low_vif_models[
            [
                "feature_1",
                "feature_2",
                "feature_3",
                "cv_r2_oof",
                "cv_r2_fold_mean",
                "cv_r2_fold_std",
                "adjusted_r2",
                "max_vif",
            ]
        ]
        .head(1)
        .to_string(
            index=False
        )
    )


if not high_vif_models.empty:

    print(
        "\nBest high-VIF raw model:"
    )

    print(
        high_vif_models[
            [
                "feature_1",
                "feature_2",
                "feature_3",
                "cv_r2_oof",
                "cv_r2_fold_mean",
                "cv_r2_fold_std",
                "adjusted_r2",
                "max_vif",
            ]
        ]
        .head(1)
        .to_string(
            index=False
        )
    )


# =====================================================================
# 5. Identify competitive high-VIF models
# =====================================================================

if low_vif_models.empty:

    best_low_vif_r2 = (
        raw_models[
            "cv_r2_oof"
        ].max()
    )

else:

    best_low_vif_r2 = (
        low_vif_models[
            "cv_r2_oof"
        ].max()
    )


contrast_threshold = (
    best_low_vif_r2
    - CONTRAST_MAX_CV_R2_GAP
)


contrast_sources = (
    high_vif_models.loc[
        high_vif_models[
            "cv_r2_oof"
        ]
        >= contrast_threshold
    ]
    .head(
        CONTRAST_SOURCE_TOP_N
    )
    .copy()
)


print(
    "\nHigh-VIF contrast search:"
)

print(
    f"  best low-VIF CV R²: "
    f"{best_low_vif_r2:.3f}"
)

print(
    f"  inspect high-VIF models with CV R² >= "
    f"{contrast_threshold:.3f}"
)

print(
    f"  source models retained: "
    f"{len(contrast_sources):,}"
)


# =====================================================================
# 6. Generate interpretable window-contrast candidates
# =====================================================================

contrast_specs = {}

for _, raw_model in (
    contrast_sources.iterrows()
):

    features = [
        raw_model[
            "feature_1"
        ],
        raw_model[
            "feature_2"
        ],
        raw_model[
            "feature_3"
        ],
    ]


    for (
        feature_a,
        feature_b,
    ) in combinations(
        features,
        2,
    ):

        pair = identify_window_pair(
            feature_a,
            feature_b,
            catalogue_lookup,
        )


        if pair is None:
            continue


        (
            local_feature,
            global_feature,
        ) = pair


        third_feature = next(
            feature
            for feature
            in features
            if feature
            not in {
                local_feature,
                global_feature,
            }
        )


        key = (
            local_feature,
            global_feature,
            third_feature,
        )


        # Keep the strongest raw source if the same transformed
        # specification is generated more than once.
        previous = contrast_specs.get(
            key
        )


        candidate_info = {
            "local_feature":
                local_feature,
            "global_feature":
                global_feature,
            "third_feature":
                third_feature,

            "source_raw_rank":
                int(
                    raw_model[
                        "raw_rank"
                    ]
                ),

            "source_raw_cv_r2":
                float(
                    raw_model[
                        "cv_r2_oof"
                    ]
                ),

            "source_raw_max_vif":
                float(
                    raw_model[
                        "max_vif"
                    ]
                ),
        }


        if (
            previous is None
            or candidate_info[
                "source_raw_cv_r2"
            ]
            >
            previous[
                "source_raw_cv_r2"
            ]
        ):

            contrast_specs[
                key
            ] = candidate_info


contrast_specs = list(
    contrast_specs.values()
)


print(
    "  unique interpretable contrast specifications:",
    f"{len(contrast_specs):,}",
)


# =====================================================================
# 7. Evaluate contrast reparameterisations
# =====================================================================

contrast_rows = []

contrast_start = (
    time.perf_counter()
)

total_contrasts = len(
    contrast_specs
)


for i, spec in enumerate(
    contrast_specs,
    start=1,
):

    local_feature = (
        spec[
            "local_feature"
        ]
    )

    global_feature = (
        spec[
            "global_feature"
        ]
    )

    third_feature = (
        spec[
            "third_feature"
        ]
    )


    local_values = (
        analysis[
            local_feature
        ].to_numpy(
            dtype=float
        )
    )

    global_values = (
        analysis[
            global_feature
        ].to_numpy(
            dtype=float
        )
    )

    third_values = (
        analysis[
            third_feature
        ].to_numpy(
            dtype=float
        )
    )


    contrast_values = (
        local_values
        - global_values
    )


    # Reparameterised model:
    #
    #   global
    #   local - global
    #   third
    #
    # This spans exactly the same information as:
    #
    #   local
    #   global
    #   third

    X = np.column_stack(
        [
            global_values,
            contrast_values,
            third_values,
        ]
    )


    valid = (
        np.isfinite(y)
        &
        np.all(
            np.isfinite(
                X
            ),
            axis=1,
        )
    )


    n = int(
        valid.sum()
    )


    if (
        n
        >= minimum_cases
    ):

        vif = calculate_vif(
            X[
                valid
            ]
        )


        max_vif = float(
            np.nanmax(
                vif
            )
        )


        passes_vif = bool(
            np.isfinite(
                max_vif
            )
            and max_vif
            <= VIF_THRESHOLD
        )


        beta, _, r2, adj_r2 = (
            fit_ols_full(
                X[
                    valid
                ],
                y[
                    valid
                ],
            )
        )


        cv = cross_validated_r2(
            X,
            y,
            splits,
        )


        contrast_name = (
            f"({local_feature})"
            f"_minus_"
            f"({global_feature})"
        )


        contrast_rows.append(
            {
                "global_feature":
                    global_feature,

                "contrast_feature":
                    contrast_name,

                "local_source_feature":
                    local_feature,

                "third_feature":
                    third_feature,

                "n":
                    n,

                "r2":
                    r2,

                "adjusted_r2":
                    adj_r2,

                **cv,

                "intercept":
                    beta[0],

                "beta_global":
                    beta[1],

                "beta_contrast":
                    beta[2],

                "beta_third":
                    beta[3],

                "vif_global":
                    vif[0],

                "vif_contrast":
                    vif[1],

                "vif_third":
                    vif[2],

                "max_vif":
                    max_vif,

                "passes_vif":
                    passes_vif,

                "source_raw_rank":
                    spec[
                        "source_raw_rank"
                    ],

                "source_raw_cv_r2":
                    spec[
                        "source_raw_cv_r2"
                    ],

                "source_raw_max_vif":
                    spec[
                        "source_raw_max_vif"
                    ],

                "vif_reduction":
                    (
                        spec[
                            "source_raw_max_vif"
                        ]
                        - max_vif
                    ),
            }
        )


    if (
        total_contrasts
        > 0
        and (
            i
            % CONTRAST_PROGRESS_EVERY
            == 0
            or i
            == total_contrasts
        )
    ):

        report_progress(
            done=i,
            total=total_contrasts,
            start_time=contrast_start,
            label="contrast search",
        )


# =====================================================================
# 8. Save transformed models
# =====================================================================

if contrast_rows:

    contrast_models = (
        pd.DataFrame(
            contrast_rows
        )
        .sort_values(
            [
                "cv_r2_oof",
                "max_vif",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    contrast_models.to_parquet(
        CONTRAST_ALL_PATH,
        index=False,
    )


    contrast_models.head(
        TOP_MODELS_TO_SAVE
    ).to_csv(
        CONTRAST_TOP_PATH,
        index=False,
    )


    print(
        "\nContrast search complete."
    )

    print(
        f"  transformed models: "
        f"{len(contrast_models):,}"
    )

    print(
        f"  transformed models with "
        f"VIF <= {VIF_THRESHOLD:g}: "
        f"{contrast_models['passes_vif'].sum():,}"
    )


    print(
        "\nTop 20 transformed candidates:\n"
    )

    print(
        contrast_models[
            [
                "global_feature",
                "contrast_feature",
                "third_feature",
                "cv_r2_oof",
                "cv_r2_fold_mean",
                "cv_r2_fold_std",
                "adjusted_r2",
                "source_raw_max_vif",
                "max_vif",
                "vif_reduction",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


else:

    contrast_models = (
        pd.DataFrame()
    )

    print(
        "\nNo interpretable same-metric / "
        "different-window high-VIF contrasts were found."
    )


# =====================================================================
# 9. Raw summaries
# =====================================================================

print(
    "\nTop 20 LOW-VIF raw candidates:\n"
)

print(
    low_vif_models[
        [
            "feature_1",
            "feature_2",
            "feature_3",
            "cv_r2_oof",
            "cv_r2_fold_mean",
            "cv_r2_fold_std",
            "adjusted_r2",
            "max_vif",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nTop 20 HIGH-VIF raw candidates:\n"
)

print(
    high_vif_models[
        [
            "feature_1",
            "feature_2",
            "feature_3",
            "cv_r2_oof",
            "cv_r2_fold_mean",
            "cv_r2_fold_std",
            "adjusted_r2",
            "max_vif",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# =====================================================================
# 10. Predictor frequency among strong low-VIF models
# =====================================================================

top_frequency_models = (
    low_vif_models.head(
        min(
            100,
            len(
                low_vif_models
            ),
        )
    )
)


if not top_frequency_models.empty:

    frequency = (
        pd.concat(
            [
                top_frequency_models[
                    "feature_1"
                ],
                top_frequency_models[
                    "feature_2"
                ],
                top_frequency_models[
                    "feature_3"
                ],
            ]
        )
        .value_counts()
    )


    print(
        "\nPredictor frequency among top "
        f"{len(top_frequency_models)} "
        "low-VIF raw models:\n"
    )


    print(
        frequency.head(
            30
        ).to_string()
    )


# =====================================================================
# Final paths
# =====================================================================

print(
    "\nSaved outputs:"
)

print(
    f"  {SINGLE_RESULTS_PATH}"
)

print(
    f"  {RAW_ALL_PATH}"
)

print(
    f"  {RAW_TOP_LOW_VIF_PATH}"
)

print(
    f"  {RAW_TOP_HIGH_VIF_PATH}"
)

if contrast_rows:

    print(
        f"  {CONTRAST_ALL_PATH}"
    )

    print(
        f"  {CONTRAST_TOP_PATH}"
    )