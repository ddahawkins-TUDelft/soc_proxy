"""Develop, validate, and refit the final 3-predictor LDES diagnostic.

Workflow
--------
1. DEVELOPMENT:
   Use NL cases only.
   - Screen individual reference-CEM-free predictors.
   - Evaluate 3-predictor linear models.
   - Use leave-one-weather-window-out cross-validation.
   - Select the model with the highest predictive CV R² subject to VIF <= 5.

2. VALIDATION:
   Fit the selected specification to ALL NL cases.
   Apply those coefficients unchanged to ALL BE cases.
   Report predictive R², RMSE, MAE, and mean error.

3. FINAL REFIT:
   Keep the predictor specification fixed.
   Refit only the coefficients using the combined NL + BE dataset.
   These are the coefficients of the final published diagnostic.

Reference-CEM outcomes are NEVER used as predictors.
They are used only to define the true signed LDES capacity-error target
during development and validation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import combinations
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneGroupOut


# =====================================================================
# USER CONFIGURATION
# =====================================================================

DEVELOPMENT_COUNTRY = "NL"
VALIDATION_COUNTRY = "BE"

PROXY_WEIGHT_FIELD = "lambda_soc"

# "positive" -> lambda_soc > 0
# "all"      -> all lambda_soc values
# float      -> specific value, e.g. 0.5
PROXY_WEIGHT = "positive"

CAPACITY_ERROR = "signed"


# ---------------------------------------------------------------------
# Weather-window blocking used ONLY within NL development
# ---------------------------------------------------------------------

CV_GROUP_FIELDS = [
    "start_date",
    "end_date",
]


# ---------------------------------------------------------------------
# Reference-CEM-free diagnostic predictors
# ---------------------------------------------------------------------

ALLOWED_ERROR_FAMILIES = [
    # "clustered_approximation",
    # "tsa",
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

# Common deployment-available normalisation basis.
NORMALISATION_BASIS = "reference_proxy_full_range"


# ---------------------------------------------------------------------
# Candidate reduction
# ---------------------------------------------------------------------

TOP_N_OVERALL = 40

# Keep several windows from every conceptual metric series so that
# potentially complementary predictors are not screened out simply
# because they are weak individually.
TOP_N_PER_SERIES = 3

KEEP_FULL_HORIZON = True

MIN_CASE_FRACTION = 0.95


# ---------------------------------------------------------------------
# Multicollinearity
# ---------------------------------------------------------------------

MAX_VIF = 5.0


# ---------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------

PROGRESS_EVERY = 5000


# ---------------------------------------------------------------------
# Saved search results
# ---------------------------------------------------------------------

TOP_MODELS_TO_SAVE = 200


# =====================================================================
# Paths
# =====================================================================

RESULTS_DIR = Path("results") / "10_year"

SIGNAL_METRICS_PATH = RESULTS_DIR / "signal_metrics.parquet"

INVESTMENT_METRICS_PATH = RESULTS_DIR / "investment_metrics.parquet"

PARAMETERS_PATH = RESULTS_DIR / "parameters.parquet"


SINGLE_RESULTS_PATH = RESULTS_DIR / "diagnostic_nl_single_predictors.csv"

SEARCH_ALL_PATH = RESULTS_DIR / "diagnostic_nl_three_predictor_all.parquet"

SEARCH_TOP_PATH = RESULTS_DIR / "diagnostic_nl_three_predictor_top.csv"

SELECTED_MODEL_PATH = RESULTS_DIR / "diagnostic_selected_model.json"

VALIDATION_PREDICTIONS_PATH = (
    RESULTS_DIR / "diagnostic_nl_to_be_validation_predictions.csv"
)

VALIDATION_SUMMARY_PATH = RESULTS_DIR / "diagnostic_nl_to_be_validation_summary.csv"

VALIDATION_FIGURE_PATH = RESULTS_DIR / "diagnostic_nl_to_be_validation.png"

FINAL_COEFFICIENTS_PATH = RESULTS_DIR / "diagnostic_final_pooled_coefficients.csv"


# =====================================================================
# General helpers
# =====================================================================


def window_label(value: object) -> str:
    """Readable label for one temporal window."""

    if pd.isna(value):
        return "full"

    return f"{int(value)}d"


def make_feature_name(row: pd.Series) -> str:
    """Unique machine-readable predictor identifier."""

    basis = (
        "none"
        if pd.isna(row["normalisation_basis"])
        else str(row["normalisation_basis"])
    )

    return (
        f"{row['error_family']}"
        f"__{row['signal_type']}"
        f"__{row['metric']}"
        f"__{basis}"
        f"__{window_label(row['window_half_width_days'])}"
    )


def format_duration(seconds: float) -> str:

    if not np.isfinite(seconds):
        return "unknown"

    seconds = int(round(seconds))

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, secs = divmod(
        remainder,
        60,
    )

    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"

    if minutes:
        return f"{minutes:d}m {secs:02d}s"

    return f"{secs:d}s"


def report_progress(
    *,
    done: int,
    total: int,
    start_time: float,
    label: str,
) -> None:

    elapsed = time.perf_counter() - start_time

    if done <= 0 or elapsed <= 0:
        return

    rate = done / elapsed

    remaining = total - done

    eta_seconds = remaining / rate if rate > 0 else np.inf

    if np.isfinite(eta_seconds):
        finish_time = datetime.now() + timedelta(seconds=eta_seconds)

        finish_string = finish_time.strftime("%H:%M:%S")

    else:
        finish_string = "unknown"

    percentage = 100 * done / total

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
# Linear regression
# =====================================================================


def fit_ols(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Fit OLS with intercept and return coefficients."""

    X_design = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    return np.linalg.lstsq(
        X_design,
        y,
        rcond=None,
    )[0]


def predict_ols(
    X: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:

    X_design = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    return X_design @ beta


def fit_ols_full(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
]:

    n = len(y)
    p = X.shape[1]

    beta = fit_ols(
        X,
        y,
    )

    y_hat = predict_ols(
        X,
        beta,
    )

    r2 = r2_score(
        y,
        y_hat,
    )

    if n <= p + 1:
        adjusted_r2 = np.nan

    else:
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

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
    """Ensure one metadata row per case."""

    if "case_id" not in parameters.columns:
        raise ValueError("parameters.parquet must contain case_id.")

    if not parameters["case_id"].duplicated().any():
        return parameters.copy()

    name_candidates = [
        "parameter",
        "parameter_name",
        "name",
        "key",
    ]

    name_column = next(
        (column for column in name_candidates if column in parameters.columns),
        None,
    )

    if name_column is None or "value" not in parameters.columns:
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
# Predictor construction
# =====================================================================


def prepare_signal_metrics(
    signal_metrics: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Construct the permitted diagnostic predictor matrix."""

    diagnostics = signal_metrics.loc[
        signal_metrics["error_family"].isin(ALLOWED_ERROR_FAMILIES)
        & signal_metrics["signal_type"].isin(ALLOWED_SIGNAL_TYPES)
        & signal_metrics["metric"].isin(ALLOWED_METRICS)
    ].copy()

    # =============================================================
    # HARD reference-CEM leakage check
    # =============================================================

    forbidden_families = {
        "reference_approximation",
        "cem",
    }

    if diagnostics["error_family"].isin(forbidden_families).any():
        raise RuntimeError(
            "Reference-CEM-dependent predictors entered the diagnostic feature pool."
        )

    # =============================================================
    # Normalisation
    # =============================================================

    normalised_mask = diagnostics["metric"].isin(
        [
            "nmbe",
            "nrmse",
        ]
    ) & (diagnostics["normalisation_basis"] == NORMALISATION_BASIS)

    pearson_mask = (diagnostics["metric"] == "pearson") & diagnostics[
        "normalisation_basis"
    ].isna()

    diagnostics = diagnostics.loc[normalised_mask | pearson_mask].copy()

    diagnostics["feature"] = diagnostics.apply(
        make_feature_name,
        axis=1,
    )

    duplicates = diagnostics.duplicated(
        subset=[
            "case_id",
            "feature",
        ],
        keep=False,
    )

    if duplicates.any():
        raise ValueError("Duplicate case_id / feature pairs detected.")

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
        .reset_index(drop=True)
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
# VIF
# =====================================================================


def calculate_vif(
    X: np.ndarray,
) -> np.ndarray:

    finite = np.all(
        np.isfinite(X),
        axis=1,
    )

    X = X[finite]

    p = X.shape[1]

    if len(X) <= p or p < 2:
        return np.full(
            p,
            np.nan,
        )

    std = np.std(
        X,
        axis=0,
        ddof=1,
    )

    if np.any(std == 0):
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

    if not np.all(np.isfinite(corr)):
        return np.full(
            p,
            np.inf,
        )

    try:
        inv_corr = np.linalg.inv(corr)

    except np.linalg.LinAlgError:
        return np.full(
            p,
            np.inf,
        )

    return np.diag(inv_corr)


# =====================================================================
# NL blocked cross-validation
# =====================================================================


def make_group_splits(
    groups: np.ndarray,
) -> list[
    tuple[
        np.ndarray,
        np.ndarray,
    ]
]:

    splitter = LeaveOneGroupOut()

    dummy = np.zeros(len(groups))

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
    """Generate truly out-of-fold predictions within NL."""

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
        X_train = X[train_idx]

        y_train = y[train_idx]

        X_test = X[test_idx]

        y_test = y[test_idx]

        train_valid = np.isfinite(y_train) & np.all(
            np.isfinite(X_train),
            axis=1,
        )

        test_valid = np.isfinite(y_test) & np.all(
            np.isfinite(X_test),
            axis=1,
        )

        if train_valid.sum() <= X.shape[1] + 1:
            continue

        if test_valid.sum() < 2:
            continue

        beta = fit_ols(
            X_train[train_valid],
            y_train[train_valid],
        )

        pred = predict_ols(
            X_test[test_valid],
            beta,
        )

        prediction_rows = test_idx[test_valid]

        predictions[prediction_rows] = pred

        if np.std(y_test[test_valid]) > 0:
            fold_r2.append(
                r2_score(
                    y_test[test_valid],
                    pred,
                )
            )

    valid = np.isfinite(y) & np.isfinite(predictions)

    if valid.sum() < 3:
        pooled_r2 = np.nan

    else:
        pooled_r2 = r2_score(
            y[valid],
            predictions[valid],
        )

    return {
        "cv_r2_oof": float(pooled_r2),
        "cv_r2_fold_mean": (float(np.mean(fold_r2)) if fold_r2 else np.nan),
        "cv_r2_fold_std": (float(np.std(fold_r2)) if fold_r2 else np.nan),
        "cv_n_predicted": int(valid.sum()),
    }


# =====================================================================
# Prediction metrics
# =====================================================================


def prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[valid]

    y_pred = y_pred[valid]

    residual = y_pred - y_true

    return {
        "n": int(len(y_true)),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        # Positive = diagnostic over-predicts capacity error on average.
        "mean_prediction_error": float(np.mean(residual)),
    }


# =====================================================================
# Load data
# =====================================================================

print(f"Reading {SIGNAL_METRICS_PATH}")

signal_metrics = pd.read_parquet(SIGNAL_METRICS_PATH)


print(f"Reading {INVESTMENT_METRICS_PATH}")

investment_metrics = pd.read_parquet(INVESTMENT_METRICS_PATH)


print(f"Reading {PARAMETERS_PATH}")

parameters = pd.read_parquet(PARAMETERS_PATH)


metadata = parameters_to_wide(parameters)


# =====================================================================
# Metadata validation
# =====================================================================

required_columns = [
    "case_id",
    "country",
    PROXY_WEIGHT_FIELD,
    *CV_GROUP_FIELDS,
]


missing = [column for column in required_columns if column not in metadata.columns]


if missing:
    raise ValueError(f"Missing metadata columns: {missing}")


metadata["proxy_weight"] = pd.to_numeric(
    metadata[PROXY_WEIGHT_FIELD],
    errors="coerce",
)


metadata["cv_group"] = (
    metadata[CV_GROUP_FIELDS]
    .astype(str)
    .agg(
        " | ".join,
        axis=1,
    )
)


# =====================================================================
# Target
# =====================================================================

target = (
    investment_metrics.loc[
        investment_metrics["metric"] == "ldes_capacity_error_signed",
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


if target["case_id"].duplicated().any():
    raise ValueError("Expected one LDES capacity-error value per case.")


if CAPACITY_ERROR == "signed":
    target["target"] = target["ldes_capacity_error_signed"]

elif CAPACITY_ERROR == "absolute":
    target["target"] = target["ldes_capacity_error_signed"].abs()

else:
    raise ValueError("CAPACITY_ERROR must be 'signed' or 'absolute'.")


# =====================================================================
# Predictor matrix
# =====================================================================

X_wide, catalogue = prepare_signal_metrics(signal_metrics)


# =====================================================================
# Master analysis table
# =====================================================================

analysis = (
    target.merge(
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
    .set_index("case_id")
    .join(
        X_wide,
        how="inner",
    )
)


# ---------------------------------------------------------------------
# W_P filter applied equally to NL and BE
# ---------------------------------------------------------------------

if PROXY_WEIGHT == "positive":
    analysis = analysis.loc[analysis["proxy_weight"] > 0].copy()

elif PROXY_WEIGHT == "all":
    pass

elif isinstance(
    PROXY_WEIGHT,
    (
        float,
        int,
    ),
):
    analysis = analysis.loc[
        np.isclose(
            analysis["proxy_weight"],
            float(PROXY_WEIGHT),
        )
    ].copy()

else:
    raise ValueError("PROXY_WEIGHT must be 'positive', 'all', or numeric.")


development = analysis.loc[analysis["country"] == DEVELOPMENT_COUNTRY].copy()


validation = analysis.loc[analysis["country"] == VALIDATION_COUNTRY].copy()


if development.empty:
    raise ValueError(f"No {DEVELOPMENT_COUNTRY} development cases found.")


if validation.empty:
    raise ValueError(f"No {VALIDATION_COUNTRY} validation cases found.")


print(
    "\nDevelopment country:",
    DEVELOPMENT_COUNTRY,
)

print(
    "Development cases:",
    len(development),
)

print(
    "Development weather-window groups:",
    development["cv_group"].nunique(),
)


print(
    "\nValidation country:",
    VALIDATION_COUNTRY,
)

print(
    "Validation cases:",
    len(validation),
)


print("\nDevelopment cases per CV group:\n")

print(development["cv_group"].value_counts().sort_index().to_string())


# =====================================================================
# NL cross-validation setup
# =====================================================================

y_dev = development["target"].to_numpy(dtype=float)


groups_dev = development["cv_group"].astype(str).to_numpy()


splits = make_group_splits(groups_dev)


print(f"\nUsing {len(splits)} leave-one-weather-window-out development folds.")


# =====================================================================
# 1. Single-predictor screening — NL ONLY
# =====================================================================

print("\nScoring individual predictors on NL only...")


single_rows = []


for feature in X_wide.columns:
    x = development[[feature]].to_numpy(dtype=float)

    valid = np.isfinite(y_dev) & np.isfinite(x[:, 0])

    if valid.sum() >= 3:
        beta, _, r2, adjusted_r2 = fit_ols_full(
            x[valid],
            y_dev[valid],
        )

    else:
        beta = np.array(
            [
                np.nan,
                np.nan,
            ]
        )

        r2 = np.nan
        adjusted_r2 = np.nan

    cv = cross_validated_r2(
        x,
        y_dev,
        splits,
    )

    single_rows.append(
        {
            "feature": feature,
            "n": int(valid.sum()),
            "r2": r2,
            "adjusted_r2": adjusted_r2,
            "intercept": beta[0],
            "coefficient": beta[1],
            **cv,
        }
    )


single = (
    pd.DataFrame(single_rows)
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
    .reset_index(drop=True)
)


single.to_csv(
    SINGLE_RESULTS_PATH,
    index=False,
)


print(f"Saved NL single-predictor results to {SINGLE_RESULTS_PATH}")


# =====================================================================
# 2. Diversity-aware candidate reduction — NL ONLY
# =====================================================================

selected_features: set[str] = set()


selected_features.update(single.head(TOP_N_OVERALL)["feature"])


series_group = [
    "error_family",
    "signal_type",
    "metric",
]


for _, subset in single.groupby(
    series_group,
    dropna=False,
):
    subset = subset.sort_values(
        "cv_r2_oof",
        ascending=False,
    )

    selected_features.update(subset.head(TOP_N_PER_SERIES)["feature"])


if KEEP_FULL_HORIZON:
    selected_features.update(
        single.loc[
            single["window_half_width_days"].isna(),
            "feature",
        ]
    )


selected_features = sorted(selected_features)


n_features = len(selected_features)


n_combinations = n_features * (n_features - 1) * (n_features - 2) // 6


print(
    "\nPredictors before screening:",
    len(single),
)

print(
    "Predictors entering NL 3-way search:",
    n_features,
)

print(
    "Three-predictor combinations:",
    f"{n_combinations:,}",
)


minimum_cases = int(np.ceil(MIN_CASE_FRACTION * len(development)))


# =====================================================================
# 3. Three-predictor search — NL ONLY
# =====================================================================

print("\nSearching NL three-predictor models...")


search_rows = []

search_start = time.perf_counter()


for i, feature_tuple in enumerate(
    combinations(
        selected_features,
        3,
    ),
    start=1,
):
    features = list(feature_tuple)

    X = development[features].to_numpy(dtype=float)

    valid = np.isfinite(y_dev) & np.all(
        np.isfinite(X),
        axis=1,
    )

    n = int(valid.sum())

    if n >= minimum_cases:
        vif = calculate_vif(X[valid])

        max_vif = float(np.nanmax(vif))

        passes_vif = bool(np.isfinite(max_vif) and max_vif <= MAX_VIF)

        beta, _, r2, adjusted_r2 = fit_ols_full(
            X[valid],
            y_dev[valid],
        )

        cv = cross_validated_r2(
            X,
            y_dev,
            splits,
        )

        search_rows.append(
            {
                "feature_1": features[0],
                "feature_2": features[1],
                "feature_3": features[2],
                "n": n,
                "r2": r2,
                "adjusted_r2": adjusted_r2,
                **cv,
                "intercept": beta[0],
                "beta_1": beta[1],
                "beta_2": beta[2],
                "beta_3": beta[3],
                "vif_1": vif[0],
                "vif_2": vif[1],
                "vif_3": vif[2],
                "max_vif": max_vif,
                "passes_vif": passes_vif,
            }
        )

    if i % PROGRESS_EVERY == 0 or i == n_combinations:
        report_progress(
            done=i,
            total=n_combinations,
            start_time=search_start,
            label="NL search",
        )


models = pd.DataFrame(search_rows)


if models.empty:
    raise ValueError("No NL three-predictor models were fitted.")


models = models.sort_values(
    [
        "cv_r2_oof",
        "max_vif",
    ],
    ascending=[
        False,
        True,
    ],
).reset_index(drop=True)


models.to_parquet(
    SEARCH_ALL_PATH,
    index=False,
)


acceptable_models = models.loc[models["passes_vif"]].copy()


if acceptable_models.empty:
    raise ValueError(f"No model satisfied VIF <= {MAX_VIF}.")


acceptable_models.head(TOP_MODELS_TO_SAVE).to_csv(
    SEARCH_TOP_PATH,
    index=False,
)


# =====================================================================
# 4. Select the diagnostic specification
# =====================================================================

winner = acceptable_models.iloc[0]


selected_features = [
    winner["feature_1"],
    winner["feature_2"],
    winner["feature_3"],
]


print(
    "\n"
    "============================================================\n"
    "SELECTED NL DIAGNOSTIC SPECIFICATION\n"
    "============================================================"
)


for j, feature in enumerate(
    selected_features,
    start=1,
):
    print(f"x{j}: {feature}")


print(f"\nNL blocked-CV R²: {winner['cv_r2_oof']:.4f}")

print(f"NL mean fold R²: {winner['cv_r2_fold_mean']:.4f}")

print(f"NL fold R² std: {winner['cv_r2_fold_std']:.4f}")

print(f"NL in-sample adjusted R²: {winner['adjusted_r2']:.4f}")

print(f"Maximum VIF: {winner['max_vif']:.3f}")


# =====================================================================
# 5. Fit selected model to ALL NL
# =====================================================================

X_nl = development[selected_features].to_numpy(dtype=float)


y_nl = development["target"].to_numpy(dtype=float)


valid_nl = np.isfinite(y_nl) & np.all(
    np.isfinite(X_nl),
    axis=1,
)


beta_nl = fit_ols(
    X_nl[valid_nl],
    y_nl[valid_nl],
)


nl_fitted = predict_ols(
    X_nl[valid_nl],
    beta_nl,
)


nl_fit_metrics = prediction_metrics(
    y_nl[valid_nl],
    nl_fitted,
)


print("\nNL coefficients:")

print(f"  intercept = {beta_nl[0]: .8f}")

for i, feature in enumerate(
    selected_features,
    start=1,
):
    print(f"  beta_{i} = {beta_nl[i]: .8f}    [{feature}]")


# =====================================================================
# 6. VALIDATE — apply NL coefficients directly to BE
# =====================================================================

X_be = validation[selected_features].to_numpy(dtype=float)


y_be = validation["target"].to_numpy(dtype=float)


valid_be = np.isfinite(y_be) & np.all(
    np.isfinite(X_be),
    axis=1,
)


y_be_pred = predict_ols(
    X_be[valid_be],
    beta_nl,
)


be_metrics = prediction_metrics(
    y_be[valid_be],
    y_be_pred,
)


print(
    "\n"
    "============================================================\n"
    "BE EXTERNAL VALIDATION USING UNCHANGED NL COEFFICIENTS\n"
    "============================================================"
)


print(f"n = {be_metrics['n']}")

print(f"Predictive R² = {be_metrics['r2']:.4f}")

print(f"RMSE = {be_metrics['rmse']:.6f}")

print(f"MAE = {be_metrics['mae']:.6f}")

print(f"Mean prediction error = {be_metrics['mean_prediction_error']:.6f}")


# ---------------------------------------------------------------------
# Save individual BE predictions
# ---------------------------------------------------------------------

be_output = validation.loc[
    valid_be,
    [
        "country",
        "proxy_weight",
        "cv_group",
    ],
].copy()


be_output["observed_capacity_error"] = y_be[valid_be]


be_output["predicted_capacity_error"] = y_be_pred


be_output["prediction_error"] = (
    be_output["predicted_capacity_error"] - be_output["observed_capacity_error"]
)


be_output.to_csv(
    VALIDATION_PREDICTIONS_PATH,
    index=True,
)


# ---------------------------------------------------------------------
# Save validation summary
# ---------------------------------------------------------------------

validation_summary = pd.DataFrame(
    [
        {
            "development_country": DEVELOPMENT_COUNTRY,
            "validation_country": VALIDATION_COUNTRY,
            "selected_feature_1": selected_features[0],
            "selected_feature_2": selected_features[1],
            "selected_feature_3": selected_features[2],
            "nl_cv_r2": winner["cv_r2_oof"],
            "nl_max_vif": winner["max_vif"],
            "validation_n": be_metrics["n"],
            "validation_r2": be_metrics["r2"],
            "validation_rmse": be_metrics["rmse"],
            "validation_mae": be_metrics["mae"],
            "validation_mean_prediction_error": be_metrics["mean_prediction_error"],
        }
    ]
)


validation_summary.to_csv(
    VALIDATION_SUMMARY_PATH,
    index=False,
)


# =====================================================================
# 7. Validation figure
# =====================================================================

fig, ax = plt.subplots(
    figsize=(
        5.5,
        5.5,
    )
)


ax.scatter(
    y_be[valid_be],
    y_be_pred,
    alpha=0.6,
)


plot_min = float(
    min(
        np.min(y_be[valid_be]),
        np.min(y_be_pred),
    )
)


plot_max = float(
    max(
        np.max(y_be[valid_be]),
        np.max(y_be_pred),
    )
)


padding = 0.05 * (plot_max - plot_min)


plot_min -= padding
plot_max += padding


ax.plot(
    [
        plot_min,
        plot_max,
    ],
    [
        plot_min,
        plot_max,
    ],
    linestyle="--",
)


ax.set_xlim(
    plot_min,
    plot_max,
)

ax.set_ylim(
    plot_min,
    plot_max,
)


ax.set_xlabel("Observed signed LDES capacity error")

ax.set_ylabel("Diagnostic-predicted signed LDES capacity error")


ax.set_title("NL-trained diagnostic applied to BE")


ax.text(
    0.04,
    0.96,
    (f"$R^2$ = {be_metrics['r2']:.2f}\nn = {be_metrics['n']}"),
    transform=ax.transAxes,
    va="top",
)


fig.tight_layout()


fig.savefig(
    VALIDATION_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


# =====================================================================
# 8. FINAL REFIT — same predictors, NL + BE
# =====================================================================

pooled = analysis.loc[
    analysis["country"].isin(
        [
            DEVELOPMENT_COUNTRY,
            VALIDATION_COUNTRY,
        ]
    )
].copy()


X_pooled = pooled[selected_features].to_numpy(dtype=float)


y_pooled = pooled["target"].to_numpy(dtype=float)


valid_pooled = np.isfinite(y_pooled) & np.all(
    np.isfinite(X_pooled),
    axis=1,
)


beta_pooled = fit_ols(
    X_pooled[valid_pooled],
    y_pooled[valid_pooled],
)


y_pooled_pred = predict_ols(
    X_pooled[valid_pooled],
    beta_pooled,
)


pooled_metrics = prediction_metrics(
    y_pooled[valid_pooled],
    y_pooled_pred,
)


pooled_vif = calculate_vif(X_pooled[valid_pooled])


print(
    "\n"
    "============================================================\n"
    "FINAL NL + BE REFIT\n"
    "============================================================"
)


print("\nPredictor specification remains FIXED:")


for j, feature in enumerate(
    selected_features,
    start=1,
):
    print(f"x{j}: {feature}")


print("\nFinal pooled coefficients:")

print(f"  intercept = {beta_pooled[0]: .8f}")


for i, feature in enumerate(
    selected_features,
    start=1,
):
    print(f"  beta_{i} = {beta_pooled[i]: .8f}    [{feature}]")


print(f"\nPooled n = {pooled_metrics['n']}")

print(f"Pooled fitted R² = {pooled_metrics['r2']:.4f}")

print(f"Final maximum VIF = {np.max(pooled_vif):.3f}")


# ---------------------------------------------------------------------
# Save final coefficients
# ---------------------------------------------------------------------

coefficient_rows = [
    {
        "term": "intercept",
        "feature": "intercept",
        "coefficient": beta_pooled[0],
        "vif": np.nan,
    }
]


for i, feature in enumerate(
    selected_features,
    start=1,
):
    coefficient_rows.append(
        {
            "term": f"x{i}",
            "feature": feature,
            "coefficient": beta_pooled[i],
            "vif": pooled_vif[i - 1],
        }
    )


final_coefficients = pd.DataFrame(coefficient_rows)


final_coefficients.to_csv(
    FINAL_COEFFICIENTS_PATH,
    index=False,
)


# =====================================================================
# 9. Save complete selected-model metadata
# =====================================================================

selected_model_metadata = {
    "development_country": DEVELOPMENT_COUNTRY,
    "validation_country": VALIDATION_COUNTRY,
    "proxy_weight_selection": str(PROXY_WEIGHT),
    "normalisation_basis": NORMALISATION_BASIS,
    "cv_group_fields": CV_GROUP_FIELDS,
    "selected_features": selected_features,
    "development": {
        "n": int(valid_nl.sum()),
        "blocked_cv_r2": float(winner["cv_r2_oof"]),
        "blocked_cv_fold_mean": float(winner["cv_r2_fold_mean"]),
        "blocked_cv_fold_std": float(winner["cv_r2_fold_std"]),
        "max_vif": float(winner["max_vif"]),
        "coefficients": beta_nl.tolist(),
    },
    "validation": {
        "n": be_metrics["n"],
        "r2": be_metrics["r2"],
        "rmse": be_metrics["rmse"],
        "mae": be_metrics["mae"],
        "mean_prediction_error": be_metrics["mean_prediction_error"],
    },
    "final_pooled_refit": {
        "n": pooled_metrics["n"],
        "fitted_r2": pooled_metrics["r2"],
        "max_vif": float(np.max(pooled_vif)),
        "coefficients": beta_pooled.tolist(),
    },
}


with open(
    SELECTED_MODEL_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        selected_model_metadata,
        file,
        indent=2,
    )


# =====================================================================
# Final output summary
# =====================================================================

print("\nSaved outputs:")

print(f"  {SINGLE_RESULTS_PATH}")

print(f"  {SEARCH_ALL_PATH}")

print(f"  {SEARCH_TOP_PATH}")

print(f"  {VALIDATION_PREDICTIONS_PATH}")

print(f"  {VALIDATION_SUMMARY_PATH}")

print(f"  {VALIDATION_FIGURE_PATH}")

print(f"  {FINAL_COEFFICIENTS_PATH}")

print(f"  {SELECTED_MODEL_PATH}")
