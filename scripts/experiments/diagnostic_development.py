"""Test a manually specified SoC diagnostic by NL -> BE transfer.

Purpose
-------
The predictor structure is specified manually from theory.

NO predictor selection is performed in this script.

1. Filter the combined 2/5/10-year results to the requested horizons.
2. Retain NL + BE, kmeans + medoid, lambda_soc > 0.
3. Fit the manually specified predictors using NL ONLY.
4. Apply the NL coefficients unchanged to BE.
5. Report training and cross-country transfer performance.
6. Generate an observed-versus-predicted plot for NL and BE.

This permits 1--10 manually selected predictors.

When multiple horizon lengths are included, each horizon class receives
equal total statistical weight during NL fitting and performance scoring.

Reference-CEM results may define the signed LDES capacity-error target,
but reference-CEM-derived quantities must NOT be included in PREDICTORS.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# =====================================================================
# CONFIGURATION
# =====================================================================

RESULTS_DIR = Path("results") / "2_5_10_year"
OUTPUT_DIR = Path("results") / "diagnostic"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SIGNAL_METRICS_PATH = RESULTS_DIR / "signal_metrics.parquet"

INVESTMENT_METRICS_PATH = RESULTS_DIR / "investment_metrics.parquet"

PARAMETERS_PATH = RESULTS_DIR / "parameters.parquet"


# ---------------------------------------------------------------------
# Horizons
# ---------------------------------------------------------------------

INCLUDE_2_YEAR = False
INCLUDE_5_YEAR = False
INCLUDE_10_YEAR = True


# If more than one horizon is included, give every horizon class equal
# total statistical weight.
EQUAL_HORIZON_WEIGHTING = False


# ---------------------------------------------------------------------
# Manually specified predictors
# ---------------------------------------------------------------------

# Enter between 1 and 10 predictor names exactly as they appear in the
# signal-metric feature names.
#
# IMPORTANT:
# These should be chosen BEFORE looking at BE performance if BE is to
# function as a country holdout.
#
# Example starting structure shown below. Replace with whichever metrics
# you wish to test from theory.

PREDICTORS = [
    "rpcc__level__nmbe__reference_proxy_full_range__full",
    # "rpcc__delta__nrmse__reference_proxy_full_range__180d",
    "rpcc__delta__pearson__none__365d",
    "rpcc__level__nmbe__reference_proxy_full_range__180d",
    # Option 2:
    # "rpcc__level__nmbe__reference_proxy_full_range__full",
    # "clustered_approximation__delta__nrmse__reference_proxy_full_range__180d",
    # "clustered_approximation__delta__pearson__none__912d",
]

# ---------------------------------------------------------------------
# Automatic multicollinearity reparameterisation
# ---------------------------------------------------------------------

AUTO_TRANSFORM_HIGH_VIF = True
VIF_THRESHOLD = 5.0

# Only transform predictors belonging to the same:
#   error_family + signal_type + metric + normalisation_basis
# but evaluated over different temporal windows.
#
# The broadest selected window is retained and narrower windows become:
#
#     narrower - broadest
#
# This is an invertible reparameterisation and therefore does NOT alter
# model predictions or R².

# ---------------------------------------------------------------------
# Model subset
# ---------------------------------------------------------------------

COUNTRIES = [
    "NL",
    "BE",
]

TRAIN_COUNTRY = "NL"
VALIDATION_COUNTRY = "BE"

CLUSTER_METHOD = "kmeans"
REPRESENTATION_METHOD = "medoid"

PROXY_WEIGHT_FIELD = "lambda_soc"
PROXY_WEIGHT = "all"


# ---------------------------------------------------------------------
# Allowed predictor construction
# ---------------------------------------------------------------------

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

NORMALISATION_BASIS = "reference_proxy_full_range"


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

PLOT_PERCENTAGE_POINTS = True

# Always draw colours from plasma.
PLASMA_TRAIN_POSITION = 0.22
PLASMA_VALIDATION_POSITION = 0.78


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

OUTPUT_STEM = "diagnostic_manual_nl_to_be"

SUMMARY_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_summary.csv"

COEFFICIENTS_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_coefficients.csv"

PREDICTIONS_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_predictions.csv"

BY_HORIZON_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_by_horizon.csv"

MODEL_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_model.json"

FIGURE_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_observed_vs_predicted.png"

# =====================================================================
# WINDOW-CONTRAST REPARAMETERISATION
# =====================================================================


def parse_feature_name(
    feature: str,
) -> dict | None:
    """Parse a standard diagnostic feature name."""

    parts = feature.split("__")

    if len(parts) != 5:
        return None

    (
        error_family,
        signal_type,
        metric,
        normalisation_basis,
        window,
    ) = parts

    if window == "full":
        window_days = np.inf

    elif window.endswith("d"):
        try:
            window_days = float(window[:-1])

        except ValueError:
            return None

    else:
        return None

    return {
        "feature": feature,
        "error_family": error_family,
        "signal_type": signal_type,
        "metric": metric,
        "normalisation_basis": normalisation_basis,
        "window": window,
        "window_days": window_days,
        "series_key": (
            error_family,
            signal_type,
            metric,
            normalisation_basis,
        ),
    }


def get_window_groups(
    features: list[str],
) -> dict[
    tuple,
    list[int],
]:
    """Find groups differing only in temporal window."""

    groups = {}

    for index, feature in enumerate(features):
        parsed = parse_feature_name(feature)

        if parsed is None:
            continue

        groups.setdefault(
            parsed["series_key"],
            [],
        ).append(index)

    # A contrast is possible only when at least two selected terms
    # belong to the same metric series.
    return {
        key: indices
        for (
            key,
            indices,
        ) in groups.items()
        if len(indices) >= 2
    }


def build_window_contrast_design(
    X_raw: np.ndarray,
    features: list[str],
    transformed_groups: set[tuple],
) -> tuple[
    np.ndarray,
    list[str],
    list[dict],
]:
    """Apply local-minus-global transformations.

    For every selected metric series:

        broadest window remains unchanged

        every narrower window becomes:

            narrower - broadest

    The number of columns remains unchanged, so the transformation is
    invertible and spans exactly the same regression model.
    """

    X = np.array(
        X_raw,
        dtype=float,
        copy=True,
    )

    terms = list(features)

    transformation_info = []

    groups = get_window_groups(features)

    for group_key in transformed_groups:
        indices = groups[group_key]

        # Largest temporal support is the global/broad term.
        broad_index = max(
            indices,
            key=lambda index: parse_feature_name(features[index])["window_days"],
        )

        broad_feature = features[broad_index]

        transformation_info.append(
            {
                "series": "__".join(group_key),
                "broad_feature": broad_feature,
                "contrast_features": [],
            }
        )

        for index in indices:
            if index == broad_index:
                continue

            local_feature = features[index]

            # Always construct from RAW features. This avoids chained
            # transformations and keeps the interpretation simple.
            X[:, index] = X_raw[:, index] - X_raw[:, broad_index]

            terms[index] = f"contrast[{local_feature} - {broad_feature}]"

            transformation_info[-1]["contrast_features"].append(
                {
                    "local": local_feature,
                    "broad": broad_feature,
                    "term": terms[index],
                }
            )

    return (
        X,
        terms,
        transformation_info,
    )


def choose_vif_reparameterisation(
    X_raw: np.ndarray,
    features: list[str],
    weights: np.ndarray,
    *,
    threshold: float = 5.0,
) -> dict:
    """Automatically apply useful window contrasts when VIF is high.

    Selection is based ONLY on VIF in the NL training data.

    No target values and no BE data are used to choose the
    transformation.
    """

    raw_vif = calculate_weighted_vif(
        X_raw,
        weights,
    )

    raw_max_vif = float(np.max(raw_vif))

    result = {
        "raw_vif": raw_vif,
        "raw_max_vif": raw_max_vif,
        "transformed": False,
        "transformed_groups": set(),
        "X": X_raw,
        "terms": list(features),
        "transformation_info": [],
        "vif": raw_vif,
        "max_vif": raw_max_vif,
    }

    if not AUTO_TRANSFORM_HIGH_VIF or raw_max_vif <= threshold:
        return result

    groups = get_window_groups(features)

    if not groups:
        return result

    active_groups = set()

    current_max_vif = raw_max_vif

    remaining_groups = set(groups.keys())

    # Greedily apply whichever physically meaningful metric-window
    # transformation reduces maximum VIF most.
    while remaining_groups:
        best_group = None
        best_max_vif = current_max_vif

        best_X = None
        best_terms = None
        best_info = None
        best_vif = None

        for group_key in remaining_groups:
            candidate_groups = active_groups | {group_key}

            (
                candidate_X,
                candidate_terms,
                candidate_info,
            ) = build_window_contrast_design(
                X_raw,
                features,
                candidate_groups,
            )

            candidate_vif = calculate_weighted_vif(
                candidate_X,
                weights,
            )

            candidate_max_vif = float(np.max(candidate_vif))

            if candidate_max_vif < best_max_vif - 1e-10:
                best_group = group_key

                best_max_vif = candidate_max_vif

                best_X = candidate_X

                best_terms = candidate_terms

                best_info = candidate_info

                best_vif = candidate_vif

        # No meaningful contrast improves VIF further.
        if best_group is None:
            break

        active_groups.add(best_group)

        remaining_groups.remove(best_group)

        current_max_vif = best_max_vif

        result.update(
            {
                "transformed": True,
                "transformed_groups": active_groups.copy(),
                "X": best_X,
                "terms": best_terms,
                "transformation_info": best_info,
                "vif": best_vif,
                "max_vif": best_max_vif,
            }
        )

        if current_max_vif <= threshold:
            break

    return result


# =====================================================================
# FEATURE HELPERS
# =====================================================================


def window_label(
    value,
) -> str:

    if pd.isna(value):
        return "full"

    return f"{int(value)}d"


def make_feature_name(
    row: pd.Series,
) -> str:

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


# =====================================================================
# HORIZON WEIGHTING
# =====================================================================


def make_weights(
    horizons: np.ndarray,
) -> np.ndarray:
    """Construct regression/scoring weights."""

    horizons = np.asarray(
        horizons,
        dtype=float,
    )

    if not EQUAL_HORIZON_WEIGHTING:
        return np.ones(
            len(horizons),
            dtype=float,
        )

    unique_horizons = np.unique(horizons)

    # If only one horizon is present, weighting reduces to ordinary LS.
    if len(unique_horizons) == 1:
        return np.ones(
            len(horizons),
            dtype=float,
        )

    weights = np.zeros(
        len(horizons),
        dtype=float,
    )

    for horizon in unique_horizons:
        mask = horizons == horizon

        n = int(mask.sum())

        if n > 0:
            weights[mask] = 1.0 / n

    # Rescale to mean 1. This changes neither WLS coefficients nor R².
    weights *= len(weights) / weights.sum()

    return weights


# =====================================================================
# LINEAR MODEL
# =====================================================================


def fit_wls(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted least-squares regression with intercept."""

    X_design = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    sqrt_w = np.sqrt(weights)

    return np.linalg.lstsq(
        (X_design * sqrt_w[:, None]),
        (y * sqrt_w),
        rcond=None,
    )[0]


def predict_linear(
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


# =====================================================================
# VIF
# =====================================================================


def calculate_weighted_vif(
    X: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:

    p = X.shape[1]

    if p == 1:
        return np.array([1.0])

    weights = weights / weights.sum()

    mean = np.sum(
        X * weights[:, None],
        axis=0,
    )

    centred = X - mean

    variance = np.sum(
        weights[:, None] * centred**2,
        axis=0,
    )

    std = np.sqrt(variance)

    if np.any(std <= 0):
        return np.full(
            p,
            np.inf,
        )

    Z = centred / std

    corr = Z.T @ (Z * weights[:, None])

    try:
        inverse = np.linalg.inv(corr)

    except np.linalg.LinAlgError:
        return np.full(
            p,
            np.inf,
        )

    return np.diag(inverse)


# =====================================================================
# PERFORMANCE METRICS
# =====================================================================


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: np.ndarray,
) -> dict[str, float]:

    weights = make_weights(horizons)

    error = y_pred - y_true

    return {
        "n": int(len(y_true)),
        "weighted_r2": float(
            r2_score(
                y_true,
                y_pred,
                sample_weight=weights,
            )
        ),
        "unweighted_r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "weighted_rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                    sample_weight=weights,
                )
            )
        ),
        "weighted_mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
                sample_weight=weights,
            )
        ),
        "weighted_mean_prediction_error": float(
            np.average(
                error,
                weights=weights,
            )
        ),
    }


# =====================================================================
# 95% DESCRIPTIVE COVARIANCE ELLIPSE
# =====================================================================


def add_covariance_ellipse(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    color,
) -> None:
    """Draw a 95% descriptive covariance ellipse.

    This represents the spread of the point cloud. It is NOT a
    confidence interval for the regression.
    """

    if len(x) < 3:
        return

    weights = weights / weights.sum()

    points = np.column_stack(
        [
            x,
            y,
        ]
    )

    centre = np.sum(
        points * weights[:, None],
        axis=0,
    )

    centred = points - centre

    covariance = centred.T @ (centred * weights[:, None])

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]

    eigenvectors = eigenvectors[:, order]

    if np.any(eigenvalues <= 0):
        return

    # 95th percentile of chi-square(df=2).
    chi2_95 = 5.991464547107979

    radii = np.sqrt(chi2_95 * eigenvalues)

    angle = np.degrees(
        np.arctan2(
            eigenvectors[1, 0],
            eigenvectors[0, 0],
        )
    )

    ellipse = Ellipse(
        xy=centre,
        width=(2 * radii[0]),
        height=(2 * radii[1]),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.12,
        linewidth=1.5,
        zorder=1,
    )

    ax.add_patch(ellipse)


# =====================================================================
# CONFIGURATION VALIDATION
# =====================================================================

if not (1 <= len(PREDICTORS) <= 10):
    raise ValueError("PREDICTORS must contain between 1 and 10 features.")


if len(PREDICTORS) != len(set(PREDICTORS)):
    raise ValueError("PREDICTORS contains duplicates.")


included_horizons = [
    horizon
    for (
        horizon,
        include,
    ) in [
        (
            2,
            INCLUDE_2_YEAR,
        ),
        (
            5,
            INCLUDE_5_YEAR,
        ),
        (
            10,
            INCLUDE_10_YEAR,
        ),
    ]
    if include
]


if not included_horizons:
    raise ValueError("At least one model horizon must be enabled.")


print(
    "============================================================\n"
    "MANUAL NL -> BE DIAGNOSTIC TEST\n"
    "============================================================"
)


print("\nIncluded horizons:")

print("  " + ", ".join(f"{x}-year" for x in included_horizons))


print("\nManually specified predictors:")


for i, feature in enumerate(
    PREDICTORS,
    start=1,
):
    print(f"  x{i}: {feature}")


# =====================================================================
# LOAD DATA
# =====================================================================

print(f"\nReading {SIGNAL_METRICS_PATH}")

signal_metrics = pd.read_parquet(SIGNAL_METRICS_PATH)


print(f"Reading {INVESTMENT_METRICS_PATH}")

investment_metrics = pd.read_parquet(INVESTMENT_METRICS_PATH)


print(f"Reading {PARAMETERS_PATH}")

parameters = pd.read_parquet(PARAMETERS_PATH)


# =====================================================================
# PARAMETERS / CASE FILTER
# =====================================================================

metadata = parameters.copy()


required_columns = [
    "case_id",
    "country",
    "horizon_years",
    "cluster_method",
    "representation_method",
    PROXY_WEIGHT_FIELD,
]


missing = [column for column in required_columns if column not in metadata.columns]


if missing:
    raise ValueError(f"Missing required parameter columns:\n{missing}")


metadata["horizon_years"] = (
    pd.to_numeric(
        metadata["horizon_years"],
        errors="coerce",
    )
    .round()
    .astype("Int64")
)


metadata["proxy_weight"] = pd.to_numeric(
    metadata[PROXY_WEIGHT_FIELD],
    errors="coerce",
)


metadata = metadata.loc[
    metadata["country"].isin(COUNTRIES)
    & metadata["horizon_years"].isin(included_horizons)
    & metadata["cluster_method"].astype(str).str.lower().eq(CLUSTER_METHOD.lower())
    & metadata["representation_method"]
    .astype(str)
    .str.lower()
    .eq(REPRESENTATION_METHOD.lower())
].copy()


if PROXY_WEIGHT == "positive":
    metadata = metadata.loc[metadata["proxy_weight"] > 0].copy()

elif PROXY_WEIGHT == "all":
    pass

elif isinstance(
    PROXY_WEIGHT,
    (
        int,
        float,
    ),
):
    metadata = metadata.loc[
        np.isclose(
            metadata["proxy_weight"],
            float(PROXY_WEIGHT),
        )
    ].copy()

else:
    raise ValueError("Invalid PROXY_WEIGHT.")


if metadata.empty:
    raise ValueError("No cases remain after filtering.")


print("\nCases after filtering:")

print(
    metadata.groupby(
        [
            "country",
            "horizon_years",
        ]
    )
    .size()
    .rename("n")
    .to_string()
)


# Because this combined parquet also contains method-development work,
# print the experiment names retained after filtering as a sanity check.
if "experiment_name" in metadata.columns:
    print("\nRetained experiment names:")

    print(metadata["experiment_name"].value_counts().to_string())


# =====================================================================
# TARGET
# =====================================================================

target = investment_metrics.loc[
    investment_metrics["metric"] == "ldes_capacity_error_signed",
    [
        "case_id",
        "value",
    ],
].rename(
    columns={
        "value": "target",
    }
)


if target["case_id"].duplicated().any():
    raise ValueError(
        "More than one signed LDES capacity-error target exists for at least one case."
    )


# =====================================================================
# SIGNAL FEATURES
# =====================================================================

diagnostics = signal_metrics.loc[
    signal_metrics["error_family"].isin(ALLOWED_ERROR_FAMILIES)
    & signal_metrics["signal_type"].isin(ALLOWED_SIGNAL_TYPES)
    & signal_metrics["metric"].isin(ALLOWED_METRICS)
].copy()


# Normalised error metrics must use the exogenous reference-proxy range.
normalised_mask = diagnostics["metric"].isin(
    [
        "nmbe",
        "nrmse",
    ]
) & (diagnostics["normalisation_basis"] == NORMALISATION_BASIS)


# Pearson has no normalisation basis.
pearson_mask = (diagnostics["metric"] == "pearson") & diagnostics[
    "normalisation_basis"
].isna()


diagnostics = diagnostics.loc[normalised_mask | pearson_mask].copy()


diagnostics["feature"] = diagnostics.apply(
    make_feature_name,
    axis=1,
)


# Hard leakage guard.
forbidden_families = {
    "reference_approximation",
    "cem",
}


if diagnostics["error_family"].isin(forbidden_families).any():
    raise RuntimeError("Reference-CEM-dependent metrics entered the predictor table.")


duplicate_features = diagnostics.duplicated(
    [
        "case_id",
        "feature",
    ],
    keep=False,
)


if duplicate_features.any():
    raise ValueError("Duplicate case/feature metric records were found.")


X_wide = diagnostics.pivot(
    index="case_id",
    columns="feature",
    values="value",
)


X_wide.columns.name = None


missing_predictors = [
    feature for feature in PREDICTORS if feature not in X_wide.columns
]


if missing_predictors:
    print("\nAvailable feature names containing similar metric terms:")

    for feature in sorted(X_wide.columns):
        if any(
            term in feature
            for term in [
                "rpcc",
                "tsa",
                "clustered_approximation",
            ]
        ):
            print(f"  {feature}")

    raise ValueError(
        "\nThe following manually selected "
        "predictors were not found:\n" + "\n".join(missing_predictors)
    )


# =====================================================================
# MASTER ANALYSIS TABLE
# =====================================================================

analysis = (
    metadata[
        [
            "case_id",
            "country",
            "horizon_years",
            "proxy_weight",
        ]
        + (
            [
                "start_date",
                "end_date",
            ]
            if ("start_date" in metadata.columns and "end_date" in metadata.columns)
            else []
        )
    ]
    .merge(
        target,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    .set_index("case_id")
    .join(
        X_wide[PREDICTORS],
        how="inner",
    )
)


# =====================================================================
# FEATURE COMPLETENESS CHECK
# =====================================================================

print("\nSelected-predictor completeness:")


for feature in PREDICTORS:
    completeness = (
        analysis.assign(available=analysis[feature].notna())
        .groupby(
            [
                "country",
                "horizon_years",
            ]
        )["available"]
        .agg(
            [
                "sum",
                "count",
            ]
        )
    )

    completeness["fraction"] = completeness["sum"] / completeness["count"]

    print(f"\n{feature}")

    print(completeness.to_string())


# Complete cases across ALL manually selected predictors.
complete_mask = np.isfinite(analysis["target"].to_numpy(dtype=float)) & np.all(
    np.isfinite(analysis[PREDICTORS].to_numpy(dtype=float)),
    axis=1,
)


n_before = len(analysis)

analysis = analysis.loc[complete_mask].copy()


n_after = len(analysis)


print("\nComplete-case filtering:")

print(f"  before = {n_before:,}")

print(f"  after  = {n_after:,}")

print(f"  removed = {n_before - n_after:,}")


# =====================================================================
# NL TRAINING / BE VALIDATION
# =====================================================================

train = analysis.loc[analysis["country"] == TRAIN_COUNTRY].copy()


validation = analysis.loc[analysis["country"] == VALIDATION_COUNTRY].copy()


if train.empty:
    raise ValueError("NL training set is empty.")


if validation.empty:
    raise ValueError("BE validation set is empty.")


print("\nTraining sample:")

print(train.groupby("horizon_years").size().rename("n").to_string())


print("\nValidation sample:")

print(validation.groupby("horizon_years").size().rename("n").to_string())


# =====================================================================
# FIT NL COEFFICIENTS
# =====================================================================

X_train_raw = train[PREDICTORS].to_numpy(dtype=float)


y_train = train["target"].to_numpy(dtype=float)


horizon_train = train["horizon_years"].to_numpy(dtype=float)


train_weights = make_weights(horizon_train)


# ---------------------------------------------------------------------
# Raw VIF + optional automatic reparameterisation
# ---------------------------------------------------------------------

reparameterisation = choose_vif_reparameterisation(
    X_train_raw,
    PREDICTORS,
    train_weights,
    threshold=VIF_THRESHOLD,
)


X_train = reparameterisation["X"]


MODEL_TERMS = reparameterisation["terms"]


vif = reparameterisation["vif"]


print(
    "\n"
    "============================================================\n"
    "MULTICOLLINEARITY CHECK\n"
    "============================================================"
)


print("\nRaw predictor VIFs:")


for feature, feature_vif in zip(
    PREDICTORS,
    reparameterisation["raw_vif"],
):
    print(f"  {feature}: {feature_vif:.3f}")


print(f"\nRaw maximum VIF = {reparameterisation['raw_max_vif']:.3f}")


if reparameterisation["transformed"]:
    print(f"\nVIF > {VIF_THRESHOLD:g}; applying window-contrast reparameterisation.")

    for transformation in reparameterisation["transformation_info"]:
        print("\nSeries:")

        print(f"  {transformation['series']}")

        print("  retained broad term:")

        print(f"    {transformation['broad_feature']}")

        for contrast in transformation["contrast_features"]:
            print("  contrast:")

            print(f"    {contrast['local']} - {contrast['broad']}")

    print("\nTransformed VIFs:")

    for term, feature_vif in zip(
        MODEL_TERMS,
        vif,
    ):
        print(f"  {term}: {feature_vif:.3f}")

    print(f"\nTransformed maximum VIF = {reparameterisation['max_vif']:.3f}")


else:
    if reparameterisation["raw_max_vif"] <= VIF_THRESHOLD:
        print("\nNo transformation required.")

    else:
        print("\nWARNING:")

        print(
            f"  Maximum VIF remains > "
            f"{VIF_THRESHOLD:g}, but no "
            "physically meaningful same-metric "
            "window contrast reduced it."
        )

        print("  Predictors have NOT been automatically altered.")


# ---------------------------------------------------------------------
# Fit the selected parameterisation
# ---------------------------------------------------------------------

beta = fit_wls(
    X_train,
    y_train,
    train_weights,
)


train_prediction = predict_linear(
    X_train,
    beta,
)


# ---------------------------------------------------------------------
# Verify that any VIF transformation is a true reparameterisation
# ---------------------------------------------------------------------

if reparameterisation["transformed"]:
    raw_beta = fit_wls(
        X_train_raw,
        y_train,
        train_weights,
    )

    raw_train_prediction = predict_linear(
        X_train_raw,
        raw_beta,
    )

    if not np.allclose(
        raw_train_prediction,
        train_prediction,
        rtol=1e-9,
        atol=1e-11,
    ):
        raise RuntimeError(
            "Window-contrast reparameterisation changed fitted predictions."
        )

    print("\nTransformation check:")

    print("  fitted predictions unchanged: YES")


train_metrics = calculate_metrics(
    y_train,
    train_prediction,
    horizon_train,
)


# =====================================================================
# APPLY IDENTICAL NL PARAMETERISATION UNCHANGED TO BE
# =====================================================================

X_validation_raw = validation[PREDICTORS].to_numpy(dtype=float)


if reparameterisation["transformed"]:
    (
        X_validation,
        validation_terms,
        _,
    ) = build_window_contrast_design(
        X_validation_raw,
        PREDICTORS,
        reparameterisation["transformed_groups"],
    )

    if validation_terms != MODEL_TERMS:
        raise RuntimeError("NL and BE transformation definitions do not match.")


else:
    X_validation = X_validation_raw


y_validation = validation["target"].to_numpy(dtype=float)


horizon_validation = validation["horizon_years"].to_numpy(dtype=float)


validation_prediction = predict_linear(
    X_validation,
    beta,
)


validation_metrics = calculate_metrics(
    y_validation,
    validation_prediction,
    horizon_validation,
)


# =====================================================================
# PRINT MODEL
# =====================================================================

print(
    "\n"
    "============================================================\n"
    "NL MODEL\n"
    "============================================================"
)


print(f"\nintercept = {beta[0]: .8f}")


for i, term in enumerate(
    MODEL_TERMS,
    start=1,
):
    print(f"beta_{i} = {beta[i]: .8f}    [{term}]")


print("\nWeighted VIF:")


for term, feature_vif in zip(
    MODEL_TERMS,
    vif,
):
    print(f"  {term}: {feature_vif:.3f}")


print(
    "\n"
    "============================================================\n"
    "NL TRAINING FIT\n"
    "============================================================"
)


print(f"n = {train_metrics['n']}")

print(f"Weighted R² = {train_metrics['weighted_r2']:.4f}")

print(f"Unweighted R² = {train_metrics['unweighted_r2']:.4f}")

print(f"Weighted RMSE = {train_metrics['weighted_rmse']:.6f}")

print(f"Weighted MAE = {train_metrics['weighted_mae']:.6f}")


print(
    "\n"
    "============================================================\n"
    "NL -> BE CROSS-COUNTRY VALIDATION\n"
    "============================================================"
)


print(f"n = {validation_metrics['n']}")

print(f"Weighted predictive R² = {validation_metrics['weighted_r2']:.4f}")

print(f"Unweighted predictive R² = {validation_metrics['unweighted_r2']:.4f}")

print(f"Weighted RMSE = {validation_metrics['weighted_rmse']:.6f}")

print(f"Weighted MAE = {validation_metrics['weighted_mae']:.6f}")

print(
    f"Weighted mean prediction error = "
    f"{validation_metrics['weighted_mean_prediction_error']:.6f}"
)


# =====================================================================
# PERFORMANCE BY HORIZON
# =====================================================================

by_horizon_rows = []


for dataset_name, frame, prediction in [
    (
        "NL training",
        train,
        train_prediction,
    ),
    (
        "BE validation",
        validation,
        validation_prediction,
    ),
]:
    horizons = frame["horizon_years"].to_numpy(dtype=float)

    observed = frame["target"].to_numpy(dtype=float)

    for horizon in sorted(np.unique(horizons)):
        mask = horizons == horizon

        metrics = calculate_metrics(
            observed[mask],
            prediction[mask],
            horizons[mask],
        )

        by_horizon_rows.append(
            {
                "dataset": dataset_name,
                "horizon_years": int(horizon),
                **metrics,
            }
        )


by_horizon = pd.DataFrame(by_horizon_rows)


print("\nPerformance by horizon:")

print(by_horizon.to_string(index=False))


by_horizon.to_csv(
    BY_HORIZON_PATH,
    index=False,
)


# =====================================================================
# SAVE PREDICTIONS
# =====================================================================

train_output = train[
    [
        "country",
        "horizon_years",
        "proxy_weight",
        "target",
    ]
    + (
        [
            "start_date",
            "end_date",
        ]
        if ("start_date" in train.columns and "end_date" in train.columns)
        else []
    )
].copy()


train_output["prediction"] = train_prediction


train_output["prediction_error"] = train_prediction - y_train


train_output["dataset"] = "NL training"


validation_output = validation[
    [
        "country",
        "horizon_years",
        "proxy_weight",
        "target",
    ]
    + (
        [
            "start_date",
            "end_date",
        ]
        if ("start_date" in validation.columns and "end_date" in validation.columns)
        else []
    )
].copy()


validation_output["prediction"] = validation_prediction


validation_output["prediction_error"] = validation_prediction - y_validation


validation_output["dataset"] = "BE validation"


prediction_output = pd.concat(
    [
        train_output,
        validation_output,
    ]
)


prediction_output.to_csv(
    PREDICTIONS_PATH,
    index=True,
)


# =====================================================================
# MAIN FIGURE
# =====================================================================

scale = 100.0 if PLOT_PERCENTAGE_POINTS else 1.0


train_x = scale * y_train

train_y = scale * train_prediction


validation_x = scale * y_validation

validation_y = scale * validation_prediction


validation_weights = make_weights(horizon_validation)


# ---------------------------------------------------------------------
# Plasma colours ONLY
# ---------------------------------------------------------------------

cmap = plt.get_cmap("plasma")


training_color = cmap(PLASMA_TRAIN_POSITION)


validation_color = cmap(PLASMA_VALIDATION_POSITION)


fig, ax = plt.subplots(
    figsize=(
        6.5,
        6.2,
    )
)


# Ellipses first so points remain visible above them.
add_covariance_ellipse(
    ax,
    train_x,
    train_y,
    train_weights,
    color=training_color,
)


add_covariance_ellipse(
    ax,
    validation_x,
    validation_y,
    validation_weights,
    color=validation_color,
)


ax.scatter(
    train_x,
    train_y,
    s=23,
    alpha=0.30,
    marker="o",
    color=training_color,
    edgecolors="none",
    label=(f"NL training (n={len(train_x)})"),
    zorder=3,
)


ax.scatter(
    validation_x,
    validation_y,
    s=29,
    alpha=0.50,
    marker="^",
    color=validation_color,
    edgecolors="none",
    label=(f"BE validation (n={len(validation_x)})"),
    zorder=4,
)


plot_min = float(
    min(
        np.min(train_x),
        np.min(train_y),
        np.min(validation_x),
        np.min(validation_y),
    )
)


plot_max = float(
    max(
        np.max(train_x),
        np.max(train_y),
        np.max(validation_x),
        np.max(validation_y),
    )
)


plot_range = plot_max - plot_min


padding = 0.05 * plot_range if plot_range > 0 else 0.05


plot_min -= padding
plot_max += padding


# The 1:1 line is neutral rather than a third data colour.
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
    linewidth=1.0,
    color="0.30",
    label="1:1",
    zorder=2,
)


ax.set_xlim(
    plot_min,
    plot_max,
)

ax.set_ylim(
    plot_min,
    plot_max,
)


ax.set_aspect(
    "equal",
    adjustable="box",
)


unit = "%" if PLOT_PERCENTAGE_POINTS else ""


ax.set_xlabel(f"Observed signed LDES capacity error ({unit})")


ax.set_ylabel(f"Predicted signed LDES capacity error ({unit})")


r2_label = (
    "weighted " if (EQUAL_HORIZON_WEIGHTING and len(included_horizons) > 1) else ""
)


ax.text(
    0.04,
    0.96,
    (
        f"NL training {r2_label}"
        f"$R^2$ = "
        f"{train_metrics['weighted_r2']:.2f}\n"
        f"BE validation {r2_label}"
        f"$R^2$ = "
        f"{validation_metrics['weighted_r2']:.2f}"
    ),
    transform=ax.transAxes,
    va="top",
)


ax.legend(
    frameon=False,
    loc="lower right",
)


fig.tight_layout()


fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


# =====================================================================
# SAVE COEFFICIENTS
# =====================================================================

coefficient_rows = [
    {
        "term": "intercept",
        "model_term": "intercept",
        "original_predictor": "intercept",
        "coefficient": beta[0],
        "weighted_vif": np.nan,
    }
]


for i, (
    original_feature,
    model_term,
    feature_vif,
) in enumerate(
    zip(
        PREDICTORS,
        MODEL_TERMS,
        vif,
    ),
    start=1,
):
    coefficient_rows.append(
        {
            "term": f"x{i}",
            "model_term": model_term,
            "original_predictor": original_feature,
            "coefficient": beta[i],
            "weighted_vif": feature_vif,
        }
    )


pd.DataFrame(coefficient_rows).to_csv(
    COEFFICIENTS_PATH,
    index=False,
)


# =====================================================================
# SAVE SUMMARY
# =====================================================================

summary = pd.DataFrame(
    [
        {
            "predictor_count": len(PREDICTORS),
            "included_horizons": ",".join(str(x) for x in included_horizons),
            "equal_horizon_weighting": EQUAL_HORIZON_WEIGHTING,
            "training_country": TRAIN_COUNTRY,
            "validation_country": VALIDATION_COUNTRY,
            "training_n": train_metrics["n"],
            "training_weighted_r2": train_metrics["weighted_r2"],
            "training_unweighted_r2": train_metrics["unweighted_r2"],
            "validation_n": validation_metrics["n"],
            "validation_weighted_r2": validation_metrics["weighted_r2"],
            "validation_unweighted_r2": validation_metrics["unweighted_r2"],
            "validation_weighted_rmse": validation_metrics["weighted_rmse"],
            "validation_weighted_mae": validation_metrics["weighted_mae"],
            "validation_weighted_mean_prediction_error": validation_metrics[
                "weighted_mean_prediction_error"
            ],
            "reparameterised": bool(reparameterisation["transformed"]),
            "raw_max_weighted_vif": float(reparameterisation["raw_max_vif"]),
            "model_max_weighted_vif": float(np.max(vif)),
        }
    ]
)


summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# =====================================================================
# SAVE MODEL DESCRIPTION
# =====================================================================

model_metadata = {
    "predictors": PREDICTORS,
    "model_terms": MODEL_TERMS,
    "predictor_selection": (
        "manual / theory specified; no predictor selection performed by this script"
    ),
    "included_horizons": included_horizons,
    "equal_horizon_weighting": EQUAL_HORIZON_WEIGHTING,
    "training_country": TRAIN_COUNTRY,
    "validation_country": VALIDATION_COUNTRY,
    "cluster_method": CLUSTER_METHOD,
    "representation_method": REPRESENTATION_METHOD,
    "reparameterised": bool(reparameterisation["transformed"]),
    "transformation_info": reparameterisation["transformation_info"],
    "raw_vif": {
        feature: float(feature_vif)
        for (
            feature,
            feature_vif,
        ) in zip(
            PREDICTORS,
            reparameterisation["raw_vif"],
        )
    },
    "model_vif": {
        term: float(feature_vif)
        for (
            term,
            feature_vif,
        ) in zip(
            MODEL_TERMS,
            vif,
        )
    },
    "coefficients": {
        "intercept": float(beta[0]),
        "terms": [
            {
                "term": term,
                "original_predictor": original_feature,
                "coefficient": float(beta[i]),
            }
            for i, (
                original_feature,
                term,
            ) in enumerate(
                zip(
                    PREDICTORS,
                    MODEL_TERMS,
                ),
                start=1,
            )
        ],
    },
    "nl_training": train_metrics,
    "be_validation": validation_metrics,
}


with open(
    MODEL_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        model_metadata,
        file,
        indent=2,
    )


# =====================================================================
# FINAL EQUATION
# =====================================================================

print(
    "\n"
    "============================================================\n"
    "FINAL NL-FITTED EQUATION\n"
    "============================================================"
)


equation = f"capacity_error = {beta[0]:+.6f}"


for i in range(
    1,
    len(beta),
):
    equation += f" {beta[i]:+.6f} * x{i}"


print(f"\n{equation}")


print("\nwhere:")


for i, term in enumerate(
    MODEL_TERMS,
    start=1,
):
    print(f"  x{i} = {term}")


print("\nSaved outputs:")


for path in [
    SUMMARY_PATH,
    COEFFICIENTS_PATH,
    PREDICTIONS_PATH,
    BY_HORIZON_PATH,
    MODEL_PATH,
    FIGURE_PATH,
]:
    print(f"  {path}")
