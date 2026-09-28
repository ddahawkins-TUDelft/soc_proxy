"""End-to-end three-state LDES risk diagnostic (Model B only).

Purpose
-------
The diagnostic is intended as a screening tool rather than an exact predictor of
LDES capacity error. For a chosen tolerance T (default 10 percentage points), it
returns three probabilities for a temporally aggregated model run:

    P(error < -T)             substantial LDES underinvestment
    P(-T <= error <= +T)      error within the comfortable range
    P(error > +T)             substantial LDES overinvestment

The model is intentionally built in two stages:

1. A continuous linear diagnostic predicts signed LDES capacity error from the
   three manually/theoretically selected signal metrics.
2. A multinomial logistic calibration model maps the continuous diagnostic
   prediction and its square onto the three outcome probabilities.

This script produces two distinct analyses.

A. Validation experiment: NL -> BE
   - The complete diagnostic is developed using NL only.
   - NL performance is estimated with nested cross-fitting of the full two-stage
     pipeline.
   - The final NL model is then applied unchanged to BE, which remains a true
     cross-country holdout.

B. Final proposed diagnostic: NL + BE
   - After the validation experiment, the diagnostic is refitted using all NL
     and BE cases for subsequent use by readers.
   - A full combined-data three-state mapping and equations are exported.
   - Apparent (in-sample) calibration is shown for the combined sample and the
     NL/BE subsets. This is descriptive only and is NOT independent validation.
   - A nested cross-fitted calibration figure is also produced as an internal
     validation estimate for the combined-data formulation.

The probability calibrator used for deployment is fitted to out-of-fold
continuous predictions, while the final continuous model itself is fitted to all
available training cases. This makes the calibration layer see continuous-model
predictions with an out-of-sample error structure during training.

The target signed LDES capacity error is stored as a fraction (0.10 = 10%).
Figures and equations report it in percentage points.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =====================================================================
# CONFIGURATION
# =====================================================================

RESULTS_DIR = Path("results") / "2_5_10_year"
OUTPUT_DIR = Path("results") / "diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_METRICS_PATH = RESULTS_DIR / "signal_metrics.parquet"
INVESTMENT_METRICS_PATH = RESULTS_DIR / "investment_metrics.parquet"
PARAMETERS_PATH = RESULTS_DIR / "parameters.parquet"

# Current diagnostic experiment: 10-year cases.
INCLUDED_HORIZONS = [10]
EQUAL_HORIZON_WEIGHTING = False

# Comfortable signed-error band, in percentage points.
ERROR_TOLERANCE_PP = 10.0

# Fixed manually/theoretically selected predictors.
PREDICTORS = [
    "rpcc__level__nmbe__reference_proxy_full_range__full",
    "rpcc__delta__pearson__none__365d",
    "rpcc__level__nmbe__reference_proxy_full_range__180d",
]

AUTO_TRANSFORM_HIGH_VIF = True
VIF_THRESHOLD = 5.0

COUNTRIES = ["NL", "BE"]
TRAIN_COUNTRY = "NL"
VALIDATION_COUNTRY = "BE"

CLUSTER_METHOD = "kmeans"
REPRESENTATION_METHOD = "medoid"
PROXY_WEIGHT_FIELD = "lambda_soc"
PROXY_WEIGHT: str | float = "all"

ALLOWED_ERROR_FAMILIES = ["clustered_approximation", "tsa", "rpcc"]
ALLOWED_SIGNAL_TYPES = ["level", "delta"]
ALLOWED_METRICS = ["nmbe", "nrmse", "pearson"]
NORMALISATION_BASIS = "reference_proxy_full_range"

CV_FOLDS = 5
CV_RANDOM_STATE = 42
TARGET_TRAIN_SENSITIVITY = 0.90
N_CALIBRATION_BINS = 8

# Plasma-derived colours, avoiding the yellow end of the map.
PLASMA_NL_POSITION = 0.08
PLASMA_BE_POSITION = 0.50
PLASMA_UNDER_POSITION = 0.08
PLASMA_OVER_POSITION = 0.50
ACCEPTABLE_GREY = "0.45"
COMBINED_GREY = "0.25"

OUTPUT_STEM = "diagnostic_three_state_modelB"

# NL -> BE validation outputs.
VALIDATION_SUMMARY_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_validation_summary.csv"
VALIDATION_SCREENING_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_validation_screening.csv"
VALIDATION_PREDICTIONS_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_validation_predictions.csv"
VALIDATION_BINS_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_validation_calibration_bins.csv"
VALIDATION_CAL_PNG = OUTPUT_DIR / f"{OUTPUT_STEM}_validation_calibration.png"
VALIDATION_CAL_PDF = OUTPUT_DIR / f"{OUTPUT_STEM}_validation_calibration.pdf"
VALIDATION_MAP_PNG = OUTPUT_DIR / f"{OUTPUT_STEM}_validation_three_state_mapping.png"
VALIDATION_MAP_PDF = OUTPUT_DIR / f"{OUTPUT_STEM}_validation_three_state_mapping.pdf"

# Final NL+BE diagnostic outputs.
FINAL_SUMMARY_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_final_combined_summary.csv"
FINAL_PREDICTIONS_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_final_combined_predictions.csv"
FINAL_BINS_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_final_combined_calibration_bins.csv"
FINAL_APPARENT_CAL_PNG = OUTPUT_DIR / f"{OUTPUT_STEM}_final_combined_calibration_apparent.png"
FINAL_APPARENT_CAL_PDF = OUTPUT_DIR / f"{OUTPUT_STEM}_final_combined_calibration_apparent.pdf"
FINAL_CV_CAL_PNG = OUTPUT_DIR / f"{OUTPUT_STEM}_final_combined_calibration_crossfit.png"
FINAL_CV_CAL_PDF = OUTPUT_DIR / f"{OUTPUT_STEM}_final_combined_calibration_crossfit.pdf"
FINAL_MAP_PNG = OUTPUT_DIR / f"{OUTPUT_STEM}_final_combined_three_state_mapping.png"
FINAL_MAP_PDF = OUTPUT_DIR / f"{OUTPUT_STEM}_final_combined_three_state_mapping.pdf"
FINAL_EQUATIONS_TXT = OUTPUT_DIR / f"{OUTPUT_STEM}_final_combined_equations.txt"
FINAL_EQUATIONS_TEX = OUTPUT_DIR / f"{OUTPUT_STEM}_final_combined_equations.tex"
MODEL_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_model.json"

STATE_UNDER = "under"
STATE_ACCEPTABLE = "acceptable"
STATE_OVER = "over"
STATE_ORDER = [STATE_UNDER, STATE_ACCEPTABLE, STATE_OVER]


# =====================================================================
# FEATURE HELPERS
# =====================================================================


def window_label(value) -> str:
    if pd.isna(value):
        return "full"
    return f"{int(value)}d"


def make_feature_name(row: pd.Series) -> str:
    basis = "none" if pd.isna(row["normalisation_basis"]) else str(row["normalisation_basis"])
    return (
        f"{row['error_family']}"
        f"__{row['signal_type']}"
        f"__{row['metric']}"
        f"__{basis}"
        f"__{window_label(row['window_half_width_days'])}"
    )


def parse_feature_name(feature: str) -> dict | None:
    parts = feature.split("__")
    if len(parts) != 5:
        return None

    error_family, signal_type, metric, normalisation_basis, window = parts

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
        "series_key": (error_family, signal_type, metric, normalisation_basis),
    }


def get_window_groups(features: list[str]) -> dict[tuple, list[int]]:
    groups: dict[tuple, list[int]] = {}
    for index, feature in enumerate(features):
        parsed = parse_feature_name(feature)
        if parsed is None:
            continue
        groups.setdefault(parsed["series_key"], []).append(index)
    return {key: indices for key, indices in groups.items() if len(indices) >= 2}


# =====================================================================
# HORIZON WEIGHTING
# =====================================================================


def make_weights(horizons: np.ndarray) -> np.ndarray:
    horizons = np.asarray(horizons, dtype=float)

    if not EQUAL_HORIZON_WEIGHTING:
        return np.ones(len(horizons), dtype=float)

    unique_horizons = np.unique(horizons)
    if len(unique_horizons) == 1:
        return np.ones(len(horizons), dtype=float)

    weights = np.zeros(len(horizons), dtype=float)
    for horizon in unique_horizons:
        mask = horizons == horizon
        n = int(mask.sum())
        if n > 0:
            weights[mask] = 1.0 / n

    weights *= len(weights) / weights.sum()
    return weights


# =====================================================================
# CONTINUOUS SIGNED-ERROR MODEL
# =====================================================================


def fit_wls(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    X_design = np.column_stack([np.ones(len(X)), X])
    sqrt_w = np.sqrt(weights)
    return np.linalg.lstsq(
        X_design * sqrt_w[:, None],
        y * sqrt_w,
        rcond=None,
    )[0]


def predict_linear(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X_design = np.column_stack([np.ones(len(X)), X])
    return X_design @ beta


# =====================================================================
# VIF + WINDOW-CONTRAST REPARAMETERISATION
# =====================================================================


def calculate_weighted_vif(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    p = X.shape[1]
    if p == 1:
        return np.array([1.0])

    weights = weights / weights.sum()
    mean = np.sum(X * weights[:, None], axis=0)
    centred = X - mean
    variance = np.sum(weights[:, None] * centred**2, axis=0)
    std = np.sqrt(variance)

    if np.any(std <= 0):
        return np.full(p, np.inf)

    Z = centred / std
    corr = Z.T @ (Z * weights[:, None])

    try:
        inverse = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        return np.full(p, np.inf)

    return np.diag(inverse)


def build_window_contrast_design(
    X_raw: np.ndarray,
    features: list[str],
    transformed_groups: set[tuple],
) -> tuple[np.ndarray, list[str], list[dict]]:
    """Apply local-minus-global window contrasts without changing model span."""
    X = np.array(X_raw, dtype=float, copy=True)
    terms = list(features)
    transformation_info: list[dict] = []
    groups = get_window_groups(features)

    for group_key in transformed_groups:
        indices = groups[group_key]
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
            X[:, index] = X_raw[:, index] - X_raw[:, broad_index]
            terms[index] = f"contrast[{local_feature} - {broad_feature}]"
            transformation_info[-1]["contrast_features"].append(
                {"local": local_feature, "broad": broad_feature, "term": terms[index]}
            )

    return X, terms, transformation_info


def choose_vif_reparameterisation(
    X_raw: np.ndarray,
    features: list[str],
    weights: np.ndarray,
    *,
    threshold: float,
) -> dict:
    """Choose same-metric window contrasts using NL predictors only."""
    raw_vif = calculate_weighted_vif(X_raw, weights)
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

    active_groups: set[tuple] = set()
    remaining_groups = set(groups.keys())
    current_max_vif = raw_max_vif

    while remaining_groups:
        best_group = None
        best_max_vif = current_max_vif
        best_X = best_terms = best_info = best_vif = None

        for group_key in remaining_groups:
            candidate_groups = active_groups | {group_key}
            candidate_X, candidate_terms, candidate_info = build_window_contrast_design(
                X_raw,
                features,
                candidate_groups,
            )
            candidate_vif = calculate_weighted_vif(candidate_X, weights)
            candidate_max_vif = float(np.max(candidate_vif))

            if candidate_max_vif < best_max_vif - 1e-10:
                best_group = group_key
                best_max_vif = candidate_max_vif
                best_X = candidate_X
                best_terms = candidate_terms
                best_info = candidate_info
                best_vif = candidate_vif

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
# TARGET HELPERS
# =====================================================================


def make_state_labels(signed_error: np.ndarray) -> np.ndarray:
    tol = ERROR_TOLERANCE_PP / 100.0
    signed_error = np.asarray(signed_error, dtype=float)
    return np.where(
        signed_error < -tol,
        STATE_UNDER,
        np.where(signed_error > tol, STATE_OVER, STATE_ACCEPTABLE),
    )


def make_binary_labels(signed_error: np.ndarray) -> np.ndarray:
    tol = ERROR_TOLERANCE_PP / 100.0
    return (np.abs(np.asarray(signed_error, dtype=float)) > tol).astype(int)


# =====================================================================
# CROSS-FITTED CONTINUOUS PREDICTIONS
# =====================================================================


def effective_n_splits(labels: np.ndarray, requested: int) -> int:
    _, counts = np.unique(labels, return_counts=True)
    n_splits = min(requested, int(np.min(counts)))
    if n_splits < 2:
        raise ValueError(
            "At least two examples of every stratification class are required "
            "for cross-fitting."
        )
    return n_splits


def make_stratified_folds(labels: np.ndarray) -> StratifiedKFold:
    n_splits = effective_n_splits(labels, CV_FOLDS)
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )


def cross_fitted_continuous_predictions(
    X: np.ndarray,
    y_signed: np.ndarray,
    horizons: np.ndarray,
    state_labels: np.ndarray,
) -> np.ndarray:
    """Generate OOF signed-error predictions from the continuous NL model."""
    prediction = np.full(len(y_signed), np.nan, dtype=float)
    splitter = make_stratified_folds(state_labels)

    for train_idx, test_idx in splitter.split(X, state_labels):
        fold_weights = make_weights(horizons[train_idx])
        beta = fit_wls(X[train_idx], y_signed[train_idx], fold_weights)
        prediction[test_idx] = predict_linear(X[test_idx], beta)

    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("OOF continuous predictions contain non-finite values.")

    return prediction


# =====================================================================
# CALIBRATION MODELS
# =====================================================================



def three_state_features(signed_prediction: np.ndarray) -> np.ndarray:
    """Quadratic signed-error features in percentage-point units."""
    e_pp = np.asarray(signed_prediction, dtype=float) * 100.0
    return np.column_stack([e_pp, e_pp**2])


def fit_three_state_calibrator(
    signed_prediction: np.ndarray,
    y_state: np.ndarray,
    weights: np.ndarray,
) -> Pipeline:
    """Model B: multinomial P(under / acceptable / over) from e_hat and e_hat^2."""
    X = three_state_features(signed_prediction)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1e12,
                    solver="lbfgs",
                    max_iter=20_000,
                    tol=1e-10,
                ),
            ),
        ]
    )
    model.fit(X, y_state, logistic__sample_weight=weights)
    return model


def predict_three_state_calibrator(
    model: Pipeline,
    signed_prediction: np.ndarray,
) -> pd.DataFrame:
    X = three_state_features(signed_prediction)
    p = model.predict_proba(X)
    classes = list(model.named_steps["logistic"].classes_)

    out = pd.DataFrame(index=np.arange(len(X)))
    for state in STATE_ORDER:
        if state in classes:
            out[state] = p[:, classes.index(state)]
        else:
            out[state] = 0.0

    out["risk"] = out[STATE_UNDER] + out[STATE_OVER]
    return out



def cross_fitted_three_state_probabilities(
    signed_oof_prediction: np.ndarray,
    y_state: np.ndarray,
    horizons: np.ndarray,
) -> pd.DataFrame:
    """Cross-fit the Model B calibration layer on NL."""
    result = pd.DataFrame(
        np.nan,
        index=np.arange(len(y_state)),
        columns=[STATE_UNDER, STATE_ACCEPTABLE, STATE_OVER, "risk"],
    )
    splitter = make_stratified_folds(y_state)
    dummy_X = np.zeros((len(y_state), 1))

    for train_idx, test_idx in splitter.split(dummy_X, y_state):
        weights = make_weights(horizons[train_idx])
        model = fit_three_state_calibrator(
            signed_oof_prediction[train_idx],
            y_state[train_idx],
            weights,
        )
        fold = predict_three_state_calibrator(model, signed_oof_prediction[test_idx])
        result.iloc[test_idx] = fold.to_numpy()

    if result.isna().any().any():
        raise RuntimeError("Cross-fitted three-state probabilities contain NaNs.")

    return result


# =====================================================================
# METRICS
# =====================================================================


def probability_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    *,
    null_probability: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    weights = np.asarray(weights, dtype=float)

    prevalence = float(np.average(y_true, weights=weights))
    if len(np.unique(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, probability, sample_weight=weights))
        ap = float(average_precision_score(y_true, probability, sample_weight=weights))
    else:
        auc = np.nan
        ap = np.nan

    brier = float(brier_score_loss(y_true, probability, sample_weight=weights))
    null_vector = np.full(len(y_true), float(null_probability))
    null_brier = float(brier_score_loss(y_true, null_vector, sample_weight=weights))
    brier_skill = float(1.0 - brier / null_brier) if null_brier > 0 else np.nan

    return {
        "n": int(len(y_true)),
        "prevalence": prevalence,
        "roc_auc": auc,
        "average_precision": ap,
        "brier_score": brier,
        "null_brier_nl_prevalence": null_brier,
        "brier_skill_vs_nl_prevalence": brier_skill,
        "log_loss": float(
            log_loss(
                y_true,
                np.column_stack([1.0 - probability, probability]),
                sample_weight=weights,
                labels=[0, 1],
            )
        ),
        "mean_predicted_probability": float(np.average(probability, weights=weights)),
    }


def threshold_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    weights = np.asarray(weights, dtype=float)
    predicted = probability >= threshold

    positive = y_true == 1
    negative = ~positive

    tp = float(weights[predicted & positive].sum())
    fn = float(weights[(~predicted) & positive].sum())
    tn = float(weights[(~predicted) & negative].sum())
    fp = float(weights[predicted & negative].sum())

    def safe_div(num: float, den: float) -> float:
        return float(num / den) if den > 0 else np.nan

    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)

    return {
        "probability_threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_ppv": safe_div(tp, tp + fp),
        "negative_predictive_value": safe_div(tn, tn + fn),
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "false_negative_rate": float(1.0 - sensitivity) if np.isfinite(sensitivity) else np.nan,
        "false_positive_rate": float(1.0 - specificity) if np.isfinite(specificity) else np.nan,
        "fraction_flagged": safe_div(tp + fp, tp + tn + fp + fn),
        "tp_count": int(np.sum(predicted & positive)),
        "fn_count": int(np.sum((~predicted) & positive)),
        "tn_count": int(np.sum((~predicted) & negative)),
        "fp_count": int(np.sum(predicted & negative)),
    }


def choose_threshold_for_sensitivity(
    y_true: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    target_sensitivity: float,
) -> float:
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    feasible: list[tuple[float, float]] = []

    for threshold in candidates:
        metrics = threshold_metrics(y_true, probability, weights, float(threshold))
        if (
            np.isfinite(metrics["sensitivity"])
            and metrics["sensitivity"] >= target_sensitivity
        ):
            feasible.append((float(threshold), float(metrics["specificity"])))

    if not feasible:
        return 0.0

    return max(feasible, key=lambda item: (item[0], item[1]))[0]


def multiclass_metrics(
    y_true: np.ndarray,
    probabilities: pd.DataFrame,
    weights: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true)
    weights = np.asarray(weights, dtype=float)
    p = probabilities[STATE_ORDER].to_numpy(dtype=float)
    pred = np.asarray(STATE_ORDER, dtype=object)[np.argmax(p, axis=1)]

    result = {
        "accuracy": float(accuracy_score(y_true, pred, sample_weight=weights)),
        "macro_f1": float(f1_score(y_true, pred, labels=STATE_ORDER, average="macro")),
        "multiclass_log_loss": float(
            np.average(
                -np.log(
                    np.clip(
                        p[
                            np.arange(len(y_true)),
                            np.array([STATE_ORDER.index(state) for state in y_true]),
                        ],
                        1e-12,
                        1.0,
                    )
                ),
                weights=weights,
            )
        ),
    }

    for i, state in enumerate(STATE_ORDER):
        binary = (y_true == state).astype(int)
        if len(np.unique(binary)) >= 2:
            result[f"auc_{state}"] = float(
                roc_auc_score(binary, p[:, i], sample_weight=weights)
            )
            result[f"brier_{state}"] = float(
                brier_score_loss(binary, p[:, i], sample_weight=weights)
            )
        else:
            result[f"auc_{state}"] = np.nan
            result[f"brier_{state}"] = np.nan

    cm = confusion_matrix(y_true, pred, labels=STATE_ORDER)
    result["confusion_matrix"] = cm.tolist()
    return result


# =====================================================================
# CALIBRATION BINS
# =====================================================================


def weighted_calibration_bins(
    y_true: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    weights = np.asarray(weights, dtype=float)

    edges = np.quantile(probability, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = 0.0
    edges[-1] = 1.0
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(0.0, 1.0, min(n_bins, 5) + 1)

    bin_index = np.digitize(probability, edges[1:-1], right=False)
    rows = []
    for i in range(len(edges) - 1):
        mask = bin_index == i
        if not np.any(mask):
            continue
        w = weights[mask]
        rows.append(
            {
                "bin": int(i + 1),
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "n": int(mask.sum()),
                "weight_sum": float(w.sum()),
                "mean_predicted_probability": float(np.average(probability[mask], weights=w)),
                "observed_exceedance_frequency": float(np.average(y_true[mask], weights=w)),
            }
        )
    return pd.DataFrame(rows)


# =====================================================================
# DATA ASSEMBLY
# =====================================================================


def load_analysis_table() -> pd.DataFrame:
    print(f"Reading {SIGNAL_METRICS_PATH}")
    signal_metrics = pd.read_parquet(SIGNAL_METRICS_PATH)
    print(f"Reading {INVESTMENT_METRICS_PATH}")
    investment_metrics = pd.read_parquet(INVESTMENT_METRICS_PATH)
    print(f"Reading {PARAMETERS_PATH}")
    parameters = pd.read_parquet(PARAMETERS_PATH)

    metadata = parameters.copy()
    required = [
        "case_id",
        "country",
        "horizon_years",
        "cluster_method",
        "representation_method",
        PROXY_WEIGHT_FIELD,
    ]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise ValueError(f"Missing required parameter columns: {missing}")

    metadata["horizon_years"] = (
        pd.to_numeric(metadata["horizon_years"], errors="coerce").round().astype("Int64")
    )
    metadata["proxy_weight"] = pd.to_numeric(metadata[PROXY_WEIGHT_FIELD], errors="coerce")

    metadata = metadata.loc[
        metadata["country"].isin(COUNTRIES)
        & metadata["horizon_years"].isin(INCLUDED_HORIZONS)
        & metadata["cluster_method"].astype(str).str.lower().eq(CLUSTER_METHOD.lower())
        & metadata["representation_method"].astype(str).str.lower().eq(REPRESENTATION_METHOD.lower())
    ].copy()

    if PROXY_WEIGHT == "positive":
        metadata = metadata.loc[metadata["proxy_weight"] > 0].copy()
    elif PROXY_WEIGHT == "all":
        pass
    elif isinstance(PROXY_WEIGHT, (int, float)):
        metadata = metadata.loc[np.isclose(metadata["proxy_weight"], float(PROXY_WEIGHT))].copy()
    else:
        raise ValueError("PROXY_WEIGHT must be 'all', 'positive', or numeric.")

    target = investment_metrics.loc[
        investment_metrics["metric"] == "ldes_capacity_error_signed",
        ["case_id", "value"],
    ].rename(columns={"value": "target"})

    if target["case_id"].duplicated().any():
        raise ValueError("More than one signed LDES capacity-error target per case.")

    diagnostics = signal_metrics.loc[
        signal_metrics["error_family"].isin(ALLOWED_ERROR_FAMILIES)
        & signal_metrics["signal_type"].isin(ALLOWED_SIGNAL_TYPES)
        & signal_metrics["metric"].isin(ALLOWED_METRICS)
    ].copy()

    normalised_mask = diagnostics["metric"].isin(["nmbe", "nrmse"]) & (
        diagnostics["normalisation_basis"] == NORMALISATION_BASIS
    )
    pearson_mask = (diagnostics["metric"] == "pearson") & diagnostics[
        "normalisation_basis"
    ].isna()
    diagnostics = diagnostics.loc[normalised_mask | pearson_mask].copy()

    forbidden_families = {"reference_approximation", "cem"}
    if diagnostics["error_family"].isin(forbidden_families).any():
        raise RuntimeError("Reference-CEM-dependent metrics entered predictor table.")

    diagnostics["feature"] = diagnostics.apply(make_feature_name, axis=1)
    if diagnostics.duplicated(["case_id", "feature"], keep=False).any():
        raise ValueError("Duplicate case/feature metric records were found.")

    X_wide = diagnostics.pivot(index="case_id", columns="feature", values="value")
    X_wide.columns.name = None

    missing_predictors = [feature for feature in PREDICTORS if feature not in X_wide.columns]
    if missing_predictors:
        raise ValueError(
            "Selected diagnostic predictors were not found:\n" + "\n".join(missing_predictors)
        )

    base_columns = ["case_id", "country", "horizon_years", "proxy_weight"]
    optional_columns = [
        column for column in ["start_date", "end_date"] if column in metadata.columns
    ]

    analysis = (
        metadata[base_columns + optional_columns]
        .merge(target, on="case_id", how="inner", validate="one_to_one")
        .set_index("case_id")
        .join(X_wide[PREDICTORS], how="inner")
    )

    complete = np.isfinite(analysis["target"].to_numpy(dtype=float)) & np.all(
        np.isfinite(analysis[PREDICTORS].to_numpy(dtype=float)), axis=1
    )
    analysis = analysis.loc[complete].copy()

    analysis["state"] = make_state_labels(analysis["target"].to_numpy(dtype=float))
    analysis["exceeds_tolerance"] = make_binary_labels(
        analysis["target"].to_numpy(dtype=float)
    )

    print("\nCases by country/state:")
    print(pd.crosstab(analysis["country"], analysis["state"]).to_string())
    print("\nBinary exceedance prevalence:")
    print(analysis.groupby("country")["exceeds_tolerance"].mean().to_string())

    return analysis




# =====================================================================
# FULL-PIPELINE CROSS-FITTING
# =====================================================================


def make_country_state_strata(countries: np.ndarray, states: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{country}|{state}" for country, state in zip(countries, states)],
        dtype=str,
    )


def cross_fitted_full_model_b(
    X: np.ndarray,
    y_signed: np.ndarray,
    y_state: np.ndarray,
    horizons: np.ndarray,
    *,
    stratify_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Nested cross-fit the complete continuous + three-state pipeline.

    For each outer fold, the continuous model is fitted only on the outer
    training cases. The calibration model is trained from inner out-of-fold
    continuous predictions generated within that outer training set. Therefore
    the outer test cases are unseen by both stages of the fitted pipeline.
    """
    X = np.asarray(X, dtype=float)
    y_signed = np.asarray(y_signed, dtype=float)
    y_state = np.asarray(y_state, dtype=str)
    horizons = np.asarray(horizons, dtype=float)

    if stratify_labels is None:
        stratify_labels = y_state
    stratify_labels = np.asarray(stratify_labels, dtype=str)

    signed_prediction = np.full(len(y_signed), np.nan, dtype=float)
    probabilities = pd.DataFrame(
        np.nan,
        index=np.arange(len(y_signed)),
        columns=[STATE_UNDER, STATE_ACCEPTABLE, STATE_OVER, "risk"],
    )

    outer = make_stratified_folds(stratify_labels)
    dummy = np.zeros((len(y_signed), 1))

    for outer_train, outer_test in outer.split(dummy, stratify_labels):
        X_train = X[outer_train]
        y_train_signed = y_signed[outer_train]
        y_train_state = y_state[outer_train]
        h_train = horizons[outer_train]
        w_train = make_weights(h_train)

        # Inner OOF continuous scores for calibration fitting.
        inner_strata = stratify_labels[outer_train]
        inner_oof_signed = cross_fitted_continuous_predictions(
            X_train,
            y_train_signed,
            h_train,
            inner_strata,
        )
        calibrator = fit_three_state_calibrator(
            inner_oof_signed,
            y_train_state,
            w_train,
        )

        # Outer continuous model and genuinely unseen outer predictions.
        beta = fit_wls(X_train, y_train_signed, w_train)
        outer_signed = predict_linear(X[outer_test], beta)
        outer_probs = predict_three_state_calibrator(calibrator, outer_signed)

        signed_prediction[outer_test] = outer_signed
        probabilities.iloc[outer_test] = outer_probs.to_numpy()

    if not np.all(np.isfinite(signed_prediction)):
        raise RuntimeError("Nested OOF continuous predictions contain non-finite values.")
    if probabilities.isna().any().any():
        raise RuntimeError("Nested OOF three-state probabilities contain NaNs.")

    return signed_prediction, probabilities


# =====================================================================
# PLOTTING
# =====================================================================


def plot_reliability_curves(
    curves: list[dict],
    *,
    title: str,
    png_path: Path,
    pdf_path: Path,
    annotation_lines: list[str] | None = None,
    bins_stage: str,
) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.0,
        color="0.35",
        label="Perfect calibration",
        zorder=1,
    )

    all_bins = []
    for curve in curves:
        bins = weighted_calibration_bins(
            curve["y"],
            curve["probability"],
            curve["weights"],
            N_CALIBRATION_BINS,
        )
        bins.insert(0, "dataset", curve["name"])
        bins.insert(0, "stage", bins_stage)
        all_bins.append(bins)

        ax.plot(
            bins["mean_predicted_probability"],
            bins["observed_exceedance_frequency"],
            marker="o",
            markersize=5.5,
            linewidth=1.4,
            color=curve["color"],
            alpha=0.84,
            label=curve["label"],
            zorder=3,
        )

    if annotation_lines:
        ax.text(
            0.04,
            0.96,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )

    ax.set_title(title)
    ax.set_xlabel(
        f"Predicted probability of |LDES capacity error| > {ERROR_TOLERANCE_PP:g}%"
    )
    ax.set_ylabel("Observed exceedance frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return pd.concat(all_bins, ignore_index=True)


def plot_three_state_mapping(
    model_b: Pipeline,
    signed_values: np.ndarray,
    *,
    title: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    cmap = plt.get_cmap("plasma")
    under_color = cmap(PLASMA_UNDER_POSITION)
    over_color = cmap(PLASMA_OVER_POSITION)

    min_pp = float(np.nanmin(signed_values) * 100.0)
    max_pp = float(np.nanmax(signed_values) * 100.0)
    padding = 0.05 * max(max_pp - min_pp, 1.0)
    grid_pp = np.linspace(min_pp - padding, max_pp + padding, 500)
    grid_fraction = grid_pp / 100.0
    probs = predict_three_state_calibrator(model_b, grid_fraction)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(
        grid_pp,
        probs[STATE_UNDER],
        color=under_color,
        linewidth=2.0,
        label=f"Error < -{ERROR_TOLERANCE_PP:g}%",
    )
    ax.plot(
        grid_pp,
        probs[STATE_ACCEPTABLE],
        color=ACCEPTABLE_GREY,
        linewidth=2.0,
        label=f"Error within ±{ERROR_TOLERANCE_PP:g}%",
    )
    ax.plot(
        grid_pp,
        probs[STATE_OVER],
        color=over_color,
        linewidth=2.0,
        label=f"Error > +{ERROR_TOLERANCE_PP:g}%",
    )
    ax.axvline(0.0, color="0.75", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Predicted signed LDES capacity error from continuous diagnostic (%)")
    ax.set_ylabel("Calibrated probability")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# EQUATION / SERIALISATION HELPERS
# =====================================================================


def pipeline_to_dict(model: Pipeline) -> dict:
    scaler: StandardScaler = model.named_steps["scale"]
    logistic: LogisticRegression = model.named_steps["logistic"]
    return {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classes": logistic.classes_.tolist(),
        "intercept": logistic.intercept_.tolist(),
        "coefficients": logistic.coef_.tolist(),
    }


def raw_multinomial_equations(model: Pipeline) -> dict[str, dict[str, float]]:
    """Convert the standardised softmax model to raw e_hat_pp polynomial logits."""
    scaler: StandardScaler = model.named_steps["scale"]
    logistic: LogisticRegression = model.named_steps["logistic"]

    raw_coef = logistic.coef_ / scaler.scale_[None, :]
    raw_intercept = logistic.intercept_ - np.sum(
        logistic.coef_ * (scaler.mean_ / scaler.scale_)[None, :],
        axis=1,
    )

    out = {}
    for i, state in enumerate(logistic.classes_):
        out[str(state)] = {
            "intercept": float(raw_intercept[i]),
            "predicted_error_pp": float(raw_coef[i, 0]),
            "predicted_error_pp_squared": float(raw_coef[i, 1]),
        }
    return out


def write_final_equations(
    beta: np.ndarray,
    model_terms: list[str],
    calibrator: Pipeline,
) -> dict:
    # Continuous equation directly in percentage-point units.
    beta_pp = 100.0 * np.asarray(beta, dtype=float)
    softmax = raw_multinomial_equations(calibrator)

    lines = []
    lines.append("FINAL NL+BE THREE-STATE LDES RISK DIAGNOSTIC")
    lines.append("=" * 58)
    lines.append("")
    lines.append("Stage 1: continuous signed-error diagnostic")
    lines.append("--------------------------------------------")
    continuous = f"e_hat_pp = {beta_pp[0]:+.10g}"
    for i, coefficient in enumerate(beta_pp[1:], start=1):
        continuous += f" {coefficient:+.10g} * x{i}"
    lines.append(continuous)
    lines.append("")
    lines.append("where:")
    for i, term in enumerate(model_terms, start=1):
        lines.append(f"  x{i} = {term}")

    lines.append("")
    lines.append("Stage 2: three-state multinomial calibration")
    lines.append("--------------------------------------------")
    lines.append("For e = e_hat_pp, define one logit per state:")
    for state in STATE_ORDER:
        coeff = softmax[state]
        lines.append(
            f"  z_{state} = {coeff['intercept']:+.10g} "
            f"{coeff['predicted_error_pp']:+.10g} * e "
            f"{coeff['predicted_error_pp_squared']:+.10g} * e^2"
        )
    lines.append("")
    lines.append("Convert logits to probabilities with softmax:")
    lines.append("  P(state=s) = exp(z_s) / sum_j exp(z_j)")
    lines.append("")
    lines.append(f"  under      : true LDES error < -{ERROR_TOLERANCE_PP:g}%")
    lines.append(
        f"  acceptable : -{ERROR_TOLERANCE_PP:g}% <= true LDES error <= +{ERROR_TOLERANCE_PP:g}%"
    )
    lines.append(f"  over       : true LDES error > +{ERROR_TOLERANCE_PP:g}%")
    lines.append("")
    lines.append("Overall exceedance risk = P(under) + P(over).")

    FINAL_EQUATIONS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Compact LaTeX fragment for manuscript use.
    tex = []
    tex.append(r"% Auto-generated final NL+BE diagnostic equations")
    tex.append(r"\begin{align}")
    rhs = f"{beta_pp[0]:+.6g}"
    for i, coefficient in enumerate(beta_pp[1:], start=1):
        rhs += f" {coefficient:+.6g} x_{{{i}}}"
    tex.append(rf"\hat{{e}}_{{\mathrm{{pp}}}} &= {rhs} \\")
    tex.append(r"\end{align}")
    tex.append("")
    tex.append(r"\begin{align}")
    for state in STATE_ORDER:
        coeff = softmax[state]
        tex.append(
            rf"z_{{\mathrm{{{state}}}}} &= "
            rf"{coeff['intercept']:+.6g} "
            rf"{coeff['predicted_error_pp']:+.6g}\hat{{e}}_{{\mathrm{{pp}}}} "
            rf"{coeff['predicted_error_pp_squared']:+.6g}\hat{{e}}_{{\mathrm{{pp}}}}^2 \\")
    tex.append(r"\end{align}")
    tex.append("")
    tex.append(
        r"\[P(s)=\frac{\exp(z_s)}{\sum_j \exp(z_j)},\qquad "
        r"P(|e|>T)=P(\mathrm{under})+P(\mathrm{over}).\]"
    )
    FINAL_EQUATIONS_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")

    return {
        "continuous_equation_percentage_points": {
            "intercept": float(beta_pp[0]),
            "terms": [
                {"term": term, "coefficient": float(coefficient)}
                for term, coefficient in zip(model_terms, beta_pp[1:])
            ],
        },
        "three_state_raw_logit_equations": softmax,
        "softmax": "P(state=s) = exp(z_s) / sum_j exp(z_j)",
    }


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    analysis = load_analysis_table()
    nl = analysis.loc[analysis["country"] == TRAIN_COUNTRY].copy()
    be = analysis.loc[analysis["country"] == VALIDATION_COUNTRY].copy()

    if nl.empty or be.empty:
        raise ValueError("NL training or BE validation subset is empty.")

    cmap = plt.get_cmap("plasma")
    nl_color = cmap(PLASMA_NL_POSITION)
    be_color = cmap(PLASMA_BE_POSITION)

    # ================================================================
    # PART A: TRUE CROSS-COUNTRY VALIDATION, NL -> BE
    # ================================================================
    X_nl_raw = nl[PREDICTORS].to_numpy(dtype=float)
    X_be_raw = be[PREDICTORS].to_numpy(dtype=float)
    y_nl_signed = nl["target"].to_numpy(dtype=float)
    y_be_signed = be["target"].to_numpy(dtype=float)
    y_nl_state = nl["state"].to_numpy(dtype=str)
    y_be_state = be["state"].to_numpy(dtype=str)
    y_nl_binary = nl["exceeds_tolerance"].to_numpy(dtype=int)
    y_be_binary = be["exceeds_tolerance"].to_numpy(dtype=int)
    h_nl = nl["horizon_years"].to_numpy(dtype=float)
    h_be = be["horizon_years"].to_numpy(dtype=float)
    w_nl = make_weights(h_nl)
    w_be = make_weights(h_be)

    nl_reparam = choose_vif_reparameterisation(
        X_nl_raw,
        PREDICTORS,
        w_nl,
        threshold=VIF_THRESHOLD,
    )
    X_nl = nl_reparam["X"]
    nl_terms = nl_reparam["terms"]

    if nl_reparam["transformed"]:
        X_be, be_terms, _ = build_window_contrast_design(
            X_be_raw,
            PREDICTORS,
            nl_reparam["transformed_groups"],
        )
        if be_terms != nl_terms:
            raise RuntimeError("NL and BE transformation definitions do not match.")
    else:
        X_be = X_be_raw

    # Honest internal NL estimate of the entire two-stage pipeline.
    nl_nested_signed, nl_nested_states = cross_fitted_full_model_b(
        X_nl,
        y_nl_signed,
        y_nl_state,
        h_nl,
        stratify_labels=y_nl_state,
    )
    nl_nested_risk = nl_nested_states["risk"].to_numpy(dtype=float)

    # Final NL diagnostic used for the untouched BE holdout.
    nl_oof_signed_for_calibration = cross_fitted_continuous_predictions(
        X_nl,
        y_nl_signed,
        h_nl,
        y_nl_state,
    )
    nl_calibrator = fit_three_state_calibrator(
        nl_oof_signed_for_calibration,
        y_nl_state,
        w_nl,
    )
    nl_beta = fit_wls(X_nl, y_nl_signed, w_nl)
    be_signed_from_nl = predict_linear(X_be, nl_beta)
    be_holdout_states = predict_three_state_calibrator(nl_calibrator, be_signed_from_nl)
    be_holdout_risk = be_holdout_states["risk"].to_numpy(dtype=float)

    nl_prevalence = float(np.average(y_nl_binary, weights=w_nl))
    validation_nl_binary_metrics = probability_metrics(
        y_nl_binary,
        nl_nested_risk,
        w_nl,
        null_probability=nl_prevalence,
    )
    validation_be_binary_metrics = probability_metrics(
        y_be_binary,
        be_holdout_risk,
        w_be,
        null_probability=nl_prevalence,
    )
    validation_nl_multi = multiclass_metrics(y_nl_state, nl_nested_states, w_nl)
    validation_be_multi = multiclass_metrics(y_be_state, be_holdout_states, w_be)

    validation_threshold = choose_threshold_for_sensitivity(
        y_nl_binary,
        nl_nested_risk,
        w_nl,
        TARGET_TRAIN_SENSITIVITY,
    )
    validation_screening = []
    for dataset, y, p, w in [
        ("NL_nested_crossfit", y_nl_binary, nl_nested_risk, w_nl),
        ("BE_holdout", y_be_binary, be_holdout_risk, w_be),
    ]:
        row = {"dataset": dataset}
        row.update(threshold_metrics(y, p, w, validation_threshold))
        validation_screening.append(row)
    pd.DataFrame(validation_screening).to_csv(VALIDATION_SCREENING_PATH, index=False)

    validation_summary = pd.DataFrame(
        [
            {
                "dataset": "NL_nested_crossfit",
                **validation_nl_binary_metrics,
                **{f"multiclass_{k}": v for k, v in validation_nl_multi.items() if k != "confusion_matrix"},
            },
            {
                "dataset": "BE_holdout",
                **validation_be_binary_metrics,
                **{f"multiclass_{k}": v for k, v in validation_be_multi.items() if k != "confusion_matrix"},
            },
        ]
    )
    validation_summary.to_csv(VALIDATION_SUMMARY_PATH, index=False)

    nl_validation_out = nl[
        ["country", "horizon_years", "proxy_weight", "target", "state", "exceeds_tolerance"]
    ].copy()
    nl_validation_out["continuous_prediction_pp"] = 100.0 * nl_nested_signed
    nl_validation_out["prediction_source"] = "nested cross-fit"
    nl_validation_out["p_under"] = nl_nested_states[STATE_UNDER].to_numpy()
    nl_validation_out["p_acceptable"] = nl_nested_states[STATE_ACCEPTABLE].to_numpy()
    nl_validation_out["p_over"] = nl_nested_states[STATE_OVER].to_numpy()
    nl_validation_out["p_exceed"] = nl_nested_risk

    be_validation_out = be[
        ["country", "horizon_years", "proxy_weight", "target", "state", "exceeds_tolerance"]
    ].copy()
    be_validation_out["continuous_prediction_pp"] = 100.0 * be_signed_from_nl
    be_validation_out["prediction_source"] = "NL model applied unchanged"
    be_validation_out["p_under"] = be_holdout_states[STATE_UNDER].to_numpy()
    be_validation_out["p_acceptable"] = be_holdout_states[STATE_ACCEPTABLE].to_numpy()
    be_validation_out["p_over"] = be_holdout_states[STATE_OVER].to_numpy()
    be_validation_out["p_exceed"] = be_holdout_risk
    pd.concat([nl_validation_out, be_validation_out]).to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=True,
    )

    validation_bins = plot_reliability_curves(
        [
            {
                "name": "NL nested cross-fit",
                "label": "NL nested cross-fit",
                "y": y_nl_binary,
                "probability": nl_nested_risk,
                "weights": w_nl,
                "color": nl_color,
            },
            {
                "name": "BE holdout",
                "label": "BE validation",
                "y": y_be_binary,
                "probability": be_holdout_risk,
                "weights": w_be,
                "color": be_color,
            },
        ],
        title="Three-state diagnostic: NL to BE validation",
        png_path=VALIDATION_CAL_PNG,
        pdf_path=VALIDATION_CAL_PDF,
        annotation_lines=[
            f"NL AUC = {validation_nl_binary_metrics['roc_auc']:.2f}, Brier = {validation_nl_binary_metrics['brier_score']:.3f}",
            f"BE AUC = {validation_be_binary_metrics['roc_auc']:.2f}, Brier = {validation_be_binary_metrics['brier_score']:.3f}",
        ],
        bins_stage="NL_to_BE_validation",
    )
    validation_bins.to_csv(VALIDATION_BINS_PATH, index=False)

    plot_three_state_mapping(
        nl_calibrator,
        np.r_[nl_oof_signed_for_calibration, be_signed_from_nl],
        title="NL-trained three-state diagnostic",
        png_path=VALIDATION_MAP_PNG,
        pdf_path=VALIDATION_MAP_PDF,
    )

    # ================================================================
    # PART B: FINAL PROPOSED DIAGNOSTIC, REFIT TO NL + BE
    # ================================================================
    combined = analysis.loc[analysis["country"].isin([TRAIN_COUNTRY, VALIDATION_COUNTRY])].copy()
    X_all_raw = combined[PREDICTORS].to_numpy(dtype=float)
    y_all_signed = combined["target"].to_numpy(dtype=float)
    y_all_state = combined["state"].to_numpy(dtype=str)
    y_all_binary = combined["exceeds_tolerance"].to_numpy(dtype=int)
    h_all = combined["horizon_years"].to_numpy(dtype=float)
    countries_all = combined["country"].to_numpy(dtype=str)
    w_all = make_weights(h_all)

    combined_reparam = choose_vif_reparameterisation(
        X_all_raw,
        PREDICTORS,
        w_all,
        threshold=VIF_THRESHOLD,
    )
    X_all = combined_reparam["X"]
    combined_terms = combined_reparam["terms"]
    combined_strata = make_country_state_strata(countries_all, y_all_state)

    # Internal validation of the complete final formulation.
    combined_cv_signed, combined_cv_states = cross_fitted_full_model_b(
        X_all,
        y_all_signed,
        y_all_state,
        h_all,
        stratify_labels=combined_strata,
    )
    combined_cv_risk = combined_cv_states["risk"].to_numpy(dtype=float)

    # Final deployable diagnostic.
    combined_oof_signed_for_calibration = cross_fitted_continuous_predictions(
        X_all,
        y_all_signed,
        h_all,
        combined_strata,
    )
    final_calibrator = fit_three_state_calibrator(
        combined_oof_signed_for_calibration,
        y_all_state,
        w_all,
    )
    final_beta = fit_wls(X_all, y_all_signed, w_all)
    final_apparent_signed = predict_linear(X_all, final_beta)
    final_apparent_states = predict_three_state_calibrator(
        final_calibrator,
        final_apparent_signed,
    )
    final_apparent_risk = final_apparent_states["risk"].to_numpy(dtype=float)

    combined_prevalence = float(np.average(y_all_binary, weights=w_all))

    # Metrics helper for full sample and country subsets.
    final_metric_rows = []
    for evaluation, signed_pred, states_df, risk in [
        ("nested_crossfit", combined_cv_signed, combined_cv_states, combined_cv_risk),
        ("apparent_final_fit", final_apparent_signed, final_apparent_states, final_apparent_risk),
    ]:
        for subset_name, mask in [
            ("NL+BE", np.ones(len(combined), dtype=bool)),
            ("NL", countries_all == "NL"),
            ("BE", countries_all == "BE"),
        ]:
            weights_subset = w_all[mask]
            binary_metrics = probability_metrics(
                y_all_binary[mask],
                risk[mask],
                weights_subset,
                null_probability=combined_prevalence,
            )
            multi = multiclass_metrics(
                y_all_state[mask],
                states_df.iloc[np.where(mask)[0]].reset_index(drop=True),
                weights_subset,
            )
            final_metric_rows.append(
                {
                    "evaluation": evaluation,
                    "subset": subset_name,
                    **binary_metrics,
                    **{f"multiclass_{k}": v for k, v in multi.items() if k != "confusion_matrix"},
                }
            )

    final_summary = pd.DataFrame(final_metric_rows)
    final_summary.to_csv(FINAL_SUMMARY_PATH, index=False)

    final_out = combined[
        ["country", "horizon_years", "proxy_weight", "target", "state", "exceeds_tolerance"]
    ].copy()
    final_out["continuous_prediction_crossfit_pp"] = 100.0 * combined_cv_signed
    final_out["p_under_crossfit"] = combined_cv_states[STATE_UNDER].to_numpy()
    final_out["p_acceptable_crossfit"] = combined_cv_states[STATE_ACCEPTABLE].to_numpy()
    final_out["p_over_crossfit"] = combined_cv_states[STATE_OVER].to_numpy()
    final_out["p_exceed_crossfit"] = combined_cv_risk
    final_out["continuous_prediction_final_fit_pp"] = 100.0 * final_apparent_signed
    final_out["p_under_final_fit"] = final_apparent_states[STATE_UNDER].to_numpy()
    final_out["p_acceptable_final_fit"] = final_apparent_states[STATE_ACCEPTABLE].to_numpy()
    final_out["p_over_final_fit"] = final_apparent_states[STATE_OVER].to_numpy()
    final_out["p_exceed_final_fit"] = final_apparent_risk
    final_out.to_csv(FINAL_PREDICTIONS_PATH, index=True)

    # Final-model calibration shown on the data used to fit it. This is
    # deliberately labelled apparent/in-sample and is not validation evidence.
    final_apparent_bins = plot_reliability_curves(
        [
            {
                "name": "NL+BE combined",
                "label": "NL+BE combined (in-sample)",
                "y": y_all_binary,
                "probability": final_apparent_risk,
                "weights": w_all,
                "color": COMBINED_GREY,
            },
            {
                "name": "NL subset",
                "label": "NL subset",
                "y": y_all_binary[countries_all == "NL"],
                "probability": final_apparent_risk[countries_all == "NL"],
                "weights": w_all[countries_all == "NL"],
                "color": nl_color,
            },
            {
                "name": "BE subset",
                "label": "BE subset",
                "y": y_all_binary[countries_all == "BE"],
                "probability": final_apparent_risk[countries_all == "BE"],
                "weights": w_all[countries_all == "BE"],
                "color": be_color,
            },
        ],
        title="Final NL+BE diagnostic: apparent calibration",
        png_path=FINAL_APPARENT_CAL_PNG,
        pdf_path=FINAL_APPARENT_CAL_PDF,
        annotation_lines=["Descriptive only: all cases used for final fitting"],
        bins_stage="final_combined_apparent",
    )

    # More defensible internal estimate for the same combined-data formulation.
    final_cv_bins = plot_reliability_curves(
        [
            {
                "name": "NL+BE combined",
                "label": "NL+BE nested cross-fit",
                "y": y_all_binary,
                "probability": combined_cv_risk,
                "weights": w_all,
                "color": COMBINED_GREY,
            },
            {
                "name": "NL subset",
                "label": "NL subset",
                "y": y_all_binary[countries_all == "NL"],
                "probability": combined_cv_risk[countries_all == "NL"],
                "weights": w_all[countries_all == "NL"],
                "color": nl_color,
            },
            {
                "name": "BE subset",
                "label": "BE subset",
                "y": y_all_binary[countries_all == "BE"],
                "probability": combined_cv_risk[countries_all == "BE"],
                "weights": w_all[countries_all == "BE"],
                "color": be_color,
            },
        ],
        title="Final NL+BE diagnostic: nested cross-fitted calibration",
        png_path=FINAL_CV_CAL_PNG,
        pdf_path=FINAL_CV_CAL_PDF,
        annotation_lines=["Internal validation; not an independent country holdout"],
        bins_stage="final_combined_nested_crossfit",
    )

    pd.concat([final_apparent_bins, final_cv_bins], ignore_index=True).to_csv(
        FINAL_BINS_PATH,
        index=False,
    )

    plot_three_state_mapping(
        final_calibrator,
        np.r_[combined_oof_signed_for_calibration, final_apparent_signed],
        title="Final NL+BE three-state diagnostic",
        png_path=FINAL_MAP_PNG,
        pdf_path=FINAL_MAP_PDF,
    )

    final_equations = write_final_equations(
        final_beta,
        combined_terms,
        final_calibrator,
    )

    final_screening_threshold = choose_threshold_for_sensitivity(
        y_all_binary,
        combined_cv_risk,
        w_all,
        TARGET_TRAIN_SENSITIVITY,
    )

    # ================================================================
    # SERIALISE COMPLETE MODEL / VALIDATION RECORD
    # ================================================================
    model_metadata = {
        "target": {
            "tolerance_percentage_points": ERROR_TOLERANCE_PP,
            "three_states": {
                STATE_UNDER: f"error < -{ERROR_TOLERANCE_PP:g}%",
                STATE_ACCEPTABLE: f"-{ERROR_TOLERANCE_PP:g}% <= error <= +{ERROR_TOLERANCE_PP:g}%",
                STATE_OVER: f"error > +{ERROR_TOLERANCE_PP:g}%",
            },
        },
        "predictors": PREDICTORS,
        "validation_NL_to_BE": {
            "design": "NL-only development; BE applied unchanged as cross-country holdout",
            "NL_nested_crossfit_binary_metrics": validation_nl_binary_metrics,
            "BE_holdout_binary_metrics": validation_be_binary_metrics,
            "NL_nested_crossfit_multiclass_metrics": validation_nl_multi,
            "BE_holdout_multiclass_metrics": validation_be_multi,
            "screening_threshold_selected_on_NL": float(validation_threshold),
            "continuous_coefficients_fraction_units": {
                "intercept": float(nl_beta[0]),
                "terms": [
                    {
                        "term": term,
                        "original_predictor": original,
                        "coefficient": float(nl_beta[i]),
                    }
                    for i, (term, original) in enumerate(zip(nl_terms, PREDICTORS), start=1)
                ],
            },
            "calibrator": pipeline_to_dict(nl_calibrator),
        },
        "final_NL_plus_BE": {
            "design": "final proposed diagnostic refitted using all NL+BE cases",
            "warning": "apparent calibration is descriptive and not independent validation",
            "reparameterised": bool(combined_reparam["transformed"]),
            "transformation_info": combined_reparam["transformation_info"],
            "raw_vif": {
                feature: float(value)
                for feature, value in zip(PREDICTORS, combined_reparam["raw_vif"])
            },
            "model_vif": {
                term: float(value)
                for term, value in zip(combined_terms, combined_reparam["vif"])
            },
            "continuous_coefficients_fraction_units": {
                "intercept": float(final_beta[0]),
                "terms": [
                    {
                        "term": term,
                        "original_predictor": original,
                        "coefficient": float(final_beta[i]),
                    }
                    for i, (term, original) in enumerate(zip(combined_terms, PREDICTORS), start=1)
                ],
            },
            "calibrator": pipeline_to_dict(final_calibrator),
            "equations": final_equations,
            "screening_threshold_from_nested_crossfit": float(final_screening_threshold),
            "nested_crossfit_metrics": final_summary.loc[
                final_summary["evaluation"] == "nested_crossfit"
            ].to_dict(orient="records"),
            "apparent_fit_metrics": final_summary.loc[
                final_summary["evaluation"] == "apparent_final_fit"
            ].to_dict(orient="records"),
        },
        "cross_fitting": {
            "folds_requested": CV_FOLDS,
            "random_state": CV_RANDOM_STATE,
            "validation_internal_estimate": "nested full-pipeline cross-fitting",
            "combined_stratification": "country x three-state target",
        },
    }

    with open(MODEL_PATH, "w", encoding="utf-8") as handle:
        json.dump(model_metadata, handle, indent=2)

    # ================================================================
    # CONSOLE SUMMARY
    # ================================================================
    print("\n============================================================")
    print("THREE-STATE LDES RISK DIAGNOSTIC (MODEL B)")
    print("============================================================")
    print(f"Tolerance: +/-{ERROR_TOLERANCE_PP:g} percentage points")

    print("\nA. NL -> BE cross-country validation")
    print(
        f"  NL nested CV: AUC={validation_nl_binary_metrics['roc_auc']:.3f}, "
        f"Brier={validation_nl_binary_metrics['brier_score']:.3f}"
    )
    print(
        f"  BE holdout:   AUC={validation_be_binary_metrics['roc_auc']:.3f}, "
        f"Brier={validation_be_binary_metrics['brier_score']:.3f}"
    )
    print(
        f"  BE directional AUC: under={validation_be_multi['auc_under']:.3f}, "
        f"acceptable={validation_be_multi['auc_acceptable']:.3f}, "
        f"over={validation_be_multi['auc_over']:.3f}"
    )

    print("\nB. Final NL+BE diagnostic")
    print("  Final equation files:")
    print(f"    {FINAL_EQUATIONS_TXT}")
    print(f"    {FINAL_EQUATIONS_TEX}")
    print("  Note: apparent calibration uses the fitting data; nested cross-fit is the")
    print("        more defensible internal performance estimate.")

    print("\nSaved outputs:")
    for path in [
        VALIDATION_SUMMARY_PATH,
        VALIDATION_SCREENING_PATH,
        VALIDATION_PREDICTIONS_PATH,
        VALIDATION_BINS_PATH,
        VALIDATION_CAL_PNG,
        VALIDATION_CAL_PDF,
        VALIDATION_MAP_PNG,
        VALIDATION_MAP_PDF,
        FINAL_SUMMARY_PATH,
        FINAL_PREDICTIONS_PATH,
        FINAL_BINS_PATH,
        FINAL_APPARENT_CAL_PNG,
        FINAL_APPARENT_CAL_PDF,
        FINAL_CV_CAL_PNG,
        FINAL_CV_CAL_PDF,
        FINAL_MAP_PNG,
        FINAL_MAP_PDF,
        FINAL_EQUATIONS_TXT,
        FINAL_EQUATIONS_TEX,
        MODEL_PATH,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
