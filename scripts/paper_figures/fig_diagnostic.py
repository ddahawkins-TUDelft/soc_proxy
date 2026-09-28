"""End-to-end NL -> BE validation of the SoC diagnostic.

This script:
1. Loads signal metrics, investment metrics, and case parameters.
2. Reconstructs the manually specified diagnostic predictors.
3. Fits the diagnostic on NL only.
4. Applies the NL-fitted coefficients unchanged to BE.
5. Reports training and cross-country validation performance.
6. Saves coefficients, predictions, summary statistics, and the fitted model.
7. Produces a publication-ready observed-vs-predicted validation figure.

The figure uses circular markers for both datasets, plasma blue for NL,
plasma pink for BE, no covariance ellipses, and a neutral 1:1 line.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================================
# CONFIGURATION
# =====================================================================

RESULTS_DIR = Path("results") / "2_5_10_year"
OUTPUT_DIR = Path("results") / "figures/diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_METRICS_PATH = RESULTS_DIR / "signal_metrics.parquet"
INVESTMENT_METRICS_PATH = RESULTS_DIR / "investment_metrics.parquet"
PARAMETERS_PATH = RESULTS_DIR / "parameters.parquet"

# Current diagnostic is evaluated for the 10-year cases.
INCLUDED_HORIZONS = [10]

# If multiple horizons are included, give each horizon equal total weight.
EQUAL_HORIZON_WEIGHTING = False

# Final manually/theoretically specified diagnostic predictors.
# Keep these fixed before evaluating BE if BE is to remain a true holdout.
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
# "all", "positive", or a numeric weight such as 0.5
PROXY_WEIGHT: str | float = "all"

ALLOWED_ERROR_FAMILIES = ["clustered_approximation", "tsa", "rpcc"]
ALLOWED_SIGNAL_TYPES = ["level", "delta"]
ALLOWED_METRICS = ["nmbe", "nrmse", "pearson"]
NORMALISATION_BASIS = "reference_proxy_full_range"

PLOT_PERCENTAGE_POINTS = True

# Plasma positions chosen to avoid the yellow end of the colormap.
PLASMA_TRAIN_POSITION = 0.05  # deep blue
PLASMA_VALIDATION_POSITION = 0.50  # pink / magenta

OUTPUT_STEM = "diagnostic_manual_nl_to_be"
SUMMARY_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_summary.csv"
COEFFICIENTS_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_coefficients.csv"
PREDICTIONS_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_predictions.csv"
BY_HORIZON_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_by_horizon.csv"
MODEL_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_model.json"
FIGURE_PNG_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_observed_vs_predicted.png"
FIGURE_PDF_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_observed_vs_predicted.pdf"


# =====================================================================
# FEATURE HELPERS
# =====================================================================


def window_label(value) -> str:
    if pd.isna(value):
        return "full"
    return f"{int(value)}d"


def make_feature_name(row: pd.Series) -> str:
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
# LINEAR MODEL + VIF
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
    threshold: float = 5.0,
) -> dict:
    """Choose a physically interpretable window-contrast reparameterisation."""
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
        best_X = None
        best_terms = None
        best_info = None
        best_vif = None

        for group_key in remaining_groups:
            candidate_groups = active_groups | {group_key}
            candidate_X, candidate_terms, candidate_info = build_window_contrast_design(
                X_raw, features, candidate_groups
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
        "weighted_r2": float(r2_score(y_true, y_pred, sample_weight=weights)),
        "unweighted_r2": float(r2_score(y_true, y_pred)),
        "weighted_rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=weights))
        ),
        "weighted_mae": float(
            mean_absolute_error(y_true, y_pred, sample_weight=weights)
        ),
        "weighted_mean_prediction_error": float(np.average(error, weights=weights)),
    }


# =====================================================================
# LOAD + ASSEMBLE ANALYSIS TABLE
# =====================================================================


def load_analysis_table() -> pd.DataFrame:
    print(f"Reading {SIGNAL_METRICS_PATH}")
    signal_metrics = pd.read_parquet(SIGNAL_METRICS_PATH)

    print(f"Reading {INVESTMENT_METRICS_PATH}")
    investment_metrics = pd.read_parquet(INVESTMENT_METRICS_PATH)

    print(f"Reading {PARAMETERS_PATH}")
    parameters = pd.read_parquet(PARAMETERS_PATH)

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
        raise ValueError(f"Missing required parameter columns: {missing}")

    metadata["horizon_years"] = (
        pd.to_numeric(metadata["horizon_years"], errors="coerce")
        .round()
        .astype("Int64")
    )
    metadata["proxy_weight"] = pd.to_numeric(
        metadata[PROXY_WEIGHT_FIELD], errors="coerce"
    )

    metadata = metadata.loc[
        metadata["country"].isin(COUNTRIES)
        & metadata["horizon_years"].isin(INCLUDED_HORIZONS)
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
    elif isinstance(PROXY_WEIGHT, (int, float)):
        metadata = metadata.loc[
            np.isclose(metadata["proxy_weight"], float(PROXY_WEIGHT))
        ].copy()
    else:
        raise ValueError("PROXY_WEIGHT must be 'all', 'positive', or numeric.")

    if metadata.empty:
        raise ValueError("No cases remain after filtering.")

    print("\nCases after filtering:")
    print(metadata.groupby(["country", "horizon_years"]).size().rename("n").to_string())

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

    # Guard against target/reference-CEM leakage into predictors.
    forbidden_families = {"reference_approximation", "cem"}
    if diagnostics["error_family"].isin(forbidden_families).any():
        raise RuntimeError("Reference-CEM-dependent metrics entered predictor table.")

    diagnostics["feature"] = diagnostics.apply(make_feature_name, axis=1)

    duplicate_features = diagnostics.duplicated(["case_id", "feature"], keep=False)
    if duplicate_features.any():
        raise ValueError("Duplicate case/feature metric records were found.")

    X_wide = diagnostics.pivot(index="case_id", columns="feature", values="value")
    X_wide.columns.name = None

    missing_predictors = [
        feature for feature in PREDICTORS if feature not in X_wide.columns
    ]
    if missing_predictors:
        raise ValueError(
            "Selected diagnostic predictors were not found:\n"
            + "\n".join(missing_predictors)
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

    complete_mask = np.isfinite(analysis["target"].to_numpy(dtype=float)) & np.all(
        np.isfinite(analysis[PREDICTORS].to_numpy(dtype=float)), axis=1
    )
    before = len(analysis)
    analysis = analysis.loc[complete_mask].copy()
    print(f"\nComplete cases: {len(analysis)} / {before}")

    return analysis


# =====================================================================
# FIT, VALIDATE, SAVE, PLOT
# =====================================================================


def main() -> None:
    analysis = load_analysis_table()

    train = analysis.loc[analysis["country"] == TRAIN_COUNTRY].copy()
    validation = analysis.loc[analysis["country"] == VALIDATION_COUNTRY].copy()

    if train.empty or validation.empty:
        raise ValueError("Training or validation subset is empty.")

    X_train_raw = train[PREDICTORS].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=float)
    horizon_train = train["horizon_years"].to_numpy(dtype=float)
    train_weights = make_weights(horizon_train)

    reparam = choose_vif_reparameterisation(
        X_train_raw,
        PREDICTORS,
        train_weights,
        threshold=VIF_THRESHOLD,
    )
    X_train = reparam["X"]
    model_terms = reparam["terms"]
    vif = reparam["vif"]

    beta = fit_wls(X_train, y_train, train_weights)
    train_prediction = predict_linear(X_train, beta)

    # Verify that any VIF transformation is a true reparameterisation.
    if reparam["transformed"]:
        raw_beta = fit_wls(X_train_raw, y_train, train_weights)
        raw_prediction = predict_linear(X_train_raw, raw_beta)
        if not np.allclose(raw_prediction, train_prediction, rtol=1e-9, atol=1e-11):
            raise RuntimeError(
                "Window-contrast reparameterisation changed predictions."
            )

    X_validation_raw = validation[PREDICTORS].to_numpy(dtype=float)
    if reparam["transformed"]:
        X_validation, validation_terms, _ = build_window_contrast_design(
            X_validation_raw,
            PREDICTORS,
            reparam["transformed_groups"],
        )
        if validation_terms != model_terms:
            raise RuntimeError("NL and BE transformation definitions do not match.")
    else:
        X_validation = X_validation_raw

    y_validation = validation["target"].to_numpy(dtype=float)
    horizon_validation = validation["horizon_years"].to_numpy(dtype=float)
    validation_prediction = predict_linear(X_validation, beta)

    train_metrics = calculate_metrics(y_train, train_prediction, horizon_train)
    validation_metrics = calculate_metrics(
        y_validation, validation_prediction, horizon_validation
    )

    print("\nNL training")
    print(f"  n    = {train_metrics['n']}")
    print(f"  R²   = {train_metrics['weighted_r2']:.4f}")
    print(f"  RMSE = {train_metrics['weighted_rmse']:.4f}")
    print(f"  MAE  = {train_metrics['weighted_mae']:.4f}")

    print("\nBE cross-country validation")
    print(f"  n    = {validation_metrics['n']}")
    print(f"  R²   = {validation_metrics['weighted_r2']:.4f}")
    print(f"  RMSE = {validation_metrics['weighted_rmse']:.4f}")
    print(f"  MAE  = {validation_metrics['weighted_mae']:.4f}")
    print(
        "  mean prediction error = "
        f"{validation_metrics['weighted_mean_prediction_error']:.4f}"
    )

    print("\nModel coefficients")
    print(f"  intercept = {beta[0]:+.8f}")
    for i, term in enumerate(model_terms, start=1):
        print(f"  beta_{i} = {beta[i]:+.8f}  [{term}]")

    # ---------------------------------------------------------------
    # Save coefficients + predictions + summaries
    # ---------------------------------------------------------------
    coefficient_rows = [
        {
            "term": "intercept",
            "model_term": "intercept",
            "original_predictor": "intercept",
            "coefficient": beta[0],
            "weighted_vif": np.nan,
        }
    ]
    for i, (original_feature, model_term, feature_vif) in enumerate(
        zip(PREDICTORS, model_terms, vif), start=1
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
    pd.DataFrame(coefficient_rows).to_csv(COEFFICIENTS_PATH, index=False)

    keep_cols = ["country", "horizon_years", "proxy_weight", "target"]
    optional_cols = [c for c in ["start_date", "end_date"] if c in analysis.columns]

    train_output = train[keep_cols + optional_cols].copy()
    train_output["prediction"] = train_prediction
    train_output["prediction_error"] = train_prediction - y_train
    train_output["dataset"] = "NL training"

    validation_output = validation[keep_cols + optional_cols].copy()
    validation_output["prediction"] = validation_prediction
    validation_output["prediction_error"] = validation_prediction - y_validation
    validation_output["dataset"] = "BE cross-country validation"

    prediction_output = pd.concat([train_output, validation_output])
    prediction_output.to_csv(PREDICTIONS_PATH, index=True)

    by_horizon_rows = []
    for dataset_name, frame, prediction in [
        ("NL training", train, train_prediction),
        ("BE cross-country validation", validation, validation_prediction),
    ]:
        horizons = frame["horizon_years"].to_numpy(dtype=float)
        observed = frame["target"].to_numpy(dtype=float)
        for horizon in sorted(np.unique(horizons)):
            mask = horizons == horizon
            metrics = calculate_metrics(
                observed[mask], prediction[mask], horizons[mask]
            )
            by_horizon_rows.append(
                {"dataset": dataset_name, "horizon_years": int(horizon), **metrics}
            )
    pd.DataFrame(by_horizon_rows).to_csv(BY_HORIZON_PATH, index=False)

    summary = pd.DataFrame(
        [
            {
                "predictor_count": len(PREDICTORS),
                "included_horizons": ",".join(str(x) for x in INCLUDED_HORIZONS),
                "equal_horizon_weighting": EQUAL_HORIZON_WEIGHTING,
                "training_country": TRAIN_COUNTRY,
                "validation_country": VALIDATION_COUNTRY,
                "training_n": train_metrics["n"],
                "training_weighted_r2": train_metrics["weighted_r2"],
                "training_weighted_rmse": train_metrics["weighted_rmse"],
                "training_weighted_mae": train_metrics["weighted_mae"],
                "validation_n": validation_metrics["n"],
                "validation_weighted_r2": validation_metrics["weighted_r2"],
                "validation_weighted_rmse": validation_metrics["weighted_rmse"],
                "validation_weighted_mae": validation_metrics["weighted_mae"],
                "validation_weighted_mean_prediction_error": validation_metrics[
                    "weighted_mean_prediction_error"
                ],
                "reparameterised": bool(reparam["transformed"]),
                "raw_max_weighted_vif": float(reparam["raw_max_vif"]),
                "model_max_weighted_vif": float(np.max(vif)),
            }
        ]
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    model_metadata = {
        "predictors": PREDICTORS,
        "model_terms": model_terms,
        "predictor_selection": "manual / theory specified",
        "included_horizons": INCLUDED_HORIZONS,
        "equal_horizon_weighting": EQUAL_HORIZON_WEIGHTING,
        "training_country": TRAIN_COUNTRY,
        "validation_country": VALIDATION_COUNTRY,
        "cluster_method": CLUSTER_METHOD,
        "representation_method": REPRESENTATION_METHOD,
        "proxy_weight": PROXY_WEIGHT,
        "reparameterised": bool(reparam["transformed"]),
        "transformation_info": reparam["transformation_info"],
        "coefficients": {
            "intercept": float(beta[0]),
            "terms": [
                {
                    "term": term,
                    "original_predictor": original_feature,
                    "coefficient": float(beta[i]),
                }
                for i, (original_feature, term) in enumerate(
                    zip(PREDICTORS, model_terms), start=1
                )
            ],
        },
        "nl_training": train_metrics,
        "be_validation": validation_metrics,
    }
    with open(MODEL_PATH, "w", encoding="utf-8") as file:
        json.dump(model_metadata, file, indent=2)

    # ---------------------------------------------------------------
    # Publication figure
    # ---------------------------------------------------------------
    scale = 100.0 if PLOT_PERCENTAGE_POINTS else 1.0
    train_x = scale * y_train
    train_y = scale * train_prediction
    validation_x = scale * y_validation
    validation_y = scale * validation_prediction

    cmap = plt.get_cmap("plasma")
    training_color = cmap(PLASMA_TRAIN_POSITION)
    validation_color = cmap(PLASMA_VALIDATION_POSITION)

    fig, ax = plt.subplots(figsize=(6.5, 6.2))

    ax.scatter(
        train_x,
        train_y,
        s=24,
        alpha=0.34,
        marker="o",
        color=training_color,
        edgecolors="none",
        label=f"NL training (n={len(train_x)})",
        zorder=3,
    )

    ax.scatter(
        validation_x,
        validation_y,
        s=24,
        alpha=0.56,
        marker="o",
        color=validation_color,
        edgecolors="none",
        label=f"BE validation (n={len(validation_x)})",
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

    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        linestyle="--",
        linewidth=1.1,
        color="0.30",
        label="1:1",
        zorder=2,
    )

    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect("equal", adjustable="box")

    unit = "%" if PLOT_PERCENTAGE_POINTS else ""
    ax.set_xlabel(f"Observed signed LDES capacity error ({unit})")
    ax.set_ylabel(f"Predicted signed LDES capacity error ({unit})")

    # Metrics are stored as fractions, so MAE is converted to percentage points.
    train_mae_pp = 100.0 * train_metrics["weighted_mae"]
    validation_mae_pp = 100.0 * validation_metrics["weighted_mae"]

    ax.text(
        0.04,
        0.96,
        (
            f"NL training $R^2$ = {train_metrics['weighted_r2']:.2f}, "
            f"MAE = {train_mae_pp:.1f} pp\n"
            f"BE validation $R^2$ = {validation_metrics['weighted_r2']:.2f}, "
            f"MAE = {validation_mae_pp:.1f} pp"
        ),
        transform=ax.transAxes,
        va="top",
    )

    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    fig.savefig(FIGURE_PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved outputs:")
    for path in [
        SUMMARY_PATH,
        COEFFICIENTS_PATH,
        PREDICTIONS_PATH,
        BY_HORIZON_PATH,
        MODEL_PATH,
        FIGURE_PNG_PATH,
        FIGURE_PDF_PATH,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
