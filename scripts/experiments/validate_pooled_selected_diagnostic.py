"""Validate a pooled-selected diagnostic by cross-country coefficient transfer.

Workflow
--------
1. Predictor structure:
   Load the best low-VIF 3-predictor model previously selected from the
   combined NL + BE ensemble using blocked weather-window CV.

2. Coefficient training:
   Fit ONLY the coefficients using NL cases.

3. Cross-country transfer:
   Apply the NL coefficients unchanged to BE cases.

4. Final refit:
   Keep the predictor structure fixed and re-estimate the coefficients
   using NL + BE together.

Interpretation
--------------
Because BE participated in predictor selection, the BE result is not an
independent external validation of the complete model-development process.

It IS a direct test of cross-country coefficient transfer for a predictor
structure selected to work across the combined NL + BE ensemble.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =====================================================================
# Configuration
# =====================================================================

RESULTS_DIR = Path("results") / "10_year"

SIGNAL_METRICS_PATH = RESULTS_DIR / "signal_metrics.parquet"
INVESTMENT_METRICS_PATH = RESULTS_DIR / "investment_metrics.parquet"
PARAMETERS_PATH = RESULTS_DIR / "parameters.parquet"

# Result produced by the pooled NL+BE combinatorial search.
POOLED_SEARCH_PATH = RESULTS_DIR / "diagnostic_search_three_predictor_raw_all.parquet"

PROXY_WEIGHT_FIELD = "lambda_soc"
PROXY_WEIGHT = "positive"

DEVELOPMENT_COUNTRY = "NL"
VALIDATION_COUNTRY = "BE"

NORMALISATION_BASIS = "reference_proxy_full_range"

MAX_VIF = 5.0


OUTPUT_SUMMARY_PATH = RESULTS_DIR / "diagnostic_pooled_selected_nl_to_be_summary.csv"

OUTPUT_PREDICTIONS_PATH = (
    RESULTS_DIR / "diagnostic_pooled_selected_nl_to_be_predictions.csv"
)

OUTPUT_COEFFICIENTS_PATH = (
    RESULTS_DIR / "diagnostic_pooled_selected_final_coefficients.csv"
)

OUTPUT_MODEL_PATH = RESULTS_DIR / "diagnostic_pooled_selected_model.json"

OUTPUT_FIGURE_PATH = RESULTS_DIR / "diagnostic_pooled_selected_nl_to_be.png"


# =====================================================================
# Helpers
# =====================================================================


def window_label(value):
    if pd.isna(value):
        return "full"

    return f"{int(value)}d"


def make_feature_name(row):
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


def fit_ols(X, y):
    """OLS with intercept."""

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


def predict_ols(X, beta):

    X_design = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    return X_design @ beta


def calculate_vif(X):
    """VIF from predictor correlation matrix."""

    X = X[
        np.all(
            np.isfinite(X),
            axis=1,
        )
    ]

    std = np.std(
        X,
        axis=0,
        ddof=1,
    )

    if np.any(std == 0):
        return np.full(
            X.shape[1],
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

    try:
        inv_corr = np.linalg.inv(corr)

    except np.linalg.LinAlgError:
        return np.full(
            X.shape[1],
            np.inf,
        )

    return np.diag(inv_corr)


def prediction_metrics(y_true, y_pred):

    residual = y_pred - y_true

    return {
        "n": len(y_true),
        "r2": r2_score(
            y_true,
            y_pred,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "mean_prediction_error": np.mean(residual),
    }


# =====================================================================
# 1. Load pooled feature-selection result
# =====================================================================

print(f"Reading pooled search results: {POOLED_SEARCH_PATH}")

search = pd.read_parquet(POOLED_SEARCH_PATH)


acceptable = (
    search.loc[np.isfinite(search["max_vif"]) & (search["max_vif"] <= MAX_VIF)]
    .sort_values(
        "cv_r2_oof",
        ascending=False,
    )
    .reset_index(drop=True)
)


if acceptable.empty:
    raise ValueError("No pooled model satisfies the VIF threshold.")


winner = acceptable.iloc[0]


selected_features = [
    winner["feature_1"],
    winner["feature_2"],
    winner["feature_3"],
]


print(
    "\n"
    "============================================================\n"
    "POOLED-SELECTED DIAGNOSTIC STRUCTURE\n"
    "============================================================"
)


for i, feature in enumerate(
    selected_features,
    start=1,
):
    print(f"x{i}: {feature}")


print(f"\nPooled selection CV R²: {winner['cv_r2_oof']:.4f}")

print(f"Pooled selection maximum VIF: {winner['max_vif']:.3f}")


# =====================================================================
# 2. Load raw data
# =====================================================================

print(f"\nReading {SIGNAL_METRICS_PATH}")

signal_metrics = pd.read_parquet(SIGNAL_METRICS_PATH)


print(f"Reading {INVESTMENT_METRICS_PATH}")

investment_metrics = pd.read_parquet(INVESTMENT_METRICS_PATH)


print(f"Reading {PARAMETERS_PATH}")

parameters = pd.read_parquet(PARAMETERS_PATH)


# =====================================================================
# 3. Build predictor matrix
# =====================================================================

# Keep only predictor families allowed in the diagnostic.
metrics = signal_metrics.loc[
    signal_metrics["error_family"].isin(
        [
            "clustered_approximation",
            "tsa",
            "rpcc",
        ]
    )
    & signal_metrics["signal_type"].isin(
        [
            "level",
            "delta",
        ]
    )
    & signal_metrics["metric"].isin(
        [
            "nmbe",
            "nrmse",
            "pearson",
        ]
    )
].copy()


normalised_mask = metrics["metric"].isin(
    [
        "nmbe",
        "nrmse",
    ]
) & (metrics["normalisation_basis"] == NORMALISATION_BASIS)


pearson_mask = (metrics["metric"] == "pearson") & metrics["normalisation_basis"].isna()


metrics = metrics.loc[normalised_mask | pearson_mask].copy()


metrics["feature"] = metrics.apply(
    make_feature_name,
    axis=1,
)


X_wide = metrics.pivot(
    index="case_id",
    columns="feature",
    values="value",
)

X_wide.columns.name = None


# Confirm that the pooled-selected predictors exist.
missing_features = [
    feature for feature in selected_features if feature not in X_wide.columns
]


if missing_features:
    raise ValueError(
        "Selected predictors are missing from "
        "signal_metrics.parquet:\n"
        f"{missing_features}"
    )


# =====================================================================
# 4. Build target + metadata table
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


metadata = parameters[
    [
        "case_id",
        "country",
        PROXY_WEIGHT_FIELD,
        "start_date",
        "end_date",
    ]
].copy()


metadata["proxy_weight"] = pd.to_numeric(
    metadata[PROXY_WEIGHT_FIELD],
    errors="coerce",
)


analysis = (
    target.merge(
        metadata,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    .set_index("case_id")
    .join(
        X_wide[selected_features],
        how="inner",
    )
)


if PROXY_WEIGHT == "positive":
    analysis = analysis.loc[analysis["proxy_weight"] > 0].copy()

elif PROXY_WEIGHT == "all":
    pass

elif isinstance(
    PROXY_WEIGHT,
    (float, int),
):
    analysis = analysis.loc[
        np.isclose(
            analysis["proxy_weight"],
            float(PROXY_WEIGHT),
        )
    ].copy()

else:
    raise ValueError("Invalid PROXY_WEIGHT.")


nl = analysis.loc[analysis["country"] == DEVELOPMENT_COUNTRY].copy()


be = analysis.loc[analysis["country"] == VALIDATION_COUNTRY].copy()


print(f"\nNL training cases: {len(nl)}")

print(f"BE validation cases: {len(be)}")


# =====================================================================
# 5. Fit coefficients using NL ONLY
# =====================================================================

X_nl = nl[selected_features].to_numpy(dtype=float)

y_nl = nl["target"].to_numpy(dtype=float)


valid_nl = np.isfinite(y_nl) & np.all(
    np.isfinite(X_nl),
    axis=1,
)


X_nl_valid = X_nl[valid_nl]

y_nl_valid = y_nl[valid_nl]


beta_nl = fit_ols(
    X_nl_valid,
    y_nl_valid,
)


nl_pred = predict_ols(
    X_nl_valid,
    beta_nl,
)


nl_metrics = prediction_metrics(
    y_nl_valid,
    nl_pred,
)


nl_vif = calculate_vif(X_nl_valid)


print(
    "\n"
    "============================================================\n"
    "NL COEFFICIENT TRAINING\n"
    "============================================================"
)


print(f"n = {len(y_nl_valid)}")

print(f"NL fitted R² = {nl_metrics['r2']:.4f}")

print(f"NL maximum VIF = {np.max(nl_vif):.3f}")


print("\nNL coefficients:")

print(f"  intercept = {beta_nl[0]: .8f}")


for i, feature in enumerate(
    selected_features,
    start=1,
):
    print(f"  beta_{i} = {beta_nl[i]: .8f}    [{feature}]")


# =====================================================================
# 6. Apply NL coefficients unchanged to BE
# =====================================================================

X_be = be[selected_features].to_numpy(dtype=float)

y_be = be["target"].to_numpy(dtype=float)


valid_be = np.isfinite(y_be) & np.all(
    np.isfinite(X_be),
    axis=1,
)


X_be_valid = X_be[valid_be]

y_be_valid = y_be[valid_be]


be_pred = predict_ols(
    X_be_valid,
    beta_nl,
)


be_metrics = prediction_metrics(
    y_be_valid,
    be_pred,
)


print(
    "\n"
    "============================================================\n"
    "NL → BE COEFFICIENT TRANSFER\n"
    "============================================================"
)


print(f"n = {be_metrics['n']}")

print(f"Predictive R² = {be_metrics['r2']:.4f}")

print(f"RMSE = {be_metrics['rmse']:.6f}")

print(f"MAE = {be_metrics['mae']:.6f}")

print(f"Mean prediction error = {be_metrics['mean_prediction_error']:.6f}")


# =====================================================================
# 7. Save BE predictions
# =====================================================================

be_output = be.loc[
    valid_be,
    [
        "country",
        "proxy_weight",
        "start_date",
        "end_date",
    ],
].copy()


be_output["observed_capacity_error"] = y_be_valid


be_output["predicted_capacity_error"] = be_pred


be_output["prediction_error"] = be_pred - y_be_valid


be_output.to_csv(
    OUTPUT_PREDICTIONS_PATH,
    index=True,
)


# =====================================================================
# 8. Validation scatter
# =====================================================================

fig, ax = plt.subplots(
    figsize=(
        5.5,
        5.5,
    )
)


ax.scatter(
    y_be_valid,
    be_pred,
    alpha=0.6,
)


plot_min = min(
    np.min(y_be_valid),
    np.min(be_pred),
)

plot_max = max(
    np.max(y_be_valid),
    np.max(be_pred),
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

ax.set_ylabel("Predicted signed LDES capacity error")


ax.set_title("Pooled-selected diagnostic: NL coefficients applied to BE")


ax.text(
    0.04,
    0.96,
    (f"$R^2$ = {be_metrics['r2']:.2f}\nn = {be_metrics['n']}"),
    transform=ax.transAxes,
    va="top",
)


fig.tight_layout()


fig.savefig(
    OUTPUT_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


# =====================================================================
# 9. Final pooled coefficient refit
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


X_pooled_valid = X_pooled[valid_pooled]

y_pooled_valid = y_pooled[valid_pooled]


beta_pooled = fit_ols(
    X_pooled_valid,
    y_pooled_valid,
)


pooled_pred = predict_ols(
    X_pooled_valid,
    beta_pooled,
)


pooled_metrics = prediction_metrics(
    y_pooled_valid,
    pooled_pred,
)


pooled_vif = calculate_vif(X_pooled_valid)


print(
    "\n"
    "============================================================\n"
    "FINAL NL + BE COEFFICIENT REFIT\n"
    "============================================================"
)


print("Predictor structure remains fixed.")


print("\nFinal pooled coefficients:")

print(f"  intercept = {beta_pooled[0]: .8f}")


for i, feature in enumerate(
    selected_features,
    start=1,
):
    print(f"  beta_{i} = {beta_pooled[i]: .8f}    [{feature}]")


print(f"\nPooled n = {len(y_pooled_valid)}")

print(f"Pooled fitted R² = {pooled_metrics['r2']:.4f}")

print(f"Pooled maximum VIF = {np.max(pooled_vif):.3f}")


# =====================================================================
# 10. Save outputs
# =====================================================================

summary = pd.DataFrame(
    [
        {
            "pooled_selection_cv_r2": winner["cv_r2_oof"],
            "pooled_selection_max_vif": winner["max_vif"],
            "feature_1": selected_features[0],
            "feature_2": selected_features[1],
            "feature_3": selected_features[2],
            "nl_training_n": len(y_nl_valid),
            "nl_fitted_r2": nl_metrics["r2"],
            "be_validation_n": be_metrics["n"],
            "be_transfer_r2": be_metrics["r2"],
            "be_transfer_rmse": be_metrics["rmse"],
            "be_transfer_mae": be_metrics["mae"],
            "be_mean_prediction_error": be_metrics["mean_prediction_error"],
            "final_pooled_n": len(y_pooled_valid),
            "final_pooled_r2": pooled_metrics["r2"],
            "final_pooled_max_vif": np.max(pooled_vif),
        }
    ]
)


summary.to_csv(
    OUTPUT_SUMMARY_PATH,
    index=False,
)


coefficient_rows = [
    {
        "term": "intercept",
        "feature": "intercept",
        "nl_coefficient": beta_nl[0],
        "final_pooled_coefficient": beta_pooled[0],
        "final_pooled_vif": np.nan,
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
            "nl_coefficient": beta_nl[i],
            "final_pooled_coefficient": beta_pooled[i],
            "final_pooled_vif": pooled_vif[i - 1],
        }
    )


pd.DataFrame(coefficient_rows).to_csv(
    OUTPUT_COEFFICIENTS_PATH,
    index=False,
)


model_metadata = {
    "selection_population": "NL + BE",
    "selection_method": "blocked weather-window cross-validation",
    "selected_features": selected_features,
    "pooled_selection_cv_r2": float(winner["cv_r2_oof"]),
    "pooled_selection_max_vif": float(winner["max_vif"]),
    "coefficient_training_country": DEVELOPMENT_COUNTRY,
    "validation_country": VALIDATION_COUNTRY,
    "nl_coefficients": beta_nl.tolist(),
    "be_transfer_metrics": {key: float(value) for key, value in be_metrics.items()},
    "final_pooled_coefficients": beta_pooled.tolist(),
}


with open(
    OUTPUT_MODEL_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        model_metadata,
        file,
        indent=2,
    )


print("\nSaved outputs:")

print(f"  {OUTPUT_SUMMARY_PATH}")

print(f"  {OUTPUT_PREDICTIONS_PATH}")

print(f"  {OUTPUT_COEFFICIENTS_PATH}")

print(f"  {OUTPUT_MODEL_PATH}")

print(f"  {OUTPUT_FIGURE_PATH}")
