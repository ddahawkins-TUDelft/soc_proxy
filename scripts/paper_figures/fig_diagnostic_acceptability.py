"""Final fixed-predictor ex-post LDES diagnostic evaluation.



Scientific design

-----------------

This script assumes the THREE signal predictors have already been selected using

Netherlands (NL) development data. Edit ``PREDICTORS`` below to the selected

triplet. Proxy weight W_P is always added as a fourth predictor, centred at 0.5.



The script then performs:



1. Continuous diagnostic

   - NL internal validation by leave-one-weather-chronology-out (LOCO).

   - Final NL fit applied unchanged to Belgium (BE) as a cross-country holdout.

   - Reports R2, RMSE, MAE, VIF, and the same outcomes for W_P = 0.5 cases only.



2. Three-state probability diagnostic

   A. Two-stage multinomial calibration (paper formulation):

        X -> e_hat -> [e_hat, e_hat^2] -> P(under, acceptable, over)

   B. Gaussian distributional sensitivity formulation:

        X -> mu(X), sigma(X) -> Normal(mu, sigma)

        -> P(under, acceptable, over)

      The scale model is fitted to chronology-held-out residuals so uncertainty

      is not estimated from optimistic in-sample residuals.



3. Acceptance-tolerance sensitivity

   Repeats the probability evaluation for +/-5, +/-10, +/-15 and +/-20

   percentage-point true-error bands. The continuous diagnostic is unchanged.



4. Action-confidence sensitivity

   For each probability formulation and tolerance, sweeps confidence thresholds

   from 0.50 to 0.95. At threshold q:

       trust       if P(acceptable) >= q

       investigate if P(under)+P(over) >= q

   and reports coverage, precision, decisive coverage, and decisive accuracy.

   A 0.75 operating point is also exported explicitly.



5. Final combined refit

   After the NL->BE validation experiment, the continuous model is refitted to

   NL+BE and probability layers are calibrated from chronology-held-out combined

   predictions/residuals. These are deployment equations/parameters, not an

   independent validation result.



Important

---------

All validation folds are grouped by weather chronology. Cases with different k

or W_P but the same underlying chronology are never split across train/test.



Run from the repository root, e.g.:

    pixi run python scripts/paper_figures/diagnostic_final_nl_be_with_gaussian.py

"""



from __future__ import annotations



import json

from dataclasses import dataclass

from pathlib import Path

from typing import Iterable



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.stats import norm

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.metrics import (

    accuracy_score,

    average_precision_score,

    f1_score,

    log_loss,

    mean_absolute_error,

    mean_squared_error,

    r2_score,

    roc_auc_score,

)

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler





# =============================================================================

# CONFIGURATION

# =============================================================================



RESULTS_DIR = Path("results") / "2_5_10_year"

OUTPUT_DIR = RESULTS_DIR / "diagnostic" / "final_fixed_predictor"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



PARAMETERS_PATH = RESULTS_DIR / "parameters.parquet"

SIGNAL_METRICS_PATH = RESULTS_DIR / "signal_metrics.parquet"

INVESTMENT_METRICS_PATH = RESULTS_DIR / "investment_metrics.parquet"



COUNTRIES = ("NL", "BE")

TRAIN_COUNTRY = "NL"

VALIDATION_COUNTRY = "BE"

HORIZON_YEARS = 10

CLUSTER_METHOD = "kmeans"

REPRESENTATION_METHOD = "medoid"

PROXY_WEIGHT_FIELD = "lambda_soc"

WP_CENTRE = 0.5



# -----------------------------------------------------------------------------

# EDIT ONLY THIS LIST when testing a different selected three-signal diagnostic.

# W_P is added automatically as the fourth predictor.

# Defaults are copied from the script supplied on 2026-10-01.

# -----------------------------------------------------------------------------

PREDICTORS = [
    "clustered_approximation__level__nrmse__reference_proxy_full_range__full",
    "rpcc__delta__pearson__none__548d",
    "clustered_approximation__level__nmbe__reference_proxy_full_range__180d",
]



# Main paper tolerance + appendix sensitivities.

MAIN_ERROR_TOLERANCE_PP = 10.0

ERROR_TOLERANCES_PP = (5.0, 10.0, 15.0, 20.0)



# Practical trust/investigate probability threshold and sweep.

MAIN_CONFIDENCE_THRESHOLD = 0.75

CONFIDENCE_THRESHOLDS = tuple(np.round(np.arange(0.50, 0.951, 0.05), 2))



# Gaussian residual-scale model.

VARIANCE_RIDGE_ALPHA = 1.0

VARIANCE_FLOOR = 1e-6

SIGMA_FLOOR = 0.005  # fraction = 0.5 percentage points



# Plotting only.

PLASMA_NL = 0.08

PLASMA_BE = 0.50

PLASMA_UNDER = 0.08

PLASMA_OVER = 0.50

ACCEPTABLE_GREY = "0.42"



STATE_ORDER = np.array(["under", "acceptable", "over"], dtype=object)





# =============================================================================

# DATA HELPERS

# =============================================================================





def window_label(value) -> str:

    if pd.isna(value):

        return "full"

    return f"{int(value)}d"





def make_feature_name(row: pd.Series) -> str:

    basis = "none" if pd.isna(row["normalisation_basis"]) else str(row["normalisation_basis"])

    return (

        f"{row['error_family']}__{row['signal_type']}__{row['metric']}__"

        f"{basis}__{window_label(row['window_half_width_days'])}"

    )





def derive_chronology_id(metadata: pd.DataFrame) -> pd.Series:

    """Return a chronology identifier shared by all k/W_P variants.



    Prefer an existing chronology_id. Otherwise use the exact start/end dates.

    Country is deliberately NOT included: if NL and BE use the same calendar

    horizon, combined-data cross-fitting holds that chronology out in both.

    """

    if "chronology_id" in metadata.columns:

        return metadata["chronology_id"].astype(str)



    if {"start_date", "end_date"}.issubset(metadata.columns):

        start = pd.to_datetime(metadata["start_date"], errors="coerce")

        end = pd.to_datetime(metadata["end_date"], errors="coerce")

        if start.isna().any() or end.isna().any():

            raise ValueError("Could not derive chronology_id because start/end dates contain NaT.")

        return start.dt.strftime("%Y-%m-%d") + "__" + end.dt.strftime("%Y-%m-%d")



    raise ValueError(

        "Need either 'chronology_id' or both 'start_date' and 'end_date' in parameters.parquet "

        "to perform chronology-grouped validation."

    )





def load_analysis_table() -> pd.DataFrame:

    parameters = pd.read_parquet(PARAMETERS_PATH).copy()

    signal_metrics = pd.read_parquet(SIGNAL_METRICS_PATH).copy()

    investment_metrics = pd.read_parquet(INVESTMENT_METRICS_PATH).copy()



    required = {

        "case_id",

        "country",

        "horizon_years",

        "cluster_method",

        "representation_method",

        PROXY_WEIGHT_FIELD,

    }

    missing = sorted(required.difference(parameters.columns))

    if missing:

        raise ValueError(f"parameters.parquet missing required columns: {missing}")



    parameters["horizon_years"] = pd.to_numeric(parameters["horizon_years"], errors="coerce")

    parameters["proxy_weight"] = pd.to_numeric(parameters[PROXY_WEIGHT_FIELD], errors="coerce")

    parameters["chronology_id"] = derive_chronology_id(parameters)



    metadata = parameters.loc[

        parameters["country"].isin(COUNTRIES)

        & np.isclose(parameters["horizon_years"].astype(float), HORIZON_YEARS)

        & parameters["cluster_method"].astype(str).str.lower().eq(CLUSTER_METHOD.lower())

        & parameters["representation_method"].astype(str).str.lower().eq(REPRESENTATION_METHOD.lower())

    ].copy()



    target = investment_metrics.loc[

        investment_metrics["metric"].eq("ldes_capacity_error_signed"),

        ["case_id", "value"],

    ].rename(columns={"value": "signed_ldes_error"})

    if target["case_id"].duplicated().any():

        raise ValueError("Multiple ldes_capacity_error_signed values found for a case_id.")



    diagnostics = signal_metrics.copy()

    diagnostics["feature"] = diagnostics.apply(make_feature_name, axis=1)

    if diagnostics.duplicated(["case_id", "feature"], keep=False).any():

        dupes = diagnostics.loc[

            diagnostics.duplicated(["case_id", "feature"], keep=False),

            ["case_id", "feature"],

        ]

        raise ValueError(f"Duplicate signal metric rows found:\n{dupes.head(20)}")



    wide = diagnostics.pivot(index="case_id", columns="feature", values="value")

    missing_predictors = [x for x in PREDICTORS if x not in wide.columns]

    if missing_predictors:

        raise ValueError("Selected predictors not found:\n  " + "\n  ".join(missing_predictors))



    optional = [c for c in ("start_date", "end_date", "k_periods") if c in metadata.columns]

    table = (

        metadata[

            ["case_id", "country", "horizon_years", "chronology_id", "proxy_weight", *optional]

        ]

        .merge(target, on="case_id", how="inner", validate="one_to_one")

        .set_index("case_id")

        .join(wide[PREDICTORS], how="inner")

        .reset_index()

    )



    numeric = ["proxy_weight", "signed_ldes_error", *PREDICTORS]

    mask = np.all(np.isfinite(table[numeric].to_numpy(dtype=float)), axis=1)

    table = table.loc[mask].copy()



    print("\nAnalysis cases")

    print("==============")

    print(f"Total: {len(table)} | NL: {(table.country == 'NL').sum()} | BE: {(table.country == 'BE').sum()}")

    print(f"NL chronologies: {table.loc[table.country.eq('NL'), 'chronology_id'].nunique()}")

    print(f"BE chronologies: {table.loc[table.country.eq('BE'), 'chronology_id'].nunique()}")

    print(f"W_P: {sorted(table['proxy_weight'].unique())}")

    print("Signal predictors:")

    for p in PREDICTORS:

        print(f"  - {p}")

    print("  - W_P - 0.5 (added automatically)")



    return table





# =============================================================================

# CONTINUOUS DIAGNOSTIC

# =============================================================================





@dataclass

class ContinuousDiagnostic:

    signal_mean: np.ndarray

    signal_scale: np.ndarray

    beta: np.ndarray



    @classmethod

    def fit(cls, frame: pd.DataFrame) -> "ContinuousDiagnostic":

        x = frame[PREDICTORS].to_numpy(dtype=float)

        mean = np.mean(x, axis=0)

        scale = np.std(x, axis=0, ddof=0)

        if np.any(scale <= 0):

            raise ValueError("At least one selected signal predictor has zero variance.")

        z = (x - mean) / scale

        wp = frame["proxy_weight"].to_numpy(dtype=float) - WP_CENTRE

        design = np.column_stack([np.ones(len(frame)), z, wp])

        y = frame["signed_ldes_error"].to_numpy(dtype=float)

        beta = np.linalg.lstsq(design, y, rcond=None)[0]

        return cls(mean, scale, beta)



    def design_matrix(self, frame: pd.DataFrame) -> np.ndarray:

        x = frame[PREDICTORS].to_numpy(dtype=float)

        z = (x - self.signal_mean) / self.signal_scale

        wp = frame["proxy_weight"].to_numpy(dtype=float) - WP_CENTRE

        return np.column_stack([np.ones(len(frame)), z, wp])



    def predictor_matrix(self, frame: pd.DataFrame) -> np.ndarray:

        """Four non-intercept predictor columns used for VIF/scale modelling."""

        x = frame[PREDICTORS].to_numpy(dtype=float)

        z = (x - self.signal_mean) / self.signal_scale

        wp = frame["proxy_weight"].to_numpy(dtype=float) - WP_CENTRE

        return np.column_stack([z, wp])



    def predict(self, frame: pd.DataFrame) -> np.ndarray:

        return self.design_matrix(frame) @ self.beta



    def raw_equation(self) -> dict:

        """Return equivalent raw-signal equation, still using (W_P-0.5)."""

        signal_beta = self.beta[1:4] / self.signal_scale

        intercept = self.beta[0] - np.sum(self.beta[1:4] * self.signal_mean / self.signal_scale)

        return {

            "intercept": float(intercept),

            "signal_coefficients": {p: float(b) for p, b in zip(PREDICTORS, signal_beta)},

            "proxy_weight_centered_coefficient": float(self.beta[4]),

        }





def grouped_oof_continuous(frame: pd.DataFrame) -> np.ndarray:

    prediction = np.full(len(frame), np.nan, dtype=float)

    groups = frame["chronology_id"].astype(str).to_numpy()

    unique = pd.unique(groups)

    if len(unique) < 3:

        raise ValueError("Need at least three weather chronologies for grouped OOF prediction.")



    for held_out in unique:

        train = frame.loc[groups != held_out]

        test_idx = np.where(groups == held_out)[0]

        test = frame.iloc[test_idx]

        model = ContinuousDiagnostic.fit(train)

        prediction[test_idx] = model.predict(test)



    if not np.all(np.isfinite(prediction)):

        raise RuntimeError("Grouped OOF continuous predictions contain non-finite values.")

    return prediction





def continuous_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:

    truth = np.asarray(truth, dtype=float)

    prediction = np.asarray(prediction, dtype=float)

    return {

        "n": int(len(truth)),

        "r2": float(r2_score(truth, prediction)),

        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),

        "mae": float(mean_absolute_error(truth, prediction)),

    }





def calculate_vif(matrix: np.ndarray) -> np.ndarray:

    x = np.asarray(matrix, dtype=float)

    out = []

    for j in range(x.shape[1]):

        y = x[:, j]

        others = np.delete(x, j, axis=1)

        if others.shape[1] == 0:

            out.append(1.0)

            continue

        design = np.column_stack([np.ones(len(others)), others])

        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]

        ss_res = np.sum((y - fitted) ** 2)

        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        out.append(np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2))

    return np.asarray(out)





# =============================================================================

# STATE LABELS + PROBABILITY METRICS

# =============================================================================





def state_from_error(error: np.ndarray, tolerance_pp: float) -> np.ndarray:

    e = np.asarray(error, dtype=float)

    t = tolerance_pp / 100.0

    return np.where(e < -t, "under", np.where(e > t, "over", "acceptable"))





def align_probabilities(raw: np.ndarray, classes: Iterable[str]) -> np.ndarray:

    classes = list(classes)

    out = np.zeros((len(raw), 3), dtype=float)

    for j, label in enumerate(STATE_ORDER):

        if label in classes:

            out[:, j] = raw[:, classes.index(label)]

    row_sum = out.sum(axis=1, keepdims=True)

    if np.any(row_sum <= 0):

        raise RuntimeError("Probability row with zero mass after class alignment.")

    return out / row_sum





def probability_metrics(truth_state: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:

    truth_state = np.asarray(truth_state, dtype=object)

    p = np.asarray(probabilities, dtype=float)

    pred = STATE_ORDER[np.argmax(p, axis=1)]

    onehot = np.column_stack([(truth_state == s).astype(float) for s in STATE_ORDER])

    true_idx = np.array([np.where(STATE_ORDER == s)[0][0] for s in truth_state], dtype=int)



    acceptable_truth = (truth_state == "acceptable").astype(int)

    p_acceptable = p[:, 1]

    try:

        auc = float(roc_auc_score(acceptable_truth, p_acceptable))

        ap = float(average_precision_score(acceptable_truth, p_acceptable))

    except ValueError:

        auc = np.nan

        ap = np.nan



    result = {

        "n": int(len(truth_state)),

        "state_accuracy": float(accuracy_score(truth_state, pred)),

        "state_macro_f1": float(

            f1_score(truth_state, pred, labels=STATE_ORDER.tolist(), average="macro", zero_division=0)

        ),

        "multiclass_log_loss": float(-np.mean(np.log(np.clip(p[np.arange(len(p)), true_idx], 1e-15, 1.0)))),

        "multiclass_brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),

        "acceptable_auc": auc,

        "acceptable_ap": ap,

    }



    for label in STATE_ORDER:

        mask_pred = pred == label

        mask_true = truth_state == label

        result[f"precision_{label}"] = (

            float(np.mean(mask_true[mask_pred])) if np.any(mask_pred) else np.nan

        )

        result[f"recall_{label}"] = (

            float(np.mean(mask_pred[mask_true])) if np.any(mask_true) else np.nan

        )

    return result





def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:

    if total <= 0:

        return np.nan, np.nan

    phat = successes / total

    denom = 1.0 + z**2 / total

    centre = (phat + z**2 / (2 * total)) / denom

    half = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denom

    return float(centre - half), float(centre + half)





def screening_metrics(

    truth_state: np.ndarray,

    probabilities: np.ndarray,

    confidence_threshold: float,

) -> dict[str, float]:

    truth_state = np.asarray(truth_state, dtype=object)

    p = np.asarray(probabilities, dtype=float)

    p_acc = p[:, 1]

    p_unacc = p[:, 0] + p[:, 2]

    truth_acc = truth_state == "acceptable"



    trust = p_acc >= confidence_threshold

    investigate = p_unacc >= confidence_threshold

    # For q > 0.5 these cannot overlap; enforce this explicitly.

    if np.any(trust & investigate):

        raise RuntimeError("Trust and investigate masks overlap. Use confidence threshold > 0.5.")



    decisive = trust | investigate

    correct = (trust & truth_acc) | (investigate & ~truth_acc)



    trust_correct = int(np.sum(trust & truth_acc))

    trust_n = int(np.sum(trust))

    inv_correct = int(np.sum(investigate & ~truth_acc))

    inv_n = int(np.sum(investigate))

    trust_lo, trust_hi = wilson_interval(trust_correct, trust_n)

    inv_lo, inv_hi = wilson_interval(inv_correct, inv_n)



    return {

        "confidence_threshold": float(confidence_threshold),

        "trust_n": trust_n,

        "trust_coverage": float(np.mean(trust)),

        "trust_precision": float(trust_correct / trust_n) if trust_n else np.nan,

        "trust_precision_ci95_low": trust_lo,

        "trust_precision_ci95_high": trust_hi,

        "investigate_n": inv_n,

        "investigate_coverage": float(np.mean(investigate)),

        "investigate_precision": float(inv_correct / inv_n) if inv_n else np.nan,

        "investigate_precision_ci95_low": inv_lo,

        "investigate_precision_ci95_high": inv_hi,

        "decisive_n": int(np.sum(decisive)),

        "decisive_coverage": float(np.mean(decisive)),

        "decisive_accuracy": float(np.mean(correct[decisive])) if np.any(decisive) else np.nan,

        "correct_decisive_fraction_all_cases": float(np.mean(correct)),

    }





# =============================================================================

# TWO-STAGE MULTINOMIAL FORMULATION

# =============================================================================





def fit_two_stage_calibrator(predicted_error: np.ndarray, truth_state: np.ndarray):

    e_pp = np.asarray(predicted_error, dtype=float) * 100.0

    x = np.column_stack([e_pp, e_pp**2])

    model = make_pipeline(

        StandardScaler(),

        LogisticRegression(C=1e12, solver="lbfgs", max_iter=20000, tol=1e-10),

    )

    model.fit(x, truth_state)

    return model





def predict_two_stage(model, predicted_error: np.ndarray) -> np.ndarray:

    e_pp = np.asarray(predicted_error, dtype=float) * 100.0

    x = np.column_stack([e_pp, e_pp**2])

    raw = model.predict_proba(x)

    return align_probabilities(raw, model.named_steps["logisticregression"].classes_)





def nested_loco_two_stage(frame: pd.DataFrame, tolerance_pp: float) -> tuple[np.ndarray, np.ndarray]:

    """Outer chronology holdout; inner chronology holdout supplies calibration scores."""

    groups = frame["chronology_id"].astype(str).to_numpy()

    unique = pd.unique(groups)

    signed = np.full(len(frame), np.nan, dtype=float)

    probs = np.full((len(frame), 3), np.nan, dtype=float)



    for held_out in unique:

        train = frame.loc[groups != held_out].copy()

        test_idx = np.where(groups == held_out)[0]

        test = frame.iloc[test_idx]



        inner_oof = grouped_oof_continuous(train)

        train_state = state_from_error(train["signed_ldes_error"].to_numpy(dtype=float), tolerance_pp)

        calibrator = fit_two_stage_calibrator(inner_oof, train_state)



        mean_model = ContinuousDiagnostic.fit(train)

        outer_signed = mean_model.predict(test)

        outer_probs = predict_two_stage(calibrator, outer_signed)

        signed[test_idx] = outer_signed

        probs[test_idx] = outer_probs



    if not np.all(np.isfinite(signed)) or not np.all(np.isfinite(probs)):

        raise RuntimeError("Nested LOCO two-stage output contains non-finite values.")

    return signed, probs





# =============================================================================

# GAUSSIAN DISTRIBUTIONAL FORMULATION

# =============================================================================





@dataclass

class GaussianScaleModel:

    pipeline: object

    variance_scale: float



    @staticmethod

    def feature_matrix(frame: pd.DataFrame) -> np.ndarray:

        x = frame[PREDICTORS].to_numpy(dtype=float)

        wp = frame["proxy_weight"].to_numpy(dtype=float)[:, None] - WP_CENTRE

        return np.column_stack([x, wp])



    @classmethod

    def fit(cls, frame: pd.DataFrame, residual: np.ndarray) -> "GaussianScaleModel":

        residual = np.asarray(residual, dtype=float)

        squared = residual**2

        target = np.log(squared + VARIANCE_FLOOR)

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=VARIANCE_RIDGE_ALPHA))

        X = cls.feature_matrix(frame)

        pipe.fit(X, target)

        base_var = np.exp(np.clip(pipe.predict(X), -30.0, 10.0))

        ratio = squared / np.maximum(base_var, VARIANCE_FLOOR)

        variance_scale = float(np.mean(ratio))

        if not np.isfinite(variance_scale) or variance_scale <= 0:

            variance_scale = 1.0

        return cls(pipe, variance_scale)



    def predict_sigma(self, frame: pd.DataFrame) -> np.ndarray:

        base_var = np.exp(np.clip(self.pipeline.predict(self.feature_matrix(frame)), -30.0, 10.0))

        var = np.maximum(base_var * self.variance_scale, SIGMA_FLOOR**2)

        return np.sqrt(var)





def gaussian_state_probabilities(mu: np.ndarray, sigma: np.ndarray, tolerance_pp: float) -> np.ndarray:

    mu = np.asarray(mu, dtype=float)

    sigma = np.maximum(np.asarray(sigma, dtype=float), SIGMA_FLOOR)

    t = tolerance_pp / 100.0

    p_under = norm.cdf((-t - mu) / sigma)

    p_over = 1.0 - norm.cdf((t - mu) / sigma)

    p_acc = np.maximum(0.0, 1.0 - p_under - p_over)

    p = np.column_stack([p_under, p_acc, p_over])

    return p / p.sum(axis=1, keepdims=True)





def nested_loco_gaussian(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:

    """Return chronology-held-out mu and sigma; tolerance is applied afterwards."""

    groups = frame["chronology_id"].astype(str).to_numpy()

    unique = pd.unique(groups)

    mu = np.full(len(frame), np.nan, dtype=float)

    sigma = np.full(len(frame), np.nan, dtype=float)



    for held_out in unique:

        train = frame.loc[groups != held_out].copy()

        test_idx = np.where(groups == held_out)[0]

        test = frame.iloc[test_idx]



        inner_oof = grouped_oof_continuous(train)

        residual = train["signed_ldes_error"].to_numpy(dtype=float) - inner_oof

        scale_model = GaussianScaleModel.fit(train, residual)

        mean_model = ContinuousDiagnostic.fit(train)



        mu[test_idx] = mean_model.predict(test)

        sigma[test_idx] = scale_model.predict_sigma(test)



    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):

        raise RuntimeError("Nested LOCO Gaussian output contains non-finite values.")

    return mu, sigma





# =============================================================================

# PLOTTING

# =============================================================================





def save_figure(fig, stem: str) -> None:

    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")

    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)





def plot_continuous_validation(
    nl: pd.DataFrame,
    nl_oof: np.ndarray,
    be: pd.DataFrame,
    be_pred: np.ndarray,
) -> None:
    """Plot NL chronology-held-out and BE transfer predictions on one axis.

    The Netherlands markers are chronology-held-out predictions from the
    development data. Belgium markers are predictions from the model fitted to
    all Dutch cases and transferred unchanged to Belgium. Colours are sampled
    from the plasma colormap; marker shape additionally distinguishes datasets.
    """
    cmap = plt.get_cmap("plasma")
    nl_color = cmap(PLASMA_NL)
    be_color = cmap(PLASMA_BE)

    nl_truth = nl["signed_ldes_error"].to_numpy(dtype=float) * 100.0
    nl_pred_pp = np.asarray(nl_oof, dtype=float) * 100.0
    be_truth = be["signed_ldes_error"].to_numpy(dtype=float) * 100.0
    be_pred_pp = np.asarray(be_pred, dtype=float) * 100.0

    truth_all = np.r_[nl_truth, be_truth]
    pred_all = np.r_[nl_pred_pp, be_pred_pp]
    lim = max(
        20.0,
        float(
            np.ceil(
                max(np.max(np.abs(truth_all)), np.max(np.abs(pred_all))) / 10.0
            )
            * 10.0
        ),
    )

    nl_metrics = continuous_metrics(nl_truth / 100.0, nl_oof)
    be_metrics = continuous_metrics(be_truth / 100.0, be_pred)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))

    # One-to-one reference line. No acceptability bands are shown here because
    # this figure reports the threshold-independent continuous prediction.
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        linestyle="--",
        linewidth=1.0,
        color="0.35",
        zorder=1,
    )

    ax.scatter(
        nl_truth,
        nl_pred_pp,
        s=20,
        alpha=0.65,
        color=nl_color,
        marker="o",
        linewidths=0,
        label="NL (held-out)",
        zorder=2,
    )
    ax.scatter(
        be_truth,
        be_pred_pp,
        s=20,
        alpha=0.65,
        color=be_color,
        marker="o",
        linewidths=0,
        label="BE (transfer)",
        zorder=3,
    )

    # Colour-matched performance callouts. Keep these in axes coordinates so
    # their placement is stable if the numerical range changes.
    callout = dict(
        boxstyle="round,pad=0.30",
        facecolor="white",
        linewidth=0.9,
        alpha=0.92,
    )
    ax.text(
        0.04,
        0.96,
        "NL (Training)\n"
        f"$R^2$ = {nl_metrics['r2']:.2f}\n"
        rf"RMSE = {100.0 * nl_metrics['rmse']:.1f} pp",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=nl_color,
        fontsize=9,
        bbox={**callout, "edgecolor": nl_color},
    )
    ax.text(
        0.96,
        0.04,
        "BE (Validation)\n"
        f"$R^2$ = {be_metrics['r2']:.2f}\n"
        rf"RMSE = {100.0 * be_metrics['rmse']:.1f} pp",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=be_color,
        fontsize=9,
        bbox={**callout, "edgecolor": be_color},
    )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed LDES capacity error (%)")
    ax.set_ylabel("Predicted LDES capacity error (%)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=2)

    fig.tight_layout()
    save_figure(fig, "fig_continuous_diagnostic_validation")

def plot_two_stage_mapping(calibrator, signed_range: np.ndarray, tolerance_pp: float) -> None:

    lo = float(np.min(signed_range) * 100.0)

    hi = float(np.max(signed_range) * 100.0)

    pad = 0.08 * max(hi - lo, 1.0)

    grid_pp = np.linspace(lo - pad, hi + pad, 600)

    p = predict_two_stage(calibrator, grid_pp / 100.0)

    cmap = plt.get_cmap("plasma")



    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    ax.plot(grid_pp, p[:, 0], label="Under-investment", color=cmap(PLASMA_UNDER), linewidth=2)

    ax.plot(grid_pp, p[:, 1], label="Acceptable", color=ACCEPTABLE_GREY, linewidth=2)

    ax.plot(grid_pp, p[:, 2], label="Over-investment", color=cmap(PLASMA_OVER), linewidth=2)

    ax.axvline(-tolerance_pp, linestyle="--", linewidth=0.9, color="0.65")

    ax.axvline(tolerance_pp, linestyle="--", linewidth=0.9, color="0.65")

    ax.set_xlabel("Predicted signed LDES capacity error (percentage points)")

    ax.set_ylabel("Calibrated probability")

    ax.set_ylim(0, 1)

    ax.legend(frameon=False)

    fig.tight_layout()

    save_figure(fig, "fig_three_state_two_stage_mapping")





def plot_gaussian_holdout(mu: np.ndarray, sigma: np.ndarray, tolerance_pp: float) -> None:

    p = gaussian_state_probabilities(mu, sigma, tolerance_pp)

    order = np.argsort(mu)

    x = mu[order] * 100.0

    cmap = plt.get_cmap("plasma")



    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    ax.scatter(x, p[order, 0], s=16, alpha=0.55, label="Under-investment", color=cmap(PLASMA_UNDER))

    ax.scatter(x, p[order, 1], s=16, alpha=0.55, label="Acceptable", color=ACCEPTABLE_GREY)

    ax.scatter(x, p[order, 2], s=16, alpha=0.55, label="Over-investment", color=cmap(PLASMA_OVER))

    ax.axvline(-tolerance_pp, linestyle="--", linewidth=0.9, color="0.65")

    ax.axvline(tolerance_pp, linestyle="--", linewidth=0.9, color="0.65")

    ax.set_xlabel("Predicted Gaussian mean signed error (percentage points)")

    ax.set_ylabel("Gaussian state probability")

    ax.set_ylim(0, 1)

    ax.legend(frameon=False)

    ax.set_title("BE holdout: heteroscedastic Gaussian probabilities")

    fig.tight_layout()

    save_figure(fig, "fig_three_state_gaussian_be_holdout")





def plot_precision_coverage(screening: pd.DataFrame) -> None:

    subset = screening.loc[

        screening["tolerance_pp"].eq(MAIN_ERROR_TOLERANCE_PP)

        & screening["subset"].eq("all")

        & screening["dataset"].eq("BE_holdout")

    ].copy()

    cmap = plt.get_cmap("plasma")

    colors = {"two_stage": cmap(0.12), "gaussian": cmap(0.55)}



    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))

    for formulation in ("two_stage", "gaussian"):

        s = subset.loc[subset["formulation"].eq(formulation)].sort_values("confidence_threshold")

        axes[0].plot(s["trust_coverage"], s["trust_precision"], marker="o", ms=3.5,

                     label=formulation.replace("_", " "), color=colors[formulation])

        axes[1].plot(s["investigate_coverage"], s["investigate_precision"], marker="o", ms=3.5,

                     label=formulation.replace("_", " "), color=colors[formulation])

    axes[0].set_title("Trust calls")

    axes[1].set_title("Investigate calls")

    for ax in axes:

        ax.set_xlabel("Coverage")

        ax.set_ylabel("Precision")

        ax.set_ylim(0, 1.02)

        ax.legend(frameon=False)

    fig.tight_layout()

    save_figure(fig, "fig_probability_formulation_precision_coverage")





# =============================================================================

# EVALUATION HELPERS

# =============================================================================





def subset_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:

    return {

        "all": np.ones(len(frame), dtype=bool),

        "WP_0.5": np.isclose(frame["proxy_weight"].to_numpy(dtype=float), 0.5),

    }





def append_probability_rows(

    probability_rows: list[dict],

    screening_rows: list[dict],

    *,

    dataset: str,

    frame: pd.DataFrame,

    tolerance_pp: float,

    formulation: str,

    probabilities: np.ndarray,

) -> None:

    truth_all = state_from_error(frame["signed_ldes_error"].to_numpy(dtype=float), tolerance_pp)

    for subset_name, mask in subset_masks(frame).items():

        truth = truth_all[mask]

        p = probabilities[mask]

        row = {

            "dataset": dataset,

            "subset": subset_name,

            "tolerance_pp": tolerance_pp,

            "formulation": formulation,

            **probability_metrics(truth, p),

        }

        probability_rows.append(row)

        for threshold in CONFIDENCE_THRESHOLDS:

            screening_rows.append(

                {

                    "dataset": dataset,

                    "subset": subset_name,

                    "tolerance_pp": tolerance_pp,

                    "formulation": formulation,

                    **screening_metrics(truth, p, threshold),

                }

            )





# =============================================================================

# FINAL PARAMETER EXPORT

# =============================================================================





def two_stage_raw_logits(model) -> dict:

    scaler = model.named_steps["standardscaler"]

    logistic = model.named_steps["logisticregression"]

    raw_coef = logistic.coef_ / scaler.scale_[None, :]

    raw_intercept = logistic.intercept_ - np.sum(

        logistic.coef_ * (scaler.mean_ / scaler.scale_)[None, :], axis=1

    )

    return {

        str(cls): {

            "intercept": float(raw_intercept[i]),

            "predicted_error_pp": float(raw_coef[i, 0]),

            "predicted_error_pp_squared": float(raw_coef[i, 1]),

        }

        for i, cls in enumerate(logistic.classes_)

    }





def export_final_models(

    combined: pd.DataFrame,

    final_mean: ContinuousDiagnostic,

    final_two_stage,

    final_scale: GaussianScaleModel,

) -> None:

    payload = {

        "signal_predictors": PREDICTORS,

        "continuous_standardised_form": {

            "intercept": float(final_mean.beta[0]),

            "signal_coefficients": {

                p: float(b) for p, b in zip(PREDICTORS, final_mean.beta[1:4])

            },

            "proxy_weight_center": WP_CENTRE,

            "proxy_weight_centered_coefficient": float(final_mean.beta[4]),

            "signal_means": {p: float(x) for p, x in zip(PREDICTORS, final_mean.signal_mean)},

            "signal_scales": {p: float(x) for p, x in zip(PREDICTORS, final_mean.signal_scale)},

        },

        "continuous_raw_form": final_mean.raw_equation(),

        "two_stage_main_tolerance_pp": MAIN_ERROR_TOLERANCE_PP,

        "two_stage_raw_logits": two_stage_raw_logits(final_two_stage),

        "gaussian_scale_model": {

            "ridge_alpha": VARIANCE_RIDGE_ALPHA,

            "variance_scale": float(final_scale.variance_scale),

            "note": "Scale model uses three raw signal predictors + (W_P-0.5); sklearn pipeline retained in script, not serialised fully here.",

        },

        "main_confidence_threshold": MAIN_CONFIDENCE_THRESHOLD,

        "n_combined": int(len(combined)),

    }

    (OUTPUT_DIR / "final_model_parameters.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")



    raw = final_mean.raw_equation()

    lines = [

        "FINAL COMBINED CONTINUOUS DIAGNOSTIC",

        "====================================",

        "",

        "Signed error is a fraction (0.10 = 10%).",

        "",

        f"intercept = {raw['intercept']:+.10g}",

    ]

    for feature, coef in raw["signal_coefficients"].items():

        lines.append(f"{feature}: {coef:+.10g}")

    lines.append(f"(W_P - {WP_CENTRE:g}): {raw['proxy_weight_centered_coefficient']:+.10g}")

    lines.extend(["", "TWO-STAGE PROBABILITY LOGITS AT MAIN TOLERANCE"])

    for state, coef in two_stage_raw_logits(final_two_stage).items():

        lines.append(

            f"{state}: z = {coef['intercept']:+.10g} "

            f"{coef['predicted_error_pp']:+.10g}*e_pp "

            f"{coef['predicted_error_pp_squared']:+.10g}*e_pp^2"

        )

    (OUTPUT_DIR / "final_equations.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")





# =============================================================================

# MAIN

# =============================================================================





def main() -> None:

    table = load_analysis_table()

    nl = table.loc[table["country"].eq(TRAIN_COUNTRY)].reset_index(drop=True)

    be = table.loc[table["country"].eq(VALIDATION_COUNTRY)].reset_index(drop=True)

    if nl.empty or be.empty:

        raise RuntimeError("Expected both NL and BE cases.")



    # ------------------------------------------------------------------

    # Continuous NL development / BE holdout.

    # ------------------------------------------------------------------

    nl_oof = grouped_oof_continuous(nl)

    nl_mean_model = ContinuousDiagnostic.fit(nl)

    be_pred = nl_mean_model.predict(be)



    continuous_rows = []

    for dataset, frame, pred in [

        ("NL_LOCO", nl, nl_oof),

        ("BE_holdout", be, be_pred),

    ]:

        for subset_name, mask in subset_masks(frame).items():

            metrics = continuous_metrics(frame.loc[mask, "signed_ldes_error"].to_numpy(dtype=float), pred[mask])

            continuous_rows.append({"dataset": dataset, "subset": subset_name, **metrics})

    continuous_summary = pd.DataFrame(continuous_rows)

    continuous_summary.to_csv(OUTPUT_DIR / "continuous_summary.csv", index=False)



    vif_labels = [*PREDICTORS, "W_P_minus_0.5"]

    vif_values = calculate_vif(nl_mean_model.predictor_matrix(nl))

    pd.DataFrame({"predictor": vif_labels, "vif_nl": vif_values}).to_csv(

        OUTPUT_DIR / "vif_nl.csv", index=False

    )



    plot_continuous_validation(nl, nl_oof, be, be_pred)



    # Gaussian mean/sigma are tolerance-independent and can be computed once.

    nl_gauss_mu_oof, nl_gauss_sigma_oof = nested_loco_gaussian(nl)

    nl_residual = nl["signed_ldes_error"].to_numpy(dtype=float) - nl_oof

    nl_scale_model = GaussianScaleModel.fit(nl, nl_residual)

    be_gauss_mu = be_pred.copy()  # same continuous mean model

    be_gauss_sigma = nl_scale_model.predict_sigma(be)



    probability_rows: list[dict] = []

    screening_rows: list[dict] = []



    main_nl_two_stage_calibrator = None

    main_be_two_stage_probs = None



    for tolerance_pp in ERROR_TOLERANCES_PP:

        # Honest NL nested two-stage validation.

        nl_two_signed, nl_two_probs = nested_loco_two_stage(nl, tolerance_pp)

        # Fit deployable NL calibrator on NL chronology-held-out continuous predictions.

        nl_state = state_from_error(nl["signed_ldes_error"].to_numpy(dtype=float), tolerance_pp)

        nl_calibrator = fit_two_stage_calibrator(nl_oof, nl_state)

        be_two_probs = predict_two_stage(nl_calibrator, be_pred)



        # Gaussian probabilities from nested NL mu/sigma and frozen NL->BE mu/sigma.

        nl_gauss_probs = gaussian_state_probabilities(nl_gauss_mu_oof, nl_gauss_sigma_oof, tolerance_pp)

        be_gauss_probs = gaussian_state_probabilities(be_gauss_mu, be_gauss_sigma, tolerance_pp)



        append_probability_rows(

            probability_rows, screening_rows,

            dataset="NL_nested_LOCO", frame=nl, tolerance_pp=tolerance_pp,

            formulation="two_stage", probabilities=nl_two_probs,

        )

        append_probability_rows(

            probability_rows, screening_rows,

            dataset="BE_holdout", frame=be, tolerance_pp=tolerance_pp,

            formulation="two_stage", probabilities=be_two_probs,

        )

        append_probability_rows(

            probability_rows, screening_rows,

            dataset="NL_nested_LOCO", frame=nl, tolerance_pp=tolerance_pp,

            formulation="gaussian", probabilities=nl_gauss_probs,

        )

        append_probability_rows(

            probability_rows, screening_rows,

            dataset="BE_holdout", frame=be, tolerance_pp=tolerance_pp,

            formulation="gaussian", probabilities=be_gauss_probs,

        )



        if np.isclose(tolerance_pp, MAIN_ERROR_TOLERANCE_PP):

            main_nl_two_stage_calibrator = nl_calibrator

            main_be_two_stage_probs = be_two_probs



    probability_summary = pd.DataFrame(probability_rows)

    screening_sweep = pd.DataFrame(screening_rows)

    probability_summary.to_csv(OUTPUT_DIR / "probability_summary_all_tolerances.csv", index=False)

    screening_sweep.to_csv(OUTPUT_DIR / "confidence_threshold_sweep_all_tolerances.csv", index=False)



    selected = screening_sweep.loc[

        np.isclose(screening_sweep["confidence_threshold"], MAIN_CONFIDENCE_THRESHOLD)

    ].copy()

    selected.to_csv(OUTPUT_DIR / "screening_at_p075.csv", index=False)



    # Compact tolerance-sensitivity table used for appendix writing.

    tolerance_summary = probability_summary.merge(

        selected,

        on=["dataset", "subset", "tolerance_pp", "formulation"],

        how="left",

        suffixes=("", "_screen"),

    )

    tolerance_summary.to_csv(OUTPUT_DIR / "tolerance_sensitivity_summary.csv", index=False)



    if main_nl_two_stage_calibrator is None or main_be_two_stage_probs is None:

        raise RuntimeError("MAIN_ERROR_TOLERANCE_PP is not present in ERROR_TOLERANCES_PP.")



    plot_two_stage_mapping(

        main_nl_two_stage_calibrator,

        np.r_[nl_oof, be_pred],

        MAIN_ERROR_TOLERANCE_PP,

    )

    plot_gaussian_holdout(be_gauss_mu, be_gauss_sigma, MAIN_ERROR_TOLERANCE_PP)

    plot_precision_coverage(screening_sweep)



    # Save case-level validation predictions at the main tolerance.

    main_state_nl = state_from_error(nl["signed_ldes_error"].to_numpy(dtype=float), MAIN_ERROR_TOLERANCE_PP)

    main_state_be = state_from_error(be["signed_ldes_error"].to_numpy(dtype=float), MAIN_ERROR_TOLERANCE_PP)

    nl_main_two_signed, nl_main_two_probs = nested_loco_two_stage(nl, MAIN_ERROR_TOLERANCE_PP)

    nl_main_gauss_probs = gaussian_state_probabilities(

        nl_gauss_mu_oof, nl_gauss_sigma_oof, MAIN_ERROR_TOLERANCE_PP

    )

    be_main_gauss_probs = gaussian_state_probabilities(

        be_gauss_mu, be_gauss_sigma, MAIN_ERROR_TOLERANCE_PP

    )



    for dataset, frame, continuous, state, p_two, p_gauss, sigma in [

        ("NL_nested_LOCO", nl, nl_main_two_signed, main_state_nl, nl_main_two_probs, nl_main_gauss_probs, nl_gauss_sigma_oof),

        ("BE_holdout", be, be_pred, main_state_be, main_be_two_stage_probs, be_main_gauss_probs, be_gauss_sigma),

    ]:

        out = frame[["case_id", "country", "chronology_id", "proxy_weight", "signed_ldes_error", *PREDICTORS]].copy()

        out["dataset"] = dataset

        out["actual_state"] = state

        out["continuous_prediction"] = continuous

        out["continuous_prediction_pp"] = 100.0 * continuous

        out["gaussian_sigma"] = sigma

        for j, label in enumerate(STATE_ORDER):

            out[f"two_stage_p_{label}"] = p_two[:, j]

            out[f"gaussian_p_{label}"] = p_gauss[:, j]

        out.to_csv(OUTPUT_DIR / f"predictions_{dataset}.csv", index=False)



    # ------------------------------------------------------------------

    # Final combined refit for deployment, after validation.

    # ------------------------------------------------------------------

    combined = table.reset_index(drop=True)

    combined_oof = grouped_oof_continuous(combined)

    final_mean = ContinuousDiagnostic.fit(combined)

    final_residual = combined["signed_ldes_error"].to_numpy(dtype=float) - combined_oof

    final_scale = GaussianScaleModel.fit(combined, final_residual)

    combined_state_main = state_from_error(

        combined["signed_ldes_error"].to_numpy(dtype=float), MAIN_ERROR_TOLERANCE_PP

    )

    final_two_stage = fit_two_stage_calibrator(combined_oof, combined_state_main)

    export_final_models(combined, final_mean, final_two_stage, final_scale)



    # ------------------------------------------------------------------

    # Console report: numbers most likely to be cited in the manuscript.

    # ------------------------------------------------------------------

    print("\n" + "=" * 100)

    print("CONTINUOUS DIAGNOSTIC")

    print("=" * 100)

    printable_cont = continuous_summary.copy()

    for c in ("rmse", "mae"):

        printable_cont[c + "_pp"] = 100.0 * printable_cont[c]

    print(printable_cont[["dataset", "subset", "n", "r2", "rmse_pp", "mae_pp"]].to_string(index=False))



    print("\nNL predictor VIFs (including W_P)")

    print(pd.DataFrame({"predictor": vif_labels, "vif": vif_values}).to_string(index=False))



    print("\n" + "=" * 100)

    print(f"THREE-STATE PERFORMANCE AT +/-{MAIN_ERROR_TOLERANCE_PP:g}%")

    print("=" * 100)

    main_prob = probability_summary.loc[

        np.isclose(probability_summary["tolerance_pp"], MAIN_ERROR_TOLERANCE_PP)

    ].copy()

    cols = [

        "dataset", "subset", "formulation", "n", "state_accuracy", "state_macro_f1",

        "multiclass_log_loss", "multiclass_brier", "acceptable_auc", "acceptable_ap",

    ]

    print(main_prob[cols].to_string(index=False))



    print("\n" + "=" * 100)

    print(f"PRACTICAL SCREENING AT CONFIDENCE >= {MAIN_CONFIDENCE_THRESHOLD:.2f}")

    print("=" * 100)

    show_cols = [

        "dataset", "subset", "formulation", "tolerance_pp",

        "trust_coverage", "trust_precision", "investigate_coverage", "investigate_precision",

        "decisive_coverage", "decisive_accuracy", "correct_decisive_fraction_all_cases",

    ]

    print(selected[show_cols].to_string(index=False))



    print("\n" + "=" * 100)

    print("TOLERANCE SENSITIVITY: BE HOLDOUT")

    print("=" * 100)

    be_tol = tolerance_summary.loc[tolerance_summary["dataset"].eq("BE_holdout")].copy()

    print(be_tol[show_cols].to_string(index=False))



    print(f"\nOutputs written to: {OUTPUT_DIR}")





if __name__ == "__main__":

    main()
