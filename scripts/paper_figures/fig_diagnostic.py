"""Paper-figure exploration for ex-post LDES diagnostic case 9657.

This script reconstructs the diagnostic case table directly from the same
three result objects used by the wider diagnostic workflow:

    parameters.parquet
    signal_metrics.parquet
    investment_metrics.parquet

It does NOT expect a pre-joined case-level CSV.

Scientific setup
----------------
Selected diagnostic (combination 9657):

    X1 = clustered_approximation, level, nMBE, ±180 d
    X2 = RPCC, delta, Pearson r, ±548 d
    X3 = clustered_approximation, level, nRMSE, full horizon
    X4 = W_P* = lambda_soc - 0.5

Two fits are produced:

1. NL -> BE transfer fit
   - continuous signed-error equation fitted on NL only;
   - probability calibrator fitted from NL chronology-held-out predictions;
   - BE cases are the ONLY case markers shown on transfer figures.

2. Combined final equation
   - same four-term continuous diagnostic refitted on NL + BE;
   - coefficients are written out separately for the equation intended for
     reporting after validation has been discussed.

The probability stage follows the corrected two-stage approach used in the
diagnostic search:

    signal metrics + W_P
        -> predicted signed LDES error
        -> multinomial model using [e_hat, e_hat^2]
        -> P(under), P(acceptable), P(over)

For NL -> BE transfer, the probability calibrator sees only NL
chronology-held-out regression scores, avoiding calibration on in-sample
continuous predictions.

Figures
-------
1. fig_9657_be_continuous_transfer
   Actual vs predicted signed error; BE markers only.

2. fig_9657_be_probability_curves
   Under / acceptable / over probability curves with BE cases overlaid.
   Incorrect top-class judgements are marked with X; false-acceptable cases
   receive an additional square outline.

3. fig_9657_be_acceptable_probability_bars
   Stacked probabilities for BE cases whose top-probability class is
   "acceptable", with actual signed error printed beside each case.

4. fig_9657_be_confusion_matrix
   Row-normalised three-state confusion matrix.

5. fig_9657_be_false_acceptable_severity
   Actual signed errors for all BE cases called acceptable, making it obvious
   whether incorrect calls are near misses (e.g. 11%) or severe misses.

Tables / text
-------------
summary_9657_metrics.csv
confidence_9657_be.csv
false_acceptable_summary_9657_be.csv
cases_9657_be.csv
equation_9657_nl_transfer.txt
equation_9657_combined_final.txt
equation_9657_probability_05pct_combined_final.txt
equation_9657_probability_10pct_combined_final.txt
equation_9657_probability_15pct_combined_final.txt
equation_9657_probability_20pct_combined_final.txt
fig_9657_threshold_sensitivity_appendix.pdf/png

All plotting colours are derived from the matplotlib ``plasma`` colormap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = Path("results") / "2_5_10_year"
OUTPUT_DIR = RESULTS_DIR / "diagnostic" / "paper_figures_case_9657"

PARAMETERS_PATH = RESULTS_DIR / "parameters.parquet"
SIGNAL_METRICS_PATH = RESULTS_DIR / "signal_metrics.parquet"
INVESTMENT_METRICS_PATH = RESULTS_DIR / "investment_metrics.parquet"

COUNTRIES = ("NL", "BE")
TRAIN_COUNTRY = "NL"
TRANSFER_COUNTRY = "BE"

HORIZON_YEARS = 10
CLUSTER_METHOD = "kmeans"
REPRESENTATION_METHOD = "medoid"

WP_FIELD = "lambda_soc"
WP_CENTRE = 0.5
WP_ORDER = (0.0, 0.25, 0.5, 0.75, 1.0)

TARGET_METRIC = "ldes_capacity_error_signed"
NORMALISATION_BASIS = "reference_proxy_full_range"

ACCEPTABLE_LIMIT = 0.10

STATE_LABELS = np.array(
    ["under", "acceptable", "over"],
    dtype=object,
)

CONFIDENCE_THRESHOLDS = (0.50, 0.70, 0.80, 0.90)

# The continuous signed-error diagnostic is independent of this tolerance.
# These values are used only to test the sensitivity of the three-state
# interpretive layer (under / acceptable / over).
THRESHOLD_SENSITIVITY_LIMITS = (0.05, 0.10, 0.15, 0.20)

# The ±5% case remains in the numerical sensitivity tables, but is omitted
# from the appendix figure because it provides little practical acceptable
# classification in the present validation set.
THRESHOLD_SENSITIVITY_PLOT_LIMITS = (0.10, 0.15, 0.20)

STATE_LINESTYLES = {
    "under": "-",
    "acceptable": "-",
    "over": "-",
}

APPENDIX_FIGURE_WIDTH_IN = 5.6
APPENDIX_FIGURE_HEIGHT_IN = 4.05

# Compact single-column main-paper figure dimensions.
MAIN_FIGURE_WIDTH_IN = 3.45
MAIN_FIGURE_HEIGHT_IN = 6.65

# If there are very many cases with "acceptable" as the top-probability class,
# cap the stacked-bar figure for readability. The CSV still contains all cases.
MAX_ACCEPTABLE_BAR_CASES = 60


FEATURE_SPECS = (
    {
        "name": "ca_level_nmbe_180d",
        "error_family": "clustered_approximation",
        "signal_type": "level",
        "metric": "nmbe",
        "window_half_width_days": 180.0,
        "normalisation_basis": NORMALISATION_BASIS,
        "paper_label": r"Clustered approximation level nMBE, $\pm180$ d",
    },
    {
        "name": "rpcc_delta_pearson_548d",
        "error_family": "rpcc",
        "signal_type": "delta",
        "metric": "pearson",
        "window_half_width_days": 548.0,
        "normalisation_basis": None,
        "paper_label": r"RPCC delta Pearson $r$, $\pm548$ d",
    },
    {
        "name": "ca_level_nrmse_full",
        "error_family": "clustered_approximation",
        "signal_type": "level",
        "metric": "nrmse",
        "window_half_width_days": None,
        "normalisation_basis": NORMALISATION_BASIS,
        "paper_label": r"Clustered approximation level nRMSE, full horizon",
    },
)

METRIC_FEATURES = [spec["name"] for spec in FEATURE_SPECS]


# =============================================================================
# Plasma styling
# =============================================================================

CMAP = plt.get_cmap("plasma")

# For the appendix threshold-sensitivity figure, colour denotes the
# acceptable-error tolerance (rather than the state).
THRESHOLD_PLOT_COLORS = {
    0.10: CMAP(0.12),
    0.15: CMAP(0.52),
    0.20: CMAP(0.8),
}

COLOR_UNDER = CMAP(0.10)
COLOR_ACCEPTABLE = CMAP(0.53)
COLOR_OVER = CMAP(0.80)

STATE_COLORS = {
    "under": COLOR_UNDER,
    "acceptable": COLOR_ACCEPTABLE,
    "over": COLOR_OVER,
}

STATE_MARKERS = {
    "under": "v",
    "acceptable": "o",
    "over": "^",
}


# =============================================================================
# General utilities
# =============================================================================

def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    name: str,
) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns:\n{sorted(frame.columns)}"
        )


def assign_horizon(parameters: pd.DataFrame) -> pd.DataFrame:
    out = parameters.copy()

    if "horizon_years" in out.columns:
        out["horizon_duration"] = (
            pd.to_numeric(
                out["horizon_years"],
                errors="coerce",
            )
            .round()
            .astype("Int64")
        )
        return out

    require_columns(
        out,
        {"start_date", "end_date"},
        name="parameters.parquet",
    )

    start = pd.to_datetime(
        out["start_date"],
        errors="coerce",
    )
    end = pd.to_datetime(
        out["end_date"],
        errors="coerce",
    )

    years = (
        (end - start).dt.total_seconds()
        / (365.2425 * 24 * 3600)
    )

    out["horizon_duration"] = years.round().astype("Int64")
    return out


def state_from_error_at_limit(
    values: np.ndarray,
    acceptable_limit: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    state = np.full(
        len(values),
        "acceptable",
        dtype=object,
    )
    state[values < -acceptable_limit] = "under"
    state[values > acceptable_limit] = "over"

    return state


def state_from_error(values: np.ndarray) -> np.ndarray:
    return state_from_error_at_limit(
        values,
        ACCEPTABLE_LIMIT,
    )


def safe_r2(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    ss_res = float(
        np.sum(
            (truth - prediction) ** 2
        )
    )
    ss_tot = float(
        np.sum(
            (truth - np.mean(truth)) ** 2
        )
    )

    if ss_tot <= 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


def regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    residual = truth - prediction

    return {
        "r2": safe_r2(
            truth,
            prediction,
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    residual**2
                )
            )
        ),
        "mae": float(
            np.mean(
                np.abs(residual)
            )
        ),
        "n": int(len(truth)),
    }


def explicit_multiclass_log_loss(
    truth_state: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Negative log likelihood in explicit STATE_LABELS column order."""
    truth_state = np.asarray(
        truth_state,
        dtype=object,
    )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    index = {
        label: i
        for i, label in enumerate(STATE_LABELS)
    }

    true_column = np.asarray(
        [index[label] for label in truth_state],
        dtype=int,
    )

    true_probability = probabilities[
        np.arange(len(truth_state)),
        true_column,
    ]

    return float(
        -np.mean(
            np.log(
                np.clip(
                    true_probability,
                    1e-15,
                    1.0,
                )
            )
        )
    )


def multiclass_brier(
    truth_state: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    truth_state = np.asarray(
        truth_state,
        dtype=object,
    )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    one_hot = np.column_stack(
        [
            (truth_state == label).astype(float)
            for label in STATE_LABELS
        ]
    )

    return float(
        np.mean(
            np.sum(
                (probabilities - one_hot) ** 2,
                axis=1,
            )
        )
    )


def state_metrics(
    truth_state: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    truth_state = np.asarray(
        truth_state,
        dtype=object,
    )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predicted_state = STATE_LABELS[
        np.argmax(
            probabilities,
            axis=1,
        )
    ]

    acceptable_truth = (
        truth_state == "acceptable"
    ).astype(int)

    acceptable_index = int(
        np.where(
            STATE_LABELS == "acceptable"
        )[0][0]
    )
    acceptable_probability = probabilities[
        :,
        acceptable_index,
    ]

    try:
        acceptable_auc = float(
            roc_auc_score(
                acceptable_truth,
                acceptable_probability,
            )
        )
        acceptable_ap = float(
            average_precision_score(
                acceptable_truth,
                acceptable_probability,
            )
        )
    except ValueError:
        acceptable_auc = np.nan
        acceptable_ap = np.nan

    return {
        "accuracy": float(
            accuracy_score(
                truth_state,
                predicted_state,
            )
        ),
        "macro_f1": float(
            f1_score(
                truth_state,
                predicted_state,
                labels=STATE_LABELS.tolist(),
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": explicit_multiclass_log_loss(
            truth_state,
            probabilities,
        ),
        "brier": multiclass_brier(
            truth_state,
            probabilities,
        ),
        "acceptable_auc": acceptable_auc,
        "acceptable_ap": acceptable_ap,
    }


# =============================================================================
# Assemble case-level table from the three parquet outputs
# =============================================================================

def filter_parameters(
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        parameters,
        {
            "case_id",
            "country",
            "start_date",
            "end_date",
            "k_periods",
            WP_FIELD,
        },
        name="parameters.parquet",
    )

    out = assign_horizon(
        parameters
    )

    out["case_id"] = out[
        "case_id"
    ].astype(str)

    out["start_date"] = pd.to_datetime(
        out["start_date"]
    )
    out["end_date"] = pd.to_datetime(
        out["end_date"]
    )

    out["k_periods"] = pd.to_numeric(
        out["k_periods"],
        errors="raise",
    ).astype(int)

    out["proxy_weight"] = pd.to_numeric(
        out[WP_FIELD],
        errors="raise",
    ).astype(float)

    mask = (
        out["country"].isin(COUNTRIES)
        & out["horizon_duration"].eq(
            HORIZON_YEARS
        )
    )

    if "cluster_method" in out.columns:
        mask &= (
            out["cluster_method"]
            .astype(str)
            .str.lower()
            .eq(CLUSTER_METHOD)
        )

    if "representation_method" in out.columns:
        mask &= (
            out["representation_method"]
            .astype(str)
            .str.lower()
            .eq(REPRESENTATION_METHOD)
        )

    wp_mask = np.zeros(
        len(out),
        dtype=bool,
    )
    for weight in WP_ORDER:
        wp_mask |= np.isclose(
            out["proxy_weight"],
            weight,
        )

    out = out.loc[
        mask & wp_mask
    ].copy()

    out["chronology_id"] = (
        out["country"].astype(str)
        + "__"
        + out["start_date"].dt.strftime(
            "%Y%m%d"
        )
        + "__"
        + out["end_date"].dt.strftime(
            "%Y%m%d"
        )
    )

    return out


def extract_signal_feature(
    signals: pd.DataFrame,
    spec: dict,
) -> pd.DataFrame:
    require_columns(
        signals,
        {
            "case_id",
            "error_family",
            "signal_type",
            "metric",
            "window_half_width_days",
            "value",
        },
        name="signal_metrics.parquet",
    )

    mask = (
        signals["error_family"].eq(
            spec["error_family"]
        )
        & signals["signal_type"].eq(
            spec["signal_type"]
        )
        & signals["metric"].eq(
            spec["metric"]
        )
    )

    if spec["window_half_width_days"] is None:
        mask &= signals[
            "window_half_width_days"
        ].isna()
    else:
        mask &= np.isclose(
            pd.to_numeric(
                signals[
                    "window_half_width_days"
                ],
                errors="coerce",
            ),
            float(
                spec[
                    "window_half_width_days"
                ]
            ),
            equal_nan=False,
        )

    if "normalisation_basis" in signals.columns:
        basis = spec[
            "normalisation_basis"
        ]

        if basis is None:
            mask &= signals[
                "normalisation_basis"
            ].isna()
        else:
            mask &= signals[
                "normalisation_basis"
            ].eq(basis)

    subset = signals.loc[
        mask,
        ["case_id", "value"],
    ].copy()

    subset["case_id"] = subset[
        "case_id"
    ].astype(str)

    if subset.empty:
        raise ValueError(
            "No rows found for feature "
            f"{spec['name']}:\n{spec}"
        )

    duplicates = subset.duplicated(
        "case_id",
        keep=False,
    )
    if duplicates.any():
        examples = subset.loc[
            duplicates,
            "case_id",
        ].head(20).tolist()

        raise ValueError(
            f"Feature {spec['name']} has duplicate rows per case_id. "
            "If signal_metrics.parquet contains multiple normalisation bases, "
            "ensure normalisation_basis is present and correctly specified. "
            f"Example duplicated case_ids: {examples}"
        )

    return subset.rename(
        columns={
            "value": spec["name"],
        }
    )


def build_case_table() -> pd.DataFrame:
    parameters = pd.read_parquet(
        PARAMETERS_PATH
    )
    signals = pd.read_parquet(
        SIGNAL_METRICS_PATH
    )
    investment = pd.read_parquet(
        INVESTMENT_METRICS_PATH
    )

    metadata = filter_parameters(
        parameters
    )

    require_columns(
        investment,
        {
            "case_id",
            "metric",
            "value",
        },
        name="investment_metrics.parquet",
    )

    target = investment.loc[
        investment["metric"].eq(
            TARGET_METRIC
        ),
        ["case_id", "value"],
    ].copy()

    target["case_id"] = target[
        "case_id"
    ].astype(str)

    if target.duplicated(
        "case_id"
    ).any():
        raise ValueError(
            f"{TARGET_METRIC!r} is duplicated for one or more case_ids."
        )

    target = target.rename(
        columns={
            "value": "signed_ldes_error",
        }
    )

    table = metadata[
        [
            "case_id",
            "country",
            "start_date",
            "end_date",
            "chronology_id",
            "k_periods",
            WP_FIELD,
            "proxy_weight",
        ]
    ].merge(
        target,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    for spec in FEATURE_SPECS:
        feature = extract_signal_feature(
            signals,
            spec,
        )

        table = table.merge(
            feature,
            on="case_id",
            how="inner",
            validate="one_to_one",
        )

    required = (
        METRIC_FEATURES
        + [
            "signed_ldes_error",
            "proxy_weight",
        ]
    )

    table = table.dropna(
        subset=required
    ).copy()

    table["actual_state"] = state_from_error(
        table[
            "signed_ldes_error"
        ].to_numpy(dtype=float)
    )

    table["case_label"] = (
        table["start_date"].dt.year.astype(str)
        + " | k="
        + table["k_periods"].astype(str)
        + " | $W_P$="
        + table["proxy_weight"].map(
            lambda value: f"{value:.2f}"
        )
    )

    return table


# =============================================================================
# Continuous diagnostic model
# =============================================================================

@dataclass
class ContinuousDiagnostic:
    metric_mean: np.ndarray
    metric_std: np.ndarray
    beta: np.ndarray

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
    ) -> "ContinuousDiagnostic":
        metrics = frame[
            METRIC_FEATURES
        ].to_numpy(dtype=float)

        metric_mean = np.mean(
            metrics,
            axis=0,
        )
        metric_std = np.std(
            metrics,
            axis=0,
            ddof=0,
        )

        if np.any(
            metric_std <= 0
        ):
            raise ValueError(
                "A selected signal metric is constant in the training data."
            )

        z = (
            metrics - metric_mean
        ) / metric_std

        wp_centered = (
            frame[
                "proxy_weight"
            ].to_numpy(dtype=float)
            - WP_CENTRE
        )

        design = np.column_stack(
            [
                np.ones(
                    len(frame)
                ),
                z,
                wp_centered,
            ]
        )

        beta = np.linalg.lstsq(
            design,
            frame[
                "signed_ldes_error"
            ].to_numpy(dtype=float),
            rcond=None,
        )[0]

        return cls(
            metric_mean=metric_mean,
            metric_std=metric_std,
            beta=beta,
        )

    def predict(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        metrics = frame[
            METRIC_FEATURES
        ].to_numpy(dtype=float)

        z = (
            metrics - self.metric_mean
        ) / self.metric_std

        wp_centered = (
            frame[
                "proxy_weight"
            ].to_numpy(dtype=float)
            - WP_CENTRE
        )

        design = np.column_stack(
            [
                np.ones(
                    len(frame)
                ),
                z,
                wp_centered,
            ]
        )

        return design @ self.beta

    def equation_text(
        self,
        *,
        title: str,
    ) -> str:
        lines = [
            title,
            "=" * len(title),
            "",
            "Standardised-metric form:",
            "",
            "e_hat = b0 + b1*z(X1) + b2*z(X2) + b3*z(X3)"
            " + bW*(W_P - 0.5)",
            "",
            f"b0 = {self.beta[0]:+.10f}",
        ]

        for i, spec in enumerate(
            FEATURE_SPECS,
            start=1,
        ):
            lines.append(
                f"b{i} = {self.beta[i]:+.10f}    "
                f"X{i} = {spec['name']}"
            )

        lines.extend(
            [
                f"bW = {self.beta[-1]:+.10f}",
                "",
                "Metric standardisation used by this fitted equation:",
            ]
        )

        for i, spec in enumerate(
            FEATURE_SPECS
        ):
            lines.append(
                f"{spec['name']}: "
                f"mean={self.metric_mean[i]:+.10f}, "
                f"sd={self.metric_std[i]:+.10f}"
            )

        # Raw metric coefficient form, retaining centered W_P.
        raw_coefficients = (
            self.beta[1:4]
            / self.metric_std
        )
        raw_intercept = (
            self.beta[0]
            - np.sum(
                self.beta[1:4]
                * self.metric_mean
                / self.metric_std
            )
        )

        lines.extend(
            [
                "",
                "Equivalent raw-metric form:",
                "",
                "e_hat = a0 + a1*X1 + a2*X2 + a3*X3"
                " + aW*(W_P - 0.5)",
                "",
                f"a0 = {raw_intercept:+.10f}",
            ]
        )

        for i, value in enumerate(
            raw_coefficients,
            start=1,
        ):
            lines.append(
                f"a{i} = {value:+.10f}"
            )

        lines.append(
            f"aW = {self.beta[-1]:+.10f}"
        )

        return "\n".join(lines)


# =============================================================================
# Probability calibration
# =============================================================================

def calibration_features(
    score: np.ndarray,
) -> np.ndarray:
    score = np.asarray(
        score,
        dtype=float,
    )

    return np.column_stack(
        [
            score,
            score**2,
        ]
    )


def make_probability_calibrator():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
        ),
    )


def align_probabilities(
    raw_probability: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros(
        (
            len(raw_probability),
            len(STATE_LABELS),
        ),
        dtype=float,
    )

    for j, label in enumerate(
        STATE_LABELS
    ):
        where = np.where(
            classes == label
        )[0]

        if len(where) != 1:
            raise RuntimeError(
                f"Probability model is missing class {label!r}."
            )

        aligned[:, j] = raw_probability[
            :,
            where[0],
        ]

    return aligned


def fit_probability_calibrator_for_limit(
    score: np.ndarray,
    truth_error: np.ndarray,
    acceptable_limit: float,
):
    truth_state = state_from_error_at_limit(
        truth_error,
        acceptable_limit,
    )

    present = set(
        np.unique(
            truth_state
        )
    )
    missing = set(
        STATE_LABELS
    ) - present

    if missing:
        raise RuntimeError(
            "Cannot fit three-state probability calibrator. "
            f"Missing classes: {missing}"
        )

    model = make_probability_calibrator()

    model.fit(
        calibration_features(
            score
        ),
        truth_state,
    )

    return model


def fit_probability_calibrator(
    score: np.ndarray,
    truth_error: np.ndarray,
):
    return fit_probability_calibrator_for_limit(
        score,
        truth_error,
        ACCEPTABLE_LIMIT,
    )


def predict_probabilities(
    calibrator,
    score: np.ndarray,
) -> np.ndarray:
    raw = calibrator.predict_proba(
        calibration_features(
            score
        )
    )

    return align_probabilities(
        raw,
        np.asarray(
            calibrator.classes_,
            dtype=object,
        ),
    )


def nl_grouped_oof_scores(
    nl: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for held_out in nl[
        "chronology_id"
    ].drop_duplicates():
        train = nl.loc[
            nl[
                "chronology_id"
            ].ne(held_out)
        ].copy()

        test = nl.loc[
            nl[
                "chronology_id"
            ].eq(held_out)
        ].copy()

        model = ContinuousDiagnostic.fit(
            train
        )

        fold = test[
            [
                "case_id",
                "chronology_id",
                "signed_ldes_error",
            ]
        ].copy()

        fold["score"] = model.predict(
            test
        )

        rows.append(
            fold
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def combined_grouped_oof_scores(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Chronology-held-out signed-error predictions across NL + BE.

    These scores are used only to calibrate the FINAL probability equations.
    The final continuous equation itself is still refitted on the complete
    NL+BE dataset. Using OOF scores here avoids calibrating probabilities on
    optimistic in-sample continuous predictions.
    """
    rows = []

    for held_out in frame[
        "chronology_id"
    ].drop_duplicates():
        train = frame.loc[
            frame[
                "chronology_id"
            ].ne(held_out)
        ].copy()

        test = frame.loc[
            frame[
                "chronology_id"
            ].eq(held_out)
        ].copy()

        model = ContinuousDiagnostic.fit(
            train
        )

        fold = test[
            [
                "case_id",
                "country",
                "chronology_id",
                "signed_ldes_error",
            ]
        ].copy()

        fold["score"] = model.predict(
            test
        )

        rows.append(
            fold
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def probability_equation_text(
    calibrator,
    *,
    acceptable_limit: float,
    title: str,
) -> str:
    """Return the deployable three-state softmax equation.

    The calibrator is a pipeline:
        StandardScaler([e_hat, e_hat^2]) -> multinomial LogisticRegression.

    Both standardised-feature and equivalent raw-polynomial logits are
    reported. Probabilities follow from a softmax over the three logits.
    """
    scaler = calibrator.steps[0][1]
    logistic = calibrator.steps[-1][1]

    means = np.asarray(
        scaler.mean_,
        dtype=float,
    )
    scales = np.asarray(
        scaler.scale_,
        dtype=float,
    )

    classes = np.asarray(
        logistic.classes_,
        dtype=object,
    )

    coefficients = np.asarray(
        logistic.coef_,
        dtype=float,
    )
    intercepts = np.asarray(
        logistic.intercept_,
        dtype=float,
    )

    lines = [
        title,
        "=" * len(title),
        "",
        f"Acceptable-error definition: |e| <= {100 * acceptable_limit:.0f}%",
        "",
        "Input to the probability model:",
        "  e_hat = signed LDES capacity error predicted by the FINAL",
        "          NL+BE combined continuous case-9657 equation.",
        "",
        "Calibration features:",
        "  q1 = e_hat",
        "  q2 = e_hat^2",
        "",
        "Standardisation:",
        f"  z1 = (q1 - {means[0]:+.10f}) / {scales[0]:.10f}",
        f"  z2 = (q2 - {means[1]:+.10f}) / {scales[1]:.10f}",
        "",
        "Class logits (standardised-feature form):",
        "  eta_c = alpha_c + gamma_c1*z1 + gamma_c2*z2",
        "",
    ]

    raw_rows = []

    for row_index, class_label in enumerate(
        classes
    ):
        alpha = intercepts[
            row_index
        ]
        gamma_1 = coefficients[
            row_index,
            0,
        ]
        gamma_2 = coefficients[
            row_index,
            1,
        ]

        lines.extend(
            [
                f"  {class_label}:",
                f"    alpha   = {alpha:+.10f}",
                f"    gamma_1 = {gamma_1:+.10f}",
                f"    gamma_2 = {gamma_2:+.10f}",
            ]
        )

        # Equivalent raw polynomial:
        # eta = A + B*e_hat + C*e_hat^2
        raw_intercept = (
            alpha
            - gamma_1
            * means[
                0
            ]
            / scales[
                0
            ]
            - gamma_2
            * means[
                1
            ]
            / scales[
                1
            ]
        )

        raw_linear = (
            gamma_1
            / scales[
                0
            ]
        )

        raw_quadratic = (
            gamma_2
            / scales[
                1
            ]
        )

        raw_rows.append(
            (
                class_label,
                raw_intercept,
                raw_linear,
                raw_quadratic,
            )
        )

    lines.extend(
        [
            "",
            "Equivalent raw-score polynomial logits:",
            "  eta_c = A_c + B_c*e_hat + C_c*e_hat^2",
            "",
        ]
    )

    for (
        class_label,
        raw_intercept,
        raw_linear,
        raw_quadratic,
    ) in raw_rows:
        lines.extend(
            [
                f"  {class_label}:",
                f"    A = {raw_intercept:+.10f}",
                f"    B = {raw_linear:+.10f}",
                f"    C = {raw_quadratic:+.10f}",
            ]
        )

    lines.extend(
        [
            "",
            "Three-state probabilities:",
            "  P(c) = exp(eta_c) / sum_j exp(eta_j)",
            "",
            "Class ordering used by the fitted estimator:",
            "  " + ", ".join(
                str(value)
                for value in classes
            ),
        ]
    )

    return "\n".join(
        lines
    )


def build_final_combined_probability_calibrators(
    table: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[float, object],
]:
    """Build final NL+BE probability mappings for every sensitivity tolerance.

    The continuous diagnostic is refitted on the full NL+BE dataset elsewhere.
    For probability calibration we use chronology-held-out scores generated
    from the combined NL+BE dataset, then fit one multinomial calibrator per
    acceptable-error tolerance.
    """
    combined_oof = combined_grouped_oof_scores(
        table
    )

    calibrators = {}

    for acceptable_limit in THRESHOLD_SENSITIVITY_LIMITS:
        calibrators[
            acceptable_limit
        ] = fit_probability_calibrator_for_limit(
            combined_oof[
                "score"
            ].to_numpy(dtype=float),
            combined_oof[
                "signed_ldes_error"
            ].to_numpy(dtype=float),
            acceptable_limit,
        )

    return (
        combined_oof,
        calibrators,
    )


# =============================================================================
# BE case-level prediction table
# =============================================================================

def build_be_predictions(
    table: pd.DataFrame,
) -> tuple[
    ContinuousDiagnostic,
    object,
    pd.DataFrame,
    pd.DataFrame,
]:
    nl = table.loc[
        table[
            "country"
        ].eq(
            TRAIN_COUNTRY
        )
    ].copy()

    be = table.loc[
        table[
            "country"
        ].eq(
            TRANSFER_COUNTRY
        )
    ].copy()

    if nl.empty or be.empty:
        raise ValueError(
            "Expected both NL and BE cases."
        )

    # Probability calibrator uses NL chronology-held-out continuous scores.
    nl_oof = nl_grouped_oof_scores(
        nl
    )

    calibrator = fit_probability_calibrator(
        nl_oof[
            "score"
        ].to_numpy(dtype=float),
        nl_oof[
            "signed_ldes_error"
        ].to_numpy(dtype=float),
    )

    # Final transfer equation is fit on all NL cases.
    transfer_model = ContinuousDiagnostic.fit(
        nl
    )

    be["predicted_signed_error"] = transfer_model.predict(
        be
    )

    probability = predict_probabilities(
        calibrator,
        be[
            "predicted_signed_error"
        ].to_numpy(dtype=float),
    )

    for j, label in enumerate(
        STATE_LABELS
    ):
        be[f"p_{label}"] = probability[
            :,
            j,
        ]

    be["predicted_state"] = STATE_LABELS[
        np.argmax(
            probability,
            axis=1,
        )
    ]

    be["class_correct"] = (
        be["predicted_state"]
        == be["actual_state"]
    )

    be["false_acceptable"] = (
        be["predicted_state"].eq(
            "acceptable"
        )
        & ~be["actual_state"].eq(
            "acceptable"
        )
    )

    be["actual_signed_error_pct"] = (
        100
        * be[
            "signed_ldes_error"
        ]
    )

    be["predicted_signed_error_pct"] = (
        100
        * be[
            "predicted_signed_error"
        ]
    )

    return (
        transfer_model,
        calibrator,
        nl_oof,
        be,
    )


# =============================================================================
# Metrics / reporting tables
# =============================================================================

def build_metric_summary(
    table: pd.DataFrame,
    transfer_model: ContinuousDiagnostic,
    calibrator,
    be: pd.DataFrame,
) -> pd.DataFrame:
    nl = table.loc[
        table[
            "country"
        ].eq(
            TRAIN_COUNTRY
        )
    ].copy()

    nl_prediction = transfer_model.predict(
        nl
    )

    rows = []

    nl_reg = regression_metrics(
        nl[
            "signed_ldes_error"
        ].to_numpy(dtype=float),
        nl_prediction,
    )

    rows.append(
        {
            "dataset": "NL training fit",
            "continuous_r2": nl_reg["r2"],
            "continuous_rmse": nl_reg["rmse"],
            "continuous_mae": nl_reg["mae"],
        }
    )

    be_reg = regression_metrics(
        be[
            "signed_ldes_error"
        ].to_numpy(dtype=float),
        be[
            "predicted_signed_error"
        ].to_numpy(dtype=float),
    )

    be_probability = be[
        [
            "p_under",
            "p_acceptable",
            "p_over",
        ]
    ].to_numpy(dtype=float)

    be_state = state_metrics(
        be[
            "actual_state"
        ].to_numpy(dtype=object),
        be_probability,
    )

    rows.append(
        {
            "dataset": "BE cross-country transfer",
            "continuous_r2": be_reg["r2"],
            "continuous_rmse": be_reg["rmse"],
            "continuous_mae": be_reg["mae"],
            "state_accuracy": be_state["accuracy"],
            "state_macro_f1": be_state["macro_f1"],
            "multiclass_log_loss": be_state["log_loss"],
            "multiclass_brier": be_state["brier"],
            "acceptable_auc": be_state["acceptable_auc"],
            "acceptable_ap": be_state["acceptable_ap"],
        }
    )

    return pd.DataFrame(
        rows
    )


def build_confidence_table(
    be: pd.DataFrame,
) -> pd.DataFrame:
    truth_acceptable = be[
        "actual_state"
    ].eq(
        "acceptable"
    ).to_numpy()

    p_acceptable = be[
        "p_acceptable"
    ].to_numpy(dtype=float)

    p_unacceptable = (
        be[
            "p_under"
        ].to_numpy(dtype=float)
        + be[
            "p_over"
        ].to_numpy(dtype=float)
    )

    rows = []

    for threshold in CONFIDENCE_THRESHOLDS:
        accept_mask = (
            p_acceptable
            >= threshold
        )

        reject_mask = (
            p_unacceptable
            >= threshold
        )

        acceptable_precision = (
            float(
                np.mean(
                    truth_acceptable[
                        accept_mask
                    ]
                )
            )
            if accept_mask.any()
            else np.nan
        )

        acceptable_recall = (
            float(
                np.sum(
                    accept_mask
                    & truth_acceptable
                )
                / np.sum(
                    truth_acceptable
                )
            )
            if np.sum(
                truth_acceptable
            ) > 0
            else np.nan
        )

        unacceptable_precision = (
            float(
                np.mean(
                    ~truth_acceptable[
                        reject_mask
                    ]
                )
            )
            if reject_mask.any()
            else np.nan
        )

        if reject_mask.any():
            predicted_tail = np.where(
                be.loc[
                    reject_mask,
                    "p_under",
                ].to_numpy()
                >= be.loc[
                    reject_mask,
                    "p_over",
                ].to_numpy(),
                "under",
                "over",
            )

            actual_tail = be.loc[
                reject_mask,
                "actual_state",
            ].to_numpy(
                dtype=object
            )

            non_acceptable = (
                actual_tail
                != "acceptable"
            )

            tail_direction_accuracy = (
                float(
                    np.mean(
                        predicted_tail[
                            non_acceptable
                        ]
                        == actual_tail[
                            non_acceptable
                        ]
                    )
                )
                if non_acceptable.any()
                else np.nan
            )
        else:
            tail_direction_accuracy = np.nan

        rows.append(
            {
                "threshold": threshold,
                "acceptable_coverage": float(
                    np.mean(
                        accept_mask
                    )
                ),
                "acceptable_precision": acceptable_precision,
                "acceptable_recall": acceptable_recall,
                "unacceptable_coverage": float(
                    np.mean(
                        reject_mask
                    )
                ),
                "unacceptable_precision": unacceptable_precision,
                "tail_direction_accuracy_when_unacceptable": (
                    tail_direction_accuracy
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def severity_statistics(
    subset: pd.DataFrame,
    *,
    label: str,
) -> dict:
    if subset.empty:
        return {
            "definition": label,
            "n_calls": 0,
            "n_false_acceptable": 0,
            "false_acceptable_rate": np.nan,
            "median_abs_actual_error_false_pct": np.nan,
            "q75_abs_actual_error_false_pct": np.nan,
            "max_abs_actual_error_false_pct": np.nan,
            "median_exceedance_over_10pct_points": np.nan,
            "q75_exceedance_over_10pct_points": np.nan,
            "max_exceedance_over_10pct_points": np.nan,
            "share_false_calls_over_15pct_abs_error": np.nan,
            "share_false_calls_over_20pct_abs_error": np.nan,
            "share_false_calls_over_30pct_abs_error": np.nan,
        }

    false = subset.loc[
        ~subset[
            "actual_state"
        ].eq(
            "acceptable"
        )
    ].copy()

    if false.empty:
        return {
            "definition": label,
            "n_calls": len(subset),
            "n_false_acceptable": 0,
            "false_acceptable_rate": 0.0,
            "median_abs_actual_error_false_pct": np.nan,
            "q75_abs_actual_error_false_pct": np.nan,
            "max_abs_actual_error_false_pct": np.nan,
            "median_exceedance_over_10pct_points": np.nan,
            "q75_exceedance_over_10pct_points": np.nan,
            "max_exceedance_over_10pct_points": np.nan,
            "share_false_calls_over_15pct_abs_error": 0.0,
            "share_false_calls_over_20pct_abs_error": 0.0,
            "share_false_calls_over_30pct_abs_error": 0.0,
        }

    absolute_error = np.abs(
        false[
            "actual_signed_error_pct"
        ].to_numpy(dtype=float)
    )

    exceedance = (
        absolute_error
        - 10.0
    )

    return {
        "definition": label,
        "n_calls": len(subset),
        "n_false_acceptable": len(false),
        "false_acceptable_rate": (
            len(false)
            / len(subset)
        ),
        "median_abs_actual_error_false_pct": float(
            np.median(
                absolute_error
            )
        ),
        "q75_abs_actual_error_false_pct": float(
            np.quantile(
                absolute_error,
                0.75,
            )
        ),
        "max_abs_actual_error_false_pct": float(
            np.max(
                absolute_error
            )
        ),
        "median_exceedance_over_10pct_points": float(
            np.median(
                exceedance
            )
        ),
        "q75_exceedance_over_10pct_points": float(
            np.quantile(
                exceedance,
                0.75,
            )
        ),
        "max_exceedance_over_10pct_points": float(
            np.max(
                exceedance
            )
        ),
        "share_false_calls_over_15pct_abs_error": float(
            np.mean(
                absolute_error
                > 15
            )
        ),
        "share_false_calls_over_20pct_abs_error": float(
            np.mean(
                absolute_error
                > 20
            )
        ),
        "share_false_calls_over_30pct_abs_error": float(
            np.mean(
                absolute_error
                > 30
            )
        ),
    }


def build_false_acceptable_summary(
    be: pd.DataFrame,
) -> pd.DataFrame:
    hard = be.loc[
        be[
            "predicted_state"
        ].eq(
            "acceptable"
        )
    ].copy()

    p80 = be.loc[
        be[
            "p_acceptable"
        ].ge(
            0.80
        )
    ].copy()

    p90 = be.loc[
        be[
            "p_acceptable"
        ].ge(
            0.90
        )
    ].copy()

    rows = [
        severity_statistics(
            hard,
            label="Top-probability class = acceptable",
        ),
        severity_statistics(
            p80,
            label="P(acceptable) >= 0.80",
        ),
        severity_statistics(
            p90,
            label="P(acceptable) >= 0.90",
        ),
    ]

    return pd.DataFrame(
        rows
    )


def confidence_rows_for_subset(
    frame: pd.DataFrame,
    *,
    subset_name: str,
) -> list[dict]:
    truth_acceptable = frame[
        "actual_state"
    ].eq(
        "acceptable"
    ).to_numpy()

    p_acceptable = frame[
        "p_acceptable"
    ].to_numpy(dtype=float)

    p_unacceptable = (
        frame[
            "p_under"
        ].to_numpy(dtype=float)
        + frame[
            "p_over"
        ].to_numpy(dtype=float)
    )

    rows = []

    for threshold in CONFIDENCE_THRESHOLDS:
        acceptable_mask = (
            p_acceptable >= threshold
        )
        unacceptable_mask = (
            p_unacceptable >= threshold
        )

        acceptable_precision = (
            float(
                np.mean(
                    truth_acceptable[
                        acceptable_mask
                    ]
                )
            )
            if acceptable_mask.any()
            else np.nan
        )

        acceptable_recall = (
            float(
                np.sum(
                    acceptable_mask
                    & truth_acceptable
                )
                / np.sum(
                    truth_acceptable
                )
            )
            if np.sum(
                truth_acceptable
            ) > 0
            else np.nan
        )

        unacceptable_precision = (
            float(
                np.mean(
                    ~truth_acceptable[
                        unacceptable_mask
                    ]
                )
            )
            if unacceptable_mask.any()
            else np.nan
        )

        tail_direction_accuracy = np.nan

        if unacceptable_mask.any():
            actual_state = frame.loc[
                unacceptable_mask,
                "actual_state",
            ].to_numpy(dtype=object)

            actually_unacceptable = (
                actual_state != "acceptable"
            )

            if actually_unacceptable.any():
                predicted_tail = np.where(
                    frame.loc[
                        unacceptable_mask,
                        "p_under",
                    ].to_numpy()
                    >= frame.loc[
                        unacceptable_mask,
                        "p_over",
                    ].to_numpy(),
                    "under",
                    "over",
                )

                tail_direction_accuracy = float(
                    np.mean(
                        predicted_tail[
                            actually_unacceptable
                        ]
                        == actual_state[
                            actually_unacceptable
                        ]
                    )
                )

        rows.append(
            {
                "subset": subset_name,
                "n_cases": int(len(frame)),
                "threshold": threshold,
                "acceptable_n_calls": int(
                    np.sum(
                        acceptable_mask
                    )
                ),
                "acceptable_coverage": float(
                    np.mean(
                        acceptable_mask
                    )
                ),
                "acceptable_precision": acceptable_precision,
                "acceptable_recall": acceptable_recall,
                "unacceptable_n_calls": int(
                    np.sum(
                        unacceptable_mask
                    )
                ),
                "unacceptable_coverage": float(
                    np.mean(
                        unacceptable_mask
                    )
                ),
                "unacceptable_precision": unacceptable_precision,
                "tail_direction_accuracy_when_unacceptable": (
                    tail_direction_accuracy
                ),
            }
        )

    return rows


def build_confidence_by_subset(
    be: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    rows.extend(
        confidence_rows_for_subset(
            be,
            subset_name="all_BE",
        )
    )

    wp_half = be.loc[
        np.isclose(
            be[
                "proxy_weight"
            ].to_numpy(dtype=float),
            0.5,
        )
    ].copy()

    rows.extend(
        confidence_rows_for_subset(
            wp_half,
            subset_name="BE_WP_0.5",
        )
    )

    return pd.DataFrame(
        rows
    )


def boundary_distance_percentage_points(
    actual_error_pct: np.ndarray,
) -> np.ndarray:
    """Absolute distance to the nearest ±10% class boundary."""
    actual_error_pct = np.asarray(
        actual_error_pct,
        dtype=float,
    )

    return np.minimum(
        np.abs(
            actual_error_pct + 10.0
        ),
        np.abs(
            actual_error_pct - 10.0
        ),
    )


def misclassification_pair_rows(
    frame: pd.DataFrame,
    *,
    subset_name: str,
) -> list[dict]:
    wrong = frame.loc[
        ~frame[
            "class_correct"
        ]
    ].copy()

    rows = []

    for predicted_state in STATE_LABELS:
        for actual_state in STATE_LABELS:
            if predicted_state == actual_state:
                continue

            group = wrong.loc[
                wrong[
                    "predicted_state"
                ].eq(
                    predicted_state
                )
                & wrong[
                    "actual_state"
                ].eq(
                    actual_state
                )
            ].copy()

            if group.empty:
                continue

            signed_error = group[
                "actual_signed_error_pct"
            ].to_numpy(dtype=float)

            absolute_error = np.abs(
                signed_error
            )

            boundary_distance = (
                boundary_distance_percentage_points(
                    signed_error
                )
            )

            rows.append(
                {
                    "subset": subset_name,
                    "predicted_state": predicted_state,
                    "actual_state": actual_state,
                    "n_misclassified": int(
                        len(group)
                    ),
                    "median_actual_signed_error_pct": float(
                        np.median(
                            signed_error
                        )
                    ),
                    "min_actual_signed_error_pct": float(
                        np.min(
                            signed_error
                        )
                    ),
                    "max_actual_signed_error_pct": float(
                        np.max(
                            signed_error
                        )
                    ),
                    "median_abs_actual_error_pct": float(
                        np.median(
                            absolute_error
                        )
                    ),
                    "min_abs_actual_error_pct": float(
                        np.min(
                            absolute_error
                        )
                    ),
                    "max_abs_actual_error_pct": float(
                        np.max(
                            absolute_error
                        )
                    ),
                    "median_distance_to_nearest_10pct_boundary_pp": float(
                        np.median(
                            boundary_distance
                        )
                    ),
                    "min_distance_to_nearest_10pct_boundary_pp": float(
                        np.min(
                            boundary_distance
                        )
                    ),
                    "max_distance_to_nearest_10pct_boundary_pp": float(
                        np.max(
                            boundary_distance
                        )
                    ),
                }
            )

    return rows


def build_misclassification_severity_by_subset(
    be: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    rows.extend(
        misclassification_pair_rows(
            be,
            subset_name="all_BE",
        )
    )

    wp_half = be.loc[
        np.isclose(
            be[
                "proxy_weight"
            ].to_numpy(dtype=float),
            0.5,
        )
    ].copy()

    rows.extend(
        misclassification_pair_rows(
            wp_half,
            subset_name="BE_WP_0.5",
        )
    )

    return pd.DataFrame(
        rows
    )


def predicted_state_summary_rows(
    frame: pd.DataFrame,
    *,
    subset_name: str,
) -> list[dict]:
    rows = []

    for predicted_state in STATE_LABELS:
        calls = frame.loc[
            frame[
                "predicted_state"
            ].eq(
                predicted_state
            )
        ].copy()

        if calls.empty:
            rows.append(
                {
                    "subset": subset_name,
                    "predicted_state": predicted_state,
                    "n_calls": 0,
                    "n_correct": 0,
                    "precision": np.nan,
                    "n_wrong": 0,
                    "wrong_median_abs_actual_error_pct": np.nan,
                    "wrong_min_abs_actual_error_pct": np.nan,
                    "wrong_max_abs_actual_error_pct": np.nan,
                    "wrong_median_distance_to_boundary_pp": np.nan,
                }
            )
            continue

        correct = calls[
            "actual_state"
        ].eq(
            predicted_state
        )

        wrong = calls.loc[
            ~correct
        ].copy()

        if wrong.empty:
            wrong_median = np.nan
            wrong_min = np.nan
            wrong_max = np.nan
            wrong_boundary = np.nan
        else:
            wrong_abs = np.abs(
                wrong[
                    "actual_signed_error_pct"
                ].to_numpy(dtype=float)
            )
            wrong_distance = (
                boundary_distance_percentage_points(
                    wrong[
                        "actual_signed_error_pct"
                    ].to_numpy(dtype=float)
                )
            )

            wrong_median = float(
                np.median(
                    wrong_abs
                )
            )
            wrong_min = float(
                np.min(
                    wrong_abs
                )
            )
            wrong_max = float(
                np.max(
                    wrong_abs
                )
            )
            wrong_boundary = float(
                np.median(
                    wrong_distance
                )
            )

        rows.append(
            {
                "subset": subset_name,
                "predicted_state": predicted_state,
                "n_calls": int(
                    len(calls)
                ),
                "n_correct": int(
                    np.sum(
                        correct
                    )
                ),
                "precision": float(
                    np.mean(
                        correct
                    )
                ),
                "n_wrong": int(
                    np.sum(
                        ~correct
                    )
                ),
                "wrong_median_abs_actual_error_pct": wrong_median,
                "wrong_min_abs_actual_error_pct": wrong_min,
                "wrong_max_abs_actual_error_pct": wrong_max,
                "wrong_median_distance_to_boundary_pp": wrong_boundary,
            }
        )

    return rows


def build_state_call_summary_by_subset(
    be: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    rows.extend(
        predicted_state_summary_rows(
            be,
            subset_name="all_BE",
        )
    )

    wp_half = be.loc[
        np.isclose(
            be[
                "proxy_weight"
            ].to_numpy(dtype=float),
            0.5,
        )
    ].copy()

    rows.extend(
        predicted_state_summary_rows(
            wp_half,
            subset_name="BE_WP_0.5",
        )
    )

    return pd.DataFrame(
        rows
    )


def confidence_at_threshold_for_probability(
    truth_state: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict:
    acceptable_index = int(
        np.where(
            STATE_LABELS
            == "acceptable"
        )[0][0]
    )

    p_acceptable = probabilities[
        :,
        acceptable_index,
    ]

    p_unacceptable = (
        probabilities[
            :,
            0,
        ]
        + probabilities[
            :,
            2,
        ]
    )

    truth_acceptable = (
        truth_state
        == "acceptable"
    )

    acceptable_mask = (
        p_acceptable
        >= threshold
    )

    unacceptable_mask = (
        p_unacceptable
        >= threshold
    )

    return {
        "acceptable_n_calls": int(
            np.sum(
                acceptable_mask
            )
        ),
        "acceptable_coverage": float(
            np.mean(
                acceptable_mask
            )
        ),
        "acceptable_precision": (
            float(
                np.mean(
                    truth_acceptable[
                        acceptable_mask
                    ]
                )
            )
            if acceptable_mask.any()
            else np.nan
        ),
        "unacceptable_n_calls": int(
            np.sum(
                unacceptable_mask
            )
        ),
        "unacceptable_coverage": float(
            np.mean(
                unacceptable_mask
            )
        ),
        "unacceptable_precision": (
            float(
                np.mean(
                    ~truth_acceptable[
                        unacceptable_mask
                    ]
                )
            )
            if unacceptable_mask.any()
            else np.nan
        ),
    }


def build_threshold_sensitivity(
    nl_oof: pd.DataFrame,
    be: pd.DataFrame,
) -> pd.DataFrame:
    """Sensitivity of the three-state interpretation to ±5/10/15/20% tolerances.

    The continuous case-9657 signed-error predictor is NOT changed. Only the
    state labels and the second-stage probability calibrator are rebuilt.
    """
    rows = []

    for acceptable_limit in THRESHOLD_SENSITIVITY_LIMITS:
        calibrator = fit_probability_calibrator_for_limit(
            nl_oof[
                "score"
            ].to_numpy(dtype=float),
            nl_oof[
                "signed_ldes_error"
            ].to_numpy(dtype=float),
            acceptable_limit,
        )

        all_probability = predict_probabilities(
            calibrator,
            be[
                "predicted_signed_error"
            ].to_numpy(dtype=float),
        )

        all_truth = state_from_error_at_limit(
            be[
                "signed_ldes_error"
            ].to_numpy(dtype=float),
            acceptable_limit,
        )

        all_predicted = STATE_LABELS[
            np.argmax(
                all_probability,
                axis=1,
            )
        ]

        subset_specs = [
            (
                "all_BE",
                np.ones(
                    len(be),
                    dtype=bool,
                ),
            ),
            (
                "BE_WP_0.5",
                np.isclose(
                    be[
                        "proxy_weight"
                    ].to_numpy(dtype=float),
                    0.5,
                ),
            ),
        ]

        for subset_name, mask in subset_specs:
            truth = all_truth[
                mask
            ]
            probability = all_probability[
                mask
            ]
            predicted = all_predicted[
                mask
            ]

            state_score = state_metrics(
                truth,
                probability,
            )

            confidence = (
                confidence_at_threshold_for_probability(
                    truth,
                    probability,
                    threshold=0.80,
                )
            )

            rows.append(
                {
                    "subset": subset_name,
                    "acceptable_limit_pct": (
                        100
                        * acceptable_limit
                    ),
                    "n_cases": int(
                        np.sum(
                            mask
                        )
                    ),
                    "state_accuracy": float(
                        accuracy_score(
                            truth,
                            predicted,
                        )
                    ),
                    "state_macro_f1": state_score[
                        "macro_f1"
                    ],
                    "acceptable_auc": state_score[
                        "acceptable_auc"
                    ],
                    "acceptable_ap": state_score[
                        "acceptable_ap"
                    ],
                    **confidence,
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Plot helpers
# =============================================================================

def save_figure(
    fig,
    stem: str,
) -> None:
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / f"{stem}.pdf",
        bbox_inches="tight",
    )
    plt.close(
        fig
    )



def plot_compact_main_figure(
    transfer_model: ContinuousDiagnostic,
    calibrator,
    table: pd.DataFrame,
    be: pd.DataFrame,
) -> None:
    """Create the proposed single-column, vertically stacked main-text figure."""
    # Matplotlib text sizes appropriate to ~3.45-inch single-column width.
    with plt.rc_context(
        {
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.4,
        }
    ):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(
                MAIN_FIGURE_WIDTH_IN,
                MAIN_FIGURE_HEIGHT_IN,
            ),
        )

        ax_a = axes[0]
        ax_b = axes[1]

        # ==============================================================
        # Panel A: actual vs predicted signed LDES error
        # ==============================================================
        actual = be[
            "actual_signed_error_pct"
        ].to_numpy(dtype=float)

        predicted = be[
            "predicted_signed_error_pct"
        ].to_numpy(dtype=float)

        max_abs = float(
            max(
                np.max(
                    np.abs(
                        actual
                    )
                ),
                np.max(
                    np.abs(
                        predicted
                    )
                ),
                20.0,
            )
        )

        limit = (
            np.ceil(
                max_abs / 10
            )
            * 10
        )

        # Neutral grey ±10% reference region: no semantic colour.
        ax_a.axvspan(
            -10,
            10,
            color="0.94",
            zorder=0,
        )
        ax_a.axhspan(
            -10,
            10,
            color="0.94",
            zorder=0,
        )

        for threshold in (
            -10,
            10,
        ):
            ax_a.axvline(
                threshold,
                linestyle="--",
                linewidth=0.75,
                color="0.62",
                zorder=1,
            )
            ax_a.axhline(
                threshold,
                linestyle="--",
                linewidth=0.75,
                color="0.62",
                zorder=1,
            )

        ax_a.plot(
            [-limit, limit],
            [-limit, limit],
            linestyle=":",
            linewidth=0.9,
            color="0.40",
            zorder=1,
        )

        correct = be[
            "class_correct"
        ].to_numpy(dtype=bool)

        false_acceptable = (
            be[
                "predicted_state"
            ].eq(
                "acceptable"
            )
            & ~be[
                "actual_state"
            ].eq(
                "acceptable"
            )
        ).to_numpy()

        false_unacceptable = (
            ~be[
                "predicted_state"
            ].eq(
                "acceptable"
            )
            & be[
                "actual_state"
            ].eq(
                "acceptable"
            )
        ).to_numpy()

        other_wrong = (
            ~correct
            & ~false_acceptable
            & ~false_unacceptable
        )

        # Plasma dark blue = correct; plasma pink = problematic false acceptable.
        ax_a.scatter(
            actual[
                correct
            ],
            predicted[
                correct
            ],
            s=12,
            marker="o",
            facecolor=CMAP(0.10),
            edgecolor="none",
            alpha=0.72,
            label="Correct",
            zorder=3,
        )

        ax_a.scatter(
            actual[
                false_acceptable
            ],
            predicted[
                false_acceptable
            ],
            s=20,
            marker="X",
            facecolor=CMAP(0.54),
            edgecolor="black",
            linewidth=0.20,
            alpha=0.90,
            label="Incorrect: predicted acceptable",
            zorder=4,
        )

        # Conservative errors are deliberately visually de-emphasised.
        ax_a.scatter(
            actual[
                false_unacceptable
            ],
            predicted[
                false_unacceptable
            ],
            s=20,
            marker="X",
            facecolor="0.6",
            edgecolor="black",
            linewidth=0.2,
            alpha=0.90,
            label="Incorrect: predicted unacceptable",
            zorder=4,
        )

        if np.any(
            other_wrong
        ):
            ax_a.scatter(
                actual[
                    other_wrong
                ],
                predicted[
                    other_wrong
                ],
                s=18,
                marker="+",
                color="0.30",
                linewidth=0.8,
                alpha=0.90,
                label="Other incorrect",
                zorder=4,
            )

        nl = table.loc[
            table[
                "country"
            ].eq(
                TRAIN_COUNTRY
            )
        ].copy()

        nl_prediction = transfer_model.predict(
            nl
        )

        nl_metrics = regression_metrics(
            nl[
                "signed_ldes_error"
            ].to_numpy(dtype=float),
            nl_prediction,
        )

        be_metrics = regression_metrics(
            be[
                "signed_ldes_error"
            ].to_numpy(dtype=float),
            be[
                "predicted_signed_error"
            ].to_numpy(dtype=float),
        )

        stats_text = (
            "NL: training\n"
            f"$R^2$ = {nl_metrics['r2']:.3f}, "
            f"RMSE = {100 * nl_metrics['rmse']:.1f} %-points\n\n"
            "BE: validation\n"
            f"$R^2$ = {be_metrics['r2']:.3f}, "
            f"RMSE = {100 * be_metrics['rmse']:.1f} %-points"
        )

        # ax_a.text(
        #     0.97,
        #     0.04,
        #     stats_text,
        #     transform=ax_a.transAxes,
        #     ha="right",
        #     va="bottom",
        #     bbox={
        #         "boxstyle": "round,pad=0.55",
        #         "facecolor": "white",
        #         "edgecolor": "0.65",
        #         "alpha": 0.94,
        #     },
        # )

        ax_a.set_xlim(
            -limit,
            limit,
        )
        ax_a.set_ylim(
            -limit,
            limit,
        )

        ax_a.set_xlabel(
            "Actual signed LDES capacity error (%)"
        )
        ax_a.set_ylabel(
            "Predicted signed LDES capacity error (%)"
        )
        ax_a.set_title(
            "(a) Signed-error diagnostic validation",
            loc="left",
        )

        ax_a.legend(
            frameon=False,
            loc="upper left",
            handletextpad=0.4,
            borderaxespad=0.15,
            labelspacing=0.35,
        )

        # ==============================================================
        # Panel B: clean three-state probability mapping
        # ==============================================================
        xmin = min(
            -0.55,
            float(
                be[
                    "predicted_signed_error"
                ].min()
            )
            - 0.04,
        )
        xmax = max(
            0.55,
            float(
                be[
                    "predicted_signed_error"
                ].max()
            )
            + 0.04,
        )

        score_grid = np.linspace(
            xmin,
            xmax,
            500,
        )

        probability_grid = predict_probabilities(
            calibrator,
            score_grid,
        )

        x_pct = (
            100
            * score_grid
        )

        curve_labels = {
            "under": "Under",
            "acceptable": "Acceptable",
            "over": "Over",
        }

        for j, state in enumerate(
            STATE_LABELS
        ):
            ax_b.plot(
                x_pct,
                probability_grid[
                    :,
                    j,
                ],
                linewidth=1.7,
                color=STATE_COLORS[
                    state
                ],
                label=curve_labels[
                    state
                ],
            )

        # The 0.8 line corresponds to the confidence threshold reported in prose.
        ax_b.axhline(
            0.80,
            linestyle="--",
            linewidth=0.75,
            color="0.58",
        )

        ax_b.text(
            0.99,
            0.81,
            "$P=0.8$",
            transform=ax_b.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color="0.45",
            fontsize=6.4,
        )

        ax_b.set_ylim(
            0,
            1.02,
        )

        ax_b.set_xlabel(
            "Predicted signed LDES capacity error (%)"
        )
        ax_b.set_ylabel(
            "Estimated class probability"
        )
        ax_b.set_title(
            "(b) Three-state probability mapping",
            loc="left",
        )

        # Direct labels are clearer than a legend for three smooth curves.
        x_span_b = 100 * (xmax - xmin)
        x_left_b = 100 * xmin + 0.15 * x_span_b
        x_right_b = 100 * xmax - 0.15 * x_span_b

        label_box_b = {
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
            "pad": 0.8,
        }

        ax_b.text(
            x_left_b,
            0.94,
            "Under",
            ha="center",
            va="center",
            fontsize=6.8,
            color=STATE_COLORS["under"],
            bbox=label_box_b,
        )
        ax_b.text(
            0.0,
            0.94,
            "Acceptable",
            ha="center",
            va="center",
            fontsize=6.8,
            color=STATE_COLORS["acceptable"],
            bbox=label_box_b,
        )
        ax_b.text(
            x_right_b,
            0.94,
            "Over",
            ha="center",
            va="center",
            fontsize=6.8,
            color=STATE_COLORS["over"],
            bbox=label_box_b,
        )

        fig.subplots_adjust(
            left=0.19,
            right=0.98,
            bottom=0.08,
            top=0.97,
            hspace=0.35,
        )

        save_figure(
            fig,
            "fig_9657_main_two_panel",
        )


def plot_threshold_sensitivity_appendix(
    final_combined_model: ContinuousDiagnostic,
    combined_calibrators: dict[float, object],
    table: pd.DataFrame,
) -> None:
    """Appendix plot: all three state curves for ±10/±15/±20%.

    Visual encoding
    ---------------
    colour -> acceptable-error tolerance

    State identity (under / acceptable / over) is communicated with direct
    floating labels rather than line style or a second legend.

    The curves represent the FINAL diagnostic, i.e. the continuous equation is
    refitted on NL+BE combined and the probability mappings are calibrated from
    chronology-held-out combined NL+BE scores.
    """
    final_score = final_combined_model.predict(
        table
    )

    xmin = min(
        -0.60,
        float(
            np.min(
                final_score
            )
        )
        - 0.05,
    )

    xmax = max(
        0.60,
        float(
            np.max(
                final_score
            )
        )
        + 0.05,
    )

    score_grid = np.linspace(
        xmin,
        xmax,
        800,
    )

    fig, ax = plt.subplots(
        figsize=(
            APPENDIX_FIGURE_WIDTH_IN,
            APPENDIX_FIGURE_HEIGHT_IN,
        )
    )

    # Nine solid curves: 3 thresholds x 3 states.
    # Colour denotes tolerance; floating labels identify state families.
    for acceptable_limit in THRESHOLD_SENSITIVITY_PLOT_LIMITS:
        calibrator = combined_calibrators[
            acceptable_limit
        ]

        probabilities = predict_probabilities(
            calibrator,
            score_grid,
        )

        for state_index, state in enumerate(
            STATE_LABELS
        ):
            ax.plot(
                100
                * score_grid,
                probabilities[
                    :,
                    state_index,
                ],
                color=THRESHOLD_PLOT_COLORS[
                    acceptable_limit
                ],
                linestyle="-",
                linewidth=2.0,
                alpha=0.95,
            )

    # Confidence reference used elsewhere in the diagnostic.
    ax.axhline(
        0.80,
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        color="0.60",
        zorder=0,
    )

    ax.text(
        0.99,
        0.81,
        "$P=0.8$",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.42",
    )

    # Floating state labels: the line style itself remains visible beneath/near
    # each region, avoiding a second bulky legend.
    x_span = (
        100
        * (
            xmax
            - xmin
        )
    )
    x_left = (
        100
        * xmin
        + 0.10
        * x_span
    )
    x_right = (
        100
        * xmax
        - 0.10
        * x_span
    )

    label_box = {
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.78,
        "pad": 1.5,
    }

    ax.text(
        x_left,
        0.93,
        "Under",
        ha="center",
        va="center",
        fontsize=9,
        color="0.18",
        bbox=label_box,
    )

    ax.text(
        0.0,
        0.93,
        "Acceptable",
        ha="center",
        va="center",
        fontsize=9,
        color="0.18",
        bbox=label_box,
    )

    ax.text(
        x_right,
        0.93,
        "Over",
        ha="center",
        va="center",
        fontsize=9,
        color="0.18",
        bbox=label_box,
    )

    # Compact colour legend for the acceptable-error tolerance.
    tolerance_handles = [
        plt.Line2D(
            [0],
            [0],
            color=THRESHOLD_PLOT_COLORS[
                acceptable_limit
            ],
            linewidth=2.2,
            linestyle="-",
            label=(
                rf"$\pm{int(100 * acceptable_limit)}\%$"
            ),
        )
        for acceptable_limit in THRESHOLD_SENSITIVITY_PLOT_LIMITS
    ]

    # One compact legend above the chart: colour identifies the
    # acceptable-error tolerance. State identity is conveyed directly by
    # the floating labels in the plotting area.
    ax.legend(
        handles=tolerance_handles,
        title="Acceptable-error tolerance",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="0.75",
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(
            0.5,
            1.03,
        ),
        columnspacing=1.2,
        handlelength=2.2,
        borderpad=0.45,
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_xlabel(
        "Predicted signed LDES capacity error (%)"
    )
    ax.set_ylabel(
        "Estimated class probability"
    )
    ax.set_title(
        "Sensitivity of the three-state probability mapping"
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.92)
    )

    save_figure(
        fig,
        "fig_9657_threshold_sensitivity_appendix",
    )

def plot_continuous_transfer(
    be: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(
        figsize=(7.2, 6.5)
    )

    max_abs = float(
        max(
            np.abs(
                be[
                    "actual_signed_error_pct"
                ]
            ).max(),
            np.abs(
                be[
                    "predicted_signed_error_pct"
                ]
            ).max(),
            20.0,
        )
    )

    limit = (
        np.ceil(
            max_abs / 10
        )
        * 10
    )

    # Acceptable region.
    ax.axvspan(
        -10,
        10,
        alpha=0.06,
        color=COLOR_ACCEPTABLE,
    )
    ax.axhspan(
        -10,
        10,
        alpha=0.06,
        color=COLOR_ACCEPTABLE,
    )

    ax.plot(
        [-limit, limit],
        [-limit, limit],
        linestyle="--",
        linewidth=1.2,
        color=CMAP(0.02),
        alpha=0.8,
    )

    ax.axvline(
        -10,
        linestyle=":",
        linewidth=1.0,
        color=CMAP(0.15),
    )
    ax.axvline(
        10,
        linestyle=":",
        linewidth=1.0,
        color=CMAP(0.85),
    )
    ax.axhline(
        -10,
        linestyle=":",
        linewidth=1.0,
        color=CMAP(0.15),
    )
    ax.axhline(
        10,
        linestyle=":",
        linewidth=1.0,
        color=CMAP(0.85),
    )

    # Colour markers by W_P, marker shape by actual class.
    norm = plt.Normalize(
        vmin=min(WP_ORDER),
        vmax=max(WP_ORDER),
    )

    scatter_for_colorbar = None

    for state in STATE_LABELS:
        subset = be.loc[
            be[
                "actual_state"
            ].eq(
                state
            )
        ].copy()

        correct = subset.loc[
            subset[
                "class_correct"
            ]
        ]
        incorrect = subset.loc[
            ~subset[
                "class_correct"
            ]
        ]

        if not correct.empty:
            scatter_for_colorbar = ax.scatter(
                correct[
                    "actual_signed_error_pct"
                ],
                correct[
                    "predicted_signed_error_pct"
                ],
                c=correct[
                    "proxy_weight"
                ],
                cmap=CMAP,
                norm=norm,
                marker=STATE_MARKERS[
                    state
                ],
                s=44,
                edgecolors="black",
                linewidths=0.35,
                alpha=0.82,
            )

        if not incorrect.empty:
            ax.scatter(
                incorrect[
                    "actual_signed_error_pct"
                ],
                incorrect[
                    "predicted_signed_error_pct"
                ],
                c=incorrect[
                    "proxy_weight"
                ],
                cmap=CMAP,
                norm=norm,
                marker="X",
                s=58,
                edgecolors="black",
                linewidths=0.40,
                alpha=0.95,
            )

    be_reg = regression_metrics(
        be[
            "signed_ldes_error"
        ].to_numpy(dtype=float),
        be[
            "predicted_signed_error"
        ].to_numpy(dtype=float),
    )

    ax.text(
        0.97,
        0.03,
        (
            "BE cross-country transfer\n"
            f"$R^2$ = {be_reg['r2']:.3f}\n"
            f"RMSE = {100 * be_reg['rmse']:.1f} %-points\n"
            f"MAE = {100 * be_reg['mae']:.1f} %-points"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.90,
        },
    )

    if scatter_for_colorbar is not None:
        cbar = fig.colorbar(
            scatter_for_colorbar,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        cbar.set_label(
            r"Proxy weight $W_P$"
        )

    ax.set_xlim(
        -limit,
        limit,
    )
    ax.set_ylim(
        -limit,
        limit,
    )
    ax.set_xlabel(
        "Actual signed LDES capacity error (%)"
    )
    ax.set_ylabel(
        "Predicted signed LDES capacity error (%)"
    )
    ax.set_title(
        "Continuous ex-post diagnostic: NL → BE transfer"
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=STATE_MARKERS[
                state
            ],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=f"Actual: {state}",
        )
        for state in STATE_LABELS
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="X",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="Incorrect three-state call",
        )
    )

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig_9657_be_continuous_transfer",
    )


def plot_probability_curves(
    calibrator,
    be: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(
        figsize=(8.0, 6.4)
    )

    xmin = min(
        -0.55,
        float(
            be[
                "predicted_signed_error"
            ].min()
        )
        - 0.04,
    )

    xmax = max(
        0.55,
        float(
            be[
                "predicted_signed_error"
            ].max()
        )
        + 0.04,
    )

    score_grid = np.linspace(
        xmin,
        xmax,
        500,
    )

    probability_grid = predict_probabilities(
        calibrator,
        score_grid,
    )

    x_pct = (
        100
        * score_grid
    )

    for j, state in enumerate(
        STATE_LABELS
    ):
        ax.plot(
            x_pct,
            probability_grid[
                :,
                j,
            ],
            linewidth=2.3,
            color=STATE_COLORS[
                state
            ],
            label=(
                "Under-investment"
                if state == "under"
                else "Acceptable"
                if state == "acceptable"
                else "Over-investment"
            ),
        )

    ax.axvline(
        -10,
        linestyle="--",
        linewidth=1.0,
        color=COLOR_UNDER,
        alpha=0.8,
    )
    ax.axvline(
        10,
        linestyle="--",
        linewidth=1.0,
        color=COLOR_OVER,
        alpha=0.8,
    )

    # Plot every BE case on the probability curve corresponding to the
    # *actual* observed class.
    for state in STATE_LABELS:
        subset = be.loc[
            be[
                "actual_state"
            ].eq(
                state
            )
        ].copy()

        probability_column = f"p_{state}"

        correct = subset.loc[
            subset[
                "class_correct"
            ]
        ]
        incorrect = subset.loc[
            ~subset[
                "class_correct"
            ]
        ]

        ax.scatter(
            correct[
                "predicted_signed_error_pct"
            ],
            correct[
                probability_column
            ],
            marker="o",
            s=34,
            facecolors=STATE_COLORS[
                state
            ],
            edgecolors="black",
            linewidths=0.25,
            alpha=0.58,
        )

        ax.scatter(
            incorrect[
                "predicted_signed_error_pct"
            ],
            incorrect[
                probability_column
            ],
            marker="X",
            s=54,
            facecolors=STATE_COLORS[
                state
            ],
            edgecolors="black",
            linewidths=0.35,
            alpha=0.90,
        )

    # Specifically ring false-acceptable calls.
    false_acceptable = be.loc[
        be[
            "false_acceptable"
        ]
    ]

    if not false_acceptable.empty:
        y = np.asarray(
            [
                row[
                    f"p_{row['actual_state']}"
                ]
                for _, row in false_acceptable.iterrows()
            ],
            dtype=float,
        )

        ax.scatter(
            false_acceptable[
                "predicted_signed_error_pct"
            ],
            y,
            marker="s",
            s=100,
            facecolors="none",
            edgecolors=CMAP(0.98),
            linewidths=1.3,
            label="False acceptable call",
        )

    probability = be[
        [
            "p_under",
            "p_acceptable",
            "p_over",
        ]
    ].to_numpy(dtype=float)

    metrics = state_metrics(
        be[
            "actual_state"
        ].to_numpy(dtype=object),
        probability,
    )

    ax.text(
        0.98,
        0.97,
        (
            "BE cross-country transfer\n"
            f"Macro-F1 = {metrics['macro_f1']:.3f}\n"
            f"Brier = {metrics['brier']:.3f}\n"
            f"Log loss = {metrics['log_loss']:.3f}\n"
            f"Acceptable AUROC/AP = "
            f"{metrics['acceptable_auc']:.3f}/"
            f"{metrics['acceptable_ap']:.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.90,
        },
    )

    ax.set_ylim(
        0,
        1,
    )
    ax.set_xlabel(
        "Predicted signed LDES capacity error (%)"
    )
    ax.set_ylabel(
        "Estimated class probability"
    )
    ax.set_title(
        "Three-state diagnostic probabilities: NL → BE transfer"
    )

    extra = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="BE case: top class correct",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="X",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="BE case: top class incorrect",
        ),
    ]

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles=handles + extra,
        labels=labels
        + [
            "BE case: top class correct",
            "BE case: top class incorrect",
        ],
        frameon=False,
        loc="upper left",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig_9657_be_probability_curves",
    )


def plot_acceptable_probability_bars(
    be: pd.DataFrame,
) -> None:
    subset = be.loc[
        be[
            "predicted_state"
        ].eq(
            "acceptable"
        )
    ].copy()

    if subset.empty:
        return

    subset = subset.sort_values(
        [
            "p_acceptable",
            "predicted_signed_error",
        ],
        ascending=[
            False,
            True,
        ],
    )

    if len(subset) > MAX_ACCEPTABLE_BAR_CASES:
        subset = subset.head(
            MAX_ACCEPTABLE_BAR_CASES
        ).copy()

    subset = subset.sort_values(
        "predicted_signed_error"
    ).reset_index(
        drop=True
    )

    y = np.arange(
        len(subset)
    )

    fig_height = max(
        6.5,
        0.24 * len(subset)
        + 1.8,
    )

    fig, ax = plt.subplots(
        figsize=(10.5, fig_height)
    )

    left = np.zeros(
        len(subset)
    )

    for state in STATE_LABELS:
        values = subset[
            f"p_{state}"
        ].to_numpy(dtype=float)

        ax.barh(
            y,
            values,
            left=left,
            height=0.75,
            color=STATE_COLORS[
                state
            ],
            edgecolor="none",
            label=state.capitalize(),
        )

        left += values

    for i, row in subset.iterrows():
        correct = (
            row[
                "actual_state"
            ]
            == "acceptable"
        )

        symbol = (
            "✓"
            if correct
            else "✗"
        )

        text = (
            f"{symbol} actual {row['actual_signed_error_pct']:+.1f}%"
            f" | pred {row['predicted_signed_error_pct']:+.1f}%"
        )

        ax.text(
            1.015,
            i,
            text,
            va="center",
            ha="left",
            fontsize=8.7,
            color=(
                CMAP(0.12)
                if correct
                else CMAP(0.96)
            ),
        )

    ax.set_yticks(
        y
    )
    ax.set_yticklabels(
        subset[
            "case_label"
        ],
        fontsize=8.0,
    )
    ax.invert_yaxis()

    ax.set_xlim(
        0,
        1,
    )
    ax.set_xlabel(
        "Predicted class probability"
    )
    ax.set_title(
        "BE cases classified as acceptable"
    )

    ax.legend(
        frameon=False,
        loc="upper left",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig_9657_be_acceptable_probability_bars",
    )


def plot_confusion_matrix(
    be: pd.DataFrame,
) -> None:
    matrix = confusion_matrix(
        be[
            "actual_state"
        ],
        be[
            "predicted_state"
        ],
        labels=STATE_LABELS.tolist(),
        normalize="true",
    )

    fig, ax = plt.subplots(
        figsize=(6.2, 5.5)
    )

    image = ax.imshow(
        matrix,
        cmap=CMAP,
        vmin=0,
        vmax=1,
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        "Share of actual class"
    )

    short_labels = [
        "Under",
        "Acceptable",
        "Over",
    ]

    ax.set_xticks(
        np.arange(3)
    )
    ax.set_yticks(
        np.arange(3)
    )

    ax.set_xticklabels(
        short_labels
    )
    ax.set_yticklabels(
        short_labels
    )

    for i in range(3):
        for j in range(3):
            value = matrix[
                i,
                j,
            ]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > 0.50
                    else "black"
                ),
                fontsize=11,
            )

    ax.set_xlabel(
        "Predicted state"
    )
    ax.set_ylabel(
        "Actual state"
    )
    ax.set_title(
        "BE three-state classification"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig_9657_be_confusion_matrix",
    )


def plot_false_acceptable_severity(
    be: pd.DataFrame,
) -> None:
    subset = be.loc[
        be[
            "predicted_state"
        ].eq(
            "acceptable"
        )
    ].copy()

    if subset.empty:
        return

    subset = subset.sort_values(
        "actual_signed_error_pct"
    ).reset_index(
        drop=True
    )

    x = np.arange(
        len(subset)
    )

    fig, ax = plt.subplots(
        figsize=(9.0, 5.3)
    )

    ax.axhspan(
        -10,
        10,
        color=COLOR_ACCEPTABLE,
        alpha=0.10,
        label="True acceptable range",
    )

    scatter = ax.scatter(
        x,
        subset[
            "actual_signed_error_pct"
        ],
        c=subset[
            "p_acceptable"
        ],
        cmap=CMAP,
        vmin=0,
        vmax=1,
        s=48,
        marker="o",
        edgecolors="black",
        linewidths=0.30,
    )

    false = subset.loc[
        ~subset[
            "actual_state"
        ].eq(
            "acceptable"
        )
    ]

    if not false.empty:
        false_positions = false.index.to_numpy()

        ax.scatter(
            false_positions,
            false[
                "actual_signed_error_pct"
            ],
            marker="X",
            s=72,
            c=false[
                "p_acceptable"
            ],
            cmap=CMAP,
            vmin=0,
            vmax=1,
            edgecolors="black",
            linewidths=0.35,
            label="False acceptable call",
        )

    ax.axhline(
        -10,
        linestyle="--",
        linewidth=1.0,
        color=COLOR_UNDER,
    )
    ax.axhline(
        10,
        linestyle="--",
        linewidth=1.0,
        color=COLOR_OVER,
    )

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        r"$P(\mathrm{acceptable})$"
    )

    ax.set_xlabel(
        "BE cases classified as acceptable"
    )
    ax.set_ylabel(
        "Actual signed LDES capacity error (%)"
    )
    ax.set_title(
        "Severity of false-acceptable BE calls"
    )

    ax.set_xticks(
        []
    )

    if not false.empty:
        absolute_false = np.abs(
            false[
                "actual_signed_error_pct"
            ].to_numpy()
        )

        exceedance = (
            absolute_false
            - 10
        )

        text = (
            f"False acceptable cases: {len(false)}/{len(subset)}\n"
            f"Median |actual error| = {np.median(absolute_false):.1f}%\n"
            f"Median exceedance = {np.median(exceedance):.1f} %-points\n"
            f"Maximum |actual error| = {np.max(absolute_false):.1f}%"
        )

        ax.text(
            0.02,
            0.97,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.6",
                "alpha": 0.90,
            },
        )

    if not false.empty:
        ax.legend(
            frameon=False,
            loc="lower right",
        )

    fig.tight_layout()

    save_figure(
        fig,
        "fig_9657_be_false_acceptable_severity",
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("Building case 9657 diagnostic directly from parquet result objects")
    print("=" * 100)

    table = build_case_table()

    print(
        f"Cases: {len(table)} | "
        f"NL={int(table['country'].eq('NL').sum())} | "
        f"BE={int(table['country'].eq('BE').sum())} | "
        f"chronologies={table['chronology_id'].nunique()} | "
        f"W_P={sorted(table['proxy_weight'].unique())}"
    )

    # ------------------------------------------------------------------
    # NL -> BE transfer model / probability calibration.
    # ------------------------------------------------------------------
    (
        transfer_model,
        calibrator,
        nl_oof,
        be,
    ) = build_be_predictions(
        table
    )

    # ------------------------------------------------------------------
    # Final equation refitted on the complete NL+BE dataset.
    # ------------------------------------------------------------------
    combined_model = ContinuousDiagnostic.fit(
        table
    )

    # Final three-state mappings for manuscript equations and appendix
    # sensitivity analysis. These are based on NL+BE combined data.
    #
    # - final continuous equation: fit on all NL+BE cases;
    # - final probability equations: calibrated from chronology-held-out
    #   predictions generated across the combined NL+BE set.
    (
        combined_oof,
        combined_calibrators,
    ) = build_final_combined_probability_calibrators(
        table
    )

    (
        OUTPUT_DIR
        / "equation_9657_nl_transfer.txt"
    ).write_text(
        transfer_model.equation_text(
            title=(
                "Case 9657 continuous diagnostic: "
                "NL training equation used for BE transfer"
            )
        ),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR
        / "equation_9657_combined_final.txt"
    ).write_text(
        combined_model.equation_text(
            title=(
                "Case 9657 continuous diagnostic: "
                "final NL+BE combined equation"
            )
        ),
        encoding="utf-8",
    )


    # Probability equations for every threshold sensitivity. These are the
    # equations suitable for reporting in the appendix: unlike the NL -> BE
    # transfer versions, they are calibrated from the combined NL+BE dataset.
    for acceptable_limit in THRESHOLD_SENSITIVITY_LIMITS:
        threshold_pct = int(
            round(
                100
                * acceptable_limit
            )
        )

        equation_path = (
            OUTPUT_DIR
            / (
                "equation_9657_probability_"
                f"{threshold_pct:02d}pct_combined_final.txt"
            )
        )

        equation_path.write_text(
            probability_equation_text(
                combined_calibrators[
                    acceptable_limit
                ],
                acceptable_limit=acceptable_limit,
                title=(
                    "Case 9657 three-state probability mapping: "
                    f"±{threshold_pct}% acceptable tolerance, "
                    "final NL+BE combined model"
                ),
            ),
            encoding="utf-8",
        )

    combined_oof.to_csv(
        OUTPUT_DIR
        / "combined_oof_scores_9657.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Metrics / case outputs.
    # ------------------------------------------------------------------
    metric_summary = build_metric_summary(
        table,
        transfer_model,
        calibrator,
        be,
    )

    metric_summary.to_csv(
        OUTPUT_DIR
        / "summary_9657_metrics.csv",
        index=False,
    )

    confidence = build_confidence_table(
        be
    )

    confidence.to_csv(
        OUTPUT_DIR
        / "confidence_9657_be.csv",
        index=False,
    )

    # Requested comparison: all BE cases vs W_P = 0.5 only.
    confidence_by_subset = build_confidence_by_subset(
        be
    )

    confidence_by_subset.to_csv(
        OUTPUT_DIR
        / "confidence_9657_be_all_vs_wp05.csv",
        index=False,
    )

    false_acceptable = build_false_acceptable_summary(
        be
    )

    false_acceptable.to_csv(
        OUTPUT_DIR
        / "false_acceptable_summary_9657_be.csv",
        index=False,
    )

    # Detailed severity of every off-diagonal three-state judgement.
    misclassification_severity = (
        build_misclassification_severity_by_subset(
            be
        )
    )

    misclassification_severity.to_csv(
        OUTPUT_DIR
        / "misclassification_severity_9657_be_all_vs_wp05.csv",
        index=False,
    )

    state_call_summary = (
        build_state_call_summary_by_subset(
            be
        )
    )

    state_call_summary.to_csv(
        OUTPUT_DIR
        / "state_call_summary_9657_be_all_vs_wp05.csv",
        index=False,
    )

    # Optional appendix evidence for the arbitrariness of the ±10% tolerance.
    # The continuous diagnostic is held fixed; only the three-state labels and
    # probability calibration are regenerated at ±5%, ±10%, ±15% and ±20%.
    threshold_sensitivity = build_threshold_sensitivity(
        nl_oof,
        be,
    )

    threshold_sensitivity.to_csv(
        OUTPUT_DIR
        / "threshold_sensitivity_9657_be.csv",
        index=False,
    )

    be_output_columns = [
        "case_id",
        "country",
        "start_date",
        "end_date",
        "chronology_id",
        "k_periods",
        WP_FIELD,
        "proxy_weight",
        *METRIC_FEATURES,
        "signed_ldes_error",
        "predicted_signed_error",
        "actual_signed_error_pct",
        "predicted_signed_error_pct",
        "actual_state",
        "predicted_state",
        "class_correct",
        "false_acceptable",
        "p_under",
        "p_acceptable",
        "p_over",
    ]

    be[
        be_output_columns
    ].to_csv(
        OUTPUT_DIR
        / "cases_9657_be.csv",
        index=False,
    )

    nl_oof.to_csv(
        OUTPUT_DIR
        / "nl_oof_scores_9657.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Main-paper figure: compact, vertically stacked, single-column.
    # ------------------------------------------------------------------
    plot_compact_main_figure(
        transfer_model,
        calibrator,
        table,
        be,
    )

    # Appendix threshold-sensitivity figure. Unlike panel (b) above, which is
    # part of the NL -> BE method validation, these curves correspond to the
    # FINAL diagnostic trained/calibrated on NL+BE combined.
    plot_threshold_sensitivity_appendix(
        combined_model,
        combined_calibrators,
        table,
    )

    # The older exploratory figure functions remain in this script for easy
    # appendix use, but are intentionally not called here.

    # ------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("NL -> BE transfer metrics")
    print("=" * 100)

    print(
        metric_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("BE confidence thresholds: all cases vs W_P = 0.5")
    print("=" * 100)

    print(
        confidence_by_subset.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("BE misclassification severity: all cases vs W_P = 0.5")
    print("=" * 100)

    print(
        misclassification_severity.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("BE state-call summary: all cases vs W_P = 0.5")
    print("=" * 100)

    print(
        state_call_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("Three-state tolerance sensitivity (±5%, ±10%, ±15%, ±20%)")
    print("=" * 100)

    print(
        threshold_sensitivity.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("Diagnostic predictors")
    print("=" * 100)

    for i, spec in enumerate(
        FEATURE_SPECS,
        start=1,
    ):
        print(
            f"X{i}: {spec['paper_label']}"
        )

    print(
        rf"X4: $W_P^* = {WP_FIELD} - {WP_CENTRE}$"
    )

    print("\nOutputs written to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
