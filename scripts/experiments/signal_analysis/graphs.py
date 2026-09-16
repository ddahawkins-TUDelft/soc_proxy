import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from scipy.stats import spearmanr
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
import itertools
import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

DF_CACHE_AllY = pd.read_csv('SoC_proxy_TSA/data/signal_cache/signal_cache_master_5and10Y_WpSubset.csv')
DF_CACHE_10Y = pd.read_csv('SoC_proxy_TSA/data/signal_cache/signal_cache_master_10Y_allWp.csv')
DF_CACHE_DIAGNOSTIC = pd.read_csv('SoC_proxy_TSA/data/signal_cache/signal_cache_master.csv')

DF_COMBINED = (
    pd.concat([DF_CACHE_AllY, DF_CACHE_10Y], ignore_index=True)
      .drop_duplicates(subset="model_id", keep="first")
)


COLOUR_MACME = "#0D0887" 
COLOUR_LDES  = "#CC4778"
COLOUR_HIGHLIGHT = "#fdb42f"
COLOUR_W0_EDGE = "#666666"
COLOUR_GREY_MEAN = "#666666"
CMAP_name = 'plasma'

COLOUR = COLOUR_LDES



cmap_density = LinearSegmentedColormap.from_list(
    "white_to_ldes",
    ["#FFFFFF", COLOUR],
    N=256
)
cmap_density.set_under("#FFFFFF") 


plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})
matplotlib.rcParams["pgf.preamble"] = r""

fig_width = 3.5
fig_height = 3.5

WINDOWS = [15, 30, 45, 90, 180, 273, 365,548,730, 912]
SUBPLOT_DIMS= [ 3,3]


# KEY Error Channels

#Approx Errors
#e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range
#e_proxy_delta_clu_vs_cem_delta_clu_nrmse_range
#e_proxy_ref_nrmse_range
#e_proxy_clu_nrmse_range

#TSA Errors
#e_tsa_proxy_nrmse_range
#e_tsa_proxy_delta_nrmse_range

#CEM Errors
#e_tsa_cem_nrmse_range
#ldes_cap_error_abs

#window metrics
#
#
#
#
#


def robust_limits(x, y, lo=1, hi=99, pad_frac=0.1):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    xmin, xmax = np.percentile(x, [lo, hi])
    ymin, ymax = np.percentile(y, [lo, hi])

    # padding
    xpad = (xmax - xmin) * pad_frac if xmax > xmin else 1e-6
    ypad = (ymax - ymin) * pad_frac if ymax > ymin else 1e-6

    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)
def run_regression(df, window, use_deltas = False, compare_soc_not_ldes_cap = False):

    x1_field = f'win{window}d_tsa_proxy_delta_nrmse_range' if use_deltas else f'win{window}d_tsa_proxy_nrmse_range'
    x2_field = f'win{window}d_proxy_delta_ref_vs_cem_delta_ref_nrmse_range' if use_deltas else f'win{window}d_proxy_ref_vs_cem_ref_nrmse_range'
    y_field = f'win{window}d_tsa_cem_nrmse_range' if compare_soc_not_ldes_cap else 'ldes_cap_error_abs'


    df =  df[[y_field,x1_field,x2_field]].dropna()

    y_all = df[y_field].to_numpy()
    x_all = df[
        [x1_field,x2_field]
    ].to_numpy()

    # X = sm.add_constant(x_all)
    X=x_all
    model = sm.OLS(y_all, X).fit()
    

    return model.rsquared, model.params, model.pvalues, model.nobs
def run_regression_general(df, y_field, x_fields):


    df =  df[[y_field]+x_fields].dropna()

    y = df[y_field].to_numpy()
    x = df[x_fields].to_numpy()

    # X = sm.add_constant(x_all
    model = sm.OLS(y, x).fit()
    

    return model.rsquared, model.params, model.pvalues, model.nobs
def add_row(rows, window, wp, r2, beta, p, n, x_fields):

    if len(beta) == len(x_fields):

        row = {
            "window": int(window),
            "Wp": wp,
            "n": n,
            "r2": float(r2),
            
        }

        # dynamically add beta fields
        for name, beta, p in zip(x_fields, beta, p):
            row[f'beta_{name}'] = float(beta)
            row[f'p_{name}'] = float(p)

        rows.append(row)

    return rows
def density_plot(fig, ax, df, y_field, x_field, show_trend=True, df_scatter=None, free_intercept=False, annotate_r2=True, y_min=0, x_lim = None, y_lim = None):


    # 1) Density background
    x_all = df[x_field].to_numpy()
    y_all = df[y_field].to_numpy()

    # hb = ax.hexbin(x_all, y_all, gridsize=45, mincnt=1)  # don’t set colors explicitly
    # fig.colorbar(hb, ax=ax, label="count")

    (xmin, xmax), (ymin, ymax) = robust_limits(x_all, y_all, lo=0.5, hi=99.5, pad_frac=0.05)
    # xmin, ymin = 0, 0
    if y_min!='auto':
        ymin=y_min

    if x_lim is not None:
        xmin, xmax = x_lim[0], x_lim[1]
    if y_lim is not None:
        ymin, ymax = y_lim[0], y_lim[1]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin,ymax)

    # Create evaluation grid
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 250), np.linspace(ymin, ymax, 250))

    # Fit KDE on all points
    kde = gaussian_kde(np.vstack([x_all, y_all]))

    # Evaluate density on the grid
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    zmin_draw = np.quantile(zz,0.25)
    zmax = zz.max()

    levels = np.linspace(zmin_draw, zmax, 20)

    # Plot as smooth gradient (filled contours)
    cf = ax.contourf(xx, yy, zz, 
                     levels=levels,
                     cmap=cmap_density, vmin=zmin_draw, vmax = zmax)
    fig.colorbar(cf, ax=ax, label="density")

    #subset of data

    if df_scatter is not None:
        x = df_scatter[x_field].to_numpy()
        y = df_scatter[y_field].to_numpy()

        ax.scatter(
            x, 
            y, 
            s=18, 
            alpha=0.9, 
            linewidths=0.3,
            color=COLOUR_HIGHLIGHT
            )
    
    #trendline
    # trendline
    if show_trend:
        xs = np.linspace(np.min(x_all), np.max(x_all), 200)

        # Fit model on the data (x_all, y_all)
        if free_intercept:
            X = sm.add_constant(x_all)          # shape: (n, 2)
        else:
            X = x_all.reshape(-1, 1)            # shape: (n, 1)

        model = sm.OLS(y_all, X).fit()

        # Predict on the grid xs (not on x_all)
        if free_intercept:
            yhat = model.params[0] + model.params[1] * xs
            intercept= model.params[0]
            beta = model.params[1]
            p_beta = model.pvalues[1]
        else:
            yhat = model.params[0] * xs
            beta = model.params[0]
            intercept= 0
            p_beta = model.pvalues[0]

        ax.plot(xs, yhat, linestyle="dashed", color=COLOUR, linewidth=2)
        if annotate_r2:
            txt = (
            rf"$R^2 = {model.rsquared:.2f}$" "\n"
            rf"$\beta = {beta:.2f}$" "\n"
            rf"$\alpha = {intercept:.2f}$" "\n"
            rf"$n = {model.nobs:.0f}$"
            )
            ax.text(
                0.05,
                0.95,
                txt,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.1),
            )


    
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)

    return fig, ax



    # print('done')


    # df = df_cache
    # #general filters
    # # df = df[df['country']=='NL'] #focus on NL
    # df = df[df['k']>30] #remove k=30
    # # df = df[df['k']<365] 

    # #horizon filtering
    # years = DF_CACHE["dates"].str.split(",", expand=True)
    # df.loc[:, "horizon_length"] = (
    # years[1].astype(int)
    # - years[0].astype(int)
    # + 1
    # )
    # df = df[df['horizon_length']>=5]
    
    # rows=[]
    # windows = WINDOWS

    # for window in windows:

    #     y_field = 'ldes_cap_error_abs' #f'win{window}d_tsa_cem_nrmse_range'
    #     x_fields = [
    #         f'win{window}d_tsa_proxy_delta_nrmse_range',
    #         f'win{window}d_proxy_delta_ref_vs_cem_delta_ref_nrmse_range',
    #         f'win{window}d_tsa_proxy_nrmse_range',
    #         f'win{window}d_proxy_ref_vs_cem_ref_nrmse_range',
    #         ]

    #     r2, beta, p, n = run_regression_general(
    #         df=df, 
    #         y_field = y_field, 
    #         x_fields=x_fields
    #     )

    #     # rows = add_row(rows,
    #     #                window,
    #     #                'all',
    #     #                r2,
    #     #                beta,
    #     #                p, 
    #     #                n)
        
    #     df_W_gt_0 = df.copy()
    #     df_W_gt_0 = df_W_gt_0[df_W_gt_0['Wp']>0]

    #     r2, beta, p, n = run_regression_general(
    #         df=df_W_gt_0, 
    #         y_field = y_field, 
    #         x_fields=x_fields
    #     )

    #     rows = add_row(rows,
    #                    window,
    #                    'Wp>0',
    #                    r2,
    #                    beta,
    #                    p, 
    #                    n)


    #     for wp in [0]:
    #         # subset
    #         df_subset = df.copy()
    #         df_subset = df_subset[df_subset['Wp']==wp] #focus on Wp=0.5

    #         r2, beta, p, n = run_regression_general(
    #             df=df_subset, 
    #             y_field = y_field, 
    #             x_fields=x_fields
    #         )
    #         rows = add_row(rows,
    #                    window,
    #                    wp,
    #                    r2,
    #                    beta,
    #                    p,
    #                    n)
            
    # df_results = pd.DataFrame(rows)
    # print(df_results.to_string(index=False))
def diagnostic_stats(x: pd.Series, y: pd.Series, show_prediction_interval=False):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return dict(r2=np.nan, beta=np.nan, p_beta=np.nan,
                    rho=np.nan, p_rho=np.nan)

    x2 = x[mask]
    y2 = y[mask]

    X = sm.add_constant(x2)
    m = sm.OLS(y2, X).fit()

    if show_prediction_interval:
        pred = m.get_prediction(X)
        sf = pred.summary_frame(alpha=0.05)

        print(sf[['mean', 'mean_ci_lower', 'mean_ci_upper', 'obs_ci_lower', 'obs_ci_upper']].head())

    rho, p_rho = spearmanr(x2, y2)

    return dict(
        r2=m.rsquared,
        beta=m.params.iloc[1],
        p_beta=m.pvalues.iloc[1],
        rho=rho,
        p_rho=p_rho,
    )
def fit_diagnostic_model(
    df: pd.DataFrame,
    *,
    y_field: str,
    x_fields: list[str],
    interactions: list[tuple[str, str]] | None = None,
    add_intercept: bool = True,
    standardize: bool = False,
    add_spearman: bool = True,
    add_vif: bool = True,
    min_n: int = 20,
    return_model: bool = False,
    return_design: bool = False,
    show_prediction_interval:bool = False
) -> dict:
    """
    Fit a linear model Y ~ X (+ interactions) and return a tidy dict of stats.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing y_field and x_fields.
    y_field : str
        Dependent variable column name.
    x_fields : list[str]
        Main-effect predictors column names.
    interactions : list[tuple[str, str]] | None
        Pairs (a, b) indicating interaction terms a*b to include.
        Example: interactions=[("X", "M")] adds X*M.
        NOTE: Only interacts raw columns (after optional standardization).
    add_intercept : bool
        If True, includes an intercept (constant) in OLS.
    standardize : bool
        If True, z-score standardizes all predictors AND y before fitting.
        Useful for comparing magnitudes; slopes become standardized betas.
    add_spearman : bool
        If True, compute Spearman rho+p between each main X and Y (pairwise).
    add_vif : bool
        If True, compute VIF for the design matrix columns (excluding intercept).
    min_n : int
        Minimum number of finite rows required to fit.

    Returns
    -------
    dict
        Keys include:
        - 'n', 'r2', 'r2_adj', 'aic', 'bic', 'rmse', 'y_field'
        - 'terms': list of term names used in the model
        - 'params': dict(term -> coefficient)
        - 'pvalues': dict(term -> pvalue)
        - 'stderr': dict(term -> std err)
        - 'tvalues': dict(term -> t stat)
        - 'spearman': dict(x -> {'rho','p'}) (optional)
        - 'vif': dict(term -> vif) (optional)
        - 'design_info': metadata about standardization / intercept / interactions
    """
    interactions = interactions or []

    # --- build a working df with required columns
    needed = [y_field] + list(dict.fromkeys(x_fields))  # preserve order, de-dupe
    for a, b in interactions:
        if a not in needed:
            needed.append(a)
        if b not in needed:
            needed.append(b)

    d = df[needed].copy()

    # --- coerce to numeric (quietly) for safety
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # --- drop non-finite rows
    mask = np.isfinite(d[y_field].to_numpy())
    for c in needed[1:]:
        mask &= np.isfinite(d[c].to_numpy())

    d = d.loc[mask].copy()
    n = int(len(d))
    if n < min_n:
        return {
            "y_field": y_field,
            "x_fields": x_fields,
            "interactions": interactions,
            "n": n,
            "error": f"Too few finite rows to fit model (n={n} < {min_n}).",
        }

    # --- construct X matrix (main effects)
    X = d[x_fields].copy()

    # optional standardization
    y = d[y_field].astype(float).to_numpy()
    if standardize:
        # z-score y and each X column
        y_mean, y_std = float(np.mean(y)), float(np.std(y))
        y = (y - y_mean) / (y_std if y_std != 0 else 1.0)

        for c in X.columns:
            m, s = float(X[c].mean()), float(X[c].std())
            X[c] = (X[c] - m) / (s if s != 0 else 1.0)

        # also standardize any interaction source columns if not already in X
        # (they'll be pulled from d and standardized similarly)
        for a, b in interactions:
            for c in (a, b):
                if c not in X.columns:
                    m, s = float(d[c].mean()), float(d[c].std())
                    d[c] = (d[c] - m) / (s if s != 0 else 1.0)

    # --- add interaction columns
    interaction_terms: list[str] = []
    for a, b in interactions:
        term = f"{a}×{b}"
        interaction_terms.append(term)
        # a or b may be in X or only in d (if user didn't include it in x_fields)
        a_vals = X[a] if a in X.columns else d[a]
        b_vals = X[b] if b in X.columns else d[b]
        X[term] = a_vals.astype(float).to_numpy() * b_vals.astype(float).to_numpy()

    # --- add intercept
    if add_intercept:
        X_design = sm.add_constant(X, has_constant="add")
    else:
        X_design = X

    # --- fit OLS
    model = sm.OLS(y, X_design.astype(float)).fit()

    # --- core stats
    yhat = model.fittedvalues
    resid = y - yhat
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    if show_prediction_interval:
        sf = model.get_prediction(X_design).summary_frame(alpha=0.05)
        half_width = (sf['obs_ci_upper'] - sf['obs_ci_lower']) / 2
        print('Prediction Interval')
        print(half_width.mean(), half_width.median(), half_width.quantile(0.9), half_width.quantile(0.95))

        
    out = {
        "y_field": y_field,
        "x_fields": list(x_fields),
        "interactions": list(interactions),
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "r2_adj": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "rmse": rmse,
        "row_index": d.index.to_list(),
        "terms": list(model.params.index),
        "params": {k: float(v) for k, v in model.params.items()},
        "stderr": {k: float(v) for k, v in model.bse.items()},
        "tvalues": {k: float(v) for k, v in model.tvalues.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
        "design_info": {
            "add_intercept": add_intercept,
            "standardize": standardize,
            "interaction_terms": interaction_terms,
        },
    }

    # --- pairwise Spearman with Y for main-effect Xs
    if add_spearman:
        sp = {}
        y_series = pd.Series(y, index=d.index)
        for c in x_fields:
            x_series = pd.Series(X[c].to_numpy(), index=d.index)
            # spearmanr can handle ties, but we still want finite
            rho, p = spearmanr(x_series, y_series)
            sp[c] = {"rho": float(rho), "p": float(p)}
        out["spearman"] = sp

    # --- VIF (exclude intercept)
    if add_vif:
        vif = {}
        cols = list(X.columns)  # includes interactions
        X_vif = X[cols].astype(float).to_numpy()
        if X_vif.shape[1] >= 2 and np.all(np.isfinite(X_vif)):
            for i, c in enumerate(cols):
                try:
                    vif[c] = float(variance_inflation_factor(X_vif, i))
                except Exception:
                    vif[c] = np.nan
        else:
            for c in cols:
                vif[c] = np.nan
        out["vif"] = vif
    
    if return_model:
        out["model"] = model  # statsmodels RegressionResults
    if return_design:
        out["X_design"] = X_design  # DataFrame with same index as d
        out["y_used"] = pd.Series(y, index=d.index, name=y_field)  # scaled or not
        out["yhat_used"] = pd.Series(yhat, index=d.index, name="y_hat_used")

    return out
def pretty_print_fit(fit: dict, *, digits: int = 3) -> None:
    """Small helper to print the most relevant pieces nicely."""
    if "error" in fit:
        print(f"[ERROR] {fit['error']}")
        return
    print(" ")
    print(" ")
    print(f"Model: {fit['y_field']} ~ {', '.join(fit['x_fields'])}"
          + (f" + interactions({fit['interactions']})" if fit["interactions"] else ""))
    print(f"n={fit['n']}, R2={fit['r2']:.{digits}f}, adjR2={fit['r2_adj']:.{digits}f}, RMSE={fit['rmse']:.{digits}f}")
    print("Coefficients:")
    for term in fit["terms"]:
        b = fit["params"][term]
        p = fit["pvalues"][term]
        print(f"  {term:>18s}: beta={b:.{digits}f}, p={p:.2e}")

    if "vif" in fit:
        print("VIF:")
        for k, v in fit["vif"].items():
            vv = "nan" if not np.isfinite(v) else f"{v:.{digits}f}"
            print(f"  {k:>18s}: {vv}")

    if "spearman" in fit:
        print("Spearman (main effects):")
        for k, v in fit["spearman"].items():
            print(f"  {k:>18s}: rho={v['rho']:.{digits}f}, p={v['p']:.2e}")

# =========================
# Moderation study additions
# =========================

def _apply_common_filters(df_cache: pd.DataFrame, min_k=0, min_horizon_years=5):
    """
    Mirrors your typical filtering pattern:
    - remove k<=min_k
    - ensure horizon_length >= min_horizon_years (computed from 'dates' column)
    """
    df = df_cache.copy()
    df = df[df["k"] > min_k]

    # horizon filtering
    years = df["dates"].str.split(",", expand=True)
    df.loc[:, "horizon_length"] = years[1].astype(int) - years[0].astype(int) + 1
    df = df[df["horizon_length"] >= min_horizon_years]

    return df
def _ols_slope_no_intercept(y, x):
    """
    Your code tends to use a zero-intercept slope model.
    Returns: slope, r2, pvalue, n
    """
    y = np.asarray(y)
    x = np.asarray(x)

    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]

    if len(y) < 3:
        return np.nan, np.nan, np.nan, len(y)

    X = x.reshape(-1, 1)
    model = sm.OLS(y, X).fit()
    slope = float(model.params[0])
    r2 = float(model.rsquared)
    p = float(model.pvalues[0])
    n = int(model.nobs)
    return slope, r2, p, n
def _ols_slope_free_intercept(y, x):
    """
    OLS with intercept: y = a + b x
    Returns: slope, intercept, r2, p_slope, n
    """
    y = np.asarray(y)
    x = np.asarray(x)

    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]

    if len(y) < 3:
        return np.nan, np.nan, np.nan, np.nan, len(y)

    X = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, X).fit()

    intercept = float(model.params[0])
    slope = float(model.params[1])
    r2 = float(model.rsquared)
    p = float(model.pvalues[1])
    n = int(model.nobs)

    return slope, intercept, r2, p, n
def moderation_table_Wp_by_M(
    df_cache: pd.DataFrame,
    y_field="e_tsa_cem_nrmse_range",
    x_field="e_tsa_proxy_delta_nrmse_range",
    m_field="e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range",
    free_intercept=True,
    m_bins=4,
):
    """
    Q3 moderation table:
      rows = (Wp_group, M_bin) where
        Wp_group in {Wp=0, Wp∈{0.25,0.5}, Wp∈{0.75,1}}
        M_bin in {Q1..Q4} from global quantiles of m_field (intrinsic proxy quality)

    For each regime, reports:
      - n, means
      - slope/intercept/R² for Y ~ X (with free intercept by default)
      - correlations among X, M, Y (useful for the 'propagation' narrative)
    """

    df = _apply_common_filters(df_cache)
    df = df[["Wp", y_field, x_field, m_field]].dropna().copy()

    if len(df) == 0:
        print("No data after filtering.")
        return pd.DataFrame()

    # --------------------
    # Wp grouping (exact values)
    # --------------------
    def _wp_group(wp):
        if np.isclose(wp, 0.0):
            return "Wp = 0"
        elif np.isclose(wp, 0.25) or np.isclose(wp, 0.5):
            return "Wp ∈ {0.25, 0.5}"
        elif np.isclose(wp, 0.75) or np.isclose(wp, 1.0):
            return "Wp ∈ {0.75, 1.0}"
        else:
            return None

    df["Wp_group"] = df["Wp"].apply(_wp_group)
    df = df[df["Wp_group"].notna()].copy()

    # --------------------
    # Global M bins (intrinsic proxy quality)
    # Q1 = best (lowest error), Q4 = worst
    # --------------------
    try:
        df["M_bin"] = pd.qcut(df[m_field], q=m_bins, labels=[f"Q{i+1}" for i in range(m_bins)], duplicates="drop")
    except ValueError:
        df["M_bin"] = pd.cut(df[m_field], bins=m_bins, labels=[f"B{i+1}" for i in range(m_bins)])

    # Helper: per-group OLS Y~X
    def _fit_group(g):
        y = g[y_field].to_numpy()
        x = g[x_field].to_numpy()
        m = g[m_field].to_numpy()

        if free_intercept:
            slope, intercept, r2, p_slope, n = _ols_slope_free_intercept(y=y, x=x)
        else:
            slope, r2, p_slope, n = _ols_slope_no_intercept(y=y, x=x)
            intercept = 0.0

        corr_xm = float(np.corrcoef(x, m)[0, 1]) if len(g) >= 3 else np.nan
        corr_xy = float(np.corrcoef(x, y)[0, 1]) if len(g) >= 3 else np.nan
        corr_my = float(np.corrcoef(m, y)[0, 1]) if len(g) >= 3 else np.nan

        return {
            "n": int(len(g)),
            "median_Wp": float(np.median(g["Wp"])),
            "median_M": float(np.median(m)),
            "median_X": float(np.median(x)),
            "median_Y": float(np.median(y)),
            "slope_Y_on_X": float(slope),
            "intercept": float(intercept),
            "p_slope": float(p_slope),
            "r2_Y_on_X": float(r2),
            "corr_X_M": corr_xm,
            "corr_X_Y": corr_xy,
            "corr_M_Y": corr_my,
        }

    rows = []
    wp_order = ["Wp = 0", "Wp ∈ {0.25, 0.5}", "Wp ∈ {0.75, 1.0}"]
    m_order = [f"Q{i+1}" for i in range(m_bins)]

    for wpg in wp_order:
        dwp = df[df["Wp_group"] == wpg]
        for mb in m_order:
            g = dwp[dwp["M_bin"].astype(str) == mb]
            if len(g) == 0:
                rows.append({
                    "Wp_group": wpg,
                    "M_bin": mb,
                    "n": 0,
                    "mean_Wp": np.nan,
                    "mean_M": np.nan,
                    "mean_X": np.nan,
                    "mean_Y": np.nan,
                    "slope_Y_on_X": np.nan,
                    "intercept": np.nan,
                    "p_slope": np.nan,
                    "r2_Y_on_X": np.nan,
                    "corr_X_M": np.nan,
                    "corr_X_Y": np.nan,
                    "corr_M_Y": np.nan,
                })
                continue

            out = _fit_group(g)
            out["Wp_group"] = wpg
            out["M_bin"] = mb
            rows.append(out)

    tab = pd.DataFrame(rows)

    print("\n=== Q3 Moderation Table: by Wp-group × proxy-quality quartile (M) ===")
    print(f"(Y={y_field}, X={x_field}, M={m_field}, free_intercept={free_intercept})\n")
    print(tab.to_string(index=False))

    return tab
def moderation_graph_density_quartiles_M_2x2(
    df_cache: pd.DataFrame,
    y_field="e_tsa_cem_nrmse_range",
    x_field="e_tsa_proxy_delta_nrmse_range",
    m_field="e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range",
    free_intercept=True
):
    """
    2x2 density figure: Q1..Q4 of M (proxy approximation error).
    Each panel: density of Y vs X with trendline + R^2 printed.
    Ticks: x in 0.01, y from 0 in 0.1.
    """

    df = _apply_common_filters(df_cache)
    df = df[[y_field, x_field, m_field, "Wp", "k"]].dropna()

    if len(df) == 0:
        print("No data after filtering.")
        return

    # Create quartiles globally
    df = df.copy()
    df["M_bin"] = pd.qcut(df[m_field], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    fig, axs = plt.subplots(2, 2, figsize=(fig_width * 2.4, fig_height * 2.2), sharex=True, sharey=True)
    bins = ["Q1", "Q2", "Q3", "Q4"]

    x_lim, y_lim = robust_limits(df[x_field], df[y_field])

    for i, b in enumerate(bins):
        r = i // 2
        c = i % 2
        ax = axs[r, c]
        d = df[df["M_bin"] == b]

        if len(d) == 0:
            ax.text(0.5, 0.5, "No data after filtering", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(b)
            continue

        # Use your existing density_plot, but pass through the free_intercept flag
        density_plot(
            fig,
            ax,
            d,
            y_field=y_field,
            x_field=x_field,
            show_trend=True,
            df_scatter=None,
            free_intercept=free_intercept,
            annotate_r2=True,
            x_lim=x_lim,
            y_lim = y_lim
        )

        ax.set_title(f"Proxy Quality {b}")

        # ticks/limits per your spec
        # ax.xaxis.set_major_locator(MultipleLocator(0.01))
        # ax.yaxis.set_major_locator(MultipleLocator(0.10))
        ax.set_ylim(bottom=0.0)

        ax.set_xlabel(None)
        ax.set_ylabel(None)

    fig.supxlabel("Proxy-Delta TSA Error (nRMSE)")
    fig.supylabel(" CEM SoC error (nRMSE)")
    plt.tight_layout()
    plt.savefig("fig_signals_3b.pdf", bbox_inches="tight")
    plt.show()
def moderation_graph_slope_vs_M(
    df_cache: pd.DataFrame,
    y_field="e_tsa_cem_nrmse_range",
    x_field="e_tsa_proxy_delta_nrmse_range",
    m_field="e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range",
    m_bins=10,
    by_country=True
):
    """
    Figure 2 (recommended): Conditional effect plot.
    Bin M into quantiles; within each bin compute slope of Y~X (zero intercept).
    Plot slope vs the median M of each bin.

    This is the cleanest way to show:
        "When proxy approximation is good (low M), the framework works well
         and TSA distortion has a weaker / different relationship with Y."
    """
    df = _apply_common_filters(df_cache)
    df = df[["country", y_field, x_field, m_field, "Wp", "k"]].dropna()

    # Bin M globally or within each country?
    # We'll bin within each country if by_country=True (more comparable within each system),
    # otherwise globally.
    fig, ax = plt.subplots(figsize=(fig_width * 1.6, fig_height * 1.2))

    def _plot_for_group(g, label):
        # Make quantile bins of M inside group
        try:
            g = g.copy()
            g["M_bin"] = pd.qcut(g[m_field], q=m_bins, duplicates="drop")
        except ValueError:
            return

        xs = []
        slopes = []
        ses = []

        for _, b in g.groupby("M_bin"):
            y = b[y_field].to_numpy()
            x = b[x_field].to_numpy()
            m = b[m_field].to_numpy()

            mask = np.isfinite(y) & np.isfinite(x) & np.isfinite(m)
            y = y[mask]
            x = x[mask]
            m = m[mask]

            if len(y) < 5:
                continue

            X = x.reshape(-1, 1)
            model = sm.OLS(y, X).fit()

            slopes.append(float(model.params[0]))
            ses.append(float(model.bse[0]))
            xs.append(float(np.median(m)))

        if len(xs) < 2:
            return

        xs = np.asarray(xs)
        slopes = np.asarray(slopes)
        ses = np.asarray(ses)

        # line + error bars
        ax.errorbar(xs, slopes, yerr=1.96 * ses, fmt="o-", capsize=2, label=label, alpha=0.9)

    if by_country:
        for country, g in df.groupby("country"):
            _plot_for_group(g, label=country)
        ax.legend(frameon=False)
    else:
        _plot_for_group(df, label="pooled")

    ax.set_xlabel("Proxy approximation error (M): median per bin (lower is better)")
    ax.set_ylabel("Conditional slope of Y on X (zero-intercept β)")
    ax.set_title("Moderation: strength of TSA distortion effect vs proxy quality")
    plt.tight_layout()
    plt.show()
def moderation_graph_density_hotspots_by_country(
    df_cache: pd.DataFrame,
    y_field="e_tsa_cem_nrmse_range",
    x_field="e_tsa_proxy_delta_nrmse_range"
):
    """
    Optional: a simpler visual to reproduce your 'hotspots' observation.
    One panel per country (density of Y vs X).
    """
    df = _apply_common_filters(df_cache)
    df = df[["country", y_field, x_field, "Wp", "k"]].dropna()

    countries = sorted(df["country"].unique().tolist())
    if len(countries) == 0:
        print("No data after filtering.")
        return

    fig, axs = plt.subplots(1, len(countries), figsize=(fig_width * 1.3 * len(countries), fig_height * 1.2),
                            sharex=False, sharey=False)

    if len(countries) == 1:
        axs = [axs]

    for ax, country in zip(axs, countries):
        d = df[df["country"] == country]
        density_plot(fig, ax, d, y_field=y_field, x_field=x_field, show_trend=True, df_scatter=None)
        ax.set_title(country)

    fig.supxlabel("TSA proxy-delta distortion (X): nRMSE range")
    fig.supylabel("Downstream CEM SoC error (Y): nRMSE range")
    plt.tight_layout()
    plt.show()
def moderation_graph_density_best25_vs_rest75(
    df_cache: pd.DataFrame,
    y_field="e_tsa_cem_nrmse_range",
    x_field="e_tsa_proxy_delta_nrmse_range",
    m_field="e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range",
    quantile_cut=0.25,
    free_intercept=True
):
    """
    Moderation figure (preferred): split by proxy approximation quality M into:
      - Best 25% (low M): "Good proxy"
      - Remaining 75% (higher M): "Rest"

    Produces a 1x2 density plot of Y vs X with:
      - trendline (uses free_intercept flag)
      - R^2 annotation on-panel (requires density_plot(..., annotate_r2=True))
      - x ticks every 0.01
      - y from 0 with ticks every 0.1
      - no country split

    Note: Uses sharey=True for comparability of downstream error.
    """

    df = _apply_common_filters(df_cache)
    df = df[[y_field, x_field, m_field, "Wp", "k"]].dropna()

    if len(df) == 0:
        print("No data after filtering.")
        return

    q_thr = df[m_field].quantile(quantile_cut)

    df_good = df[df[m_field] <= q_thr]
    df_rest = df[df[m_field] > q_thr]

    # Simple diagnostics for your sanity checks
    def _rng(arr):
        return (float(np.min(arr)), float(np.max(arr))) if len(arr) else (np.nan, np.nan)

    gx0, gx1 = _rng(df_good[x_field].to_numpy())
    rx0, rx1 = _rng(df_rest[x_field].to_numpy())

    print(
        f"[best {int(quantile_cut*100)}% vs rest] N total={len(df)} | "
        f"good={len(df_good)} | rest={len(df_rest)}"
    )
    print(f"  good X range: {gx0:.3f}–{gx1:.3f} | rest X range: {rx0:.3f}–{rx1:.3f}")

    fig, axs = plt.subplots(
        1, 2,
        figsize=(fig_width * 2.4, fig_height * 1.2),
        sharey=True,
        sharex=False
    )

    panels = [
        ("Good proxy (best 25%: low M)", df_good),
        ("Rest (remaining 75%)", df_rest),
    ]

    for ax, (title, d) in zip(axs, panels):
        if len(d) == 0:
            ax.text(0.5, 0.5, "No data after filtering", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
        else:
            density_plot(
                fig,
                ax,
                d,
                y_field=y_field,
                x_field=x_field,
                show_trend=True,
                df_scatter=None,
                free_intercept=free_intercept,
                annotate_r2=True
            )
            ax.set_title(title)

        # Axis formatting per your spec
        # ax.xaxis.set_major_locator(MultipleLocator(0.01))
        # ax.yaxis.set_major_locator(MultipleLocator(0.10))
        ax.set_ylim(bottom=0.0)

    fig.supxlabel("TSA proxy-delta distortion (X): nRMSE range")
    fig.supylabel("Downstream CEM SoC error (Y): nRMSE range")
    plt.tight_layout()
    plt.show()
def moderation_graph_density_Wp_bins_by_M(
    df_cache: pd.DataFrame,
    y_field="e_tsa_cem_nrmse_range",
    x_field="e_tsa_proxy_delta_nrmse_range",
    m_field="e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range",
    wp_field="Wp",
    mode="quartiles",          # "quartiles" | "tertiles" | "best_vs_rest" | "quantiles"
    q=4,                       # only used if mode="quantiles"
    quantile_cut=0.25,         # used for best_vs_rest (and can be used for custom quantiles if you like)
    free_intercept=True,
    sharey=True,
    sharex=False,
    print_counts=True,
):
    """
    Creates a grid of density plots of Y vs X:
      Rows: Wp bins [Wp=0, Wp in {0.25,0.5}, Wp in {0.75,1.0}]
      Cols: bins of M according to `mode`

    mode options:
      - "quartiles": M split into 4 quantiles (Q1..Q4)
      - "tertiles":  M split into 3 quantiles (T1..T3)
      - "best_vs_rest": best 25% (low M) vs rest 75%
      - "quantiles": M split into `q` quantiles (Q1..Qq)

    Axis formatting:
      - x ticks every 0.01
      - y from 0 with ticks every 0.1
      - trendline uses free_intercept flag
      - each panel annotates R^2 (requires density_plot(..., annotate_r2=True))
    """

    df = _apply_common_filters(df_cache)
    df = df[[wp_field, y_field, x_field, m_field, "k"]].dropna()

    if len(df) == 0:
        print("No data after filtering.")
        return

    # --------------------
    # Define Wp bins
    # --------------------
    def _wp_group(wp):
        # robust equality for floats
        if np.isclose(wp, 0.0):
            return "Wp = 0"
        elif np.isclose(wp, 0.25) or np.isclose(wp, 0.5):
            return "Wp ∈ {0.25, 0.5}"
        elif np.isclose(wp, 0.75) or np.isclose(wp, 1.0):
            return "Wp ∈ {0.75, 1.0}"
        else:
            return None

    df = df.copy()
    df["Wp_group"] = df[wp_field].apply(_wp_group)
    df = df[df["Wp_group"].notna()]

    wp_groups = ["Wp = 0", "Wp ∈ {0.25, 0.5}", "Wp ∈ {0.75, 1.0}"]

    # --------------------
    # Define M bins per mode (within each Wp group!)
    # This prevents one group dominating the quantile thresholds.
    # --------------------
    def _make_m_bins(d: pd.DataFrame):
        d = d.copy()

        if mode == "best_vs_rest":
            thr = d[m_field].quantile(quantile_cut)
            d["M_bin"] = np.where(d[m_field] <= thr, f"Best {int(quantile_cut*100)}%", f"Rest {int((1-quantile_cut)*100)}%")
            labels = [f"Best {int(quantile_cut*100)}%", f"Rest {int((1-quantile_cut)*100)}%"]
            return d, labels

        if mode == "tertiles":
            nb = 3
            labels = [f"T{i+1}" for i in range(nb)]
            d["M_bin"] = pd.qcut(d[m_field], q=nb, labels=labels, duplicates="drop")
            labels = [str(x) for x in d["M_bin"].cat.categories]
            return d, labels

        if mode == "quartiles":
            nb = 4
            labels = [f"Q{i+1}" for i in range(nb)]
            d["M_bin"] = pd.qcut(d[m_field], q=nb, labels=labels, duplicates="drop")
            labels = [str(x) for x in d["M_bin"].cat.categories]
            return d, labels

        if mode == "quantiles":
            nb = int(q)
            if nb < 2:
                raise ValueError("mode='quantiles' requires q>=2")
            labels = [f"Q{i+1}" for i in range(nb)]
            d["M_bin"] = pd.qcut(d[m_field], q=nb, labels=labels, duplicates="drop")
            labels = [str(x) for x in d["M_bin"].cat.categories]
            return d, labels

        raise ValueError(f"Unknown mode: {mode}")

    # Figure layout
    # number of columns depends on mode; we'll infer from first non-empty group
    col_labels = None
    df_bins = []

    x_lim = [1,0]
    y_lim = [1,0] 

    for gname in wp_groups:
        d = df[df["Wp_group"] == gname]
        if len(d) == 0:
            df_bins.append((gname, d, []))
            continue

        d2, labels = _make_m_bins(d)
        df_bins.append((gname, d2, labels))
        if col_labels is None:
            col_labels = labels

        x_lim[0]=min(x_lim[0],d[x_field].min())
        x_lim[1]=max(x_lim[1],d[x_field].max())
        y_lim[0]=min(y_lim[0],d[y_field].min())
        y_lim[1]=max(y_lim[1],d[y_field].max())

    # If still none, bail
    if col_labels is None:
        print("No data in any Wp group after filtering.")
        return

    nrows = len(wp_groups)
    ncols = len(col_labels)

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(fig_width * 1.3 * ncols, fig_height * 1.1 * nrows),
        sharey=sharey,
        sharex=sharex
    )

    # Ensure axs is 2D
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = np.array([axs])
    elif ncols == 1:
        axs = np.array([[ax] for ax in axs])

    # --------------------
    # Plot each panel
    # --------------------
    for r, (gname, d2, labels_this_group) in enumerate(df_bins):
        # If this group had fewer bins due to duplicates, we still keep fixed col_labels.
        # Panels missing bins will show "No data".
        for c, mlabel in enumerate(col_labels):
            ax = axs[r, c]

            if len(d2) == 0 or "M_bin" not in d2.columns:
                ax.text(0.5, 0.5, "No data after filtering", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(mlabel)
            else:
                dsub = d2[d2["M_bin"].astype(str) == str(mlabel)]
                if len(dsub) == 0:
                    ax.text(0.5, 0.5, "No data after filtering", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(mlabel)
                else:
                    density_plot(
                        fig, ax, dsub,
                        y_field=y_field,
                        x_field=x_field,
                        show_trend=True,
                        df_scatter=None,
                        free_intercept=free_intercept,
                        annotate_r2=True,
                        x_lim=x_lim,
                        y_lim=y_lim
                    )
                    ax.set_title(mlabel)

            # Axis formatting (your requirements)
            # ax.xaxis.set_major_locator(MultipleLocator(0.01))
            # ax.yaxis.set_major_locator(MultipleLocator(0.10))
            ax.set_ylim(bottom=0.0)

            # Left-most column: put row label on y-axis
            if c == 0:
                ax.set_ylabel(f"{gname}\n\nAbsolute LDES Capacity Error (Y)")

    fig.supxlabel("TSA proxy-delta distortion (X): nRMSE range")
    plt.tight_layout()
    plt.show()

    # --------------------
    # Optional counts printout
    # --------------------
    if print_counts:
        print(f"\n[Wp×M plot] mode={mode}, free_intercept={free_intercept}, ncols={ncols}")
        for gname, d2, _ in df_bins:
            n_total = len(d2)
            print(f"  {gname}: n={n_total}")
            if n_total > 0 and "M_bin" in d2.columns:
                vc = d2["M_bin"].astype(str).value_counts().reindex([str(x) for x in col_labels]).fillna(0).astype(int)
                print(f"    M bins: {vc.to_dict()}")

# =========================
# Research Questions
# =========================

def proxy_vs_proxyDeltas_regression(df_cache: pd.DataFrame):

    df = df_cache
    #general filters
    # df = df[df['country']=='NL'] #focus on NL
    df = df[df['k']>3] #remove k=30
    # df = df[df['k']<365] 

    # subset
    df_subset = df.copy()
    df_subset = df_subset[df_subset['Wp']==0.5] #focus on Wp=0.5

    fig1, ax1 = plt.subplots()
    density_plot(fig1, ax1, df, y_field='e_tsa_cem_nrmse_range',x_field='e_proxy_ref_nrmse_range', free_intercept=True) #e_proxy_clu_nrmse_range  e_proxy_delta_clu_vs_cem_delta_clu_nrmse_range
    # ax1.set_ylim(0,0.4)
    # ax1.set_xlim(0,1)
    plt.show()

    fig2, ax2  = plt.subplots()
    density_plot(fig2, ax2, df, y_field='e_tsa_cem_nrmse_range',x_field='e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range', free_intercept=True) #e_proxy_clu_nrmse_range  e_proxy_delta_clu_vs_cem_delta_clu_nrmse_range
    # ax2.set_ylim(0,0.4)
    # ax2.set_xlim(0,1)
    plt.show()

def Q1_soc_vs_cap_errors(df_cache: pd.DataFrame):
    df = _apply_common_filters(df_cache)

    soc_field = "e_tsa_cem_nrmse_range"
    cap_field = "ldes_cap_error_abs"
    peak_field = "peak_dt_days_cem_clu_vs_cem_ref"


    #peak_dt_days_cem_clu_vs_cem_ref,peak_dt_days_proxy_ref_vs_cem_ref,peak_dt_days_proxy_clu_vs_cem_ref,peak_dt_days_proxy_clu_vs_proxy_ref,

    df = df[[soc_field, cap_field, peak_field]].dropna()

    df[peak_field] = df[peak_field].abs()/(10*365)

    # -----------------------
    # REGRESSION TABLE
    # -----------------------

    rows = []

    def _run_and_store(name, y, X, x_names):
        Xmat = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xmat).fit()

        row = {
            "model": name,
            "n": int(model.nobs),
            "r2": float(model.rsquared),
        }

        for i, xn in enumerate(["const"] + x_names):
            row[f"beta_{xn}"] = float(model.params[i])
            row[f"p_{xn}"] = float(model.pvalues[i])

        rows.append(row)

    y = df[cap_field].to_numpy()
    soc = df[soc_field].to_numpy()
    
    peak = df[peak_field].to_numpy()

    rho, pval = spearmanr(soc, y)
    print("\n=== Spearman rank correlation ===")
    print(f"rho = {rho:.3f}, p = {pval:.2e}")

    # A: SoC only
    _run_and_store(
        "Cap ~ SoC",
        y,
        soc.reshape(-1, 1),
        ["SoC"],
    )

    # B: Peak only
    _run_and_store(
        "Cap ~ Peak",
        y,
        peak.reshape(-1, 1),
        ["PeakDays"],
    )

    # C: additive
    _run_and_store(
        "Cap ~ SoC + Peak",
        y,
        np.column_stack([soc, peak]),
        ["SoC", "PeakDays"],
    )

    # D: interaction
    _run_and_store(
        "Cap ~ SoC + Peak + SoC×Peak",
        y,
        np.column_stack([soc, peak, soc * peak]),
        ["SoC", "PeakDays", "SoC×Peak"],
    )

    df_tab = pd.DataFrame(rows)

    print("\n=== Q1: Explaining LDES capacity error ===")
    print(df_tab.to_string(index=False))

    # -----------------------
    # FIGURE: SoC vs capacity
    # -----------------------

    fig1, ax1 = plt.subplots()
    density_plot(
        fig1,
        ax1,
        df,
        y_field=cap_field,
        x_field=soc_field,
        free_intercept=True,
    )
    ax1.set_xlabel('CEM SoC Error (nRMSE)')
    ax1.set_ylabel('Absolute LDES Capacity Error')
    # ax1.set_title("LDES capacity error vs SoC trajectory error")
    plt.savefig('fig_signals_1.pdf')
    plt.show()

def Q2_global_errors(df_cache: pd.DataFrame):
    
    df = _apply_common_filters(df_cache)

    delta_approx= "e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range"
    delta_tsa = "e_tsa_proxy_delta_nrmse_range"
    level_approx = 'e_proxy_ref_nrmse_range'
    level_tsa = 'e_tsa_proxy_nrmse_range'
    soc = "e_tsa_cem_nrmse_range" #ldes_cap_error_abs e_tsa_cem_nrmse_range

    # -----------------------------
    #
    #       Joint regression
    #
    # -------------------------------

    y = df[soc].to_numpy()
    x_d_a = df[delta_approx].to_numpy()
    x_d_t = df[delta_tsa].to_numpy()
    x_l_a = df[level_approx].to_numpy()
    x_l_t = df[level_tsa].to_numpy()

    # ---- Model A: Y ~ X_level_approx
    XA = sm.add_constant(x_l_a, has_constant="add")
    mA = sm.OLS(y, XA).fit()

    # ---- Model B: Y ~ X_delta_approx
    XB = sm.add_constant(x_d_a, has_constant="add")
    mB = sm.OLS(y, XB).fit()

    # ---- Model A: Y ~ X_level_tsa
    XAa = sm.add_constant(x_l_t, has_constant="add")
    mAa = sm.OLS(y, XAa).fit()

    # ---- Model B: Y ~ X_delta_tsa
    XBa = sm.add_constant(x_d_t, has_constant="add")
    mBa = sm.OLS(y, XBa).fit()

    # ---- Model C: Y ~ X_level_approx + X_level_tsa
    XC = sm.add_constant(np.column_stack([x_l_a, x_l_t]), has_constant="add")
    mC = sm.OLS(y, XC).fit()

    # ---- Model D: Y ~ X_delta_approx + X_delta_tsa
    XD = sm.add_constant(np.column_stack([x_d_a, x_d_t]), has_constant="add")
    mD = sm.OLS(y, XD).fit()

    # ---- Model E: Y ~ X_level_approx + X_level_tsa + X_delta_approx + X_delta_tsa
    XE = sm.add_constant(np.column_stack([x_l_a, x_l_t,x_d_a, x_d_t]), has_constant="add")
    mE = sm.OLS(y, XE).fit()

    def _row(name, model, cols):
        out = {
            "model": name,
            "n": int(model.nobs),
            "r2": float(model.rsquared_adj),
        }
        for c, b, p in zip(cols, model.params, model.pvalues):
            out[f"beta_{c}"] = float(b)
            out[f"p_{c}"] = float(p)
        return out
    
    rows = []
    rows.append(_row("Y ~ X_level_approx", mA, ["const", "X_level"]))
    rows.append(_row("Y ~ X_delta_approx", mB, ["const", "X_delta"]))
    rows.append(_row("Y ~ X_level_tsa", mAa, ["const", "X_level"]))
    rows.append(_row("Y ~ X_delta_tsa", mBa, ["const", "X_delta"]))
    rows.append(_row("Y ~ X_level_approx + X_level_tsa", mC, ["const", "X_level_approx", "X_level_tsa"]))
    rows.append(_row("Y ~ X_delta_approx + X_delta_tsa", mD, ["const", "X_delta_approx", "X_delta_tsa"]))
    rows.append(_row("Y ~ X_level_approx + X_level_tsa + X_delta_approx + X_delta_tsa", mE,["const","X_level_approx", "X_level_tsa", "X_delta_approx", "X_delta_tsa"]))

    tab = pd.DataFrame(rows)
    print("\n=== Q2 joint regression: TSA level vs delta distortion ===")
    print(tab.to_string(index=False))

    # -----------------------------
    #
    #       Figure proxy deltas vs levels
    #
    # -------------------------------
    fig1, axes1 = plt.subplots(2,1, sharey=True, figsize=(fig_width, fig_height*2) )
    density_plot(
        fig1,
        axes1[0],
        df,
        y_field=soc,
        x_field=level_approx,
        free_intercept=True,
    )
    density_plot(
        fig1,
        axes1[1],
        df,
        y_field=soc,
        x_field=delta_approx,
        free_intercept=True,
    )
    axes1[0].set_xlabel('Reference Proxy Approximation Error (nRMSE)')
    axes1[1].set_xlabel('Reference Proxy-Delta Approximation Error (nRMSE)')
    axes1[0].set_ylabel(None)
    axes1[1].set_ylabel(None)
    
    fig1.supylabel('CEM SoC Error (nRMSE)')
    fig1.set_size_inches(5, 8)
    plt.savefig("fig_signals_2a.pdf", bbox_inches="tight")
    # plt.savefig('fig_signals_2a.pdf')
    plt.show()

    fig2, axes2 = plt.subplots(2,1, sharey=True, figsize=(fig_width, fig_height*2) )
    density_plot(
        fig2,
        axes2[0],
        df,
        y_field=soc,
        x_field=level_tsa,
        free_intercept=True,
    )
    density_plot(
        fig2,
        axes2[1],
        df,
        y_field=soc,
        x_field=delta_tsa,
        free_intercept=True,
    )

    axes2[0].set_xlabel('Proxy TSA Error (nRMSE)')
    axes2[1].set_xlabel('Proxy-Delta TSA Error (nRMSE)')
    axes2[0].set_ylabel(None)
    axes2[1].set_ylabel(None)
    fig2.supylabel('CEM SoC Error (nRMSE)')
   
    fig2.set_size_inches(5, 8)
    plt.savefig("fig_signals_2b.pdf", bbox_inches="tight")
    plt.show()

def Q3_weight_quality_dependence(df_cache: pd.DataFrame):
    df = _apply_common_filters(df_cache)

    # Core fields for Q3 (stick to the mechanistic chain)
    y_field = "e_tsa_cem_nrmse_range" #e_tsa_cem_nrmse_range ldes_cap_error_abs
    x_field = "e_tsa_proxy_delta_nrmse_range"
    m_field = "e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range"

    # -----------------------------
    # Q3a: split by Wp regimes
    # -----------------------------
    Q3a_plot_by_Wp_groups(
        df_cache=df_cache,
        y_field=y_field,
        x_field=x_field,
        free_intercept=True
    )

    # -----------------------------
    # Q3b: split by proxy-quality quartiles
    # -----------------------------
    Q3b_plot_by_proxy_quality_quartiles(
        df_cache=df_cache,
        y_field=y_field,
        x_field=x_field,
        m_field=m_field,
        free_intercept=True
    )

    # -----------------------------
    # Moderation table: Wp-group × proxy-quality quartile
    # -----------------------------
    moderation_table_Wp_by_M(
        df_cache=df_cache,
        y_field=y_field,
        x_field=x_field,
        m_field=m_field,
        free_intercept=True,
        m_bins=4
    )

    # -----------------------------
    # Q3c: combined grid (leave on/off as you decide)
    # Start with quartiles; later you can switch to:
    #   mode="best_vs_rest"  (best 25% vs rest 75%)
    #   mode="tertiles"
    #   mode="quantiles", q=2  (median split)
    # -----------------------------
    Q3c_combined_grid(
        df_cache=df_cache,
        y_field=y_field,
        x_field=x_field,
        m_field=m_field,
        mode="quartiles",
        free_intercept=True
    )
def Q3a_plot_by_Wp_groups(
    df_cache: pd.DataFrame,
    y_field="e_tsa_cem_nrmse_range",
    x_field="e_tsa_proxy_delta_nrmse_range",
    free_intercept=True,
):
    """
    Q3a figure: Y vs X split into 3 Wp regimes:
      Wp=0, Wp∈{0.25,0.5}, Wp∈{0.75,1}
    """
    df = _apply_common_filters(df_cache)
    df = df[["Wp", y_field, x_field, "k"]].dropna().copy()

    def _wp_group(wp):
        if np.isclose(wp, 0.0):
            return "Wp = 0"
        elif np.isclose(wp, 0.25) or np.isclose(wp, 0.5):
            return "Wp ∈ {0.25, 0.5}"
        elif np.isclose(wp, 0.75) or np.isclose(wp, 1.0):
            return "Wp ∈ {0.75, 1.0}"
        else:
            return None

    df["Wp_group"] = df["Wp"].apply(_wp_group)
    df = df[df["Wp_group"].notna()].copy()

    wp_groups = ["Wp = 0", "Wp ∈ {0.25, 0.5}", "Wp ∈ {0.75, 1.0}"]
    fig, axs = plt.subplots(3, 1, figsize=(fig_width * 1.5, fig_height * 3), sharex=True)

    for ax, gname in zip(axs, wp_groups):
        d = df[df["Wp_group"] == gname]
        if len(d) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(gname)
            continue
        density_plot(fig, ax, d, y_field=y_field, x_field=x_field, free_intercept=free_intercept, annotate_r2=True)
        ax.set_title(gname)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    x_label = 'Proxy-Delta TSA Error (nRMSE)' if x_field =='e_tsa_proxy_delta_nrmse_range' else ''
    y_label  = 'CEM SoC Error (nRMSE)' if y_field == 'e_tsa_cem_nrmse_range' else''

    fig.supxlabel(x_label)
    fig.supylabel(y_label)
    plt.tight_layout()
    plt.savefig("fig_signals_3a.pdf", bbox_inches="tight")
    plt.show()
def Q3b_plot_by_proxy_quality_quartiles(
    df_cache: pd.DataFrame,
    y_field="e_tsa_cem_nrmse_range",
    x_field="e_tsa_proxy_delta_nrmse_range",
    m_field="e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range",
    free_intercept=True,
):
    """
    Q3b figure: reuse your existing quartile plot function
    (2x2 grid) for M quartiles.
    """
    moderation_graph_density_quartiles_M_2x2(
        df_cache=df_cache,
        y_field=y_field,
        x_field=x_field,
        m_field=m_field,
        free_intercept=free_intercept
    )
def Q3c_combined_grid(
    df_cache: pd.DataFrame,
    y_field="e_tsa_cem_nrmse_range",
    x_field="e_tsa_proxy_delta_nrmse_range",
    m_field="e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range",
    mode="quartiles",          # "quartiles" | "tertiles" | "best_vs_rest" | "quantiles"
    q=4,                       # used only if mode="quantiles"
    quantile_cut=0.25,         # used if mode="best_vs_rest"
    free_intercept=True,
):
    moderation_graph_density_Wp_bins_by_M(
        df_cache=df_cache,
        y_field=y_field,
        x_field=x_field,
        m_field=m_field,
        mode=mode,
        q=q,
        quantile_cut=quantile_cut,
        free_intercept=free_intercept,
        sharey=True,
        sharex=True,
        print_counts=True,
    )


def Q4a_diagnostics(df_cache: pd.DataFrame):
    """
    Q4a:
      - Apply common filters
      - (Optional) print single-metric & window-metric diagnostics tables for Wp>0
      - Fit 3-predictor diagnostic model on Wp>0 (for KDE background)
      - Plot: KDE contours (Wp>0) of predicted vs observed
             + x=0,y=0 lines + 45-degree line
             + single legend containing: R2 + model equation + reference-line labels
             + axes shown as percentages (ticks formatted, data unchanged)
    """
    # -----------------------------
    # 0) Filters / setup
    # -----------------------------
    df = _apply_common_filters(df_cache).copy()

    windows = WINDOWS
    error_types = ["nrmse_range", "nmbe_range", "pearson_r"]

    metrics_tsa_error = ["e_tsa_proxy_", "e_tsa_proxy_delta_"]
    window_metrics_tsa_error = ["d_tsa_proxy_", "d_tsa_proxy_delta_"]

    metrics_approximation_error = ["e_proxy_clu_", "e_proxy_delta_clu_vs_cem_delta_clu_"]
    window_metrics_approximation_error = ["d_proxy_clu_vs_cem_clu_", "d_proxy_delta_clu_vs_cem_delta_clu_"]

    metrics_refProxy_vs_cluCem_error = ["e_proxy_ref_vs_cem_clu_", "e_proxy_delta_ref_vs_cem_delta_clu_"]
    window_metrics_refProxy_vs_cluCem_error = ["d_proxy_ref_vs_cem_clu_", "d_proxy_delta_ref_vs_cem_delta_clu_"]

    y_field = "ldes_cap_error_signed"

    # Derived low-collinearity term (as you found)
    df["local_minus_global"] = (
        df["win180d_proxy_ref_vs_cem_clu_nmbe_range"] - df["win912d_proxy_ref_vs_cem_clu_nmbe_range"]
    )

    # Focus on Wp>0 for the “active” population
    df_Wp_gt_0 = df.loc[df["Wp"] > 0].copy()
    if df_Wp_gt_0.empty:
        raise ValueError("No rows left after filtering to Wp>0. Cannot run Q4a_diagnostics.")

    # -----------------------------
    # 1) Single-metric analysis (optional printout)
    # -----------------------------
    metric_set = metrics_tsa_error + metrics_approximation_error + metrics_refProxy_vs_cluCem_error

    rows = []
    y = df_Wp_gt_0[y_field]

    for metric in metric_set:
        for error_type in error_types:
            x_field = f"{metric}{error_type}"
            if x_field not in df_Wp_gt_0.columns:
                continue
            x = df_Wp_gt_0[x_field]
            diagnosis = diagnostic_stats(x=x, y=y)
            diagnosis["metric"] = x_field
            rows.append(diagnosis)

    if rows:
        df_results = pd.DataFrame(rows)
        print(f"Results: regression and Spearman rank on {y_field} (Wp>0)")
        print(df_results.to_string(index=False))
        print(" ")

    # -----------------------------
    # 2) Window-metric analysis (optional saveout)
    # -----------------------------
    window_rows = []
    window_metric_set = (
        window_metrics_tsa_error
        + window_metrics_approximation_error
        + window_metrics_refProxy_vs_cluCem_error
    )

    for window in [30, 90, 180, 273, 365, 548]:
        for metric in window_metric_set:
            for error_type in error_types:
                x_field = f"win{window}{metric}{error_type}"
                if x_field not in df_Wp_gt_0.columns:
                    continue
                x = df_Wp_gt_0[x_field]
                diagnosis = diagnostic_stats(x=x, y=y)
                diagnosis["window"] = window
                diagnosis["metric"] = metric
                diagnosis["error_type"] = error_type
                window_rows.append(diagnosis)

    if window_rows:
        df_window = pd.DataFrame(window_rows)
        df_window.to_csv("single_predictor_window_results.csv", index=False)

    # -----------------------------
    # 3) Three-predictor diagnostic model
    # -----------------------------
    x_fields = [
        "win273d_proxy_delta_ref_vs_cem_delta_clu_pearson_r",
        "win912d_proxy_ref_vs_cem_clu_nmbe_range",
        "local_minus_global",
    ]

    missing = [c for c in [y_field] + x_fields + ["Wp"] if c not in df_Wp_gt_0.columns]
    if missing:
        raise KeyError(f"Missing required columns for Q4a diagnostic plot: {missing}")

    # Fit on Wp>0: used for KDE background + R²(Wp>0)
    fit_all = fit_diagnostic_model(
        df_Wp_gt_0,
        y_field=y_field,
        x_fields=x_fields,
        add_intercept=True,
        standardize=False,
        return_model=True,
        return_design=True,
        show_prediction_interval=True
    )
    if "error" in fit_all:
        raise ValueError(f"fit_all failed: {fit_all['error']}")

    pretty_print_fit(fit_all)

    df_all = df_Wp_gt_0.loc[fit_all["row_index"]].copy()
    df_all["y_hat"] = fit_all["model"].predict(fit_all["X_design"])
    r2_all = fit_all["r2_adj"]

    # Grab coefficients for the legend “predictive model box”
    # NOTE: This assumes statsmodels-like params; if your model object differs, adjust accordingly.
    try:
        params = fit_all["model"].params
        alpha = float(params.get("const", params[0]))
        b1 = float(params.get(x_fields[0], params[1]))
        b2 = float(params.get(x_fields[1], params[2]))
        b3 = float(params.get(x_fields[2], params[3]))
    except Exception:
        # Fallback: don’t crash if param extraction differs
        alpha, b1, b2, b3 = np.nan, np.nan, np.nan, np.nan

    # -----------------------------
    # 4) Plot: KDE contours (Wp>0)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    x_all = df_all["y_hat"].to_numpy(dtype=float)
    y_all = df_all[y_field].to_numpy(dtype=float)

    # Use symmetric limits for signed errors (usually clearer)
    lim = np.max(np.abs(np.r_[x_all, y_all])) * 1.05
    xmin, xmax = -lim, lim
    ymin, ymax = -lim, lim
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # KDE evaluation grid
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 250),
        np.linspace(ymin, ymax, 250),
    )
    kde = gaussian_kde(np.vstack([x_all, y_all]))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    zmin_draw = np.quantile(zz, 0.25)
    zmax = float(zz.max())
    levels = np.linspace(zmin_draw, zmax, 20)

    cf = ax.contourf(
        xx, yy, zz,
        levels=levels,
        cmap=cmap_density,
        vmin=zmin_draw,
        vmax=zmax,
    )
    fig.colorbar(cf, ax=ax, label="density (Wp>0)")

    # Reference lines
    ax.axhline(0, linestyle="-", linewidth=1, color="grey")
    ax.axvline(0, linestyle="-", linewidth=1, color="grey")
    ax.plot([xmin, xmax], [ymin, ymax], linestyle=":", linewidth=1.5, color="grey", label="Perfect prediction")

    # Labels
    ax.set_xlabel("Predicted signed LDES capacity error (diagnostic)")
    ax.set_ylabel("Observed signed LDES capacity error")

    # ---- Format axes as percentages (ticks only; data unchanged)
    # Choose decimals to taste: 0 for whole %, 1 for 0.1%
    # pct_fmt = FuncFormatter(lambda v, pos: f"{v*100:.0f}%")
    # ax.xaxis.set_major_formatter(pct_fmt)
    # ax.yaxis.set_major_formatter(pct_fmt)

    # -----------------------------
# Annotation box: R² + model (no x-field names)
    # -----------------------------
    annot_text = (
        rf"Adjusted $R^2 = {r2_all:.2f}$" + "\n"
        # rf"$\hat{{\epsilon}}^C_{{LDES}}"
        # rf" = {alpha:.3f}"
        # rf"{b1:+.3f}x_1"
        # rf"{b2:+.3f}x_2"
        # rf"{b3:+.3f}x_3$"
    )

    ax.text(
        0.03,
        0.97,
        annot_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="none",
            alpha=0.9,
        ),
    )

    # -----------------------------
    # Legend ONLY for graphical items
    # -----------------------------
    ax.legend(frameon=False, loc="lower right")


    plt.tight_layout()
    plt.savefig("fig_signals_4a.pdf", bbox_inches="tight")
    plt.show()

    return {
        "fit_all": fit_all,
        "df_all": df_all,
        "single_metric_table": pd.DataFrame(rows) if rows else None,
        "window_metric_table": pd.DataFrame(window_rows) if window_rows else None,
    }
def Q4b_heatmap():

    windows = [15, 30, 45, 90, 180, 273, 365, 548, 730, 912]

    #data saved from Q4a
    data = {
        ("Proxy Approx.", "nMBE"):     [0.00,0.00,0.00,0.01,0.01,0.01,0.01,0.01,0.02,0.02],
        ("Proxy Approx.", "nRMSE"):    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.01,0.01],
        ("Proxy Approx.", "Pearson"):  [0.07,0.00,0.00,0.00,0.01,0.01,0.02,0.01,0.01,0.03],

        ("Proxy TSA", "nMBE"):     [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.01],
        ("Proxy TSA", "nRMSE"):    [0.03,0.03,0.03,0.05,0.04,0.05,0.05,0.05,0.06,0.08],
        ("Proxy TSA", "Pearson"):  [0.11,0.10,0.22,0.08,0.04,0.08,0.10,0.07,0.10,0.12],

        ("Proxy RPCC", "nMBE"):     [0.00,0.00,0.00,0.01,0.01,0.01,0.01,0.02,0.03,0.04],
        ("Proxy RPCC", "nRMSE"):    [0.00,0.00,0.00,0.00,0.01,0.01,0.01,0.01,0.01,0.01],
        ("Proxy RPCC", "Pearson"):  [0.11,0.03,0.22,0.10,0.00,0.01,0.04,0.00,0.04,0.14],

        ("Proxy-Delta Approx.", "nMBE"):    [0.28,0.13,0.07,0.02,0.00,0.09,0.08,0.02,0.12,0.06],
        ("Proxy-Delta Approx.", "nRMSE"):   [0.19,0.18,0.19,0.14,0.09,0.07,0.09,0.08,0.09,0.09],
        ("Proxy-Delta Approx.", "Pearson"): [0.16,0.08,0.05,0.05,0.14,0.23,0.18,0.19,0.19,0.18],

        ("Proxy-Delta TSA", "nMBE"):     [0.26,0.27,0.28,0.23,0.01,0.00,0.03,0.01,0.02,0.00],
        ("Proxy-Delta TSA", "nRMSE"):    [0.27,0.30,0.31,0.31,0.24,0.23,0.25,0.24,0.24,0.26],
        ("Proxy-Delta TSA", "Pearson"):  [0.14,0.18,0.17,0.19,0.23,0.27,0.25,0.25,0.24,0.26],

        ("Proxy-Delta RPCC", "nMBE"):     [0.03,0.25,0.23,0.19,0.00,0.06,0.01,0.00,0.04,0.05],
        ("Proxy-Delta RPCC", "nRMSE"):    [0.15,0.16,0.18,0.20,0.12,0.10,0.13,0.12,0.12,0.12],
        ("Proxy-Delta RPCC", "Pearson"):  [0.15,0.13,0.12,0.15,0.21,0.26,0.24,0.23,0.22,0.23],
    }

    index = pd.MultiIndex.from_tuples(data.keys(), names=["Metric family", "Error type"])
    df = pd.DataFrame(list(data.values()), index=index, columns=windows)

    plot_hier_heatmap_clean(df)
def plot_hier_heatmap_clean(
    df,
    vmax=None,
    cmap="plasma",
    sep_color="#B0B0B0",   # light grey
    sep_lw=1.2,
    left_label_pad=0.32,   # how much space to reserve on the left (axes fraction)
    cbar_pad=0.02,
    cbar_width="3.5%",
    cbar_height="90%",
):
    """
    df: DataFrame with MultiIndex rows: (Metric family, Error type)
        columns: windows
    """
    # df = df.sort_index(level=[0, 1])

    vals = df.values
    windows = list(df.columns)
    fams = df.index.get_level_values(0)
    err_types = df.index.get_level_values(1).tolist()
    unique_fams = fams.unique()

    if vmax is None:
        vmax = float(np.nanmax(vals))

    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Heatmap
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)

    # --- X axis
    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels(windows)
    ax.set_xlabel("Window half-width around proxy peak (days)")

    # --- Y axis: error types only
    ax.set_yticks(np.arange(len(err_types)))
    ax.set_yticklabels(err_types)
    ax.set_ylabel(None)

    # --- Compute group boundaries and centres (in row coordinates)
    group_ends = []
    group_centres = []

    start = 0
    for fam in unique_fams:
        n = int((fams == fam).sum())
        end = start + n  # exclusive
        group_ends.append(end)
        group_centres.append((start + end - 1) / 2)
        start = end

    # --- Reserve some left margin for metric-family labels
    # This shifts the axes rightwards without messing with figure layout too much.
    pos = ax.get_position()
    ax.set_position([pos.x0 + left_label_pad * pos.width, pos.y0,
                     pos.width * (1 - left_label_pad), pos.height])

    # --- Draw metric family labels as plain text (NO axis/spine)
    # We'll use axis coordinates for x and data coordinates for y.
    # x in axes coords: negative values go into the left margin we created.
    for fam, yc in zip(unique_fams, group_centres):
        ax.text(
            -0.2, yc, fam,
            transform=ax.get_yaxis_transform(),  # x in axes, y in data
            ha="left", va="center",
            fontsize=8
        )

    # --- Group separator lines across heatmap + extended into left margin
    # Use y in data coords; x in axes coords so we can extend beyond [0,1].
    for end in group_ends[:-1]:
        y = end - 0.5
        ax.hlines(
            y=y,
            xmin=-0.2, xmax=1.0,                 # extend left into label margin
            transform=ax.get_yaxis_transform(),    # x in axes coords, y in data coords
            color=sep_color,
            linewidth=sep_lw,
            zorder=10,
            clip_on=False,
        )

    # --- Colorbar in its own inset axis (prevents overlap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean single-predictor, adjusted $R^2$") 

    # Aesthetic: remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig("fig_signals_4b.pdf", bbox_inches="tight")
    plt.show()
    
def proxy_intro(df_cache: pd.DataFrame):

    df = df_cache.copy()

    # Keep reference-only proxy metrics
    cols = [
        "country",
        "proxy_ref_pearson_r",
        "proxy_ref_rmse",
    ]

    df_ref = df[cols].dropna()

    summary = df_ref.groupby("country").agg({
        "proxy_ref_pearson_r": ["median", "min", "max"],
        "proxy_ref_rmse": ["median", "min", "max"],
    })

    print(summary)

def main():
    # Q1_soc_vs_cap_errors(df_cache=DF_CACHE_10Y)
    # Q2_global_errors(df_cache=DF_CACHE_10Y)
    # Q3_weight_quality_dependence(df_cache=DF_CACHE_10Y)
    Q4a_diagnostics(df_cache=DF_CACHE_DIAGNOSTIC)
    # Q4b_heatmap()
    # proxy_intro(df_cache=DF_CACHE_10Y)

    # moderation_table_1(
    # df_cache=DF_CACHE_10Y,
    # y_field="e_tsa_cem_nrmse_range",
    # x_field="e_tsa_proxy_delta_nrmse_range",
    # m_field="e_proxy_delta_ref_vs_cem_delta_ref_nrmse_range",
    # m_bins=4,
    # by_country=False,
    # free_intercept=True
    # )

    # moderation_graph_density_Wp_bins_by_M(
    #     df_cache=DF_CACHE_10Y,
    #     mode="quartiles",
    #     y_field = 'ldes_cap_error_abs',
    #     x_field='e_tsa_proxy_nrmse_range',
    #     free_intercept=True
    # )

    # moderation_graph_density_Wp_bins_by_M(
    #     df_cache=DF_CACHE_10Y,
    #     mode="quantiles",
    #     y_field = 'ldes_cap_error_abs',
    #     x_field='e_tsa_proxy_nrmse_range',
    #     free_intercept=True,
    #     q=2
    # )





if __name__ == "__main__":
    main()
