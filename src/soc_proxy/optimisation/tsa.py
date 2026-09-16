
# === Core dependencies  ==========================================
from __future__ import annotations
import numpy as np
import pandas as pd
import multiprocessing
import pyomo.environ as pyo
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple
import re

# === External dependency ==========================================
from .biases import generate_endogenous_biases

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
max_threads = multiprocessing.cpu_count()

ENDO_BIAS_CONFIG = dict(
            normalize_mode="minmax",
            smooth_window=1,
            extrema_window=121,
            min_prominence=0.05,
            tau_decay=12,
            weight_global_extreme_boost=True, coef_global_extreme=3.0, tau_global_extreme=5.0,
            weight_endpoints=True, coef_endpoints=2.0,
            priority_floor_multiplier=1.5,
            weight_extremes=True, coef_extremes=0.75,
            weight_span=True, coef_span=1.25, span_alpha=2.0,
            weight_tail_softmax=False, weight_ramp=False, weight_curvature=False, weight_rarity=False,
            plot=False,
        )

# =============================================================================
# Section A: Distance / features helpers
# =============================================================================

def distance_matrix(
    feature_df: pd.DataFrame,
    *,
    matrix_weights: dict = {"renewables": 1.0, "demand": 1.0, "proxy": 1.0},
    metric: str = "euclidean",
    column_prefixes_renewables: List[str] = ('solar','onshore_wind','offshore_wind'),
    column_prefixes_demand:     List[str] = ('demand_power',),
    column_prefixes_proxy:      List[str] = (),
    proxy_window: tuple[str, str] | None = None,  # (start, end) inclusive
) -> np.ndarray:
    """
    Returns a weighted distance matrix D = wR*D_R + wD*D_D + wP*(M ⊙ D_P),
    where M masks rows (days) in the proxy term so it only contributes for rows
    whose index falls within proxy_window.
    """
    def cols_by_prefix(prefixes):
        return [c for c in feature_df.columns if any(c.startswith(p) for p in prefixes)]

    renew_cols = cols_by_prefix(column_prefixes_renewables)
    dem_cols   = cols_by_prefix(column_prefixes_demand)
    prox_cols  = cols_by_prefix(column_prefixes_proxy)

    def scaled_block(cols):
        if not cols:
            return None
        X = feature_df[cols].to_numpy()
        Xz = StandardScaler().fit_transform(X)
        return Xz / np.sqrt(len(cols))

    X_R = scaled_block(renew_cols)
    X_D = scaled_block(dem_cols)
    X_P = scaled_block(prox_cols) if prox_cols else None

    D_R = cdist(X_R, X_R, metric=metric) if X_R is not None else 0.0
    D_D = cdist(X_D, X_D, metric=metric) if X_D is not None else 0.0
    D_P = cdist(X_P, X_P, metric=metric) if X_P is not None else 0.0

    present = {"renewables": X_R is not None, "demand": X_D is not None, "proxy": X_P is not None}
    denom = sum(matrix_weights[k] for k, ok in present.items() if ok) or 1.0
    wR = (matrix_weights["renewables"] if present["renewables"] else 0.0) / denom
    wD = (matrix_weights["demand"]     if present["demand"]     else 0.0) / denom
    wP = (matrix_weights["proxy"]      if present["proxy"]      else 0.0) / denom

    if wP == 0.0:
        M = 0.0
    else:
        if proxy_window is None or proxy_window == 'None':
            M = 1.0
        else:
            if not isinstance(feature_df.index, pd.DatetimeIndex):
                raise ValueError("feature_df.index must be a DatetimeIndex to use proxy_window.")
            start = pd.Timestamp(proxy_window[0]); end = pd.Timestamp(proxy_window[1])
            in_window = (feature_df.index >= start) & (feature_df.index <= end)
            m = np.asarray(in_window, dtype=float).reshape(-1, 1)
            M = m @ np.ones((1, feature_df.shape[0]))

    D = wR * D_R + wD * D_D
    if wP > 0:
        if np.isscalar(M):
            if M != 0.0:
                D += wP * D_P
        else:
            if np.any(M):
                D += wP * (D_P * M)
    return D

# ---- Feature grouping / ORDO matrix (existing logic preserved) ---------------

_HPAT = re.compile(r"^(?P<base>.+)_h(?P<hour>\d{2})$")

def _group_feature_columns(df_features: pd.DataFrame, preferred_features: List[str] | None = None):
    """Return (mode, groups). Mode: 'daily' or 'hourly24'. Groups: base -> cols."""
    cols = list(df_features.columns)
    if preferred_features:
        def keep(c):
            m = _HPAT.match(c); base = m.group("base") if m else c
            return base in preferred_features
        cols = [c for c in cols if keep(c)]

    groups = {}
    for c in cols:
        m = _HPAT.match(c)
        if m:
            base = m.group("base")
            groups.setdefault(base, []).append(c)

    if groups and all(len(sorted(v)) >= 24 for v in groups.values()):
        groups = {k: sorted([c for c in v if _HPAT.match(c)])[:24] for k, v in groups.items()}
        return "hourly24", groups

    if preferred_features:
        groups = {f: [f] for f in preferred_features if f in df_features.columns}
    else:
        groups = {c: [c] for c in df_features.columns}
    return "daily", groups

def _minmax_signed(col: np.ndarray) -> np.ndarray:
    cmin = np.nanmin(col); cmax = np.nanmax(col)
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
        return np.zeros_like(col, dtype=float)
    z = (col - cmin) / (cmax - cmin)
    return 2.0 * z - 1.0

def _ordo_cost_matrix_from_features(
    df_features: pd.DataFrame,
    preferred_features: List[str] | None,
    feature_weights: Dict[str, float] | None,
    normalize: str = "minmax_signed",
) -> np.ndarray:
    """
    Build D_{ij} = Σ_f w_f * ||X_f[i,:] - X_f[j,:]||^2
    (daily or hourly24; per-column normalization; per-block weight via sqrt(w)).
    """
    mode, groups = _group_feature_columns(df_features, preferred_features)
    N = len(df_features)
    if feature_weights is None:
        feature_weights = {}

    X_blocks, W_blocks = [], []
    for base, cols in groups.items():
        block = df_features[cols].to_numpy(dtype=float, copy=False)
        if normalize == "minmax_signed":
            block = np.column_stack([_minmax_signed(block[:, j]) for j in range(block.shape[1])])
        elif normalize != "none":
            raise ValueError(f"Unknown normalize='{normalize}'")
        X_blocks.append(block)
        w = float(feature_weights.get(base, 0.0))
        w = max(0.0, w)
        W_blocks.append((np.sqrt(w) * np.ones(block.shape[1], dtype=float)) if w > 0 else np.zeros(block.shape[1]))

    if not X_blocks:
        raise ValueError("No features selected for cost matrix.")

    X = np.concatenate(X_blocks, axis=1)
    wcol = np.concatenate(W_blocks, axis=0)
    Xw = X * wcol[None, :]

    norms = np.sum(Xw * Xw, axis=1)
    G = Xw @ Xw.T
    D = norms[:, None] + norms[None, :] - 2.0 * G
    D[D < 0] = 0.0
    np.fill_diagonal(D, 0.0)
    return D

# =============================================================================
# Section B: SoC helpers
# =============================================================================

def _apply_linear_soc(s_hat: np.ndarray, a: np.ndarray, cyc: bool = True) -> np.ndarray:
    """Apply the linearised SoC operator in O(T)."""
    y = np.cumsum(a * s_hat)
    if not cyc:
        return y
    soc_end = float(y[-1])
    r = np.linspace(0.0, 1.0, len(s_hat))
    return y - r * soc_end

def _build_a_and_soc_ref(reference_surplus: np.ndarray, eta_ch: float, eta_dis: float, cyc: bool = True):
    """Frozen gains a_t from reference sign pattern + corresponding reference SoC."""
    # If you reintroduce charging/discharging efficiencies, uncomment below.
    # a = np.where(reference_surplus >= 0.0, eta_ch, 1.0 / eta_dis).astype(float)
    a = np.where(reference_surplus >= 0.0, 1.0, 1.0).astype(float)
    soc_ref = _apply_linear_soc(reference_surplus, a, cyc=cyc)
    return a, soc_ref

# =============================================================================
# Section C: MILP
# =============================================================================

def _milp_tsa_core(
    *,
    # --- Geometry / assignment ---
    D: np.ndarray,                        # (N x N) distance matrix
    k: int,                               # number of representatives
    candidates: list[int] | None = None,  # None => unrestricted; else GLOBAL row ids
    # --- SoC / proxy term toggles & data ---
    use_soc_term: bool = False,            # False => pure ORDO, skip all SoC machinery
    a: np.ndarray | None = None,          # (T,), required if use_soc_term
    soc_ref: np.ndarray | None = None,    # (T,), required if use_soc_term
    surplus_daily_for_rows: dict[int, float] | None = None,  # row-id -> daily surplus (required if use_soc_term)
    # --- Objective mixing ---
    lambda_soc: float = 0.5,              # weight on SoC term (ignored if use_soc_term=False)
    # --- Endogenous bias handling (SoC) ---
    endogenous_biases: np.ndarray | None = None,  # (T,)
    bias_minmax: tuple[float, float] = (0.0, 1.0),
    bias_zero_threshold: float = 0.25,    # <= threshold => drop residuals for that t
    # --- SoC bounds (optional soft guardrails) ---
    soc_bounds: tuple[float | None, float | None] = (None, None),  # (lo, hi) in SoC units; None keeps old auto-bounds
    # --- Solver knobs ---
    solver: str = "gurobi",
    MIPGap: float = 0.01,
    threads: int | None = 12,
    timelimit: int = 2000,
    verbose: bool = True,
    root_lp: str = "barrier",
) -> dict:
    """
    Unified TSA MILP with optional SoC term and optional candidate restriction.
    - If candidates is None => unrestricted rows (0..N-1).
    - If use_soc_term=False => pure ORDO; SoC variables are not created at all.
    """
    import numpy as np
    import pyomo.environ as pyo

    N = int(D.shape[0])
    assert D.shape == (N, N), "D must be (N,N)"

    # -----------------------------
    # Select row set (restricted/unrestricted)
    # -----------------------------
    if candidates is None:
        row_ids = list(range(N))               # unrestricted
        # map into D directly
        def D_ij(i, j): return float(D[i, j])
    else:
        print('[TSA] Rebuilding D to acknowledge pre-selection of candidates...')
        row_ids = list(map(int, candidates))   # restricted on GLOBAL ids
        row_of_global = {g: r for r, g in enumerate(row_ids)}
        def D_ij(i, j): return float(D[row_of_global[int(i)], j])

    # -----------------------------
    # Optional SoC block pre-checks
    # -----------------------------
    use_soc = bool(use_soc_term)
    if use_soc:
        assert a is not None and soc_ref is not None and surplus_daily_for_rows is not None, \
            "a, soc_ref, surplus_daily_for_rows required when use_soc_term=True"
        T = int(len(soc_ref))
        assert T == N, "This implementation assumes daily layout: T == N"
        a = np.asarray(a, dtype=float).reshape(-1)
        soc_ref = np.asarray(soc_ref, dtype=float).reshape(-1)
        # Bias handling
        def _rescale_biases(b, lo, hi):
            b = np.asarray(b, dtype=float).reshape(-1)
            if b.size != T or not np.all(np.isfinite(b)): return np.ones(T)*hi
            bmin, bmax = float(np.min(b)), float(np.max(b))
            if bmax <= bmin + 1e-12: return np.ones(T)*hi
            return lo + (b - bmin) * (hi - lo) / (bmax - bmin)
        if endogenous_biases is None:
            b_full = np.ones(T, dtype=float) * bias_minmax[1]
        else:
            b_full = _rescale_biases(endogenous_biases, *bias_minmax)
        active_mask = b_full > float(bias_zero_threshold)
        T_pos = sorted(set([t for t in range(T) if active_mask[t]] +
                           [0, T-1, int(np.argmax(soc_ref)), int(np.argmin(soc_ref))]))
        b_pos = np.array([float(b_full[t]) for t in T_pos], dtype=float)
        b_avg = float(np.mean(b_pos)) if b_pos.size else 1.0
        if not np.isfinite(b_avg) or b_avg <= 0: b_avg = 1.0
        if verbose:
            print(f"[TSA] SoC Objective term enabled. Active residuals: {len(T_pos)}/{T} "
                  f"({100*len(T_pos)/max(1,T):.1f}%), bias_avg_active={b_avg:.3f}")
    
    # -----------------------------
    # Build model
    # -----------------------------

    print('[TSA] Building pyomo model.')
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=row_ids, ordered=True)   # row ids (GLOBAL ids if restricted)
    m.J = pyo.RangeSet(0, N - 1)

    # Distance param
    m.D = pyo.Param(m.I, m.J, initialize={(i, j): D_ij(i, j) for i in row_ids for j in range(N)},
                    within=pyo.NonNegativeReals)

    # Decision vars
    m.y = pyo.Var(m.I, domain=pyo.Binary)          # choose reps
    m.x = pyo.Var(m.I, m.J, domain=pyo.Binary)     # assign day j to row i

    # ORDO constraints
    m.assign_once = pyo.Constraint(m.J, rule=lambda _m, j: sum(_m.x[i, j] for i in _m.I) == 1)
    m.link        = pyo.Constraint(m.I, m.J, rule=lambda _m, i, j: _m.x[i, j] <= _m.y[i])
    m.num_reps    = pyo.Constraint(rule=lambda _m: sum(_m.y[i] for i in _m.I) == k)

    # -----------------------------
    # Optional SoC block
    # -----------------------------
    if use_soc:
        # set index sets for SoC
        m.T = pyo.RangeSet(0, T - 1)
        m.Tz = pyo.Set(initialize=T_pos, ordered=True)

        # map surplus per chosen row to time t==j (daily layout)
        S_map = {int(i): float(surplus_daily_for_rows[int(i)]) for i in row_ids}
        def surplus_hat_rule(_m, t):
            j = int(t)
            return sum(S_map[int(i)] * _m.x[i, j] for i in _m.I)
        m.surplus_hat = pyo.Expression(m.T, rule=surplus_hat_rule)

        # SoC bounds
        if soc_bounds != (None, None):
            lo = float(soc_bounds[0]) if soc_bounds[0] is not None else float(np.min(soc_ref)) - 1e-9
            hi = float(soc_bounds[1]) if soc_bounds[1] is not None else float(np.max(soc_ref)) + 1e-9
        else:
            UB = 1.1 * max(1e-9, float(np.max(np.abs(soc_ref))))
            lo, hi = -UB, UB

        m.soc_raw   = pyo.Var(m.T, bounds=(lo, hi))
        m.soc_raw_0 = pyo.Constraint(expr=m.soc_raw[0] == a[0] * m.surplus_hat[0])
        def soc_raw_dyn_rule(_m, t):
            if t == _m.T.first(): return pyo.Constraint.Skip
            return _m.soc_raw[t] == _m.soc_raw[t - 1] + a[int(t)] * _m.surplus_hat[t]
        m.soc_raw_dyn = pyo.Constraint(m.T, rule=soc_raw_dyn_rule)
        m.soc_end     = pyo.Var(bounds=(lo, hi))
        m.soc_end_def = pyo.Constraint(expr=m.soc_end == m.soc_raw[T - 1])

        # L1 residuals on active set
        def _z_bounds(_m, t): return (0.0, (hi - lo) + abs(float(soc_ref[int(t)])))
        m.socdiff = pyo.Var(m.Tz)
        m.z       = pyo.Var(m.Tz, domain=pyo.NonNegativeReals, bounds=_z_bounds)
        m.soc_def = pyo.Constraint(m.Tz, rule=lambda _m, t: _m.socdiff[t] == soc_ref[int(t)] - _m.soc_raw[t])
        m.z_pos   = pyo.Constraint(m.Tz, rule=lambda _m, t: _m.z[t] >=  _m.socdiff[t])
        m.z_neg   = pyo.Constraint(m.Tz, rule=lambda _m, t: _m.z[t] >= -_m.socdiff[t])

    # -----------------------------
    # Objective 
    # -----------------------------
    # Distance part (mean-scale)
    if candidates is None:
        dist_scale = max(1e-12, float(np.max(D)))
    else:
        # mean over restricted rows vs all columns
        dist_scale = max(1e-12, float(np.max(D[[{j:i for i,j in enumerate(row_ids)}[i] for i in row_ids], :])))

    m.dist_term_scaled = pyo.Expression(
        rule=lambda _m: sum(_m.x[i, j] * _m.D[i, j] for i in _m.I for j in _m.J) / dist_scale
    )
    if verbose:
        print(f'[TSA] Optimisation Lambda-SoC set to {lambda_soc}') 
        #TODO would be so much better with a dynamic lambda, given that we are less confident in the soc near 0, 
        # we could encourage the model to focus on timeseries errors here rather than the proxy
    if use_soc:
        prox_scale = max(1e-12, float(np.max(np.abs(soc_ref)))) * b_avg
        b_for_Tz = {int(t): float(b_full[int(t)]) for t in T_pos}
        m.prox_term_scaled = pyo.Expression(
            rule=lambda _m: (sum(b_for_Tz[int(t)] * _m.z[t] for t in _m.Tz) / prox_scale) if len(T_pos) else 0.0
        )
        m.obj = pyo.Objective(
            rule=lambda _m: (1 - lambda_soc) * _m.dist_term_scaled + lambda_soc * _m.prox_term_scaled,
            sense=pyo.minimize
        )
    else:
        # pure ORDO
        m.obj = pyo.Objective(rule=lambda _m: _m.dist_term_scaled, sense=pyo.minimize)

    # -----------------------------
    # Solve
    # -----------------------------
    if solver == "gurobi":
        opt = pyo.SolverFactory("gurobi_persistent"); opt.set_instance(m)
        opt.set_gurobi_param('MIPGap', MIPGap)
        if threads is not None: opt.set_gurobi_param('Threads', int(threads))
        opt.set_gurobi_param('LogToConsole', int(verbose))
        opt.set_gurobi_param('TimeLimit', int(timelimit))
        opt.set_gurobi_param('Presolve', -1)
        opt.set_gurobi_param('Aggregate', 2)
        if root_lp == "barrier":
            opt.set_gurobi_param('Method', 2); opt.set_gurobi_param('Crossover', 0)
        elif root_lp == "dual":
            opt.set_gurobi_param('Method', 1)
        res = opt.solve(tee=verbose)
    else:
        res = pyo.SolverFactory(solver).solve(m, tee=verbose, options={'MIPGap': MIPGap, 'TimeLimit': int(timelimit)})

    # -----------------------------
    # Extract + report shares
    # -----------------------------
    selected_days = [int(i) for i in m.I if pyo.value(m.y[i]) > 0.5]
    assignments   = {int(j): max([int(i) for i in m.I], key=lambda i: pyo.value(m.x[i, j])) for j in m.J}

    try:
        obj_val     = float(pyo.value(m.obj))
        dist_scaled = float(pyo.value(m.dist_term_scaled))
        if use_soc:
            prox_scaled = float(pyo.value(m.prox_term_scaled))
            dist_contrib = (1 - lambda_soc) * dist_scaled
            prox_contrib =      lambda_soc  * prox_scaled
            share_dist = (dist_contrib / obj_val) if obj_val else 0.0
            share_prox = (prox_contrib / obj_val) if obj_val else 0.0
            print(f"[TSA] [Obj.] total={obj_val:.6g} | dist={dist_contrib:.6g} | prox={prox_contrib:.6g} "
                  f"| shares: dist={share_dist:.3f}, prox={share_prox:.3f}")
            breakdown = dict(obj_total=obj_val, dist_scaled=dist_scaled, prox_scaled=prox_scaled,
                             dist_contrib=dist_contrib, prox_contrib=prox_contrib,
                             share_dist=share_dist, share_prox=share_prox)
        else:
            print(f"[TSA] [Obj.] total={obj_val:.6g} | dist={dist_scaled:.6g} (SoC off)")
            breakdown = dict(obj_total=obj_val, dist_scaled=dist_scaled, share_dist=1.0, share_prox=0.0)
    except Exception:
        breakdown = None

    out = {
        "selected_days": selected_days,
        "assignments": assignments,
        "model": m,
        "results": res,
        "objective_breakdown": breakdown,
    }
    if use_soc:
        out.update({
            "proxy_active_timesteps": T_pos,
            "bias_avg_active": b_avg,
        })
    return out

def milp_tsa(D: np.ndarray, k: int, solver="gurobi", MIPGap=0.01, verbose=True):
    return _milp_tsa_core(D=D, k=k, candidates=None, use_soc_term=False, solver=solver, MIPGap=MIPGap, verbose=verbose)

def solve_ordo_with_endogenous_soc(
    *, df_features, k, feature_weights, preferred_features, eta_ch, eta_dis,
    lambda_soc, reference_surplus, normalize="minmax_signed", solver="gurobi",
    MIPGap=0.01, verbose=True, surplus_by_day=None, use_endogenous_biases=False, timelimit=1200
):
    D = _ordo_cost_matrix_from_features(df_features, preferred_features, feature_weights, normalize)
    N = len(df_features)
    s_scale = max(1.0, float(np.percentile(np.abs(surplus_by_day), 99.5)))
    S_scaled = surplus_by_day / s_scale
    ref_scaled = reference_surplus / s_scale
    a, soc_ref = _build_a_and_soc_ref(ref_scaled, eta_ch, eta_dis, cyc=True)
    biases = (generate_endogenous_biases(surplus_by_day.cumsum(), params=ENDO_BIAS_CONFIG)) if use_endogenous_biases else None
    surplus_map = {i: float(S_scaled[i]) for i in range(N)}
    return _milp_tsa_core(D=D, k=k, candidates=None,
                          use_soc_term=True, a=a, soc_ref=soc_ref,
                          surplus_daily_for_rows=surplus_map,
                          lambda_soc=lambda_soc,
                          endogenous_biases=biases, bias_minmax=(0.0,1.0), bias_zero_threshold=0.25,
                          solver=solver, MIPGap=MIPGap, timelimit=timelimit, verbose=verbose)

def solve_ordo_with_endogenous_soc_restricted(
    *, df_features, k, feature_weights, preferred_features, candidates, fix_reps, warm_start_rep_for_day,
    eta_ch, eta_dis, lambda_soc, surplus_by_day, normalize="minmax_signed",
    solver="gurobi", MIPGap=0.01, threads=12, timelimit=1200, verbose=True, lp_method="barrier",
    use_endogenous_biases=False, bias_minmax=(0.0,1.0), bias_zero_threshold=0.0,
):  
    
    
    D = _ordo_cost_matrix_from_features(df_features, preferred_features, feature_weights, normalize)
    N = len(df_features)
    s_scale = max(1.0, float(np.percentile(np.abs(surplus_by_day), 99.5)))
    S_scaled = surplus_by_day / s_scale
    ref_scaled = surplus_by_day.reshape(-1) / s_scale
    a, soc_ref = _build_a_and_soc_ref(ref_scaled, eta_ch, eta_dis, cyc=True)
    biases = (generate_endogenous_biases(surplus_by_day.cumsum(), params=ENDO_BIAS_CONFIG)) if use_endogenous_biases else None
    S_map = {int(i): float(S_scaled[int(i)]) for i in range(N)}
    return _milp_tsa_core(D=D, k=k, candidates=list(map(int, candidates)),
                          use_soc_term=True, a=a, soc_ref=soc_ref,
                          surplus_daily_for_rows=S_map,
                          lambda_soc=lambda_soc,
                          endogenous_biases=biases, bias_minmax=bias_minmax, bias_zero_threshold=bias_zero_threshold,
                          solver=solver, MIPGap=MIPGap, threads=threads, timelimit=timelimit, verbose=verbose)

def solve_tsa(
    *,
    # --- Required core inputs ---
    df_features: pd.DataFrame,           # daily rows (N), feature columns (hourly or daily)
    k: int,                               # number of representatives
    feature_weights: Dict[str, float] | None = None,
    preferred_features: List[str] | None = None,
    normalize: str = "minmax_signed",     # for feature blocks -> D
    # --- Candidate restriction (None => unrestricted) ---
    candidates: List[int] | None = None,  # global day IDs to restrict the reps to
    # --- SoC / endogenous proxy toggle & data ---
    use_soc_term: bool = False,
    surplus_by_day: np.ndarray | None = None,   # shape (N,), daily net surplus (same order as df_features)
    reference_surplus: np.ndarray | None = None,# optional; if None, uses surplus_by_day as the reference
    eta_ch: float = 1.0, eta_dis: float = 1.0,  # efficiencies if you re-enable them in _build_a_and_soc_ref
    lambda_soc: float = 0.5,                    # weight on SoC term when use_soc_term=True
    # --- Optional endogenous biasing ---
    use_endogenous_biases: bool = False,
    biases_params: Dict | None = None,          # params for generate_endogenous_biases(...)
    bias_minmax: tuple[float, float] = (0.0, 1.0),
    bias_zero_threshold: float = 0.25,
    # --- Optional distance normalization ---
    normalize_D_to_unit: bool = False,          # if True, scale/clamp D to [0,1] (global)
    D_unit_method: str = "p99_clip",
    # --- Solver knobs ---
    solver: str = "gurobi",
    MIPGap: float = 0.01,
    threads: int | None = 12,
    timelimit: int = 2000,
    verbose: bool = True,
    root_lp: str = "barrier",
) -> dict:
    """
    One-stop TSA solver:
      - If `candidates` is None -> unrestricted reps; else restricted to those global day IDs.
      - If `use_soc_term` is False -> pure ORDO; SoC variables/constraints are not created.
      - If `use_endogenous_biases` is True -> weights are generated and applied to the SoC L1 residuals.

    Returns the same dict structure as before (selected_days, assignments, model, results, objective_breakdown, ...).
    """
    # --- 1) Build distance matrix from features
    D_full = _ordo_cost_matrix_from_features(
        df_features=df_features,
        preferred_features=preferred_features,
        feature_weights=feature_weights,
        normalize=normalize,
    )

    # Optional: normalize D to [0,1] globally (safe scaling)
    if normalize_D_to_unit:
        D_full = _normalize_D_unit(D_full, method=D_unit_method)

    N = len(df_features)

    # --- 2) If SoC term is off, delegate immediately
    if not use_soc_term:
        return _milp_tsa_core(
            D=D_full, k=k, candidates=candidates,
            use_soc_term=False,
            solver=solver, MIPGap=MIPGap, threads=threads, timelimit=timelimit, verbose=verbose, root_lp=root_lp,
        )

    # --- 3) Prepare SoC inputs (only if needed)
    if surplus_by_day is None:
        raise ValueError("surplus_by_day must be provided when use_soc_term=True.")
    if surplus_by_day.shape[0] != N:
        raise ValueError(f"surplus_by_day length {surplus_by_day.shape[0]} != N {N}")

    # Robust surplus scaling (same recipe you’ve been using)
    s_scale = max(1.0, float(np.percentile(np.abs(surplus_by_day), 99.5)))
    S_scaled = surplus_by_day.reshape(-1) / s_scale

    # Reference for building a_t and soc_ref: explicit reference_surplus if given, else use the same surplus
    ref_surplus = reference_surplus.reshape(-1) / s_scale if reference_surplus is not None else S_scaled
    a_vec, soc_ref = _build_a_and_soc_ref(ref_surplus, eta_ch=eta_ch, eta_dis=eta_dis, cyc=True)

    # Optional endogenous biases
    biases = None
    if use_endogenous_biases:
        params = dict(
            normalize_mode="minmax", smooth_window=1, extrema_window=121, min_prominence=0.05,
            tau_decay=12, weight_global_extreme_boost=True, coef_global_extreme=3.0, tau_global_extreme=5.0,
            weight_endpoints=True, coef_endpoints=2.0, priority_floor_multiplier=1.5,
            weight_extremes=True, coef_extremes=0.75, weight_span=True, coef_span=1.25, span_alpha=2.0,
            weight_tail_softmax=False, weight_ramp=False, weight_curvature=False, weight_rarity=False,
            plot=False,
        )
        if biases_params: params.update(biases_params)
        # Typical input for biases: a SoC-like proxy; using cumulative surplus here
        biases = generate_endogenous_biases(pd.Series(np.cumsum(S_scaled)), params=params)

    # Map surplus by GLOBAL row id (the core expects dict[int->float])
    surplus_map = {int(i): float(S_scaled[int(i)]) for i in range(N)}

    # --- 4) Delegate to the unified core
    return _milp_tsa_core(
        D=D_full,
        k=k,
        candidates=candidates,           # None => unrestricted; list => restricted
        use_soc_term=True,
        a=a_vec,
        soc_ref=soc_ref,
        surplus_daily_for_rows=surplus_map,
        lambda_soc=lambda_soc,
        endogenous_biases=biases,
        bias_minmax=bias_minmax,
        bias_zero_threshold=bias_zero_threshold,
        solver=solver, MIPGap=MIPGap, threads=threads, timelimit=timelimit, verbose=verbose, root_lp=root_lp,
    )


# ===== Helper: global unit normalization for D =========================
def _normalize_D_unit(D: np.ndarray, method: str = "p99_clip") -> np.ndarray:
    """
    Return D' in [0, 1] using a single global scale.
    - 'max'      : divide by global max of off-diagonals, clip to 1
    - 'p99_clip' : divide by 95th percentile of off-diagonals, then clip to 1 (robust)
    - 'median2'  : divide by 2*median(off-diagonals) (no clip)
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = D[mask]
    if vals.size == 0:
        return D
    if method == "max":
        s = max(1e-12, float(np.max(vals)))
        return np.minimum(D / s, 1.0)
    elif method == "p99_clip":
        s = max(1e-12, float(np.quantile(vals, 0.99)))
        return np.minimum(D / s, 1.0)
    elif method == "median2":
        s = max(1e-12, 2.0 * float(np.median(vals)))
        return D / s
    else:
        raise ValueError(f"Unknown D unit-normalization method: {method}")
    
# =============================================================================
# Section E: Utilities (unchanged)
# =============================================================================

def rebuild_from_assignments(df_features: pd.DataFrame, rep_for_day: np.ndarray) -> pd.DataFrame:
    """Build a synthetic df by copying the representative row chosen for each day."""
    assert len(rep_for_day) == len(df_features)
    return df_features.iloc[rep_for_day].reset_index(drop=True)

def save_milp_result_to_cluster_map(result: dict, dates_index: pd.DatetimeIndex, output_path: str):
    """Save TSA result in Calliope-compatible format (timesteps, PeriodNum)."""
    if not isinstance(dates_index, pd.DatetimeIndex):
        raise ValueError("dates_index must be a DatetimeIndex")
    daily_dates = dates_index.to_list()
    day_to_rep = {daily_dates[j]: daily_dates[result['assignments'][j]] for j in result['assignments']}
    mapping_df = pd.DataFrame.from_dict(day_to_rep, orient="index", columns=["PeriodNum"])
    mapping_df.index.name = "timesteps"
    mapping_df = mapping_df.reset_index()
    mapping_df["timesteps"] = mapping_df["timesteps"].dt.strftime("%Y-%m-%d")
    mapping_df["PeriodNum"] = mapping_df["PeriodNum"].dt.strftime("%Y-%m-%d")
    mapping_df.set_index('timesteps', inplace=True)
    mapping_df.to_csv(output_path)
    print('[TSA] Saved Cluster Map')