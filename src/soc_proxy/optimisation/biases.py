"""
helper_endongeous_biases.py

Generate endogenous TSA weighting biases for a timestamped SoC proxy.
Each weight component is a separate, well-commented subfunction and can be
toggled via params. Includes optional visualization.

Usage:
    from helper_endongeous_biases import generate_endogenous_biases
    w = generate_endogenous_biases(df_reference, params={"column": "soc_proxy", "plot": True})
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Utilities
# =========================

def _normalize_series(s: pd.Series, mode: str = "minmax") -> Tuple[pd.Series, Dict[str, float]]:
    """
    Normalize a pandas Series according to the chosen mode.

    Parameters
    ----------
    s : pd.Series
        Input series to normalize.
    mode : str
        'minmax' scales to [0, 1].
        'std' scales to zero mean, unit variance.
        'none' returns a copy.

    Returns
    -------
    s_norm : pd.Series
        Normalized series.
    meta : dict
        Parameters used for normalization.
    """
    if mode == "minmax":
        smin, smax = float(s.min()), float(s.max())
        span = smax - smin if smax > smin else 1.0
        return (s - smin) / span, {"mode": mode, "min": smin, "max": smax, "span": span}
    elif mode == "std":
        mu, sd = float(s.mean()), float(s.std(ddof=0))
        sd = sd if sd > 0 else 1.0
        return (s - mu) / sd, {"mode": mode, "mean": mu, "std": sd}
    else:
        return s.copy(), {"mode": "none"}


def _smooth_series(s: pd.Series, window: int) -> pd.Series:
    """
    Apply a simple centered rolling mean with given integer window size (in timesteps).

    Notes
    -----
    - 'center=True' aligns stationary/curvature calculations with the original index.
    - Edge NaNs are forward/backward filled to avoid artifacts.
    """
    if window <= 1:
        return s.copy()
    sm = s.rolling(window=window, center=True, min_periods=max(2, window // 3)).mean()
    return sm.ffill().bfill()


def _finite_differences(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Compute first and second finite differences on a series aligned to original index.
    Uses central differences where possible for better symmetry.

    Returns
    -------
    ds : pd.Series
        First difference approximation.
    d2s : pd.Series
        Second difference approximation.
    """
    values = s.values.astype(float)
    ds = np.zeros_like(values)
    d2s = np.zeros_like(values)

    # First diff (central inside, forward/backward at ends)
    ds[1:-1] = (values[2:] - values[:-2]) / 2.0
    ds[0] = values[1] - values[0]
    ds[-1] = values[-1] - values[-2]

    # Second diff (central; copy neighbors for ends)
    d2s[1:-1] = values[2:] - 2 * values[1:-1] + values[:-2]
    d2s[0] = d2s[1]
    d2s[-1] = d2s[-2]

    return pd.Series(ds, index=s.index), pd.Series(d2s, index=s.index)


def _local_extrema_with_prominence(
    s: pd.Series, window: int, min_prominence: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify prominent local maxima and minima without SciPy (heuristic).

    Heuristic:
    - A point is a local max if it equals the max within a centered window and
      is strictly greater than its immediate neighbors.
    - Prominence ≈ difference between the point and the larger of the two side-window minima.
      (Analogous definition for minima.)
    - Keep only points whose prominence >= min_prominence * span(s).

    Parameters
    ----------
    s : pd.Series
        Input series (preferably smoothed/normalized).
    window : int
        Odd integer neighborhood; if even, incremented by 1.
    min_prominence : float
        Minimum prominence as a fraction of the total series span in [0, 1].

    Returns
    -------
    peaks_idx, troughs_idx : np.ndarray
        Integer positions of detected peaks and troughs (relative to s.index order).
    """
    n = len(s)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    half = window // 2

    vals = s.values.astype(float)
    absolute_peak = vals.max()
    absolute_trough =  vals.min()
    starting_soc = vals[0]
    span = float(absolute_peak - absolute_trough)
    
    prom_thresh = min_prominence * (span if span > 0 else 1.0)

    peaks, troughs = [], []
    peak_vals, trough_vals = [], []
    for i in range(1, n - 1):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        local = vals[l:r]
        v = vals[i]

        # Peak test
        if v >= local.max() and v > vals[i - 1] and v > vals[i + 1]:
            left_min = vals[l:i].min() if i > l else vals[i]
            right_min = vals[i + 1:r].min() if i + 1 < r else vals[i]
            prom = v - max(left_min, right_min)
            if prom >= prom_thresh:
                peaks.append(i)
                peak_vals.append(v)
            if v in peak_vals and i not in peaks: #adds any double maxima
                peaks.append(i)
                peak_vals.append(v)

        # Trough test
        if v <= local.min() and v < vals[i - 1] and v < vals[i + 1]:
            left_max = vals[l:i].max() if i > l else vals[i]
            right_max = vals[i + 1:r].max() if i + 1 < r else vals[i]
            prom = min(left_max, right_max) - v
            if prom >= prom_thresh:
                troughs.append(i)
                trough_vals.append(v)

    return np.array(peaks, dtype=int), np.array(troughs, dtype=int), np.array(peak_vals), np.array(trough_vals)


# =========================
# Weight components
# =========================

def _w_extremum_proximity_scaled(index: pd.Index, peaks: np.ndarray, troughs: np.ndarray,
                                 p_scale: np.ndarray, q_scale: np.ndarray, tau: float) -> np.ndarray:
    """Like _w_extremum_proximity but each center is scaled by its relative prominence (0..1)."""
    n = len(index)
    t = np.arange(n)
    w = np.zeros(n, dtype=float)
    tau = max(float(tau), 1.0)
    for k, p in enumerate(peaks):
        w += (p_scale[k] if len(p_scale) else 1.0) * np.exp(-np.abs(t - p) / tau)
    for k, q in enumerate(troughs):
        w += (q_scale[k] if len(q_scale) else 1.0) * np.exp(-np.abs(t - q) / tau)
    return w


def _relative_prominence_weights(s_smooth: pd.Series, peaks: np.ndarray, troughs: np.ndarray,
                                 window: int, min_prominence: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative prominence (0..1) for each detected peak/trough using the same heuristic
    used in _local_extrema_with_prominence. The most prominent extreme gets weight 1.
    """
    n = len(s_smooth)
    if n == 0:
        return np.array([]), np.array([])
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    half = window // 2
    vals = s_smooth.values.astype(float)

    def prom_at(i: int, is_peak: bool) -> float:
        l = max(0, i - half)
        r = min(n, i + half + 1)
        v = vals[i]
        if is_peak:
            left_min  = vals[l:i].min() if i > l else v
            right_min = vals[i+1:r].min() if i+1 < r else v
            return v - max(left_min, right_min)
        else:
            left_max  = vals[l:i].max() if i > l else v
            right_max = vals[i+1:r].max() if i+1 < r else v
            return min(left_max, right_max) - v

    p_prom = np.array([max(0.0, prom_at(i, True))  for i in peaks])   if len(peaks)   else np.array([])
    q_prom = np.array([max(0.0, prom_at(i, False)) for i in troughs]) if len(troughs) else np.array([])

    # Normalize to 0..1 (most prominent = 1)
    if p_prom.size:
        p_prom = p_prom / max(p_prom.max(), 1e-12)
    if q_prom.size:
        q_prom = q_prom / max(q_prom.max(), 1e-12)
    return p_prom, q_prom


def _w_rarity_state_space(s: pd.Series, bins: int = 20, eps: float = 1e-6) -> np.ndarray:
    """
    Inverse-density weight over SoC state space using a histogram proxy.

    Steps
    -----
    - Bin SoC values into 'bins' equal-width bins between min and max.
    - Weight at timestep t is 1 / (eps + density_of_bin(s_t)), where densities sum to 1.
    """
    n = len(s)
    if n == 0:
        return np.array([])
    vals = s.values.astype(float)
    smin, smax = float(vals.min()), float(vals.max())
    if smax == smin:
        return np.ones(n, dtype=float)

    hist, edges = np.histogram(vals, bins=bins, range=(smin, smax), density=False)
    density = hist / float(n)  # convert counts to probabilities
    bin_idx = np.clip(np.digitize(vals, edges) - 1, 0, bins - 1)
    w = 1.0 / (eps + density[bin_idx])
    return w


def _w_curvature(d2s: pd.Series) -> np.ndarray:
    """
    Curvature-based weight using absolute second difference, normalized by its max.
    """
    a = np.abs(d2s.values.astype(float))
    m = a.max()
    if m == 0:
        return np.zeros_like(a)
    return a / m


def _w_ramp(ds: pd.Series) -> np.ndarray:
    """
    Ramp-based weight using absolute first difference, normalized by its max.
    """
    a = np.abs(ds.values.astype(float))
    m = a.max()
    if m == 0:
        return np.zeros_like(a)
    return a / m


def _w_endpoints(
    n: int,
    include_start: bool = True,
    include_end: bool = True,
    weight_start: float = 1.0,
    weight_end: float = 1.0
) -> np.ndarray:
    """
    Endpoint guard weights. Places mass at t=0 and/or t=T.
    """
    w = np.zeros(n, dtype=float)
    if n == 0:
        return w
    if include_start:
        w[0] = float(weight_start)
    if include_end:
        w[-1] = float(weight_end)
    return w


def _w_span_distance(s: pd.Series, alpha: float = 1.0) -> np.ndarray:
    """
    Distance-from-endpoints (span) weight.

    Emphasizes states far from both s(0) and s(T), which tends to
    favor the global extremes (max/min) and their neighborhoods.

    w_t = ((|s_t - s_0|)/span)^alpha + ((|s_t - s_T|)/span)^alpha
    """
    vals = s.values.astype(float)
    s0 = vals[0]
    sT = vals[-1]
    smin, smax = float(vals.min()), float(vals.max())
    span = (smax - smin) if (smax > smin) else 1.0
    d0 = np.abs(vals - s0) / span
    dT = np.abs(vals - sT) / span
    return np.power(d0, alpha) + np.power(dT, alpha)

def _w_tail_softmax(s_norm: pd.Series, k: float = 8.0, q: float = 0.98) -> np.ndarray:
    """
    Emphasize upper and lower tails with an exponential softmax around high/low quantiles.
    - s_norm should be min-max normalized (0..1).
    - k controls sharpness (larger => more mass at the very extremes).
    - q is the high quantile (low tail uses 1-q).
    """
    x = s_norm.values.astype(float)
    hi = np.quantile(x, q)
    lo = np.quantile(x, 1.0 - q)
    up = np.exp(k * (x - hi))        # lights up near top tail
    dn = np.exp(k * (lo - x))        # lights up near bottom tail
    return up + dn

def _w_global_extreme_plateau(
    s: pd.Series,
    window: int = 10,
    tau: float = 3.0,
    level: float = 1.0,
    tol_frac: float = 1e-6,
) -> np.ndarray:
    """
    Create a *flat* plateau at height `level` for all timesteps within +/- `window`
    of the global max *or* global min. Outside that window, decay exponentially
    with scale `tau`. This yields a 'flat top and steep sides' around both extremes.

    - Handles flat-topped peaks/valleys by treating any point within `tol_frac * span(s)`
      of the global max/min as part of the 'ridge center' set.
    - Plateau is continuous with the decay: w(d=window) = level.

    Parameters
    ----------
    s : pd.Series
        Input series (ideally the smoothed/normalized proxy used for long-window behavior).
    window : int
        Half-width of the plateau around each extreme (±window).
    tau : float
        Exponential decay scale outside the plateau; smaller => steeper sides.
    level : float
        Plateau height (same for max and min plateaus).
    tol_frac : float
        Tolerance vs. span(s) for including near-equal values in the ridge centers.

    Returns
    -------
    w : np.ndarray
        Weight array with flat plateaus around both extremes and steep exponential sides.
    """
    n = len(s)
    if n == 0:
        return np.array([])

    v = s.values.astype(float)
    smin, smax = float(v.min()), float(v.max())
    span = smax - smin if smax > smin else 1.0
    tol = tol_frac * span

    # Ridge centers for max/min (handle flat tops/valleys)
    max_centers = np.where(np.abs(v - smax) <= tol)[0]
    min_centers = np.where(np.abs(v - smin) <= tol)[0]
    if max_centers.size == 0:
        max_centers = np.array([int(np.argmax(v))])
    if min_centers.size == 0:
        min_centers = np.array([int(np.argmin(v))])

    t = np.arange(n)

    def nearest_distance(centers: np.ndarray) -> np.ndarray:
        if centers.size == 1:
            return np.abs(t - centers[0])
        return np.min(np.abs(t[:, None] - centers[None, :]), axis=1)

    # Distance to the *closest* ridge (either max-side or min-side)
    d_max = nearest_distance(max_centers)
    d_min = nearest_distance(min_centers)
    d = np.minimum(d_max, d_min)

    w = np.zeros(n, dtype=float)
    if window < 0:
        window = 0
    inside = d <= window
    outside = ~inside

    # Flat plateau inside ±window at height = level
    w[inside] = level

    # Steep exponential decay outside; continuous at boundary (d = window)
    if tau <= 0:
        # Step drop to 0 outside (extremely steep)
        w[outside] = 0.0
    else:
        w[outside] = level * np.exp(-(d[outside] - window) / float(tau))

    return w



# =========================
# Public API
# =========================

def generate_endogenous_biases(
    df_reference,
    params: Optional[Dict] = None
) -> np.ndarray:
    """
    Generate endogenous TSA weighting biases based on a reference SoC proxy signal.

    Parameters
    ----------
    df_reference : pd.DataFrame
        Must contain a single column (or a column named via params['column']) representing the SoC proxy.
        Index should be timestamped; if not, it still works (windows are in timesteps).
    params : dict, optional
        Configuration dictionary. Defaults below.

        Keys (with defaults)
        --------------------
        column: str | None
            Which column to use if multiple; default uses the first column.
        normalize_mode: str
            'minmax' | 'std' | 'none'. Default 'minmax'.
        smooth_window: int
            Rolling window (timesteps) for smoothing prior to derivatives/extrema. Default 169.
        extrema_window: int
            Neighborhood (timesteps) for peak/trough detection. Default 169.
        min_prominence: float
            Minimum prominence as fraction of series span (0..1). Default 0.1.
        tau_decay: float
            Time-decay constant (timesteps) for extremum proximity. Default 48.
        rarity_bins: int
            Number of bins for inverse-density weighting. Default 20.
        span_alpha: float
            Exponent for span distance weighting. Default 1.5.
        include_start_end: bool
            Whether to place endpoint weights. Default True.
        endpoint_weight_start: float
            Start endpoint mass. Default 1.0.
        endpoint_weight_end: float
            End endpoint mass. Default 1.0.

        Toggles (booleans) for each component
        -------------------------------------
        weight_extremes: bool = True
        weight_rarity: bool = True
        weight_curvature: bool = False
        weight_ramp: bool = True
        weight_endpoints: bool = True
        weight_span: bool = True

        Blend coefficients (non-negative floats)
        ----------------------------------------
        coef_extremes: float = 1.0
        coef_rarity: float = 0.5
        coef_curvature: float = 0.5
        coef_ramp: float = 0.25
        coef_endpoints: float = 1.0
        coef_span: float = 1.0

        Plotting
        --------
        plot: bool = True
            If True, produce a line plot of the proxy with an overlapping scatter of weights (scaled).
        plot_scale: float = 0.25
            Fraction of proxy span used to vertically scale the weight markers.

    Returns
    -------
    weights : np.ndarray
        Final weight array (length equal to df_reference length), normalized so sum(weights) == N.
    """
    # Defaults
    dflt = dict(
        normalize_mode="minmax",
        smooth_window=1,       # ~weekly if hourly data
        extrema_window=91,
        min_prominence=0.05,
        tau_decay=30,
        rarity_bins=20,
        span_alpha=1.5,
        include_start_end=True,
        endpoint_weight_start=1.0,
        endpoint_weight_end=1.0,
        tau_global_extreme = 7.0,
        tail_k = 8.0,
        tail_q = 0.98,
        priority_floor_multiplier = 1.25,
        gep_window=0.01*len(df_reference),
        
        # toggles
        weight_extremes=True,
        weight_rarity=False,
        weight_curvature=True,
        weight_ramp=True,
        weight_endpoints=True,
        weight_span=True,
        weight_global_extreme_boost = True,
        weight_tail_softmax = True,
        weight_global_extreme_halo=True,
        # coefficients
        coef_extremes=1.0,
        coef_rarity=0.0, #this one is weird, don't use
        coef_curvature=0.25,
        coef_ramp=1,
        coef_endpoints=2.0,
        coef_span=0.5,
        coef_global_extreme = 1.0,    
        coef_global_extreme_halo=2.0,    
        # plotting
        plot=False,
        plot_scale=1,
    )
    if params is None:
        params = {}
    cfg = {**dflt, **params}

    # Extract series
    s = pd.Series(df_reference.copy())

    # Keep original for plotting
    s_plot = s.copy()

    # Normalize and smooth for feature extraction
    s_norm, _ = _normalize_series(s, mode=cfg["normalize_mode"])
    s_smooth = _smooth_series(s_norm, window=int(cfg["smooth_window"]))

    print('[TSA] Generating Endogenous Biases')

    # Differences for curvature and ramp
    ds, d2s = _finite_differences(s_smooth)

    # Local extrema (prominent)
    peaks, troughs, _, _ = _local_extrema_with_prominence(
        s_smooth, window=int(cfg["extrema_window"]), min_prominence=float(cfg["min_prominence"])
    )

    add_manual_peaks = 246
    peaks = np.sort(np.append(peaks,add_manual_peaks))
    

    p_scale, q_scale = _relative_prominence_weights(s_smooth, peaks, troughs,
                                                window=int(cfg["extrema_window"]),
                                                min_prominence=float(cfg["min_prominence"]))
    


    n = len(s)
    accum = np.zeros(n, dtype=float)

    # === Component assembly ===

    # 2. Global extreme boost (dominates)
    if cfg.get("weight_global_extreme_plateau", True):
        w_gep = _w_global_extreme_plateau(
            s_smooth,
            window=int(cfg.get("gep_window", 10)),
            tau=float(cfg.get("gep_tau", 3)),
            level=float(cfg.get("gep_level", 1.0)),
            tol_frac=float(cfg.get("gep_tol_frac", 1e-6)),
        )
        accum += float(cfg.get("coef_global_extreme_plateau", 5.0)) * w_gep

    # 3. Tail softmax (distributional)
    if cfg.get("weight_tail_softmax", True):
        w_tail = _w_tail_softmax(s_norm, k=float(cfg.get("tail_k", 8.0)), q=float(cfg.get("tail_q", 0.98)))
        accum += cfg.get("coef_tail_softmax", 1.0) * w_tail

    if cfg["weight_extremes"]:
        w_ext = _w_extremum_proximity_scaled(s.index, peaks, troughs, p_scale, q_scale, tau=float(cfg["tau_decay"]))
        accum += cfg["coef_extremes"] * w_ext

    if cfg["weight_rarity"]:
        w_rare = _w_rarity_state_space(s_smooth, bins=int(cfg["rarity_bins"]))
        accum += cfg["coef_rarity"] * w_rare

    if cfg["weight_curvature"]:
        w_curve = _w_curvature(d2s)
        accum += cfg["coef_curvature"] * w_curve

    if cfg["weight_ramp"]:
        w_r = _w_ramp(ds)
        accum += cfg["coef_ramp"] * w_r

    if cfg["weight_endpoints"] and cfg["include_start_end"]:
        w_end = _w_endpoints(
            n,
            include_start=True,
            include_end=True,
            weight_start=float(cfg["endpoint_weight_start"]),
            weight_end=float(cfg["endpoint_weight_end"]),
        )
        accum += cfg["coef_endpoints"] * w_end

    if cfg["weight_span"]:
        w_span = _w_span_distance(s_smooth, alpha=float(cfg["span_alpha"]))
        accum += cfg["coef_span"] * w_span

    # Avoid all-zero vector
    if not np.isfinite(accum).all() or np.all(accum <= 0):
        accum = np.ones(n, dtype=float)

    # Normalize to sum to N
    total = accum.sum()
    if total <= 0 or not np.isfinite(total):
        weights = np.ones(n, dtype=float)
    else:
        weights = (n * accum) / total

    def _rescale_biases(b: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Rescale b to [lo, hi]. If constant or invalid, return ones * hi."""
        b = np.asarray(b, dtype=float).reshape(-1)
        bmin, bmax = float(np.min(b)), float(np.max(b))
        if bmax <= bmin + 1e-12:
            return np.ones(len(b), dtype=float) * hi
        return lo + (b - bmin) * (hi - lo) / (bmax - bmin)
    
    weights = _rescale_biases(weights, 0.0, 1)

    # Optional plot
    if cfg["plot"]:
        # Scale weights to overlay on proxy nicely
        span = float(s_plot.max() - s_plot.min())
        s_plot = s_plot / s_plot.max()
        span = span if span > 0 else 1.0
        w_vis = (weights / max(weights.max(), 1e-12)) * (cfg["plot_scale"])
        baseline = float(s_plot.min())
        s_smooth = s_smooth / s_smooth.max()
        
        plt.figure(figsize=(10, 4))
        plt.plot(s_plot.index, s_plot.values, label="SoC proxy", color='black')
        plt.plot(s_smooth.index, s_smooth.values,label='Smoothed Signal', color='blue')
        plt.scatter(s_plot.index, w_vis, s=6, label="Relative weights", color='red')
        plt.title("Reference proxy with relative endogenous weights")
        plt.xlabel("Time")
        plt.ylabel("Proxy / Relative weight (scaled)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return weights


__all__ = ["generate_endogenous_biases"]
