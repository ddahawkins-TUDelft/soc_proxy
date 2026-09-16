import time
import numpy as np
import pandas as pd
from scipy.signal import convolve
from numpy.fft import rfft, irfft, rfftfreq


# ------------------------------------------------------------------------------
# Provenance notes (for write-up / citations)
# - Rotation start via minimum prefix sum (a.k.a. "gas station" / circular subarray feasibility trick).
# - LIFO stack allocator for historic borrowing (energy reserved last is used first).
# - 1-D monotone root finding (bracket + bisection) to tune curtailment_factor via lost-load.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Core SoC proxy helper
# ------------------------------------------------------------------------------
def compute_soc_proxy(surplus: np.ndarray, charging_eff: float, discharging_eff: float) -> np.ndarray:
    """
    Compute a simple SoC proxy as the cumulative sum of 'surplus'.
    The caller is responsible for providing the surplus time series they want integrated.
    """
    # surplus is already post-allocation (charge + discharge), so just integrate

    x = np.cumsum(surplus.astype(np.float64))

    return  x

def apply_temporal_rte_to_soc(
    df: pd.DataFrame,
    surplus_col: str,
    charging_eff: float,
    discharging_eff: float,
    return_corrected_surplus: bool = True,
    check_cyclical: bool = False
):
    """
    Wrapper to apply compute_soc_proxy to a DataFrame column and return Series.

    Returns:
        pd.Series: SoC proxy series (cumsum of surplus_col).
        (Optional) pd.Series: delta SoC (same as surplus_col, aligned as a Series).
    Notes:
        - The 'corrected_surplus' naming from earlier drafts caused confusion. Here we
          simply return delta SoC (the first difference of SoC), which equals 'surplus'.
    """
    soc_proxy = compute_soc_proxy(df[surplus_col].values.astype(np.float64), charging_eff, discharging_eff)
    soc_series = pd.Series(soc_proxy, index=df.index)

    if check_cyclical:
        residual = soc_series.iloc[-1] - soc_series.iloc[0]
        if abs(residual) > 1e4:
            print(f"[SoC Proxy] ⚠ Warning: SoC Proxy severely non-cyclical (end-start = {residual:.3e})")

    if return_corrected_surplus:
        # delta SoC equals the original surplus (by definition)
        delta_soc = soc_series.diff().fillna(0.0)
        return soc_series, delta_soc
    else:
        return soc_series


# ------------------------------------------------------------------------------
# Fast circular LIFO allocator (+ lost load)  --- O(n)
# ------------------------------------------------------------------------------
def allocate_circular_lifo(pos: np.ndarray, neg: np.ndarray, eta_ch: float, eta_dis: float):
    """
    O(n) circular LIFO allocator.

    Inputs:
        pos, neg : non-negative arrays of raw surplus/deficit per step (before efficiencies)
        eta_ch, eta_dis : charging/discharging efficiencies (0<eta<=1)

    Returns:
        surplus : np.ndarray = charge + discharge (charge>=0, discharge<=0) in original order
        lost_sum: float, total unmet deficit (in 'stored units' after applying efficiencies)

    Provenance:
        - Rotation start = argmin prefix sum of (pos*η_ch - neg/η_dis)  [gas-station/min-prefix trick]
        - LIFO via stack of (idx, remaining)                            [stack allocation]
    """
    pos = pos.astype(np.float64, copy=False)
    neg = neg.astype(np.float64, copy=False)

    pos_eff = pos * float(eta_ch)
    neg_eff = neg / float(eta_dis)
    net = pos_eff - neg_eff

    n = pos.shape[0]
    prefix = np.cumsum(net)
    start = (int(np.argmin(prefix)) + 1) % n

    pos_r = np.roll(pos_eff, -start)
    neg_r = np.roll(neg_eff, -start)

    stack_idx = []
    stack_rem = []

    charge_r    = np.zeros(n, dtype=np.float64)
    discharge_r = np.zeros(n, dtype=np.float64)
    lost_r      = np.zeros(n, dtype=np.float64)

    for i in range(n):
        s = pos_r[i]
        d = neg_r[i]

        # Push today's stored surplus, if any
        if s > 1e-12:
            stack_idx.append(i)
            stack_rem.append(s)

        # Cover today's deficit by popping LIFO
        if d > 1e-12:
            discharge_r[i] = -d
            rem = d
            while rem > 1e-12 and stack_idx:
                j = stack_idx[-1]
                avail = stack_rem[-1]
                take = avail if avail < rem else rem
                charge_r[j]  += take
                avail        -= take
                rem          -= take
                if avail <= 1e-12:
                    stack_idx.pop(); stack_rem.pop()
                else:
                    stack_rem[-1] = avail
            if rem > 1e-9:
                discharge_r[i] = -(d - rem)
                lost_r[i] = rem

    charge    = np.roll(charge_r, start)
    discharge = np.roll(discharge_r, start)
    surplus   = charge + discharge
    lost_sum  = float(lost_r.sum())
    return surplus, lost_sum


# ------------------------------------------------------------------------------
# Curtailment tuner (monotone bracket + bisection)
# ------------------------------------------------------------------------------
def evaluate_lost_load(curta_factor: float, base_gen: np.ndarray, demand: np.ndarray, eta_ch: float, eta_dis: float) -> float:
    """
    Given curtailment_factor, compute lost load using the fast allocator.

    base_gen: array independent of curtailment (mean_capacity_factor * weighted_installed_capacity)
    demand:   demand vector (after subtracting known dispatchable capacity)
    """
    g_curtailed = base_gen / curta_factor
    rnd = g_curtailed - demand
    pos = np.maximum(rnd, 0.0)
    neg = np.maximum(-rnd, 0.0)
    _, lost_sum = allocate_circular_lifo(pos, neg, eta_ch, eta_dis)
    return lost_sum

def compute_volatility_margin(base_gen: np.ndarray, demand: np.ndarray, k=5e-2, min_m=2e-2, max_m=7e-2):
    """
    Simple, model-free margin from variability.
    vol = std(base_gen - demand) / mean(demand)
    margin = clip(k * vol, [min_m, max_m])
    Defaults give ~1–8% depending on volatility.
    """
    rnd_base = base_gen - demand
    mu_d = float(np.maximum(np.mean(demand), 1e-9))
    vol = float(np.std(rnd_base) / mu_d)
    margin = float(np.clip(k * vol, min_m, max_m))
    return margin

def find_min_feasible_curtailment(
    base_gen: np.ndarray,
    demand: np.ndarray,
    eta_ch: float,
    eta_dis: float,
    tol: float = 1e-12,
    c_init: float = 0.80,
    c_min: float = 0.65,
    c_max: float = 1,
    max_evals: int = 60,
):
    """
    Return the smallest curtailment_factor in (c_min, c_max] with lost_load <= tol.
    Monotone in c: smaller c => more supply => lost_load non-increasing.

    Strategy:
        1) Evaluate at c_init; expand to get a bracket [c_lo(feasible), c_hi(infeasible)] or vice versa.
        2) Bisection inside the bracket until lost_load <= tol and interval is small.
    """
    eta_ch = float(eta_ch); eta_dis = float(eta_dis)

    c = float(np.clip(c_init, c_min, c_max))
    f = evaluate_lost_load(c, base_gen, demand, eta_ch, eta_dis)

    evals = 1

    # If feasible already, try to relax (increase c) until it just becomes infeasible
    if f <= tol:
        c_lo, f_lo = c, f
        c_hi = min(c * 1.111111, c_max)
        while True:
            f_hi = evaluate_lost_load(c_hi, base_gen, demand, eta_ch, eta_dis); evals += 1
            if f_hi > tol or c_hi >= c_max or evals >= max_evals//3:
                break
            c_lo, f_lo = c_hi, f_hi
            c_hi = min(c_hi * 1.111111, c_max)
        # If still feasible at upper bound, return last tested (most relaxed feasible)
        if f_hi <= tol:
            return c_hi, f_hi, evals
    else:
        # Infeasible: shrink c until feasible (or hit c_min)
        c_hi, f_hi = c, f
        c_lo = max(c * 0.9, c_min)
        while True:
            f_lo = evaluate_lost_load(c_lo, base_gen, demand, eta_ch, eta_dis); evals += 1
            if f_lo <= tol or c_lo <= c_min or evals >= max_evals//3:
                break
            c_hi, f_hi = c_lo, f_lo
            c_lo = max(c_lo * 0.9, c_min)
        if f_lo > tol:  # still infeasible at c_min
            return c_lo, f_lo, evals

    # We now have a bracket [c_lo (feasible), c_hi (infeasible)] — bisection to minimal feasible
    for _ in range(max_evals - evals):
        evals += 1
        c_mid = 0.5 * (c_lo + c_hi)
        f_mid = evaluate_lost_load(c_mid, base_gen, demand, eta_ch, eta_dis)
        if f_mid <= tol:
            c_lo, f_lo = c_mid, f_mid
        else:
            c_hi, f_hi = c_mid, f_mid
        if abs(c_hi - c_lo) <= 1e-6 and f_lo <= tol:
            break


    return c_lo, f_lo, evals


# ------------------------------------------------------------------------------
# Surplus decomposition (LDES/SDES) and SoC proxies
# ------------------------------------------------------------------------------
def decompose_surplus(
    df: pd.DataFrame,
    soc_decomposition_method: str = 'gaussian',
    time_horizon_hours: int = 24,
    timestamp_col: str = None,
    charging_efficiency: float = 1.0,
    discharging_efficiency: float = 1.0
) -> pd.DataFrame:
    """
    Decomposes 'surplus' into LDES and SDES components and computes SoC proxies.

    Adds to df:
        - surplus_LDES, surplus_SDES
        - soc_proxy_LDES, soc_proxy_SDES
    """
    if 'surplus' not in df.columns:
        raise ValueError("DataFrame must contain a 'surplus' column.")

    # Timestamps
    if timestamp_col:
        timestamps = pd.to_datetime(df[timestamp_col])
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Timestamp column not provided and index is not a DatetimeIndex.")
        timestamps = df.index.to_series()
    timestamps = timestamps.sort_values()
    if timestamps.nunique() < 2:
        raise ValueError("Not enough unique timestamps to compute time intervals.")

    timestep_hours = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds() / 3600.0
    if timestep_hours == 0:
        raise ValueError("Timestamps have zero time difference. Check the index or timestamp column.")

    surplus = df['surplus'].values.astype(np.float64)
    n = surplus.shape[0]
    samples = int(round(time_horizon_hours / timestep_hours)) | 1  # odd length

    # Smoothing to get LDES
    if soc_decomposition_method == 'fft_lowpass':
        freqs = rfftfreq(n, d=timestep_hours)
        surplus_fft = rfft(surplus)
        mask = (freqs <= (1.0 / time_horizon_hours)).astype(surplus_fft.real.dtype)
        surplus_LDES = irfft(surplus_fft * mask, n=n)
    else:
        half_window = samples // 2
        if soc_decomposition_method == 'moving_average':
            kernel = np.ones(samples, dtype=np.float64) / samples
        elif soc_decomposition_method == 'triangular':
            w = np.arange(samples, dtype=np.float64)
            kernel = 1.0 - np.abs(w - half_window) / max(half_window, 1)
            kernel /= kernel.sum()
        elif soc_decomposition_method == 'gaussian':
            sigma = samples / 6.0  # ~99% within ±3σ
            x = np.arange(samples) - half_window
            kernel = np.exp(-0.5 * (x / sigma) ** 2)
            kernel /= kernel.sum()
        else:
            raise ValueError(f"Unsupported method: {soc_decomposition_method}")
        surplus_LDES = convolve(surplus, kernel, mode='same')

    surplus_SDES = surplus - surplus_LDES

    df['surplus_LDES'] = surplus_LDES
    df['surplus_SDES'] = surplus_SDES

    # Compute SoC proxies (and return deltas for convenience, though we only keep proxies)
    soc_LDES, _ = apply_temporal_rte_to_soc(
        df.assign(temp_surplus=surplus_LDES),
        'temp_surplus',
        charging_efficiency,
        discharging_efficiency,
        return_corrected_surplus=True,
        check_cyclical=True
    )
    soc_SDES, _ = apply_temporal_rte_to_soc(
        df.assign(temp_surplus=surplus_SDES),
        'temp_surplus',
        charging_efficiency,
        discharging_efficiency,
        return_corrected_surplus=True,
        check_cyclical=False
    )

    df['soc_proxy_LDES'] = soc_LDES.values
    df['soc_proxy_SDES'] = soc_SDES.values

    return df


# ------------------------------------------------------------------------------
# External API (unchanged signature)
# ------------------------------------------------------------------------------
def generate_soc_proxy(
    df: pd.DataFrame,
    demand_field: str = 'demand_power',
    renewables_fields_and_weights: dict = {'solar': 1, 'onshore_wind': 1, 'offshore_wind': 1},
    dispatchable_techs: dict = {'known_dispatchable_capacity': 3300},
    storage_process_losses: dict = {
        'charging_efficiency': 0.65 * 0.99,
        'discharging_efficiency': 0.56 * 0.99
    },
    soc_decomposition: dict = {
        'method': 'fft_lowpass',
        'time_horizon_hours': 24
    },
    timestamp_col: str = None,
    margin_mode: str = 'fixed',          # 'fixed' | 'auto_volatility' | 'none'
    margin_value: float = 0.04,          # only used for 'fixed'
    margin_bounds: tuple = (0.0, 0.5),  # clamp any margin we compute
):
    """
    Takes time series data and returns SoC proxies for LDES and SDES.

    Returns:
        df (pd.DataFrame): augmented with 'surplus', decomposition, and SoC proxies
        capacity_factors (dict): mean CFs per technology and weighted mean
        installed_caps_nominal (dict): nominal installed capacities per tech and total
    """
    t0 = time.time()

    supported_methods = {'fft_lowpass', 'gaussian', 'triangular', 'moving_average'}
    if soc_decomposition['method'] not in supported_methods:
        raise ValueError(f"Method '{soc_decomposition['method']}' not supported. Choose from: {supported_methods}")

    renewable_fields = list(renewables_fields_and_weights.keys())
    weights = np.array(list(renewables_fields_and_weights.values()), dtype=float)
    weights /= weights.sum()

    # Validate required fields
    required = [demand_field] + renewable_fields
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing fields in dataframe: {missing}")

    # Subtract known dispatchable baseline capacity from demand
    df = df.copy()  # avoid mutating caller's df
    df[demand_field] = df[demand_field].astype(np.float64) - float(dispatchable_techs.get('known_dispatchable_capacity', 0.0))

    # Capacity factors (unweighted) and weighted mean
    capacity_factors = {tech: float(df[tech].mean()) for tech in renewable_fields}
    capacity_factors['weighted_mean'] = float(sum(weights[i] * capacity_factors[tech] for i, tech in enumerate(renewable_fields)))

    # Blended mean CF over time
    df['mean_capacity_factor'] = df[renewable_fields].mul(weights, axis=1).sum(axis=1)

    # Total installed capacity to meet average demand on average
    weighted_installed_capacity = float(df[demand_field].sum() / df['mean_capacity_factor'].sum())

    # Build base generation (independent of curtailment) and demand vectors
    base_gen = df['mean_capacity_factor'].to_numpy(np.float64) * weighted_installed_capacity
    demand   = df[demand_field].to_numpy(np.float64)

    eta_ch = float(storage_process_losses['charging_efficiency'])
    eta_dis = float(storage_process_losses['discharging_efficiency'])

    # Tune curtailment_factor to (nearly) eliminate lost load
    c_star, lost_final, _ = find_min_feasible_curtailment(
        base_gen, demand, eta_ch, eta_dis,
        tol=1e-12,
    )

    # Decide margin
    lb, ub = margin_bounds
    if margin_mode == 'fixed':
        margin = float(np.clip(margin_value, lb, ub))
    elif margin_mode == 'auto_volatility':
        margin = float(np.clip(compute_volatility_margin(base_gen, demand), lb, ub))
    elif margin_mode == 'none':
        margin = 0.0
    else:
        raise ValueError(f"Unknown margin_mode: {margin_mode}")

    # Apply margin as extra curtailment headroom
    curtailment_factor = max(1e-3, c_star * (1.0 - margin))
    print(f"[SoC Proxy] Tuner: c*={c_star:.6f} lost={lost_final:.3e} | margin={margin:.3%} -> curtailment={curtailment_factor:.6f}")

    # Final allocation using tuned curtailment
    g_curtailed = base_gen / curtailment_factor
    rnd = g_curtailed - demand
    pos = np.maximum(rnd, 0.0)
    neg = np.maximum(-rnd, 0.0)

    df['surplus'], _ = allocate_circular_lifo(pos, neg, eta_ch, eta_dis)

    # Decomposition + SoC proxies
    df = decompose_surplus(
        df,
        soc_decomposition_method=soc_decomposition['method'],
        time_horizon_hours=soc_decomposition['time_horizon_hours'],
        timestamp_col=timestamp_col,
        charging_efficiency=eta_ch,
        discharging_efficiency=eta_dis
    )

    # Clean tiny numerical noise
    for col in ['soc_proxy_LDES', 'soc_proxy_SDES', 'surplus_LDES', 'surplus_SDES']:
        arr = df[col].to_numpy(np.float64)
        arr[np.abs(arr) < 1e-9] = 0.0
        df[col] = arr

    # Rescale SoC proxies indirectly by curtailment factor (preserve your earlier behavior)
    # df['soc_proxy_LDES'] *= curtailment_factor
    # df['soc_proxy_SDES'] *= curtailment_factor

    # Shift proxies so they begin from 0
    df['soc_proxy_LDES'] -= df['soc_proxy_LDES'].min()
    df['soc_proxy_SDES'] -= df['soc_proxy_SDES'].min()

    # Installed nominal capacities (based on weights)
    installed_caps_nominal = {tech: float(weights[i] * weighted_installed_capacity) for i, tech in enumerate(renewable_fields)}
    installed_caps_nominal['total'] = float(weighted_installed_capacity)

    # print(f"[timing] total = {time.time() - t0:.2f}s")
    print(f"[SoC Proxy] curtailment forecast: {curtailment_factor:.2%}")

    return df, capacity_factors, installed_caps_nominal
