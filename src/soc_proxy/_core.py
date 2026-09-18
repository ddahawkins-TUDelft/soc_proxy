"""Numerical primitives used by the SoC Proxy.

This module is intentionally private. It contains the array-level operations
shared by the public proxy generator and the automatic margin selector so the
selector can sweep candidate margins without rebuilding pandas objects or
rerunning the feasibility tuner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq
from scipy.signal import convolve


@dataclass(frozen=True)
class DecompositionPlan:
    """Precomputed filter information for repeated surplus decomposition."""

    method: str
    n: int
    timestep_hours: float
    time_horizon_hours: float
    fft_mask: np.ndarray | None = None
    kernel: np.ndarray | None = None


@dataclass(frozen=True)
class ProxyArrays:
    """Array-level result for one candidate margin."""

    generation: np.ndarray
    surplus: np.ndarray
    surplus_ldes: np.ndarray
    surplus_sdes: np.ndarray
    soc_ldes: np.ndarray
    soc_sdes: np.ndarray
    lost_load: float


def compute_soc_proxy(surplus: np.ndarray) -> np.ndarray:
    """Integrate a storage delta signal into an arbitrary-level SoC proxy."""
    return np.cumsum(np.asarray(surplus, dtype=np.float64))


def allocate_circular_lifo(
    pos: np.ndarray,
    neg: np.ndarray,
    eta_ch: float,
    eta_dis: float,
) -> tuple[np.ndarray, float]:
    """Allocate surplus to later deficits with a circular LIFO stack in O(n).

    Parameters
    ----------
    pos, neg
        Non-negative raw surplus and deficit arrays before storage losses.
    eta_ch, eta_dis
        Charging and discharging efficiencies in ``(0, 1]``.

    Returns
    -------
    np.ndarray
        Storage delta signal in stored-energy units. Positive values charge
        the store and negative values discharge it.
    float
        Unserved deficit in stored-energy units.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)

    if pos.shape != neg.shape:
        raise ValueError("pos and neg must have identical shapes.")

    pos_eff = pos * float(eta_ch)
    neg_eff = neg / float(eta_dis)
    net = pos_eff - neg_eff

    n = pos.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64), 0.0

    prefix = np.cumsum(net)
    start = (int(np.argmin(prefix)) + 1) % n

    pos_r = np.roll(pos_eff, -start)
    neg_r = np.roll(neg_eff, -start)

    stack_idx: list[int] = []
    stack_rem: list[float] = []

    charge_r = np.zeros(n, dtype=np.float64)
    discharge_r = np.zeros(n, dtype=np.float64)
    lost_r = np.zeros(n, dtype=np.float64)

    for i in range(n):
        surplus = float(pos_r[i])
        deficit = float(neg_r[i])

        if surplus > 1e-12:
            stack_idx.append(i)
            stack_rem.append(surplus)

        if deficit <= 1e-12:
            continue

        discharge_r[i] = -deficit
        remaining = deficit

        while remaining > 1e-12 and stack_idx:
            j = stack_idx[-1]
            available = stack_rem[-1]
            take = min(available, remaining)

            charge_r[j] += take
            available -= take
            remaining -= take

            if available <= 1e-12:
                stack_idx.pop()
                stack_rem.pop()
            else:
                stack_rem[-1] = available

        if remaining > 1e-9:
            discharge_r[i] = -(deficit - remaining)
            lost_r[i] = remaining

    charge = np.roll(charge_r, start)
    discharge = np.roll(discharge_r, start)
    return charge + discharge, float(lost_r.sum())


def evaluate_lost_load(
    curtailment_factor: float,
    base_gen: np.ndarray,
    demand: np.ndarray,
    eta_ch: float,
    eta_dis: float,
) -> float:
    """Evaluate unmet demand for one renewable overbuild factor."""
    generation = base_gen / float(curtailment_factor)
    residual = generation - demand
    positive = np.maximum(residual, 0.0)
    negative = np.maximum(-residual, 0.0)
    _, lost_sum = allocate_circular_lifo(positive, negative, eta_ch, eta_dis)
    return lost_sum


def find_min_feasible_curtailment(
    base_gen: np.ndarray,
    demand: np.ndarray,
    eta_ch: float,
    eta_dis: float,
    *,
    tol: float = 1e-12,
    c_init: float = 0.80,
    c_min: float = 0.65,
    c_max: float = 1.0,
    max_evals: int = 60,
) -> tuple[float, float, int]:
    """Find the least-overbuilt feasible renewable scaling by bisection.

    Smaller ``c`` means more renewable generation because generation is
    evaluated as ``base_gen / c``. The returned value is the largest feasible
    ``c`` within the search tolerance, i.e. the minimum renewable overbuild
    required to eliminate lost load.
    """
    eta_ch = float(eta_ch)
    eta_dis = float(eta_dis)

    c = float(np.clip(c_init, c_min, c_max))
    f = evaluate_lost_load(c, base_gen, demand, eta_ch, eta_dis)
    evals = 1

    if f <= tol:
        c_lo, f_lo = c, f
        c_hi = min(c * 1.111111, c_max)
        while True:
            f_hi = evaluate_lost_load(
                c_hi,
                base_gen,
                demand,
                eta_ch,
                eta_dis,
            )
            evals += 1
            if f_hi > tol or c_hi >= c_max or evals >= max_evals // 3:
                break
            c_lo, f_lo = c_hi, f_hi
            c_hi = min(c_hi * 1.111111, c_max)

        if f_hi <= tol:
            return c_hi, f_hi, evals
    else:
        c_hi, f_hi = c, f
        c_lo = max(c * 0.9, c_min)
        while True:
            f_lo = evaluate_lost_load(
                c_lo,
                base_gen,
                demand,
                eta_ch,
                eta_dis,
            )
            evals += 1
            if f_lo <= tol or c_lo <= c_min or evals >= max_evals // 3:
                break
            c_hi, f_hi = c_lo, f_lo
            c_lo = max(c_lo * 0.9, c_min)

        if f_lo > tol:
            return c_lo, f_lo, evals

    for _ in range(max_evals - evals):
        evals += 1
        c_mid = 0.5 * (c_lo + c_hi)
        f_mid = evaluate_lost_load(
            c_mid,
            base_gen,
            demand,
            eta_ch,
            eta_dis,
        )

        if f_mid <= tol:
            c_lo, f_lo = c_mid, f_mid
        else:
            c_hi, f_hi = c_mid, f_mid

        if abs(c_hi - c_lo) <= 1e-6 and f_lo <= tol:
            break

    return c_lo, f_lo, evals


def build_decomposition_plan(
    *,
    n: int,
    timestep_hours: float,
    method: str,
    time_horizon_hours: float,
) -> DecompositionPlan:
    """Precompute the filter used repeatedly during an automatic margin sweep."""
    supported = {"fft_lowpass", "gaussian", "triangular", "moving_average"}
    if method not in supported:
        raise ValueError(
            f"Unsupported decomposition method {method!r}. "
            f"Choose from {sorted(supported)}."
        )
    if n < 2:
        raise ValueError("At least two timesteps are required.")
    if timestep_hours <= 0:
        raise ValueError("timestep_hours must be positive.")
    if time_horizon_hours <= 0:
        raise ValueError("time_horizon_hours must be positive.")

    if method == "fft_lowpass":
        freqs = rfftfreq(n, d=timestep_hours)
        mask = (freqs <= (1.0 / time_horizon_hours)).astype(np.float64)
        return DecompositionPlan(
            method=method,
            n=n,
            timestep_hours=timestep_hours,
            time_horizon_hours=time_horizon_hours,
            fft_mask=mask,
        )

    samples = int(round(time_horizon_hours / timestep_hours)) | 1
    half_window = samples // 2

    if method == "moving_average":
        kernel = np.ones(samples, dtype=np.float64) / samples
    elif method == "triangular":
        w = np.arange(samples, dtype=np.float64)
        kernel = 1.0 - np.abs(w - half_window) / max(half_window, 1)
        kernel /= kernel.sum()
    else:  # gaussian
        sigma = samples / 6.0
        x = np.arange(samples, dtype=np.float64) - half_window
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()

    return DecompositionPlan(
        method=method,
        n=n,
        timestep_hours=timestep_hours,
        time_horizon_hours=time_horizon_hours,
        kernel=kernel,
    )


def decompose_surplus(
    surplus: np.ndarray,
    plan: DecompositionPlan,
) -> tuple[np.ndarray, np.ndarray]:
    """Split storage deltas into LDES and SDES components."""
    surplus = np.asarray(surplus, dtype=np.float64)
    if surplus.shape != (plan.n,):
        raise ValueError(
            f"Expected surplus shape {(plan.n,)}, got {surplus.shape}."
        )

    if plan.method == "fft_lowpass":
        assert plan.fft_mask is not None
        surplus_fft = rfft(surplus)
        surplus_ldes = irfft(surplus_fft * plan.fft_mask, n=plan.n)
    else:
        assert plan.kernel is not None
        surplus_ldes = convolve(surplus, plan.kernel, mode="same")

    surplus_sdes = surplus - surplus_ldes
    return surplus_ldes, surplus_sdes


def build_proxy_arrays(
    *,
    base_gen: np.ndarray,
    demand: np.ndarray,
    curtailment_factor: float,
    eta_ch: float,
    eta_dis: float,
    decomposition_plan: DecompositionPlan,
) -> ProxyArrays:
    """Construct all proxy arrays for one already-chosen scaling factor."""
    generation = np.asarray(base_gen, dtype=np.float64) / float(
        curtailment_factor
    )
    residual = generation - demand
    positive = np.maximum(residual, 0.0)
    negative = np.maximum(-residual, 0.0)

    surplus, lost_load = allocate_circular_lifo(
        positive,
        negative,
        eta_ch,
        eta_dis,
    )
    surplus_ldes, surplus_sdes = decompose_surplus(
        surplus,
        decomposition_plan,
    )

    return ProxyArrays(
        generation=generation,
        surplus=surplus,
        surplus_ldes=surplus_ldes,
        surplus_sdes=surplus_sdes,
        soc_ldes=compute_soc_proxy(surplus_ldes),
        soc_sdes=compute_soc_proxy(surplus_sdes),
        lost_load=lost_load,
    )
