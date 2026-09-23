"""Greedy proxy-guided chronological remapping (PGCR).

PGCR is an experimental post-TSA mapping step.

It assumes that clustering and representative-period selection have already
been completed. It then revisits the chronological assignment of those fixed
representatives using a fixed ex-ante SoC-proxy delta signal.

The central idea is:

1. Keep the representative-period library fixed.
2. Reconstruct the proxy level implied by assigning each representative.
3. Walk through the original chronology and greedily choose the representative
   that best tracks the fixed target proxy level.
4. Optionally use a short look-ahead based on the current map.
5. Repeat around the circular horizon so changes near the end can influence
   assignments near the beginning.

The method does *not* constrain representative occurrence counts. This is
intentional for the initial experiment: representative frequencies are an
output diagnostic rather than a hard constraint. However, when a representative
is an actual observed period (for example a medoid), that original period can be
locked to its own representative. This guarantees that every selected real
representative remains present at least once in the remapped chronology.

The target proxy is never recomputed from the remapped physical time series.
It therefore remains fixed ex-ante information. Optionally, an external exact
physical-proxy evaluator can be called after a cheap sweep (or a fraction of its
proposed changes) to decide whether that batch should actually be accepted.
The expensive evaluator is never called inside the O(n*k) candidate loop.

Complexity is approximately O(S * n * (k * h + L * h)), where:
    S = number of circular sweeps
    n = number of representative periods in the original chronology
    k = number of candidate representatives
    h = timesteps per representative period
    L = look-ahead periods

For fixed h and modest L, this behaves like O(S * n * k).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Hashable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ObjectArray = NDArray[np.object_]


@dataclass(frozen=True)
class ProxyRemapDiagnostics:
    """Error diagnostics for one chronological map."""

    delta_rmse: float
    delta_nrmse: float
    level_rmse: float
    level_nrmse: float
    level_max_abs_error: float
    level_max_abs_error_normalised: float
    target_level_range: float
    reconstructed_level_range: float
    level_range_error: float
    level_range_error_fraction: float
    target_cycle_change: float
    reconstructed_cycle_change: float
    closure_error: float
    closure_error_normalised: float


@dataclass(frozen=True)
class SweepRecord:
    """Summary of one circular greedy sweep.

    ``diagnostics`` always describe the inexpensive frozen-representative
    surrogate after the accepted changes. When an exact evaluator is supplied,
    ``exact_diagnostics`` describe the recomputed physical proxy used for
    acceptance.
    """

    sweep: int
    assignments_changed: int
    changed_fraction: float
    diagnostics: ProxyRemapDiagnostics
    proposal_assignments_changed: int = 0
    acceptance_fraction: float | None = None
    exact_diagnostics: ProxyRemapDiagnostics | None = None
    exact_evaluations: int = 0


@dataclass(frozen=True)
class ProxyRemapResult:
    """Result of proxy-guided chronological remapping."""

    cluster_map: ObjectArray
    initial_cluster_map: ObjectArray

    # Same maps after rotation to the selected circular starting point.
    rotated_cluster_map: ObjectArray
    rotated_initial_cluster_map: ObjectArray

    start_period: int
    representative_ids: tuple[Hashable, ...]
    representative_period_indices: dict[Hashable, int]
    locked_period_count: int

    target_delta: FloatArray
    target_level: FloatArray
    initial_reconstructed_delta: FloatArray
    initial_reconstructed_level: FloatArray
    reconstructed_delta: FloatArray
    reconstructed_level: FloatArray

    initial_diagnostics: ProxyRemapDiagnostics
    diagnostics: ProxyRemapDiagnostics

    initial_usage: dict[Hashable, int]
    remapped_usage: dict[Hashable, int]
    changed_assignments: int
    changed_fraction: float

    sweep_history: tuple[SweepRecord, ...]

    # Optional exact-feedback diagnostics. These are populated only when
    # ``exact_delta_evaluator`` is supplied to the greedy remapper.
    exact_initial_reconstructed_delta: FloatArray | None = None
    exact_initial_reconstructed_level: FloatArray | None = None
    exact_reconstructed_delta: FloatArray | None = None
    exact_reconstructed_level: FloatArray | None = None
    exact_initial_diagnostics: ProxyRemapDiagnostics | None = None
    exact_diagnostics: ProxyRemapDiagnostics | None = None
    exact_evaluations: int = 0


def _as_period_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    """Return a finite 2D float matrix shaped [period, timestep]."""

    array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        raise ValueError(
            f"{name} must be a 2D array shaped [period, timestep]; "
            f"got shape {array.shape}."
        )

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 2D; got {array.ndim} dimensions."
        )

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")

    return array


def _normalise_representatives(
    representative_delta: Mapping[Hashable, ArrayLike],
    *,
    timesteps_per_period: int,
) -> tuple[tuple[Hashable, ...], FloatArray, dict[Hashable, int]]:
    """Build a representative delta matrix and ID-to-row lookup."""

    if not representative_delta:
        raise ValueError("representative_delta must contain at least one representative.")

    representative_ids = tuple(representative_delta.keys())

    rows: list[FloatArray] = []
    for representative_id in representative_ids:
        row = np.asarray(
            representative_delta[representative_id],
            dtype=float,
        )

        if row.ndim != 1:
            raise ValueError(
                "Each representative delta profile must be one-dimensional; "
                f"representative {representative_id!r} has shape {row.shape}."
            )

        if len(row) != timesteps_per_period:
            raise ValueError(
                f"Representative {representative_id!r} has {len(row)} "
                f"timesteps; expected {timesteps_per_period}."
            )

        if not np.all(np.isfinite(row)):
            raise ValueError(
                f"Representative {representative_id!r} contains non-finite values."
            )

        rows.append(row)

    matrix = np.vstack(rows)
    lookup = {
        representative_id: index
        for index, representative_id in enumerate(representative_ids)
    }

    return representative_ids, matrix, lookup


def _validate_cluster_map(
    cluster_map: Sequence[Hashable],
    *,
    n_periods: int,
    representative_lookup: Mapping[Hashable, int],
) -> ObjectArray:
    """Validate a chronological representative-assignment map."""

    mapping = np.asarray(cluster_map, dtype=object)

    if mapping.ndim != 1:
        raise ValueError(
            f"initial_cluster_map must be one-dimensional; got shape {mapping.shape}."
        )

    if len(mapping) != n_periods:
        raise ValueError(
            "initial_cluster_map length must equal the number of target periods: "
            f"{len(mapping)} != {n_periods}."
        )

    unknown = {
        value
        for value in mapping.tolist()
        if value not in representative_lookup
    }
    if unknown:
        raise ValueError(
            "initial_cluster_map refers to representatives missing from "
            f"representative_delta: {sorted(unknown, key=str)}"
        )

    return mapping


def _map_to_indices(
    cluster_map: Sequence[Hashable],
    representative_lookup: Mapping[Hashable, int],
) -> NDArray[np.int64]:
    return np.fromiter(
        (representative_lookup[item] for item in cluster_map),
        dtype=np.int64,
        count=len(cluster_map),
    )


def reconstruct_proxy_delta(
    cluster_map: Sequence[Hashable],
    representative_delta: Mapping[Hashable, ArrayLike],
) -> FloatArray:
    """Reconstruct a flattened proxy-delta series from a cluster map."""

    mapping = np.asarray(cluster_map, dtype=object)

    if mapping.ndim != 1:
        raise ValueError("cluster_map must be one-dimensional.")

    if len(mapping) == 0:
        return np.empty(0, dtype=float)

    representative_ids = tuple(representative_delta.keys())
    if not representative_ids:
        raise ValueError("representative_delta must not be empty.")

    first = np.asarray(
        representative_delta[representative_ids[0]],
        dtype=float,
    )
    if first.ndim != 1:
        raise ValueError("Representative delta profiles must be one-dimensional.")

    _, matrix, lookup = _normalise_representatives(
        representative_delta,
        timesteps_per_period=len(first),
    )

    indices = _map_to_indices(mapping, lookup)
    return matrix[indices].reshape(-1)


def _levels_after_deltas(
    delta: ArrayLike,
    *,
    initial_level: float = 0.0,
) -> FloatArray:
    """Return level after every delta timestep."""

    delta_array = np.asarray(delta, dtype=float).reshape(-1)
    return initial_level + np.cumsum(delta_array)


def _safe_normalise(value: float, scale: float) -> float:
    """Normalise a value, returning NaN when the scale is effectively zero."""

    if np.isclose(scale, 0.0):
        return float("nan")
    return float(value / scale)


def proxy_error_metrics(
    target_delta: ArrayLike,
    reconstructed_delta: ArrayLike,
    *,
    initial_level: float = 0.0,
) -> ProxyRemapDiagnostics:
    """Compare reconstructed and fixed-target proxy deltas and levels.

    nRMSE quantities use the full range of the corresponding target signal.
    The level comparison retains the chosen common initial level, so cumulative
    drift and cycle-closure errors remain visible.
    """

    target = np.asarray(target_delta, dtype=float).reshape(-1)
    reconstructed = np.asarray(reconstructed_delta, dtype=float).reshape(-1)

    if target.shape != reconstructed.shape:
        raise ValueError(
            "target_delta and reconstructed_delta must have identical shapes: "
            f"{target.shape} != {reconstructed.shape}."
        )

    target_level = _levels_after_deltas(target, initial_level=initial_level)
    reconstructed_level = _levels_after_deltas(
        reconstructed,
        initial_level=initial_level,
    )

    delta_error = reconstructed - target
    level_error = reconstructed_level - target_level

    delta_rmse = float(np.sqrt(np.mean(delta_error**2)))
    level_rmse = float(np.sqrt(np.mean(level_error**2)))
    level_max_abs_error = float(np.max(np.abs(level_error)))

    target_delta_range = float(np.ptp(target))
    target_level_range = float(np.ptp(target_level))
    reconstructed_level_range = float(np.ptp(reconstructed_level))

    level_range_error = reconstructed_level_range - target_level_range

    target_cycle_change = float(np.sum(target))
    reconstructed_cycle_change = float(np.sum(reconstructed))
    closure_error = reconstructed_cycle_change - target_cycle_change

    return ProxyRemapDiagnostics(
        delta_rmse=delta_rmse,
        delta_nrmse=_safe_normalise(delta_rmse, target_delta_range),
        level_rmse=level_rmse,
        level_nrmse=_safe_normalise(level_rmse, target_level_range),
        level_max_abs_error=level_max_abs_error,
        level_max_abs_error_normalised=_safe_normalise(
            level_max_abs_error,
            target_level_range,
        ),
        target_level_range=target_level_range,
        reconstructed_level_range=reconstructed_level_range,
        level_range_error=level_range_error,
        level_range_error_fraction=_safe_normalise(
            level_range_error,
            target_level_range,
        ),
        target_cycle_change=target_cycle_change,
        reconstructed_cycle_change=reconstructed_cycle_change,
        closure_error=closure_error,
        closure_error_normalised=_safe_normalise(
            closure_error,
            target_level_range,
        ),
    )


def _period_start_levels(target_delta: FloatArray) -> FloatArray:
    """Return target level at the start of every period."""

    n_periods, timesteps_per_period = target_delta.shape
    flat = target_delta.reshape(-1)

    level_boundaries = np.empty(len(flat) + 1, dtype=float)
    level_boundaries[0] = 0.0
    np.cumsum(flat, out=level_boundaries[1:])

    return level_boundaries[
        np.arange(n_periods, dtype=int) * timesteps_per_period
    ]


def _validate_representative_period_indices(
    representative_period_indices: Mapping[Hashable, int] | None,
    *,
    initial_cluster_map: ObjectArray,
    n_periods: int,
    representative_lookup: Mapping[Hashable, int],
) -> dict[Hashable, int]:
    """Validate real representative periods that must remain self-assigned.

    Parameters
    ----------
    representative_period_indices
        Mapping ``representative_id -> original_period_index``. This should be
        supplied only for representatives that correspond to real observed
        periods, such as medoids/maxoids. Synthetic representatives do not have
        an original period to lock and should therefore be omitted.

    Notes
    -----
    The initial TSA map is expected already to assign every supplied
    representative period to its own representative. A mismatch is treated as
    an integration error rather than silently rewriting the baseline map.
    """

    if representative_period_indices is None:
        return {}

    normalised: dict[Hashable, int] = {}
    used_periods: dict[int, Hashable] = {}

    for representative_id, raw_period_index in representative_period_indices.items():
        if representative_id not in representative_lookup:
            raise ValueError(
                "representative_period_indices contains an unknown "
                f"representative ID: {representative_id!r}."
            )

        if not isinstance(raw_period_index, (int, np.integer)):
            raise TypeError(
                "Representative period indices must be integers; "
                f"{representative_id!r} has {raw_period_index!r}."
            )

        period_index = int(raw_period_index)
        if not 0 <= period_index < n_periods:
            raise ValueError(
                f"Representative {representative_id!r} refers to original "
                f"period {period_index}, outside [0, {n_periods - 1}]."
            )

        if period_index in used_periods:
            raise ValueError(
                "Two representatives cannot be locked to the same original "
                f"period {period_index}: {used_periods[period_index]!r} and "
                f"{representative_id!r}."
            )

        assigned = initial_cluster_map[period_index]
        if assigned != representative_id:
            raise ValueError(
                "A real representative period must initially be assigned to "
                "itself. "
                f"Period {period_index} is declared as representative "
                f"{representative_id!r}, but the initial TSA map assigns it "
                f"to {assigned!r}."
            )

        normalised[representative_id] = period_index
        used_periods[period_index] = representative_id

    return normalised


def _rotate_locked_periods(
    representative_period_indices: Mapping[Hashable, int],
    *,
    start: int,
    n_periods: int,
) -> dict[int, Hashable]:
    """Return ``rotated_period_index -> representative_id`` locks."""

    return {
        (period_index - start) % n_periods: representative_id
        for representative_id, period_index in representative_period_indices.items()
    }


def _choose_start_period(
    target_delta: FloatArray,
    start_period: int | str,
) -> int:
    """Resolve circular starting point."""

    n_periods = target_delta.shape[0]

    if start_period == "minimum":
        starts = _period_start_levels(target_delta)
        return int(np.argmin(starts))

    if isinstance(start_period, (int, np.integer)):
        start = int(start_period)
        if not 0 <= start < n_periods:
            raise ValueError(
                f"start_period must be in [0, {n_periods - 1}]; got {start}."
            )
        return start

    raise ValueError(
        "start_period must be an integer or 'minimum'; "
        f"got {start_period!r}."
    )


def _rotate_period_matrix(values: FloatArray, start: int) -> FloatArray:
    if start == 0:
        return values.copy()
    return np.concatenate((values[start:], values[:start]), axis=0)


def _rotate_map(values: ObjectArray, start: int) -> ObjectArray:
    if start == 0:
        return values.copy()
    return np.concatenate((values[start:], values[:start]))


def _unrotate_map(values: ObjectArray, start: int) -> ObjectArray:
    if start == 0:
        return values.copy()
    return np.roll(values, start)


def _usage_counts(
    cluster_map: Sequence[Hashable],
    representative_ids: Sequence[Hashable],
) -> dict[Hashable, int]:
    mapping = np.asarray(cluster_map, dtype=object)
    return {
        representative_id: int(np.count_nonzero(mapping == representative_id))
        for representative_id in representative_ids
    }


def _target_period_levels(target_delta: FloatArray) -> FloatArray:
    """Return target levels after each timestep, with cycle start at zero."""

    return np.cumsum(target_delta.reshape(-1)).reshape(target_delta.shape)


def _target_window(
    target_levels: FloatArray,
    *,
    start_period: int,
    periods: int,
    cycle_number: int,
    cycle_change: float,
) -> FloatArray:
    """Return target levels for a possibly wrapped multi-period window."""

    if periods <= 0:
        return np.empty(0, dtype=float)

    n_periods = target_levels.shape[0]
    blocks: list[FloatArray] = []

    for offset in range(periods):
        absolute_period = start_period + offset
        wrap = absolute_period // n_periods
        period_index = absolute_period % n_periods

        blocks.append(
            target_levels[period_index]
            + (cycle_number + wrap) * cycle_change
        )

    return np.concatenate(blocks)


def _future_relative_levels(
    map_indices: NDArray[np.int64],
    representative_matrix: FloatArray,
    *,
    start_period: int,
    periods: int,
) -> FloatArray:
    """Return future levels relative to zero under the current working map."""

    if periods <= 0:
        return np.empty(0, dtype=float)

    n_periods = len(map_indices)
    indices = np.fromiter(
        (
            map_indices[(start_period + offset) % n_periods]
            for offset in range(periods)
        ),
        dtype=np.int64,
        count=periods,
    )

    future_delta = representative_matrix[indices].reshape(-1)
    return np.cumsum(future_delta)


def _objective_value(diagnostics: ProxyRemapDiagnostics) -> float:
    """Return the scalar objective used to compare maps."""

    value = diagnostics.level_nrmse
    if np.isnan(value):
        value = diagnostics.level_rmse
    return float(value)


def _rotate_flat_by_period(
    values: ArrayLike,
    *,
    start_period: int,
    n_periods: int,
    timesteps_per_period: int,
    name: str,
) -> FloatArray:
    """Validate and rotate a full-resolution signal by whole periods."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        if array.shape != (n_periods, timesteps_per_period):
            raise ValueError(
                f"{name} has shape {array.shape}; expected "
                f"{(n_periods, timesteps_per_period)}."
            )
        matrix = array
    elif array.ndim == 1:
        expected = n_periods * timesteps_per_period
        if len(array) != expected:
            raise ValueError(
                f"{name} contains {len(array)} values; expected {expected}."
            )
        matrix = array.reshape(n_periods, timesteps_per_period)
    else:
        raise ValueError(
            f"{name} must be one- or two-dimensional; got shape {array.shape}."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")

    return _rotate_period_matrix(matrix, start_period).reshape(-1)


def _normalise_acceptance_fractions(
    fractions: Sequence[float],
) -> tuple[float, ...]:
    """Validate exact-feedback batch fractions and return unique descending values."""

    if not fractions:
        raise ValueError("exact_acceptance_fractions must not be empty.")

    values: list[float] = []
    for raw in fractions:
        value = float(raw)
        if not 0.0 < value <= 1.0:
            raise ValueError(
                "exact_acceptance_fractions must contain values in (0, 1]; "
                f"got {value}."
            )
        values.append(value)

    return tuple(sorted(set(values), reverse=True))


def _candidate_scores(
    *,
    current_level: float,
    period_index: int,
    sweep: int,
    target_levels: FloatArray,
    target_cycle_change: float,
    representative_cumsum: FloatArray,
    representative_totals: FloatArray,
    working_map_indices: NDArray[np.int64],
    representative_matrix: FloatArray,
    lookahead_periods: int,
) -> FloatArray:
    """Score all representative candidates for one period.

    Future assignments are held at their current working-map values. Because a
    candidate changes only today's trajectory and the level entering the
    look-ahead window, the future SSE can be evaluated analytically for every
    candidate using its endpoint level. This avoids rebuilding the complete
    look-ahead series k times.
    """

    target_today = _target_window(
        target_levels,
        start_period=period_index,
        periods=1,
        cycle_number=sweep,
        cycle_change=target_cycle_change,
    )

    predicted_today = current_level + representative_cumsum
    today_error = predicted_today - target_today[None, :]
    sse_today = np.sum(today_error**2, axis=1)

    if lookahead_periods <= 0:
        return sse_today / representative_matrix.shape[1]

    future_relative = _future_relative_levels(
        working_map_indices,
        representative_matrix,
        start_period=period_index + 1,
        periods=lookahead_periods,
    )

    target_future = _target_window(
        target_levels,
        start_period=period_index + 1,
        periods=lookahead_periods,
        cycle_number=sweep,
        cycle_change=target_cycle_change,
    )

    future_base_error = current_level + future_relative - target_future
    m = len(future_base_error)

    future_sum = float(np.sum(future_base_error))
    future_sq_sum = float(np.sum(future_base_error**2))

    sse_future = (
        future_sq_sum
        + 2.0 * representative_totals * future_sum
        + m * representative_totals**2
    )

    n_points = representative_matrix.shape[1] + m
    return (sse_today + sse_future) / n_points


def greedy_proxy_chronology_remap(
    target_delta: ArrayLike,
    representative_delta: Mapping[Hashable, ArrayLike],
    initial_cluster_map: Sequence[Hashable],
    *,
    representative_period_indices: Mapping[Hashable, int] | None = None,
    start_period: int | str = "minimum",
    lookahead_periods: int = 7,
    max_sweeps: int = 3,
    tie_tolerance: float = 1e-12,
    stop_changed_fraction: float = 0.0,
    exact_delta_evaluator: Callable[[ObjectArray], ArrayLike] | None = None,
    exact_acceptance_fractions: Sequence[float] = (1.0, 0.5, 0.25, 0.125),
    exact_improvement_tolerance: float = 0.0,
) -> ProxyRemapResult:
    """Greedily remap fixed TSA representatives to preserve proxy chronology.

    Parameters
    ----------
    target_delta
        Fixed ex-ante proxy delta values, shaped
        ``[n_periods, timesteps_per_period]``. For daily TSA with hourly data,
        this will normally be ``[n_days, 24]``.
    representative_delta
        Mapping from representative/cluster ID to its inexpensive frozen
        proxy-delta profile for one representative period. These profiles drive
        the O(nk) proposal sweep; they do not need to be the final acceptance
        objective when ``exact_delta_evaluator`` is supplied.
    initial_cluster_map
        Existing TSA chronological map. It provides both the initial solution
        and the future assignments used during short look-ahead scoring.
    representative_period_indices
        Optional mapping ``representative_id -> original_period_index`` for
        representatives that are real observed periods (e.g. medoids). These
        periods are permanently locked to their own representative during every
        greedy sweep.
    start_period
        ``"minimum"`` starts at the period boundary with the lowest target
        cumulative proxy level. An integer can be supplied to force another
        circular starting period.
    lookahead_periods
        Number of *future* representative periods included when evaluating
        today's surrogate candidate. Future assignments are taken from the
        current working map. Set to 0 for pure one-period greedy selection.
    max_sweeps
        Maximum number of complete proposal/acceptance passes.
    tie_tolerance
        If the existing assignment is within this absolute surrogate-score
        tolerance of the best candidate, retain it.
    stop_changed_fraction
        Stop after an accepted sweep if the fraction of assignments changed in
        that sweep is less than or equal to this value.
    exact_delta_evaluator
        Optional callback for exact physical feedback. It receives an
        **unrotated copy** of a candidate cluster map and must return the
        full-resolution proxy-delta signal implied by that physical chronology,
        either flattened or shaped ``[n_periods, timesteps_per_period]``.

        When supplied, a cheap greedy sweep becomes only a proposal generator.
        Proposed changes are ranked by their local surrogate improvement, then
        batches containing ``exact_acceptance_fractions`` of the proposed
        changes are evaluated with this callback. The best batch is accepted
        only if its exact proxy-level objective improves on the current map.
        This keeps expensive proxy regeneration outside the O(n*k) candidate
        loop.
    exact_acceptance_fractions
        Fractions of the proposed changes to test with the exact evaluator.
        Values must lie in ``(0, 1]``. The default tests 100%, 50%, 25% and
        12.5% of the highest-surrogate-gain changes.
    exact_improvement_tolerance
        Minimum absolute reduction in the exact level objective required before
        accepting a candidate batch. The objective is level nRMSE when defined,
        otherwise level RMSE.

    Returns
    -------
    ProxyRemapResult
        The accepted map plus surrogate and, when requested, exact physical
        proxy diagnostics.

    Notes
    -----
    Without ``exact_delta_evaluator`` this preserves the original PGCR v0
    behaviour: every greedy sweep is accepted and the best surrogate map is
    retained.

    With ``exact_delta_evaluator`` the immutable target is still the original
    ex-ante proxy. The callback should therefore hold fixed any system-level
    parameters that should not change merely because chronology was remapped
    (for the SoC proxy experiment this means using the original selected system
    scaling / curtailment factor and decomposition assumptions). The callback
    itself is deliberately external so this lightweight module remains
    independent of TSAM and ``soc_proxy``.
    """

    if lookahead_periods < 0:
        raise ValueError("lookahead_periods must be >= 0.")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be >= 1.")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be >= 0.")
    if not 0.0 <= stop_changed_fraction <= 1.0:
        raise ValueError("stop_changed_fraction must lie in [0, 1].")
    if exact_improvement_tolerance < 0:
        raise ValueError("exact_improvement_tolerance must be >= 0.")

    print("[PGCR] Running Proxy-guided Chronological Remapping.", flush=True)

    target = _as_period_matrix(target_delta, name="target_delta")
    n_periods, timesteps_per_period = target.shape

    (
        representative_ids,
        representative_matrix,
        representative_lookup,
    ) = _normalise_representatives(
        representative_delta,
        timesteps_per_period=timesteps_per_period,
    )

    initial_map = _validate_cluster_map(
        initial_cluster_map,
        n_periods=n_periods,
        representative_lookup=representative_lookup,
    )

    fixed_representative_periods = _validate_representative_period_indices(
        representative_period_indices,
        initial_cluster_map=initial_map,
        n_periods=n_periods,
        representative_lookup=representative_lookup,
    )

    selected_start = _choose_start_period(target, start_period)
    rotated_target = _rotate_period_matrix(target, selected_start)
    rotated_initial_map = _rotate_map(initial_map, selected_start)

    representative_cumsum = np.cumsum(representative_matrix, axis=1)
    representative_totals = representative_cumsum[:, -1]

    target_flat = rotated_target.reshape(-1)
    target_levels = _target_period_levels(rotated_target)
    target_cycle_change = float(np.sum(target_flat))

    locked_periods = _rotate_locked_periods(
        fixed_representative_periods,
        start=selected_start,
        n_periods=n_periods,
    )

    initial_indices = _map_to_indices(
        rotated_initial_map,
        representative_lookup,
    )
    initial_reconstructed_delta = representative_matrix[initial_indices].reshape(-1)
    initial_diagnostics = proxy_error_metrics(
        target_flat,
        initial_reconstructed_delta,
    )

    history: list[SweepRecord] = []

    # ------------------------------------------------------------------
    # Original surrogate-only PGCR path. Keep this separate so adding exact
    # feedback does not subtly change existing experiments.
    # ------------------------------------------------------------------
    if exact_delta_evaluator is None:
        working_map = rotated_initial_map.copy()
        working_indices = initial_indices.copy()

        best_map = working_map.copy()
        best_indices = working_indices.copy()
        best_diagnostics = initial_diagnostics

        current_level = 0.0

        for sweep in range(max_sweeps):
            sweep_start = perf_counter()
            print(f"[PGCR] Commencing sweep: {sweep + 1}.", flush=True)
            changed = 0

            for period_index in range(n_periods):
                current_rep_index = int(working_indices[period_index])
                locked_representative = locked_periods.get(period_index)

                if locked_representative is not None:
                    chosen_rep_index = representative_lookup[locked_representative]
                    if current_rep_index != chosen_rep_index:
                        raise RuntimeError(
                            "A locked representative period changed unexpectedly. "
                            "This indicates an internal PGCR state inconsistency."
                        )
                else:
                    scores = _candidate_scores(
                        current_level=current_level,
                        period_index=period_index,
                        sweep=sweep,
                        target_levels=target_levels,
                        target_cycle_change=target_cycle_change,
                        representative_cumsum=representative_cumsum,
                        representative_totals=representative_totals,
                        working_map_indices=working_indices,
                        representative_matrix=representative_matrix,
                        lookahead_periods=lookahead_periods,
                    )

                    best_rep_index = int(np.argmin(scores))
                    best_score = float(scores[best_rep_index])
                    current_score = float(scores[current_rep_index])

                    if current_score <= best_score + tie_tolerance:
                        chosen_rep_index = current_rep_index
                    else:
                        chosen_rep_index = best_rep_index

                    if chosen_rep_index != current_rep_index:
                        changed += 1
                        working_indices[period_index] = chosen_rep_index
                        working_map[period_index] = representative_ids[chosen_rep_index]

                current_level += representative_totals[chosen_rep_index]

            reconstructed_delta = representative_matrix[working_indices].reshape(-1)
            diagnostics = proxy_error_metrics(target_flat, reconstructed_delta)
            changed_fraction = changed / n_periods

            history.append(
                SweepRecord(
                    sweep=sweep + 1,
                    assignments_changed=changed,
                    changed_fraction=changed_fraction,
                    diagnostics=diagnostics,
                    proposal_assignments_changed=changed,
                    acceptance_fraction=1.0,
                )
            )

            print(
                f"[PGCR] Sweep {sweep + 1} completed in "
                f"{perf_counter() - sweep_start:.3f} s.",
                flush=True,
            )

            if _objective_value(diagnostics) < _objective_value(best_diagnostics):
                best_map = working_map.copy()
                best_indices = working_indices.copy()
                best_diagnostics = diagnostics

            if changed_fraction <= stop_changed_fraction:
                break

        best_reconstructed_delta = representative_matrix[best_indices].reshape(-1)
        target_level = _levels_after_deltas(target_flat)
        initial_reconstructed_level = _levels_after_deltas(initial_reconstructed_delta)
        best_reconstructed_level = _levels_after_deltas(best_reconstructed_delta)
        output_map = _unrotate_map(best_map, selected_start)
        changed_assignments = int(np.count_nonzero(output_map != initial_map))

        return ProxyRemapResult(
            cluster_map=output_map,
            initial_cluster_map=initial_map.copy(),
            rotated_cluster_map=best_map,
            rotated_initial_cluster_map=rotated_initial_map,
            start_period=selected_start,
            representative_ids=representative_ids,
            representative_period_indices=dict(fixed_representative_periods),
            locked_period_count=len(fixed_representative_periods),
            target_delta=target_flat,
            target_level=target_level,
            initial_reconstructed_delta=initial_reconstructed_delta,
            initial_reconstructed_level=initial_reconstructed_level,
            reconstructed_delta=best_reconstructed_delta,
            reconstructed_level=best_reconstructed_level,
            initial_diagnostics=initial_diagnostics,
            diagnostics=best_diagnostics,
            initial_usage=_usage_counts(initial_map, representative_ids),
            remapped_usage=_usage_counts(output_map, representative_ids),
            changed_assignments=changed_assignments,
            changed_fraction=changed_assignments / n_periods,
            sweep_history=tuple(history),
        )

    # ------------------------------------------------------------------
    # Exact-feedback PGCR.
    # ------------------------------------------------------------------
    acceptance_fractions = _normalise_acceptance_fractions(
        exact_acceptance_fractions
    )

    accepted_map = rotated_initial_map.copy()
    accepted_indices = initial_indices.copy()

    exact_evaluations = 0

    def evaluate_exact(rotated_map: ObjectArray) -> tuple[FloatArray, ProxyRemapDiagnostics]:
        nonlocal exact_evaluations

        unrotated_map = _unrotate_map(rotated_map, selected_start)
        exact_delta_unrotated = exact_delta_evaluator(unrotated_map.copy())
        exact_delta_rotated = _rotate_flat_by_period(
            exact_delta_unrotated,
            start_period=selected_start,
            n_periods=n_periods,
            timesteps_per_period=timesteps_per_period,
            name="exact_delta_evaluator result",
        )
        exact_evaluations += 1
        return (
            exact_delta_rotated,
            proxy_error_metrics(target_flat, exact_delta_rotated),
        )

    exact_initial_delta, exact_initial_diagnostics = evaluate_exact(accepted_map)
    accepted_exact_delta = exact_initial_delta.copy()
    accepted_exact_diagnostics = exact_initial_diagnostics

    for sweep in range(max_sweeps):
        sweep_start = perf_counter()
        print(f"[PGCR] Commencing sweep: {sweep + 1}.", flush=True)

        # Build one cheap surrogate proposal from the *currently accepted* map.
        proposal_map = accepted_map.copy()
        proposal_indices = accepted_indices.copy()

        # Treat each outer pass as the next traversal of the accepted circular
        # map. This preserves the original closure-aware sweep idea while
        # ensuring rejected proposals never leak state into the next pass.
        accepted_cycle_total = float(
            np.sum(representative_matrix[accepted_indices])
        )
        current_level = sweep * accepted_cycle_total

        proposed_periods: list[int] = []
        proposal_gains: list[float] = []

        for period_index in range(n_periods):
            current_rep_index = int(proposal_indices[period_index])
            locked_representative = locked_periods.get(period_index)

            if locked_representative is not None:
                chosen_rep_index = representative_lookup[locked_representative]
                if current_rep_index != chosen_rep_index:
                    raise RuntimeError(
                        "A locked representative period changed unexpectedly. "
                        "This indicates an internal PGCR state inconsistency."
                    )
            else:
                scores = _candidate_scores(
                    current_level=current_level,
                    period_index=period_index,
                    sweep=sweep,
                    target_levels=target_levels,
                    target_cycle_change=target_cycle_change,
                    representative_cumsum=representative_cumsum,
                    representative_totals=representative_totals,
                    working_map_indices=proposal_indices,
                    representative_matrix=representative_matrix,
                    lookahead_periods=lookahead_periods,
                )

                best_rep_index = int(np.argmin(scores))
                best_score = float(scores[best_rep_index])
                current_score = float(scores[current_rep_index])

                if current_score <= best_score + tie_tolerance:
                    chosen_rep_index = current_rep_index
                else:
                    chosen_rep_index = best_rep_index

                if chosen_rep_index != current_rep_index:
                    proposed_periods.append(period_index)
                    proposal_gains.append(max(0.0, current_score - best_score))
                    proposal_indices[period_index] = chosen_rep_index
                    proposal_map[period_index] = representative_ids[chosen_rep_index]

            current_level += representative_totals[chosen_rep_index]

        proposal_changed = len(proposed_periods)

        if proposal_changed == 0:
            surrogate_delta = representative_matrix[accepted_indices].reshape(-1)
            surrogate_diagnostics = proxy_error_metrics(target_flat, surrogate_delta)
            history.append(
                SweepRecord(
                    sweep=sweep + 1,
                    assignments_changed=0,
                    changed_fraction=0.0,
                    diagnostics=surrogate_diagnostics,
                    proposal_assignments_changed=0,
                    acceptance_fraction=None,
                    exact_diagnostics=accepted_exact_diagnostics,
                    exact_evaluations=0,
                )
            )
            print(
                f"[PGCR] Sweep {sweep + 1}: no surrogate changes proposed; "
                f"stopping after {perf_counter() - sweep_start:.3f} s.",
                flush=True,
            )
            break

        # Rank the proposed changes by local surrogate improvement. Stable sort
        # makes ties deterministic in chronological proposal order.
        gains = np.asarray(proposal_gains, dtype=float)
        order = np.argsort(-gains, kind="stable")
        ranked_periods = np.asarray(proposed_periods, dtype=int)[order]

        candidate_counts: list[int] = []
        for fraction in acceptance_fractions:
            count = max(1, int(np.ceil(proposal_changed * fraction)))
            count = min(count, proposal_changed)
            if count not in candidate_counts:
                candidate_counts.append(count)

        best_candidate_map: ObjectArray | None = None
        best_candidate_indices: NDArray[np.int64] | None = None
        best_candidate_exact_delta: FloatArray | None = None
        best_candidate_exact_diagnostics: ProxyRemapDiagnostics | None = None
        best_candidate_count = 0
        evaluations_this_sweep = 0

        current_exact_objective = _objective_value(accepted_exact_diagnostics)
        best_exact_objective = current_exact_objective

        for count in candidate_counts:
            candidate_map = accepted_map.copy()
            candidate_indices = accepted_indices.copy()

            selected_periods = ranked_periods[:count]
            candidate_map[selected_periods] = proposal_map[selected_periods]
            candidate_indices[selected_periods] = proposal_indices[selected_periods]

            exact_delta, exact_diagnostics = evaluate_exact(candidate_map)
            evaluations_this_sweep += 1
            exact_objective = _objective_value(exact_diagnostics)

            if exact_objective < best_exact_objective:
                best_exact_objective = exact_objective
                best_candidate_map = candidate_map
                best_candidate_indices = candidate_indices
                best_candidate_exact_delta = exact_delta
                best_candidate_exact_diagnostics = exact_diagnostics
                best_candidate_count = count

        improvement = current_exact_objective - best_exact_objective
        accepted = (
            best_candidate_map is not None
            and improvement > exact_improvement_tolerance
        )

        if accepted:
            assert best_candidate_indices is not None
            assert best_candidate_exact_delta is not None
            assert best_candidate_exact_diagnostics is not None

            previous_map = accepted_map
            accepted_map = best_candidate_map
            accepted_indices = best_candidate_indices
            accepted_exact_delta = best_candidate_exact_delta
            accepted_exact_diagnostics = best_candidate_exact_diagnostics

            accepted_changed = int(np.count_nonzero(accepted_map != previous_map))
            acceptance_fraction = best_candidate_count / proposal_changed
        else:
            accepted_changed = 0
            acceptance_fraction = None

        surrogate_delta = representative_matrix[accepted_indices].reshape(-1)
        surrogate_diagnostics = proxy_error_metrics(target_flat, surrogate_delta)
        changed_fraction = accepted_changed / n_periods

        history.append(
            SweepRecord(
                sweep=sweep + 1,
                assignments_changed=accepted_changed,
                changed_fraction=changed_fraction,
                diagnostics=surrogate_diagnostics,
                proposal_assignments_changed=proposal_changed,
                acceptance_fraction=acceptance_fraction,
                exact_diagnostics=accepted_exact_diagnostics,
                exact_evaluations=evaluations_this_sweep,
            )
        )

        acceptance_text = (
            f"accepted {accepted_changed:,}/{proposal_changed:,} proposed changes "
            f"({acceptance_fraction:.1%} batch)"
            if accepted and acceptance_fraction is not None
            else f"rejected all {proposal_changed:,} proposed changes"
        )
        print(
            f"[PGCR] Sweep {sweep + 1}: {acceptance_text}; "
            f"exact level nRMSE="
            f"{accepted_exact_diagnostics.level_nrmse:.4%}; "
            f"{evaluations_this_sweep} exact evaluation(s); "
            f"{perf_counter() - sweep_start:.3f} s.",
            flush=True,
        )

        # If not even the smallest tested high-gain batch improves the exact
        # objective, another sweep from the identical accepted map would propose
        # the same changes. Stop rather than spending more exact evaluations.
        if not accepted:
            break

        if changed_fraction <= stop_changed_fraction:
            break

    best_map = accepted_map
    best_indices = accepted_indices
    best_reconstructed_delta = representative_matrix[best_indices].reshape(-1)
    best_diagnostics = proxy_error_metrics(target_flat, best_reconstructed_delta)

    target_level = _levels_after_deltas(target_flat)
    initial_reconstructed_level = _levels_after_deltas(initial_reconstructed_delta)
    best_reconstructed_level = _levels_after_deltas(best_reconstructed_delta)

    exact_initial_level = _levels_after_deltas(exact_initial_delta)
    exact_best_level = _levels_after_deltas(accepted_exact_delta)

    output_map = _unrotate_map(best_map, selected_start)
    changed_assignments = int(np.count_nonzero(output_map != initial_map))

    return ProxyRemapResult(
        cluster_map=output_map,
        initial_cluster_map=initial_map.copy(),
        rotated_cluster_map=best_map,
        rotated_initial_cluster_map=rotated_initial_map,
        start_period=selected_start,
        representative_ids=representative_ids,
        representative_period_indices=dict(fixed_representative_periods),
        locked_period_count=len(fixed_representative_periods),
        target_delta=target_flat,
        target_level=target_level,
        initial_reconstructed_delta=initial_reconstructed_delta,
        initial_reconstructed_level=initial_reconstructed_level,
        reconstructed_delta=best_reconstructed_delta,
        reconstructed_level=best_reconstructed_level,
        initial_diagnostics=initial_diagnostics,
        diagnostics=best_diagnostics,
        initial_usage=_usage_counts(initial_map, representative_ids),
        remapped_usage=_usage_counts(output_map, representative_ids),
        changed_assignments=changed_assignments,
        changed_fraction=changed_assignments / n_periods,
        sweep_history=tuple(history),
        exact_initial_reconstructed_delta=exact_initial_delta,
        exact_initial_reconstructed_level=exact_initial_level,
        exact_reconstructed_delta=accepted_exact_delta,
        exact_reconstructed_level=exact_best_level,
        exact_initial_diagnostics=exact_initial_diagnostics,
        exact_diagnostics=accepted_exact_diagnostics,
        exact_evaluations=exact_evaluations,
    )

