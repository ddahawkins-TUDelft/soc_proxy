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
Therefore the remapping target is fixed and the method is not circular in the
sense of repeatedly regenerating its own objective.

Complexity is approximately O(S * n * (k * h + L * h)), where:
    S = number of circular sweeps
    n = number of representative periods in the original chronology
    k = number of candidate representatives
    h = timesteps per representative period
    L = look-ahead periods

For fixed h and modest L, this behaves like O(S * n * k).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from time import perf_counter

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
    """Summary of one circular greedy sweep."""

    sweep: int
    assignments_changed: int
    changed_fraction: float
    diagnostics: ProxyRemapDiagnostics


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
) -> ProxyRemapResult:
    """Greedily remap fixed TSA representatives to preserve proxy chronology.

    Parameters
    ----------
    target_delta
        Fixed ex-ante proxy delta values, shaped
        ``[n_periods, timesteps_per_period]``. For daily TSA with hourly data,
        this will normally be ``[n_days, 24]``.
    representative_delta
        Mapping from representative/cluster ID to its fixed proxy-delta
        profile for one representative period.
    initial_cluster_map
        Existing TSA chronological map. It provides both the initial solution
        and the future assignments used during short look-ahead scoring.
    representative_period_indices
        Optional mapping ``representative_id -> original_period_index`` for
        representatives that are real observed periods (e.g. medoids). These
        periods are permanently locked to their own representative during every
        greedy sweep. This guarantees each supplied real representative has at
        least one occurrence. Synthetic representatives should be omitted.
    start_period
        ``"minimum"`` starts at the period boundary with the lowest target
        cumulative proxy level. An integer can be supplied to force another
        circular starting period.
    lookahead_periods
        Number of *future* representative periods included when evaluating
        today's candidate. Future assignments are taken from the current
        working map. Set to 0 for pure one-period greedy selection.
    max_sweeps
        Maximum number of complete circular passes. State is carried from the
        end of one sweep into the beginning of the next so cycle-closure error
        can influence later decisions.
    tie_tolerance
        If the existing assignment is within this absolute score tolerance of
        the best candidate, retain it. This avoids needless map churn.
    stop_changed_fraction
        Stop after a sweep if the fraction of assignments changed in that
        sweep is less than or equal to this value.

    Returns
    -------
    ProxyRemapResult
        The best map encountered according to full-cycle proxy-level nRMSE,
        plus reconstructed signals, diagnostics, usage counts, and per-sweep
        history.

    Notes
    -----
    This implementation intentionally allows representative usage frequencies
    to change. Use-count changes are returned as diagnostics. Real representative
    periods supplied through ``representative_period_indices`` are the only hard
    assignment constraints.

    ``target_delta`` is treated as immutable ex-ante information. The function
    never recomputes the proxy from a TSAM reconstruction or from the remapped
    physical timeseries.

    No physical demand/wind/solar distance is included in the score. This is
    deliberate: v0 answers the cleanest diagnostic question -- how well can
    this fixed representative library preserve the cumulative proxy if the
    chronological map is allowed to adapt cheaply?
    """

    print("[PGCR] Running Proxy-guided Chronological Remapping.")

    if lookahead_periods < 0:
        raise ValueError("lookahead_periods must be >= 0.")

    if max_sweeps < 1:
        raise ValueError("max_sweeps must be >= 1.")

    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be >= 0.")

    if not 0.0 <= stop_changed_fraction <= 1.0:
        raise ValueError("stop_changed_fraction must lie in [0, 1].")

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

    working_map = rotated_initial_map.copy()
    working_indices = _map_to_indices(
        working_map,
        representative_lookup,
    )

    locked_periods = _rotate_locked_periods(
        fixed_representative_periods,
        start=selected_start,
        n_periods=n_periods,
    )

    representative_cumsum = np.cumsum(
        representative_matrix,
        axis=1,
    )
    representative_totals = representative_cumsum[:, -1]

    target_flat = rotated_target.reshape(-1)
    target_levels = _target_period_levels(rotated_target)
    target_cycle_change = float(np.sum(target_flat))

    initial_reconstructed_delta = (
        representative_matrix[working_indices].reshape(-1)
    )
    initial_diagnostics = proxy_error_metrics(
        target_flat,
        initial_reconstructed_delta,
    )

    best_map = working_map.copy()
    best_indices = working_indices.copy()
    best_diagnostics = initial_diagnostics

    current_level = 0.0
    history: list[SweepRecord] = []

    for sweep in range(max_sweeps):
        changed = 0
        sweep_start = perf_counter()
        print(f"[PGCR] Commencing sweep: {sweep+1}.")
        for period_index in range(n_periods):
            current_rep_index = int(working_indices[period_index])

            # A real representative period (e.g. a medoid's source day) is
            # immutable: that day always maps to itself. This preserves at least
            # one occurrence of every supplied observed representative while all
            # other periods remain free to reuse representatives arbitrarily.
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

        reconstructed_delta = (
            representative_matrix[working_indices].reshape(-1)
        )
        diagnostics = proxy_error_metrics(
            target_flat,
            reconstructed_delta,
        )

        changed_fraction = changed / n_periods

        history.append(
            SweepRecord(
                sweep=sweep + 1,
                assignments_changed=changed,
                changed_fraction=changed_fraction,
                diagnostics=diagnostics,
            )
        )

        print(f"[PGCR] Sweep {sweep+1} completed in {perf_counter() - sweep_start} seconds.")

        candidate_objective = diagnostics.level_nrmse
        best_objective = best_diagnostics.level_nrmse

        if np.isnan(candidate_objective):
            candidate_objective = diagnostics.level_rmse
        if np.isnan(best_objective):
            best_objective = best_diagnostics.level_rmse

        if candidate_objective < best_objective:
            best_map = working_map.copy()
            best_indices = working_indices.copy()
            best_diagnostics = diagnostics

        if changed_fraction <= stop_changed_fraction:
            break

        

    best_reconstructed_delta = (
        representative_matrix[best_indices].reshape(-1)
    )

    target_level = _levels_after_deltas(target_flat)
    initial_reconstructed_level = _levels_after_deltas(
        initial_reconstructed_delta
    )
    best_reconstructed_level = _levels_after_deltas(
        best_reconstructed_delta
    )

    output_map = _unrotate_map(best_map, selected_start)
    initial_output_map = initial_map.copy()

    changed_assignments = int(
        np.count_nonzero(output_map != initial_output_map)
    )

    return ProxyRemapResult(
        cluster_map=output_map,
        initial_cluster_map=initial_output_map,
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
        initial_usage=_usage_counts(
            initial_output_map,
            representative_ids,
        ),
        remapped_usage=_usage_counts(
            output_map,
            representative_ids,
        ),
        changed_assignments=changed_assignments,
        changed_fraction=changed_assignments / n_periods,
        sweep_history=tuple(history),
    )
