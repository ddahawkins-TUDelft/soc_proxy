"""Tests for the conservative TSAM -> PGCR adapter."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from proxy_guided_chronological_remapping.tsam_adapter import (
    prepare_tsam_proxy_chronology_inputs,
)


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="h")


def test_medoid_uses_original_proxy_at_real_representative_periods() -> None:
    index = _index(6)
    target = pd.Series(
        [1.0, 2.0, 10.0, 20.0, -1.0, -2.0],
        index=index,
        name="surplus_LDES",
    )

    clustering = SimpleNamespace(
        representation="medoid",
        cluster_centers=(0, 2),
    )

    result = SimpleNamespace(
        original=pd.DataFrame({"demand": range(6)}, index=index),
        cluster_assignments=np.array([0, 0, 1]),
        n_timesteps_per_period=2,
        n_segments=None,
        clustering=clustering,
        period_index=[0, 1],
        cluster_representatives=pd.DataFrame(),
    )

    inputs = prepare_tsam_proxy_chronology_inputs(
        result,
        target,
        proxy_column_name="surplus_LDES",
    )

    np.testing.assert_allclose(inputs.target_delta, [[1, 2], [10, 20], [-1, -2]])
    np.testing.assert_allclose(inputs.representative_delta[0], [1, 2])
    np.testing.assert_allclose(inputs.representative_delta[1], [-1, -2])
    assert inputs.representative_period_indices == {0: 0, 1: 2}


def test_synthetic_representation_uses_existing_represented_proxy() -> None:
    index = _index(4)
    target = pd.Series([1.0, 2.0, 3.0, 4.0], index=index, name="surplus_LDES")

    representative_index = pd.MultiIndex.from_product(
        [[0, 1], [0, 1]],
        names=["cluster", "timestep"],
    )
    representatives = pd.DataFrame(
        {"surplus_LDES": [0.5, 1.5, 3.5, 4.5]},
        index=representative_index,
    )

    result = SimpleNamespace(
        original=pd.DataFrame({"demand": range(4)}, index=index),
        cluster_assignments=np.array([0, 1]),
        n_timesteps_per_period=2,
        n_segments=None,
        clustering=SimpleNamespace(
            representation=SimpleNamespace(),
            cluster_centers=None,
        ),
        period_index=[0, 1],
        cluster_representatives=representatives,
    )

    inputs = prepare_tsam_proxy_chronology_inputs(
        result,
        target,
        proxy_column_name="surplus_LDES",
    )

    np.testing.assert_allclose(inputs.representative_delta[0], [0.5, 1.5])
    np.testing.assert_allclose(inputs.representative_delta[1], [3.5, 4.5])
    assert inputs.representative_period_indices == {}


def test_synthetic_representation_refuses_to_invent_missing_proxy() -> None:
    index = _index(4)
    target = pd.Series([1.0, 2.0, 3.0, 4.0], index=index, name="surplus_LDES")

    representative_index = pd.MultiIndex.from_product(
        [[0, 1], [0, 1]],
        names=["cluster", "timestep"],
    )
    representatives = pd.DataFrame(
        {"demand": [0.5, 1.5, 3.5, 4.5]},
        index=representative_index,
    )

    result = SimpleNamespace(
        original=pd.DataFrame({"demand": range(4)}, index=index),
        cluster_assignments=np.array([0, 1]),
        n_timesteps_per_period=2,
        n_segments=None,
        clustering=SimpleNamespace(
            representation=SimpleNamespace(),
            cluster_centers=None,
        ),
        period_index=[0, 1],
        cluster_representatives=representatives,
    )

    with pytest.raises(ValueError, match="will not construct"):
        prepare_tsam_proxy_chronology_inputs(
            result,
            target,
            proxy_column_name="surplus_LDES",
        )
