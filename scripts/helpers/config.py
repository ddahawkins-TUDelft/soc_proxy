"""Experiment configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load and resolve all experiments defined in a YAML configuration.

    Each experiment is recursively overlaid onto the top-level ``defaults``.
    Missing values therefore inherit their defaults, while explicitly supplied
    values replace them.

    Parameters
    ----------
    path
        Path to the experiment YAML file.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from experiment name to its fully resolved configuration.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")

    defaults = config.get("defaults")
    experiments = config.get("experiments")

    if not isinstance(defaults, dict):
        raise ValueError("Configuration must define a 'defaults' mapping.")

    if not isinstance(experiments, dict):
        raise ValueError("Configuration must define an 'experiments' mapping.")

    resolved: dict[str, dict[str, Any]] = {}

    for name, overrides in experiments.items():
        if overrides is None:
            overrides = {}

        if not isinstance(overrides, dict):
            raise ValueError(
                f"Experiment '{name}' must contain a mapping of overrides."
            )

        experiment = _deep_merge(defaults, overrides)

        # Useful provenance which does not need to be repeated in the YAML.
        experiment["experiment_name"] = name

        resolved[name] = experiment

    return resolved


def _deep_merge(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge ``overrides`` onto ``base``.

    Dictionaries are merged recursively. All other values, including lists
    and ``None``, replace the value in ``base``.
    """
    merged = deepcopy(base)

    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged
