"""Experiment configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(
    path: str | Path,
    *,
    scenarios_path: str | Path | None = None,
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

    if scenarios_path is None:
        scenarios_path = path.parent / "calliope" / "scenarios.yaml"

    with Path(scenarios_path).open("r", encoding="utf-8") as file:
        calliope_scenarios = yaml.safe_load(file)

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

        _resolve_country_assumptions(
            experiment,
            calliope_scenarios,
        )

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
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def _resolve_country_assumptions(
    config: dict[str, Any],
    scenarios_config: dict[str, Any],
) -> None:
    """Resolve country-specific Calliope and SoC Proxy assumptions."""

    country = str(config["data_params"]["country"])

    scenarios = scenarios_config.get("scenarios", {})
    overrides = scenarios_config.get("overrides", {})

    if country not in scenarios:
        raise ValueError(
            f"No Calliope scenario is defined for country {country!r}."
        )

    capacity_override = f"fixing_capacities_{country}"

    scenario_overrides = scenarios[country]

    if capacity_override not in scenario_overrides:
        raise ValueError(
            f"Scenario {country!r} does not include expected override "
            f"{capacity_override!r}."
        )

    if capacity_override not in overrides:
        raise ValueError(
            f"Calliope override {capacity_override!r} is not defined."
        )

    try:
        nuclear = overrides[capacity_override]["techs"]["nuclear"]
        dispatchable_capacity = float(nuclear["flow_cap_max"])
    except KeyError as exc:
        raise ValueError(
            f"{capacity_override!r} must define "
            "techs.nuclear.flow_cap_max so that the SoC Proxy "
            "baseload assumption can be resolved."
        ) from exc

    if dispatchable_capacity < 0:
        raise ValueError(
            "Nuclear flow_cap_max cannot be negative."
        )

    config.setdefault("calliope_params", {})
    config["calliope_params"]["scenario"] = country

    config.setdefault("soc_proxy_params", {})
    config["soc_proxy_params"]["dispatchable_capacity"] = (
        dispatchable_capacity
    )