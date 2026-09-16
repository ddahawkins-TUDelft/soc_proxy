#!/usr/bin/env python3
"""
reference_dispatcher.py

Runs full (non-clustered) Calliope dev7 reference models across year ranges,
using your helper: standardised_model_config(params) -> (model, filename).

Edit DEFAULT_YEAR_RANGES and BASE_PARAMS below, then run this file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from copy import deepcopy

# Import your helper (ensure this script is next to helper_model_config.py or
# that its directory is in PYTHONPATH)
from soc_proxy.calliope.model_config import standardised_model_config


# =================== EDIT THESE DEFAULTS ===================

# You can use "2015-2019", "2010-2014", or single years like "2018"
DEFAULT_YEAR_RANGES: List[str] = [
    "2010-2019",
    # "2010-2019",
    ]  
shuffled_dir = "time_varying_parameters__"
shuffled_source = []
# tvp_source = 'time_varying_parameters'
# tvp_source = 'time_varying_parameters_GB'

tvp_source = 'time_varying_parameters_BE'
scenario_calliope = 'belgium'


# Base params passed into your helper; these are merged with per-range values.
# NOTE: your helper indexes calliope_full_log[0], so keep it as a 1-length tuple/list.
BASE_PARAMS: Dict[str, Any] = {
    "config_yaml_name": "model",  # e.g., "model"
    "scenario_name": scenario_calliope, #belgium, spain, italy
    "calliope_full_log": (True,),          # helper uses [0] / (0)
    # Optionally:
    # "filename_time_varying_parameters": shuffled_dir+shuffled_source[0] if shuffled_source else None,
    # "dict_additional_overrides": {...},
}

# Where to save the NetCDF/NetCDF-like file (filename provided by your helper)
OUTPUT_DIR = Path("SoC_proxy_TSA/data/calliope_models")

# ===========================================================


def parse_ranges(ranges: List[str]) -> List[Tuple[int, int]]:
    """Parse items like '2015-2019' or '2018' into (start, end) integer tuples."""
    parsed: List[Tuple[int, int]] = []
    for item in ranges:
        s = item.strip()
        if not s:
            continue
        if "-" in s:
            a, b = s.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(s)
        if end < start:
            start, end = end, start  # normalize if accidentally reversed
        parsed.append((start, end))
    return parsed


def main(ranges: List[str] | None = None) -> int:
    year_ranges = parse_ranges(ranges or DEFAULT_YEAR_RANGES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    failures: List[Tuple[Tuple[int, int], Exception]] = []

    for start_year, end_year in year_ranges:
        # for source in shuffled_source:
            # Build params for this range (end year inclusive)
            params = deepcopy(BASE_PARAMS)
            params["horizon_start"] = f"{start_year}-01-01"
            params["horizon_end"] = f"{end_year}-12-31"
            params['filename_time_varying_parameters'] = tvp_source

            print(f"\n=== Running {start_year}-{end_year} ===")
            print(f"=== {params['filename_time_varying_parameters']}")
            print(f"=== {params['scenario_name']}\n")
            try:
                # Your helper returns (model, filename)
                model, filename = standardised_model_config(params)

                if tvp_source == 'time_varying_parameters_GB':
                    filename = f'standard_{start_year}_{end_year}_reference_GB.nc'
                elif tvp_source == 'time_varying_parameters_GB_availability_NL_demand':
                    filename = f'standard_{start_year}_{end_year}_GB_availability_NL_demand.nc'
                elif tvp_source == 'time_varying_parameters_BE':
                    filename = f'standard_{start_year}_{end_year}_reference_BE.nc'
                elif tvp_source == 'time_varying_parameters_IT':
                    filename = f'standard_{start_year}_{end_year}_reference_IT.nc'
                elif tvp_source == 'time_varying_parameters_ES':
                    filename = f'standard_{start_year}_{end_year}_reference_ES.nc'

                solver_options = {
                    # "BarHomogeneous": 1,
                    # "DualReductions": 0,
                    # Optional, if it still complains:
                    "NumericFocus": 2,
                    # "ScaleFlag": 2,
                    'Threads': 6
                }

                # Build, solve, save
                model.build()
                model.backend.shadow_prices.activate() #for tracking of duals
                model.solve(shadow_prices=[
                    "storage_max", # shadow price of energy capacity i.e. increasing storage cap, usually only >0 for 1 time step
                    "balance_storage", # shadow price of storing one unit of energy to the next time step
                    "flow_out_max", # shadow price of power capacity, for electrolyser (charge) and h2 ccgt (discharge)
                    "flow_in_max" # shadow price of power capacity in, for salt cavern injection
                    ], solver_options=solver_options)
                outfile = OUTPUT_DIR / filename
                outfile.parent.mkdir(parents=True, exist_ok=True)

                # Save exactly with the filename provided by the helper
                model.to_netcdf(str(outfile))
                print(f"  - Saved: {outfile}")

            except Exception as e:
                failures.append(((start_year, end_year), e))
                print(f"  ! FAILED {start_year}-{end_year}: {e}")

    if failures:
        print("\nSome runs failed:")
        for (sy, ey), err in failures:
            print(f"  - {sy}-{ey}: {repr(err)}")
        return 1

    print("\nAll runs completed successfully.")
    return 0


if __name__ == "__main__":
    # If you want to override ranges from the command line you still can:
    #   python reference_dispatcher.py 2018 2015-2017
    # Otherwise it uses DEFAULT_YEAR_RANGES.
    cli_ranges = sys.argv[1:] if len(sys.argv) > 1 else None
    sys.exit(main(cli_ranges))
