"""Run one experiment and plot the SoC Proxy diagnostic."""

import calliope
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scripts.helpers.config import load_experiment_config
from scripts.helpers.reference_models import load_reference_model
from scripts.pipeline import run_case
from scripts.plots.multi_experiment_proxy_comparison import (
    plot_proxy_experiment_comparison,
)
from scripts.helpers.results import (
    record_case_results,
    consolidate_results,
    generate_case_id,
)

calliope.set_log_verbosity(
    "info",
    include_solver_output=False,
)

# ---------------------------------------------------------------------------
# 1. Load experiments
# ---------------------------------------------------------------------------

configs = load_experiment_config("config/proxy_development.yaml")

# ---------------------------------------------------------------------------
# 2. Run clustered experiments
# ---------------------------------------------------------------------------

case_ids = []


for experiment_name, config in configs.items():
    print(f"Running {experiment_name}")

    case_id = generate_case_id(config)

    check_path = Path(f"results/_fragments/parameters/{case_id}.parquet")

    if check_path.exists():
        print("Case:", case_id, "already exists. Skipping.")

    else:
        timeseries_path = (
            "resources/raw_timeseries/"
            f"time_varying_parameters_{config['data_params']['country']}.csv"
        )

        result = run_case(
            config,
            timeseries_path,
            model_path="config/calliope/model.yaml",
        )

        reference_model = load_reference_model(config)

        case_id = record_case_results(
            config,
            result,
            reference_model,
        )

        print(f"Recorded {experiment_name}: {case_id}")

    case_ids.append(case_id)

consolidate_results()


# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------


plot_proxy_experiment_comparison(
    case_ids,
    output_path="results/figures/proxy_experiment_comparison.png",
)
