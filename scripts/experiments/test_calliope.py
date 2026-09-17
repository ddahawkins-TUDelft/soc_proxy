from scripts.helpers.config import load_experiment_config
from scripts.pipeline import run_case

configs = load_experiment_config("config/experiment_config.yaml")
config = configs["smoke_no_proxy"]

result = run_case(
    config,
    "resources/raw_timeseries/time_varying_parameters_NL.csv",
    model_path="config/calliope/model.yaml",
)

print(result.calliope_model.results)
