from scripts.helpers.config import load_experiment_config
from scripts.pipeline import load_case_timeseries, run_tsa_case

configs = load_experiment_config("config/experiment_config.yaml")
config = configs["smoke_no_proxy"]

timeseries = load_case_timeseries(
    config,
    "resources/raw_timeseries/time_varying_parameters_NL.csv",
)

artifacts = run_tsa_case(config, timeseries)

print(timeseries.shape)
print(artifacts.tsa_result.assignments.head())
print(artifacts.cluster_map)
print(artifacts.reconstructed_timeseries.shape)