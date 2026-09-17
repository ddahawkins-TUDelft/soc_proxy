from soc_proxy.tsa_model import tsa_model
import os
import calliope
import yaml
from copy import deepcopy
from soc_proxy.calliope.io import read_clustered_netcdf
import pandas as pd
import time

# -------------------------------------------------------------------------------------------
# ----------------------------------- CONFIGURE ----------------------------------------------

show_soc = False
show_visual = False

country = "NL"

dispatchable_by_country = {
    "NL": 3300,
    "IT": 0,
    "ES": 0,
    "BE": 2501,
    "GB": 9323,
}

overrides_by_country = {
    "NL": "standard",
    "IT": "italy",
    "ES": "spain",
    "BE": "belgium",
    "GB": "GB",
}

scenarios = [
    # 'Sensitivity_W_reps14',
    "Sensitivity_W_reps30",
    "Sensitivity_W_reps45",
    "Sensitivity_W_reps60",
    "Sensitivity_W_reps90",
    "Sensitivity_W_reps180",
    "Sensitivity_W_reps365",
    # 'Sensitivity_margin_W1_reps60',
    # 'margin_sensitivity_60-180_m=0.04',
    # 'margin_sensitivity_60-180_m=0.1',
    # 'margin_sensitivity_60-180_m=0.14',
    # 'margin_sensitivity_60-180_m=0.02',
    # 'margin_sensitivity_60-180_m=0.04',
    # 'margin_sensitivity_60-180_m=0.06',
    # 'standard_60'
]
dispatch_config = [
    ["SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv", [2006, 2015]],
    ["SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv", [2008, 2017]],
    ["SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv", [2010, 2019]],
    ["SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv", [2012, 2021]],
    ["SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv", [2014, 2023]],
    # [f'SoC_proxy_TSA/data/timeseries/time_varying_parameters_{country}.csv', [2006,2015]],
    # [f'SoC_proxy_TSA/data/timeseries/time_varying_parameters_{country}.csv', [2008,2017]],
    # [f'SoC_proxy_TSA/data/timeseries/time_varying_parameters_{country}.csv', [2010,2019]],
    # [f'SoC_proxy_TSA/data/timeseries/time_varying_parameters_{country}.csv', [2012,2021]],
    # [f'SoC_proxy_TSA/data/timeseries/time_varying_parameters_{country}.csv', [2014,2023]],
]

# for running the shorter horizon models too
for i in range(2006, 2020, 5):
    dispatch_config.append(
        ["SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv", [i, i + 4]]
    )

    dispatch_config.append(
        ["SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv", [i, i + 1]]
    )

# for i in range(2006,2020,5):
#     dispatch_config.append([f'SoC_proxy_TSA/data/timeseries/time_varying_parameters_{country}.csv',[i,i+4]])
# for i in range(2006,2023,2):
#     dispatch_config.append([f'SoC_proxy_TSA/data/timeseries/time_varying_parameters_{country}.csv',[i,i+1]])


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


def run(
    calliope_params, soc_proxy_params, tsa_params, tsa_type, tvp_source: str = None
):

    # MODEL SETUP FUNCTIONS -------------------------------------------------------------------------------------------------
    s_time = time.time()
    m = tsa_model(
        tsa_type=tsa_type,
        path_timeseries=tvp_source
        if tvp_source
        else "SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv",
    )

    m.soc_proxy.set_params(soc_proxy_params)
    m.tsa.set_params(tsa_params)
    m.calliope_model.set_params(calliope_params)
    m.compute_id()
    print(f" -----      {m.id}      ----- ")
    print(" ------------------------------------------------------- ")
    m.assign_paths(directory="SoC_proxy_TSA/data")
    m.save_params()

    if os.path.exists(m.paths["calliope_model"]):
        print(f"> Model: {m.paths['calliope_model']} already exists. Reading file...")
        # m.calliope_model.model = calliope.read_netcdf(m.paths['calliope_model'])
        m.calliope_model.model = read_clustered_netcdf(m.paths["calliope_model"])

    else:
        # TSA FUNCTIONS -------------------------------------------------------------------------------------------------

        m.apply_tsa_pipeline()

        # CALLIOPE FUNCTIONS -------------------------------------------------------------------------------------------------

        m.configure_calliope()
        m.build_calliope()
        m.solve_and_save_calliope()
        print("[Dispatch] Instance Runtime: ", time.time() - s_time)
    # CALLIOPE FUNCTIONS -------------------------------------------------------------------------------------------------

    m.generate_soc_proxy_expost()

    # RETURN FUNCTIONS -------------------------------------------------------------------------------------------------

    return m


calliope_params = {
    "type": "cluster",
    "config_yaml_name": "model",
    "date_range": "error",
    "calliope_full_log": [False, False],
}

if country != "NL":
    calliope_params["scenario_name"] = overrides_by_country[country]

soc_proxy_params = {
    "capacity_weights": {
        "solar": 1,
        "onshore_wind": 0.5,  # making the baseline assumption of an even distribution between solar and wind -based products i.e. the sum of onshore and offshore wind equals solar
        "offshore_wind": 0.5,
    },
    "storage_process_losses": {
        "charging_efficiency": 0.65
        * 0.99,  # electrolyser efficiency * ldes injection efficiency
        "discharging_efficiency": 0.56
        * 0.99,  # electrolyser efficiency * ldes injection efficiency
    },
    "dispatchable_techs": {
        "known_dispatchable_capacity": dispatchable_by_country[
            country
        ]  # we know that 3.3GW nuclear makes up c.25% of 13GW mean hourly demand with a high uptime
    },
    "soc_decomposition": {"method": "fft_lowpass", "time_horizon_hours": 24},
}

tsa_params = {
    "k_periods": 37,
    # 'matrix_weights': {  DEPRECRATE
    #         'renewables': 1,
    #         'demand': 1,
    #         'proxy': 1,
    #     },
    "lambda_soc": {"cluster": None, "optimisation": None},
    "names_renewables": list(soc_proxy_params["capacity_weights"].keys()),
    "name_demand": ["demand_power"],
    "soc_proxy": {
        "use_soc_proxy": False,
        "proxy_inputs_to_consider": [
            "surplus_LDES"
        ],  # Options: surplus_LDES, soc_proxy_LDES
        "proxy_window": None,
        "optimisation_proxy_mode": "",  # options: endogenous, exogenous
    },
    "cluster_method": "hierarchical",  # Options: k_medoids, k_means, hierarchical
    "representation_method": "medoidRepresentation",  # Options: medoidRepresentation, meanRepresentation, distributionRepresentation
    "hours_per_period": 24,
    "soc_features": {},
    "extremes_spec": {},
    "soft_prune": False,
    "post_cluster_optimisation_params": {},
}


# DISPATCH CONFIGURATION  -------------------------------------------------------------------------------------------------

# open yaml config file which defines the model runs
with open("SoC_proxy_TSA/model_config/batch_run_config.yaml", "r") as f:
    batch_config = yaml.safe_load(f)


# EXECUTION FUNCTIONS -------------------------------------------------------------------------------------------------

# loop over the model runs, update parameters, and run
list_model_dict = []

cases_log = []

print(" ------------------------------------------------------- ")
print("[Dispatch] Scenarios include...")
print(scenarios)
print(dispatch_config)
print(" ------------------------------------------------------- ", flush=True)


for dispatch in dispatch_config:
    tvp_source = dispatch[0]
    date_range = dispatch[1]

    for scenario_name, scenario_batch in batch_config.items():
        if scenario_name in scenarios:
            print(" ------------------------------------------------------- ")
            print("  ")
            print(f"[Dispatch] running scenario {scenario_name}")
            print("   ")
            print(
                " ------------------------------------------------------- ", flush=True
            )
            for model_name, config in scenario_batch.items():
                print(" ------------------------------------------------------- ")
                print(f" ----- {model_name}  -----")

                calliope_p = deepcopy(calliope_params)
                soc_proxy_p = deepcopy(soc_proxy_params)
                tsa_p = deepcopy(tsa_params)

                calliope_p["date_range"] = date_range

                if config["calliope_params"]:
                    for key, value in config["calliope_params"].items():
                        if value:
                            calliope_p[key] = value
                if tvp_source:
                    calliope_p["tvp_source"] = tvp_source
                if config["soc_proxy_params"]:
                    for key, value in config["soc_proxy_params"].items():
                        if value:
                            soc_proxy_p[key] = value
                if config["tsa_params"]:
                    for key, value in config["tsa_params"].items():
                        if value or value == 0:
                            tsa_p[key] = value

                dispatch_mode = config["dispatch_mode"]
                t_start = time.time()
                m = run(
                    calliope_p,
                    soc_proxy_p,
                    tsa_p,
                    tsa_type=dispatch_mode,
                    tvp_source=tvp_source
                    if tvp_source
                    else "SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv",
                )
                t_end = time.time()

                if show_visual:
                    list_model_dict.append(
                        {
                            "model": m,
                            "name": f"{model_name}",  # proxy_wt={tsa_params['matrix_weights']['proxy']} {'with proxy' if use_proxy else 'without proxy'}, k={m.tsa.params['k_periods']}, agg={cluster_method}, rep={rep_method}'
                            "params": {},
                        }
                    )

                cases_log.append(
                    {
                        "tvp": tvp_source,
                        "scenario_name": scenario_name,
                        "model_name": model_name,
                        "dates": ",".join(map(str, calliope_p["date_range"])),
                        "date_range": calliope_p["date_range"][1]
                        - calliope_p["date_range"][0]
                        + 1,
                        "id": m.id,
                        "runtime": t_end - t_start,
                    }
                )


timestamp = time.strftime("%Y%m%d_%H%M%S")
df = pd.DataFrame(cases_log)
df.to_csv(f"SoC_proxy_TSA/data/notes/log_{timestamp}.csv", index=False)

# VISUALISATION FUNCTIONS -------------------------------------------------------------------------------------------------

# add the reference case

if len(dispatch_config) <= 1 and show_visual:
    print(
        f"[Dispatch] Loading reference standard_{date_range[0]}_{date_range[-1]}_reference.nc"
    )
    ref_path = f"SoC_proxy_TSA/data/calliope_models/standard_{date_range[0]}_{date_range[-1]}_reference_{country}.nc"

    list_model_dict.append(
        {
            "model": calliope.read_netcdf(ref_path),
            "name": "reference",
            "params": {
                "date_range": date_range,
                "path_timeseries": f"SoC_proxy_TSA/data/timeseries/time_varying_parameters_{country}.csv",
                "soc_proxy_params": soc_proxy_params,
            },
        }
    )


if show_visual:
    print("> Dispatch: Visualising results")
    from scripts.helpers.visualisation.visualise import visualise

    visualise(
        list_model_dict=list_model_dict,
        x_field="Time",  #'Time'
        y_field="State of Charge"
        if show_soc
        else "SoC Proxy",  #'State of Charge', 'SoC Proxy'
        # colour_field='MAGMe',
        # show_tsa_internal_surplus_accumulation=True,
        # save_fig=True
    )
