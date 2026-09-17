import calliope
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from soc_proxy import generate_soc_proxy
from scripts.experiments.signal_analysis.plot_results import _load_timeseries_reference

RUN_DATA = True
ref_path, TS_WINDOW = (
    "SoC_proxy_TSA/data/calliope_models/standard_2006_2015_reference.nc",
    ["2006-01-01", "2015-12-31"],
)

sources = [
    # ['SoC_proxy_TSA/data/calliope_models/standard_2006_2015_reference.nc', ["2006-01-01", "2015-12-31"],'SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv'],
    # ['SoC_proxy_TSA/data/calliope_models/standard_2008_2017_reference.nc', ["2008-01-01", "2017-12-31"],'SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv'],
    # ['SoC_proxy_TSA/data/calliope_models/standard_2010_2019_reference.nc', ["2010-01-01", "2019-12-31"],'SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv'],
    # ['SoC_proxy_TSA/data/calliope_models/standard_2012_2021_reference.nc', ["2012-01-01", "2021-12-31"],'SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv'],
    # ['SoC_proxy_TSA/data/calliope_models/standard_2014_2023_reference.nc', ["2014-01-01", "2023-12-31"],'SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv'],
    [
        "SoC_proxy_TSA/data/calliope_models/standard_2006_2015_GB_reference.nc",
        ["2006-01-01", "2015-12-31"],
        "SoC_proxy_TSA/data/timeseries/time_varying_parameters_GB.csv",
    ],
    [
        "SoC_proxy_TSA/data/calliope_models/standard_2008_2017_GB_reference.nc",
        ["2008-01-01", "2017-12-31"],
        "SoC_proxy_TSA/data/timeseries/time_varying_parameters_GB.csv",
    ],
    [
        "SoC_proxy_TSA/data/calliope_models/standard_2010_2019_GB_reference.nc",
        ["2010-01-01", "2019-12-31"],
        "SoC_proxy_TSA/data/timeseries/time_varying_parameters_GB.csv",
    ],
    [
        "SoC_proxy_TSA/data/calliope_models/standard_2012_2021_GB_reference.nc",
        ["2012-01-01", "2021-12-31"],
        "SoC_proxy_TSA/data/timeseries/time_varying_parameters_GB.csv",
    ],
    [
        "SoC_proxy_TSA/data/calliope_models/standard_2014_2023_GB_reference.nc",
        ["2014-01-01", "2023-12-31"],
        "SoC_proxy_TSA/data/timeseries/time_varying_parameters_GB.csv",
    ],
]

step = 0.01
max_margin = 0.2

margins = np.linspace(0, max_margin, int(max_margin / step) + 1)

# margins = [
#     # 'auto',
#     0,
#     0.01,
#     0.02,
#     0.03,
#     0.04,
#     0.05,
#     0.06,
#     0.07,
#     0.08,
#     0.09,
#     0.1,
# ]


# tvp_csv = 'SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv'

# mpl.rcParams.update({
#     "text.usetex": True,
#     "pgf.texsystem": "pdflatex",
#     "pgf.rcfonts": False,
#     "axes.unicode_minus": False,
# })
# mpl.rcParams["pgf.preamble"] = r""

DEMAND_FIELD = "demand_power"


def _build_proxy(df: pd.DataFrame, demand_field: str, params, margin) -> pd.Series:
    df_proxy, _, _ = generate_soc_proxy(
        df=df,
        demand_field=demand_field,
        renewables_fields_and_weights=params["capacity_weights"],
        dispatchable_techs=params["dispatchable_techs"],
        storage_process_losses=params["storage_process_losses"],
        soc_decomposition=params["soc_decomposition"],
        timestamp_col=None,
        margin_mode="auto_volatility" if margin == "auto" else "fixed",
        margin_value=0.04 if margin == "auto" else margin,
        margin_bounds=(0, 1),
    )
    return df_proxy["soc_proxy_LDES"].rename("soc_proxy_LDES")


SOC_PROXY_PARAMS = {
    "capacity_weights": {"solar": 1, "onshore_wind": 0.5, "offshore_wind": 0.5},
    "storage_process_losses": {
        "charging_efficiency": 0.65 * 0.99,
        "discharging_efficiency": 0.56 * 0.99,
    },
    "dispatchable_techs": {"known_dispatchable_capacity": 3300},
    "soc_decomposition": {"method": "fft_lowpass", "time_horizon_hours": 24},
}


if RUN_DATA:
    data = []

    for source in sources:
        ref_path, TS_WINDOW, tvp_csv = source[0], source[1], source[2]

        ref_model = calliope.read_netcdf(ref_path)
        df_ref_ts = _load_timeseries_reference(tvp_csv, TS_WINDOW)
        # load CEM storage
        df_storage = (
            ref_model.results["storage"]
            .fillna(0)
            .to_series()
            .dropna()
            .to_frame("soc")
            .reset_index()
            .drop(columns=["nodes"], errors="ignore")
        )
        df_storage = df_storage[df_storage["techs"] == "h2_salt_cavern"]
        df_storage.set_index("timesteps", inplace=True)

        # scale to TWh
        cem_soc = (df_storage["soc"] * 1e-7).rename("cem_soc")

        for margin in margins:
            soc_ref_proxy = (
                _build_proxy(df_ref_ts, DEMAND_FIELD, SOC_PROXY_PARAMS, margin) * 1e-7
            )

            mean_error = (cem_soc.mean() - soc_ref_proxy.mean()) / cem_soc.mean()
            rmse = float(
                np.sqrt(np.mean(np.square(cem_soc - soc_ref_proxy))) / np.max(cem_soc)
            )

            ref_max = cem_soc.max()
            proxy_max = soc_ref_proxy.max()
            max_adjusted_rmse = float(
                np.sqrt(
                    np.mean(np.square(cem_soc - soc_ref_proxy * ref_max / proxy_max))
                )
                / np.max(cem_soc)
            )

            pearson_r = float(cem_soc.corr(soc_ref_proxy))

            data.append(
                {
                    "start_year": int(TS_WINDOW[0][:4]),
                    "margin": str(margin),
                    "mean_error": mean_error,
                    "norm_rmse": rmse,
                    "max_adj_rmse": max_adjusted_rmse,
                    "pearson_r": pearson_r,
                }
            )

    results = pd.DataFrame.from_dict(data)
    results.to_csv("margin_test_results.csv")
else:
    results = pd.read_csv("margin_test_results.csv")

mean_results = results.groupby("margin")
results = mean_results.mean()
results["margin"] = results.index


# Colors
COLOUR_1 = "#0d0887"  #
COLOUR_2 = "#6a00a8"  #
COLOUR_3 = "#b12a90"  #
COLOUR_4 = "#e16462"  #
COLOUR_5 = "#fca636"  #
COLOUR_6 = "#f0f921"  #
ANNOT_COLOUR = "0.2"  #


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_axisbelow(True)

ax.scatter(
    results["margin"], results["norm_rmse"], label="RMSE (Normalised)", color=COLOUR_1
)


ax.scatter(
    results["margin"], 1 - results["pearson_r"], label="1-R (Pearson)", color=COLOUR_2
)

ax.scatter(results["margin"], results["mean_error"], label="Mean Error", color=COLOUR_3)

compound_signal = (
    np.abs(results["mean_error"]) + (1 - results["pearson_r"]) + results["norm_rmse"]
) / 3

ax.plot(
    results["margin"], compound_signal, label="Compound (Mean, Abs.)", color=COLOUR_4
)

ax.axhline(y=0, color="black", linestyle="dashed")

ax.set_ylabel("Metric")
ax.set_xlabel("Margin")
ax.xaxis.grid(True, which="major", linestyle=":", alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="best", frameon=False)

fig.tight_layout()
plt.show()
fig.savefig("margin_sensitivity_test_NL.pdf", dpi=600, bbox_inches="tight")
