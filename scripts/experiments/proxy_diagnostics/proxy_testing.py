import calliope
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from soc_proxy import generate_soc_proxy
from scripts.experiments.signal_analysis.plot_results import _load_timeseries_reference

country = 'BE'
year_s=2010
year_e=year_s+9

dispatchable_by_country = {
    'NL': 3300,
    'IT': 0,
    'ES': 0,
    'BE': 2501,
    'GB': 9323,
}


ref_path, TS_WINDOW = f'SoC_proxy_TSA/data/calliope_models/standard_{year_s}_{year_e}_reference_{country}.nc', [f"{year_s}-01-01", f"{year_e}-12-31"]

# sources = [
#     ['SoC_proxy_TSA/data/calliope_models/standard_2006_2015_reference.nc', ["2006-01-01", "2015-12-31"]],
#     ['SoC_proxy_TSA/data/calliope_models/standard_2006_2015_reference.nc', ["2006-01-01", "2015-12-31"]],
#     ['SoC_proxy_TSA/data/calliope_models/standard_2006_2015_reference.nc', ["2006-01-01", "2015-12-31"]],
# ]


tvp_csv = f'SoC_proxy_TSA/data/timeseries/time_varying_parameters_{country}.csv'


DEMAND_FIELD = "demand_power"

SOC_PROXY_PARAMS = {
    "capacity_weights": {"solar": 1, "onshore_wind": 0.5, "offshore_wind": 0.5},
    "storage_process_losses": {"charging_efficiency": 0.65 * 0.99, "discharging_efficiency": 0.56 * 0.99},
    "dispatchable_techs": {"known_dispatchable_capacity": dispatchable_by_country[country]},
    "soc_decomposition": {"method": "fft_lowpass", "time_horizon_hours": 24},
}




ref_model = calliope.read_netcdf(ref_path)

# Colors
COLOUR_R = "#0d0887"   # CEM
COLOUR_EC = "#b12a90"  # proxy
ANNOT_COLOUR = "0.2"   # dark grey

def _build_proxy(df: pd.DataFrame, demand_field: str, params) -> pd.Series:
    df_proxy, _, _ = generate_soc_proxy(
        df=df,
        demand_field=demand_field,
        renewables_fields_and_weights=params["capacity_weights"],
        dispatchable_techs=params["dispatchable_techs"],
        storage_process_losses=params["storage_process_losses"],
        soc_decomposition=params["soc_decomposition"],
        timestamp_col=None,
        margin_mode='fixed',
        margin_value = 0.04,
    )
    return df_proxy["soc_proxy_LDES"].rename("soc_proxy_LDES")

# load tvp + proxy
df_ref_ts = _load_timeseries_reference(tvp_csv, TS_WINDOW)
soc_ref_proxy = _build_proxy(df_ref_ts, DEMAND_FIELD, SOC_PROXY_PARAMS)
df_ref_ts['soc_proxy_LDES'] = soc_ref_proxy

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
df_storage = df_storage[df_storage['techs'] == 'h2_salt_cavern']
df_storage.set_index('timesteps', inplace=True)

# scale to TWh
cem_soc = (df_storage["soc"] * 1e-7).rename("cem_soc")
proxy_soc = (df_ref_ts["soc_proxy_LDES"] * 1e-7).rename("proxy_soc")

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 1, 1)
ax.set_axisbelow(True)


rescalar = cem_soc.max() / proxy_soc.max()

ax.plot(cem_soc.index, cem_soc,
        label="SoC (CEM)",
        color=COLOUR_R, linewidth=1.2)
ax.plot(proxy_soc.index, proxy_soc*rescalar,
        label="SoC Proxy",
        color=COLOUR_EC, linewidth=1.2, linestyle='dashed')

ax.set_ylabel('State of Charge (TWh)')
ax.set_xlabel('Time')
ax.xaxis.grid(True, which='major', linestyle=':', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='best', frameon=False)

# ------------------------
# 1) peak annotations
# ------------------------
# CEM peak
cem_peak_ts = cem_soc.idxmax()
cem_peak_val = cem_soc.max()

ax.annotate(
    f"SoC (CEM)\nmax={cem_peak_val:.2f} TWh\n{cem_peak_ts.date()}",
    xy=(cem_peak_ts, cem_peak_val),
    xytext=(30, 0),  
    textcoords="offset points",
    ha="left",
    va="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ANNOT_COLOUR, lw=0.8),
    arrowprops=dict(arrowstyle="->", color=ANNOT_COLOUR, lw=0.8),
    color=ANNOT_COLOUR,
)

# Proxy peak
proxy_peak_ts = proxy_soc.idxmax()
proxy_peak_val = proxy_soc.max()

ax.annotate(
    f"SoC Proxy\nmax={proxy_peak_val:.2f} TWh\n{proxy_peak_ts.date()}",
    xy=(proxy_peak_ts, proxy_peak_val),
    xytext=(30, 0), 
    textcoords="offset points",
    ha="left",
    va="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ANNOT_COLOUR, lw=0.8),
    arrowprops=dict(arrowstyle="->", color=ANNOT_COLOUR, lw=0.8),
    color=ANNOT_COLOUR,
)

# ------------------------
# 2) RMSE and Pearson r
# ------------------------
# align on common index
aligned = pd.concat([cem_soc, proxy_soc], axis=1).dropna()
rmse = np.sqrt(((aligned["cem_soc"] - aligned["proxy_soc"]) ** 2).mean())
pearson_r = aligned["cem_soc"].corr(aligned["proxy_soc"])

# ymax = ax.get_ylim()[1]
text_x = pd.to_datetime("2010-01-01") - pd.DateOffset(months=4)
text_y = 0

ax.text(
    text_x,
    text_y,
    f"RMSE = {rmse:.3f} TWh\nPearon R = {pearson_r:.3f}",
    ha="left",
    va="bottom",
    color=ANNOT_COLOUR,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec='white', lw=0.8),
)

fig.tight_layout()
plt.show()
# fig.savefig('soc_proxy_comparison.pdf', dpi=600, bbox_inches="tight")
