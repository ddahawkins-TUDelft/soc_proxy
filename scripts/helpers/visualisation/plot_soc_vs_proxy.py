from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import calliope
from soc_proxy import generate_soc_proxy
from soc_proxy.calliope.timeseries import calliope_ts_to_pandas

def _reference_soc_series(model, storage_tech: str) -> pd.Series:
    """Extract actual LDES SoC from a full (unclustered) Calliope model."""
    df = (model.results['storage'].fillna(0).to_series().dropna().to_frame('soc').reset_index())
    df = df[df['techs'] == storage_tech]
    df = df.set_index('timesteps').sort_index()
    return df['soc']

def _load_reference_timeseries_df(timeseries_csv_path: Optional[Path], ts_window: List[str]) -> pd.DataFrame:
    if timeseries_csv_path is None:
        raise FileNotFoundError('timeseries_csv_path not provided. Please pass the reference timeseries CSV used to build the model.')
    df = calliope_ts_to_pandas(timeseries_csv_path, ts_window[0], ts_window[1])
    df.set_index('timesteps', inplace=True)
    return df

def _load_soc_proxy_params(model, explicit_params: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    """Return (params_dict, demand_field_name)."""
    demand_field = 'demand_power'
    if explicit_params is not None:
        return explicit_params, demand_field
    if hasattr(model, 'params') and isinstance(model.params, dict) and 'soc_proxy_params' in model.params:
        return model.params['soc_proxy_params'], demand_field
    try:
        if 'soc_proxy_params' in model.attrs:
            return model.attrs['soc_proxy_params'], demand_field
    except Exception:
        pass
    raise KeyError("Could not find 'soc_proxy_params' on the model. Pass soc_proxy_params=... explicitly.")

def plot_soc_actual_vs_proxy(reference_nc_path: Path,
    timeseries_csv_path: Optional[Path] = None,
    ts_window=List[str],
    storage_tech: str = 'h2_salt_cavern',
    soc_proxy_params: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None):
    # Load model & actual SoC
    model = calliope.read_netcdf(str(reference_nc_path))
    soc_actual = _reference_soc_series(model, storage_tech=storage_tech)
    # Build df for proxy
    df = _load_reference_timeseries_df(timeseries_csv_path, ts_window)
    params, demand_field = _load_soc_proxy_params(model, soc_proxy_params)
    df_proxy, _, _ = generate_soc_proxy(
        df=df,
        demand_field=demand_field,
        renewables_fields_and_weights=params['capacity_weights'],
        dispatchable_techs=params['dispatchable_techs'],
        storage_process_losses=params['storage_process_losses'],
        soc_decomposition=params['soc_decomposition'],
        timestamp_col=None,
    )
    soc_proxy = df_proxy['soc_proxy_LDES'].rename('soc_proxy_LDES')
    surplus = df_proxy['surplus_LDES'].rename('surplus_LDES')
    soc_actual, soc_proxy = soc_actual.align(soc_proxy, join='inner')
    _, surplus = soc_actual.align(surplus, join='inner')
    fig = plt.figure(figsize=(9, 8))
    colour1='#0D0887'
    colour2='#CC4778'
    colour3='#ED7953'
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(soc_actual.index, soc_actual.values, label='Actual SoC', color=colour1)
    ax1.plot(soc_proxy.index, soc_proxy.values, label='SoC Proxy', color=colour2)
    ax1.set_ylabel('State of Charge')
    ax1.set_xlabel('Time')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.bar(surplus.index, surplus.values, label='∆SoC', color=colour3)
    ax2.set_ylabel('∆SoC')
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper right')
    if title: ax1.set_title(title)
    fig.tight_layout()
    fig.show()
    return fig