import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import calliope
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import Normalize
from netCDF4 import Dataset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats

# --- Utilities from your codebase ---
from soc_proxy import generate_soc_proxy
from soc_proxy.calliope.timeseries import (
    calliope_ts_to_pandas,
    extrapolate_ts_from_cluster_map,
)


#CONSTANTS:

country = 'BE'

DISPATCHABLE_BY_COUNTRY = {
    'NL': 3300,
    'IT': 0,
    'ES': 0,
    'BE': 2501,
    'GB': 9323,
}

SOC_PROXY_PARAMS = {
        'capacity_weights': {
            'solar': 1,
            'onshore_wind': 0.5, # making the baseline assumption of an even distribution between solar and wind -based products i.e. the sum of onshore and offshore wind equals solar
            'offshore_wind': 0.5 
        },
        'storage_process_losses': {
            'charging_efficiency': 0.65 * 0.99, #electrolyser efficiency * ldes injection efficiency
            'discharging_efficiency': 0.56 * 0.99 #electrolyser efficiency * ldes injection efficiency
        },
        'dispatchable_techs': {
            'known_dispatchable_capacity': DISPATCHABLE_BY_COUNTRY[country] #we know that 3.3GW nuclear makes up c.25% of 13GW mean hourly demand with a high uptime
        },
        'soc_decomposition': {
            'method': 'fft_lowpass',
            'time_horizon_hours': 24
        },
}


def _soc_from_clustered_model(m: calliope.Model, path_clustermap):


    cluster_map = pd.read_csv(path_clustermap)
    cluster_map = cluster_map.rename(columns={
    'timesteps': 'datesteps',
    'PeriodNum': 'mapped_datesteps'
    })
    cluster_map['datesteps'] = pd.to_datetime(cluster_map['datesteps'], format='%Y-%m-%d')
    cluster_map['mapped_datesteps'] = pd.to_datetime(cluster_map['mapped_datesteps'], format='%Y-%m-%d')

    #pull the intracluster soc
    df_intracluster_soc = (   
            (m.results['storage'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('intra_soc')
            .reset_index()
        )
    df_intracluster_soc=df_intracluster_soc[df_intracluster_soc['techs'] == 'h2_salt_cavern']
    df_intracluster_soc['mapped_datesteps'] = pd.to_datetime(df_intracluster_soc['timesteps'], format='%Y-%m-%d')


    #pull the intercluster soc
    df_intercluster_soc = (   
        (m.results['storage_inter_cluster'].fillna(0))
        .to_series()
        # .where(lambda x: x != 0)
        .dropna()
        .to_frame('inter_soc')
        .reset_index()
    )
    df_intercluster_soc=df_intercluster_soc[df_intercluster_soc['techs'] == 'h2_salt_cavern']
    df_intracluster_soc['mapped_datesteps'] = df_intracluster_soc['mapped_datesteps'].dt.normalize()
    
    #merge everything and filter
    df_result = df_intercluster_soc.merge(cluster_map, on='datesteps', how='left')
    df_result = df_result.merge(df_intracluster_soc, on='mapped_datesteps', how='left')
    df_result = df_result[['datesteps','timesteps','inter_soc','intra_soc']]

    #create a proper measure of timestamps
    time_only = df_result['timesteps'].dt.time
    df_result['full_timestamp'] = df_result['datesteps'].dt.normalize() + pd.to_timedelta(time_only.astype(str))    
    df_result = df_result.set_index('full_timestamp')


    #compute a comprehensive SoC, combining intracluster variatinos and intercluster variations
    df_result['soc'] = df_result['inter_soc']+df_result['intra_soc']

    return df_result['soc']

def build_signal_metrics(path_tvp, id):

    m = calliope.read_netcdf(f'SoC_proxy_TSA/data/calliope_models/{id}.nc')
    path_clustermap = f'SoC_proxy_TSA/data/cluster_maps/{id}.csv'
    is_clustered =  m.inputs.clusters.any()

    if is_clustered:
        df, _ = extrapolate_ts_from_cluster_map(
            source_cluster_map=path_clustermap,
            source_original_ts=path_tvp
        )
        df_soc = _soc_from_clustered_model(m,path_clustermap)
        x,_,_ = generate_soc_proxy(
                df=df,
                demand_field='demand_power',
                renewables_fields_and_weights= SOC_PROXY_PARAMS['capacity_weights'], 
                dispatchable_techs=SOC_PROXY_PARAMS['dispatchable_techs'],
                storage_process_losses=SOC_PROXY_PARAMS['storage_process_losses'],
                soc_decomposition = SOC_PROXY_PARAMS['soc_decomposition'],
                timestamp_col='timesteps',
                margin_value = 0.04,
                margin_mode='fixed'
            )
        df_proxy = x[['timesteps','soc_proxy_LDES']]
        df_proxy.set_index('timesteps', inplace=True)

        pearson_r = float(df_soc.corr(df_proxy['soc_proxy_LDES']))
        nrmse = float(np.sqrt(np.mean(np.square(df_soc - df_proxy['soc_proxy_LDES']))) / np.max(df_soc))
        
    else:
        raise Exception('Incomplete code')



    return pearson_r, nrmse





# path_tvp='SoC_proxy_TSA/data/timeseries/time_varying_parameters_BE.csv'
# id='02360a8cfc765f0a83da'
# years = (2006,2015)


# build_signal_metrics(path_tvp, id, years)
