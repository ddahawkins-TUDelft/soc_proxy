import calliope
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from pandas import Timestamp
from scipy.signal import convolve
from numpy.fft import fft, ifft, fftfreq
from soc_proxy import generate_soc_proxy
from soc_proxy.calliope import timeseries as tt

def compare_models(
        model_reference, 
        model_test, 
        df_clustermap_test_model: pd.DataFrame,
        proxy_parameters: dict = None
    ):

    # -------------------------------------------------------------
    # 
    #              technology installed capacities
    # 
    # -------------------------------------------------------------

    
    #Reference storage cap
    df_reference_storage_capacities = (   
        (model_reference.results['storage_cap'].fillna(0))
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame('reference')
        .reset_index()
        )
    df_reference_storage_capacities.drop(['nodes'], axis=1, inplace=True)
    df_reference_storage_capacities.replace(['battery','h2_salt_cavern'], ['SDES_energy','LDES_energy'], inplace=True)
    df_reference_storage_capacities.set_index('techs', inplace=True)


    #test storage cap
    df_test_storage_capacities = (
        (model_test.results['storage_cap'].fillna(0))
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame('test')
        .reset_index()
    )
    df_test_storage_capacities.drop(['nodes'], axis=1, inplace=True)
    df_test_storage_capacities.replace(['battery','h2_salt_cavern'], ['SDES_energy','LDES_energy'], inplace=True)
    df_test_storage_capacities.set_index('techs', inplace=True)



    #Reference techs cap
    df_reference_technology_capacities = (   
        (model_reference.results['flow_cap'].fillna(0))
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame('reference')
        .reset_index()
        )
    df_reference_technology_capacities=df_reference_technology_capacities[df_reference_technology_capacities['carriers'] == 'power']
    df_reference_technology_capacities=df_reference_technology_capacities[df_reference_technology_capacities['techs'] != 'demand_power']
    df_reference_technology_capacities.drop(['nodes', 'carriers'], axis=1, inplace=True)
    df_reference_technology_capacities.replace(['battery','h2_elec_conversion'], ['SDES_power','LDES_power'], inplace=True)
    df_reference_technology_capacities.set_index('techs', inplace=True)
    df_reference_technology_capacities = pd.concat([df_reference_technology_capacities,df_reference_storage_capacities])

    #test techs cap
    df_test_technology_capacities = (   
        (model_test.results['flow_cap'].fillna(0))
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame('test')
        .reset_index()
        )
    df_test_technology_capacities=df_test_technology_capacities[df_test_technology_capacities['carriers'] == 'power']
    df_test_technology_capacities=df_test_technology_capacities[df_test_technology_capacities['techs'] != 'demand_power']
    df_test_technology_capacities.drop(['nodes', 'carriers'], axis=1, inplace=True)
    df_test_technology_capacities.replace(['battery','h2_elec_conversion'], ['SDES_power','LDES_power'], inplace=True)
    df_test_technology_capacities.set_index('techs', inplace=True)
    df_test_technology_capacities = pd.concat([df_test_technology_capacities,df_test_storage_capacities])

    #isolate a single concatenated dataframe 
    df_tech_capacities = df_reference_technology_capacities.merge(right=df_test_technology_capacities, how='left', on='techs')
    df_tech_capacities['error'] = df_tech_capacities['reference']-df_tech_capacities['test']
    df_tech_capacities['error_absolute'] = np.abs(df_tech_capacities['error'])
    df_tech_capacities['error_absolute_normalised'] = np.abs(df_tech_capacities['error'] / df_tech_capacities['reference'])

    result = {
        'df_capacities': df_tech_capacities,
        'capacity_metrics': {
            'mean_absolute_capacity_error': np.mean(df_tech_capacities['error_absolute_normalised']),
            'ldes_capacity_error': df_tech_capacities.loc['LDES_energy']['error_absolute_normalised']
        }
    }

    # -------------------------------------------------------------
    # 
    #              state of charge profiles
    # 
    # -------------------------------------------------------------

    #reference soc
    df_reference_soc = (   
            (model_reference.results['storage'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('soc')
            .reset_index()
        )
    df_reference_soc=df_reference_soc[df_reference_soc['techs'] == 'h2_salt_cavern']
    df_reference_soc = df_reference_soc.set_index('timesteps')
    df_reference_soc.drop(['nodes','techs'], axis=1, inplace=True)


    #test soc
    if not df_clustermap_test_model.empty:
        
        df_clustermap_test_model = df_clustermap_test_model.rename(columns={
        'timesteps': 'datesteps',
        'PeriodNum': 'mapped_datesteps'
        })
        df_clustermap_test_model['datesteps'] = pd.to_datetime(df_clustermap_test_model['datesteps'], format='%Y-%m-%d')
        df_clustermap_test_model['mapped_datesteps'] = pd.to_datetime(df_clustermap_test_model['mapped_datesteps'], format='%Y-%m-%d')

        #pull the intracluster soc
        df_intracluster_soc = (   
                (model_test.results['storage'].fillna(0))
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
            (model_test.results['storage_inter_cluster'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('inter_soc')
            .reset_index()
        )
        df_intercluster_soc=df_intercluster_soc[df_intercluster_soc['techs'] == 'h2_salt_cavern']
        df_intracluster_soc['mapped_datesteps'] = df_intracluster_soc['mapped_datesteps'].dt.normalize()
        
        #merge everything and filter
        df_test_soc = df_intercluster_soc.merge(df_clustermap_test_model, on='datesteps', how='left')
        df_test_soc = df_test_soc.merge(df_intracluster_soc, on='mapped_datesteps', how='left')
        df_test_soc = df_test_soc[['datesteps','timesteps','inter_soc','intra_soc']]

        #create a proper measure of timestamps
        time_only = df_test_soc['timesteps'].dt.time
        df_test_soc['full_timestamp'] = df_test_soc['datesteps'].dt.normalize() + pd.to_timedelta(time_only.astype(str))
        
        
        #compute a comprehensive SoC, combining intracluster variatinos and intercluster variations
        df_test_soc['soc'] = df_test_soc['inter_soc']+df_test_soc['intra_soc']
        df_test_soc.drop(['datesteps','timesteps','inter_soc','intra_soc'], axis=1, inplace=True)
        df_test_soc.rename(columns={'full_timestamp': 'timesteps' }, inplace=True)
        df_test_soc = df_test_soc.set_index('timesteps')

    else:
        df_test_soc = (   
            (model_test.results['storage'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('soc')
            .reset_index()
        )
        df_test_soc=df_test_soc[df_test_soc['techs'] == 'h2_salt_cavern']
        df_test_soc.drop(['nodes','techs'], axis=1, inplace=True)
        df_test_soc = df_test_soc.set_index('timesteps')

    #compute error metrics

    y_pred= df_test_soc['soc']
    y_true= df_reference_soc['soc']

    e_time_full_charge, e_time_full_discharge = peak_timing_error(y_true, y_pred,)

    result['df_soc'] = df_test_soc
    result['soc_metrics'] ={
        'full_charge_datetime_error': e_time_full_charge,
        'full_discharge_datetime_error': e_time_full_discharge,
        'pearson_r': pearson_r_standardised(y_true, y_pred),
        'cosine_similarity': cos_similarity(y_true, y_pred),
        'first_derivative_correlation': first_derivative_correlation(y_true, y_pred, standardise=True),
        'normalised_rmse': rmse(y_true, y_pred)[1]
    }

    # -------------------------------------------------------------
    # 
    #              CAPEX Error
    # 
    # -------------------------------------------------------------

    df_reference_capex= (   
            (model_reference.results['cost_investment'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('reference')
            .reset_index()
        )
    df_reference_capex.drop(['costs', 'nodes'], axis=1, inplace=True)
    df_reference_capex.replace(['battery','h2_elec_conversion','h2_salt_cavern'], ['SDES_power','LDES_power','LDES_energy'], inplace=True)
    df_reference_capex=df_reference_capex[df_reference_capex['techs'] != 'demand_power']
    df_reference_capex.set_index('techs', inplace=True)

    df_test_capex= (   
            (model_test.results['cost_investment'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('test')
            .reset_index()
        )
    df_test_capex.drop(['costs', 'nodes'], axis=1, inplace=True)
    df_test_capex.replace(['battery','h2_elec_conversion','h2_salt_cavern'], ['SDES_power','LDES_power','LDES_energy'], inplace=True)
    df_test_capex=df_test_capex[df_test_capex['techs'] != 'demand_power']
    df_test_capex.set_index('techs', inplace=True)

    #isolate single frame
    df_capex = df_reference_capex.merge(right=df_test_capex, how='left', on='techs')
    df_capex['error'] = df_capex['reference']-df_capex['test']
    df_capex['error_absolute'] = np.abs(df_capex['error'])
    df_capex['error_absolute_normalised_total_reference_cost'] = df_capex['error_absolute'] / np.sum(df_capex['reference'])
    df_capex['error_absolute_normalised'] = np.abs(df_capex['error'] / df_capex['reference'])

    result['df_capex'] = df_capex
    result['capex_metrics'] = {
            'mean_absolute_capex_error': np.mean(df_capex['error_absolute_normalised']),
            'ldes_capex_error': df_capex.loc['LDES_energy']['error_absolute_normalised']
        }
    # -------------------------------------------------------------
    # 
    #              CAPEX Error
    # 
    # -------------------------------------------------------------


    if proxy_parameters:

        df_reference_soc_proxy = (
        model_reference.inputs[[
            k for k, v in model_reference.inputs.data_vars.items()
            if "timesteps" in v.dims and len(v.dims) > 1
        ]]
        .to_dataframe()
        .stack()
        .unstack("timesteps")
        .T
        )
        

        df_reference_soc_proxy.columns = [col[1] if isinstance(col, tuple) else col for col in df_reference_soc_proxy.columns]

        #apply the soc proxy for input into TSAM clustering
        df_reference_soc_proxy, _, _ = generate_soc_proxy(
                df=df_reference_soc_proxy,
                demand_field='demand_power',
                renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
                dispatchable_techs=proxy_parameters['dispatchable_techs'],
                storage_process_losses=proxy_parameters['storage_process_losses'],
                soc_decomposition = proxy_parameters['soc_decomposition'],
        )

        # #now drop the capacity factor, general surplus, dynamic soc, and SDES fields, instead retaining only the LDES Surplus field which serves as the static input: SoC stresses
        df_reference_soc_proxy.drop(columns=['mean_capacity_factor','surplus','surplus_SDES', 'soc_proxy_SDES'], inplace=True)
        df_reference_soc_proxy.drop(columns=['onshore_wind','demand_power','offshore_wind', 'solar'], inplace=True)

        #cluster soc proxy
        df_test_soc_proxy, _ = tt.extrapolate_ts_from_cluster_map(
            source_cluster_map=df_clustermap_test_model,
            source_original_ts='SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv' 
        )
        df_test_soc_proxy.set_index('timesteps', inplace=True)

        #apply the soc proxy for input into TSAM clustering
        df_test_soc_proxy, _, _ = generate_soc_proxy(
                df=df_test_soc_proxy,
                demand_field='demand_power',
                renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
                dispatchable_techs=proxy_parameters['dispatchable_techs'],
                storage_process_losses=proxy_parameters['storage_process_losses'],
                soc_decomposition = proxy_parameters['soc_decomposition'],
        )

        # #now drop the capacity factor, general surplus, dynamic soc, and SDES fields, instead retaining only the LDES Surplus field which serves as the static input: SoC stresses
        df_test_soc_proxy.drop(columns=['mean_capacity_factor','surplus','surplus_SDES', 'soc_proxy_SDES'], inplace=True)
        df_test_soc_proxy.drop(columns=['onshore_wind','demand_power','offshore_wind', 'solar'], inplace=True)

        #combine
        df_soc_proxies = df_reference_soc_proxy.merge(df_test_soc_proxy, how='left', on='timesteps',suffixes=['_reference','_test'])
        y_pred= df_soc_proxies['surplus_LDES_test']
        y_true= df_soc_proxies['surplus_LDES_reference']

        e_time_full_charge, e_time_full_discharge = peak_timing_error(y_true, y_pred,)

        result['df_soc_proxies'] = df_soc_proxies
        result['soc_proxy_metrics'] ={
            'full_charge_datetime_error': e_time_full_charge,
            'full_discharge_datetime_error': e_time_full_discharge,
            'pearson_r': pearson_r_standardised(y_true, y_pred),
            'cosine_similarity': cos_similarity(y_true, y_pred),
            'first_derivative_correlation': first_derivative_correlation(y_true, y_pred, standardise=True),
            'normalised_rmse': rmse(y_true, y_pred)[1]
        }


    return result, df_reference_soc


def first_derivative_correlation(y_true, y_pred, standardise=False):
    """
    Computes the Pearson correlation between the first derivatives of two time series.
    
    Parameters:
    - y_true: pd.Series — the reference time series
    - y_pred: pd.Series — the predicted time series
    - standardise: bool — whether to standardise the differences before correlation

    Returns:
    - float — Pearson correlation of the first derivatives
    """
    # Compute first differences
    y_true_diff = y_true.diff().dropna()
    y_pred_diff = y_pred.diff().dropna()
    
    # Align lengths
    min_len = min(len(y_true_diff), len(y_pred_diff))
    y_true_diff = y_true_diff.iloc[:min_len]
    y_pred_diff = y_pred_diff.iloc[:min_len]

    # Optionally standardise
    if standardise:
        y_true_diff = (y_true_diff - y_true_diff.mean()) / y_true_diff.std()
        y_pred_diff = (y_pred_diff - y_pred_diff.mean()) / y_pred_diff.std()
    
    # Compute Pearson correlation
    corr, _ = pearsonr(y_true_diff, y_pred_diff)
    return corr

def peak_timing_error(y_true, y_pred, field_reference: str = 'index'):

     #get id of prediction peak,
    id_peak_true = y_true.idxmax() #get id of true profile peak, representing the hour
    id_peak_pred = y_pred.idxmax()
    id_trough_true = y_true.idxmin() #get id of true profile peak, representing the hour
    id_trough_pred = y_pred.idxmin()
    
    #if index is to be used for hour in period
    if field_reference == 'index':
        peak_true = id_peak_true
        peak_pred = id_peak_pred
        trough_true = id_trough_true
        trough_pred = id_trough_pred

        #for scaling
        series_max = y_true.index.max()
        series_min = y_true.index.min()

    #if another column is to be used for hour in period
    else:
        peak_true = y_true.loc[field_reference,id_peak_true]
        peak_pred = y_pred.loc[field_reference,id_peak_pred]

        trough_true = y_true.loc[field_reference,id_trough_true]
        trough_pred = y_pred.loc[field_reference,id_trough_pred]

        series_max = y_true[field_reference].max()
        series_min = y_true[field_reference].min()

    if isinstance(peak_true,Timestamp) and isinstance(peak_pred,Timestamp):
        
        max_span_hours  = (series_max - series_min).total_seconds() 

        error_peak = (peak_true-peak_pred).total_seconds() / max_span_hours
        error_trough = (trough_true-trough_pred).total_seconds() / max_span_hours
    
    else:
        
        max_span_hours  = (series_max - series_min).total_seconds() 

        error_peak = (peak_true-peak_pred)/ max_span_hours
        error_trough = (trough_true-trough_pred) / max_span_hours
    
    return error_peak, error_trough 

def pearson_r_standardised(y_true, y_pred):

    #normalise with respect to standard dev
    ref_stardardised = (y_true-np.mean(y_true)) / np.std(y_true)
    proxy_stardardised = (y_pred-np.mean(y_pred)) / np.std(y_pred)

    #calculate pearson r coeff
    r, _ = pearsonr(ref_stardardised, proxy_stardardised)

    return r

def cos_similarity(y_true, y_pred):
    return cosine_similarity([y_true], [y_pred])[0][0]

def rmse(y_true, y_pred):
    """
    Compute RMSE (Root Mean Squared Error) between two signals.

    Parameters:
        y_true (pd.Series or 1D np.array): Ground truth values.
        y_pred (pd.Series or 1D np.array): Predicted values.

    Returns:
        float: RMSE between y_true and y_pred.
    """

    # Ensure both inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Compute RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    #normalised version
    nrmse = rmse / (y_true.max() - y_true.min())

    return rmse, nrmse

def extract_timeseries_from_calliope(model: calliope.Model):
        #extract timeseries from calliope model
    raw_data = (
    model.inputs[[
        k for k, v in model.inputs.data_vars.items()
        if "timesteps" in v.dims and len(v.dims) > 1
    ]]
    .to_dataframe()
    .stack()
    .unstack("timesteps")
    .T
    )
    

    raw_data.columns = [col[1] if isinstance(col, tuple) else col for col in raw_data.columns]

    # #apply the soc proxy for input into TSAM clustering
    # raw_data, capacity_factors, nominal_capacities = generate_soc_proxy(
    #         df=raw_data,
    #         demand_field='demand_power',
    #         renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
    #         dispatchable_techs=proxy_parameters['dispatchable_techs'],
    #         storage_process_losses=proxy_parameters['storage_process_losses'],
    #         soc_decomposition = proxy_parameters['soc_decomposition'],
    # )

    # #now drop the capacity factor, general surplus, dynamic soc, and SDES fields, instead retaining only the LDES Surplus field which serves as the static input: SoC stresses
    # raw_data.drop(columns=['mean_capacity_factor','surplus','surplus_SDES', 'soc_proxy_LDES', 'soc_proxy_SDES'], inplace=True)
    # raw_data.rename(columns={'surplus_LDES': 'soc_stresses'}, inplace=True)

    return raw_data