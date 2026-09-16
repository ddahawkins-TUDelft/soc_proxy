import pandas as pd
from datetime import datetime
import calliope
import numpy as np

def _format_standard_calliope_ts(df: pd.DataFrame):
    df['timesteps'] = pd.to_datetime(df['timesteps'])
    columns_to_make_numeric = df.columns.difference(['timesteps'])
    df[columns_to_make_numeric] = df[columns_to_make_numeric].apply(pd.to_numeric)

    return df

def calliope_ts_to_pandas(source: str, date_range_lower_bound: str="", date_range_upper_bound: str=""):
    # import timeseries
    df_timeseries = pd.read_csv(source, low_memory=False)
    # reformat
    index_header = df_timeseries[df_timeseries.iloc[:,0].str.contains('techs')].index[0] #identify row containing new header
    df_timeseries.iloc[index_header,0]='timesteps'
    df_timeseries.columns=df_timeseries.iloc[index_header]
    df_timeseries = df_timeseries.iloc[(index_header+3):].reset_index(drop=True) #remove first row containing comment

    #ensure python reads values in the correct format
    df_timeseries = _format_standard_calliope_ts(df_timeseries)

    #filtering out all dates outside of desired range
    if date_range_lower_bound:
        df_timeseries= df_timeseries[df_timeseries['timesteps'] >= date_range_lower_bound]
    if date_range_upper_bound:
        df_timeseries= df_timeseries[df_timeseries['timesteps'] < (pd.to_datetime(date_range_upper_bound) + pd.Timedelta(days=1))]

    return df_timeseries
    
def extrapolate_ts_from_cluster_map(source_cluster_map, source_original_ts: str):

    if isinstance(source_cluster_map, str):
        #import clustering map and apply date format to all columns
        df = pd.read_csv(source_cluster_map)
    elif isinstance(source_cluster_map, pd.DataFrame):
        df=source_cluster_map
        df.rename(columns={df.columns[0]: 'timesteps', df.columns[1]: 'PeriodNum'}, inplace=True)
    else:
        raise Exception('source_cluster_map has not been assigned a valid input')
    #capture proper date formats
    for col in df.columns:
        if df[col].dtype=='object':
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass #skips all columns that arent datetime, should be none.

    #import original timeseries
    df_original_ts = calliope_ts_to_pandas(source_original_ts)
    column_names_original = df_original_ts.columns.values

    #create date-only column in the original dataset  for joining to map
    df_original_ts['date'] = df_original_ts['timesteps'].dt.normalize()

    df = pd.merge(
        df,
        df_original_ts,
        how='left',
        left_on='PeriodNum',
        right_on='date'
    )

    #carry the time allocation from the aggregated date-time column to the original date column
    df['timesteps'] = pd.to_datetime(df['timesteps_x'].astype(str)+' '+df['timesteps_y'].dt.time.astype(str))

    #extract clustered datetimes in case that is of interest to the user
    s_clustered_datetimes = df['timesteps_y']

    #filter out the columns we are uninterested in exporting
    df = df[column_names_original]

    return df, s_clustered_datetimes

#convert datetime columns into hour of year
def convert_to_hour_of_year(timestamp_series):
    """
    Converts a Pandas Series of 'yyyy-mm-dd hh:mm:ss' strings to hour of the year (0–8759).
    
    Parameters:
        timestamp_series (pd.Series): Series of timestamps as strings.
        
    Returns:
        pd.Series: Series of integers representing the hour of the year.
    """
    # Convert to datetime
    dt = pd.to_datetime(timestamp_series)

    # Calculate hour of year
    hour_of_year = (dt - pd.Timestamp(year=dt.dt.year.min(), month=1, day=1)).dt.total_seconds() // 3600

    return hour_of_year.astype(int)

def hour_of_year_from_timestamp(timestamp):

    return int((timestamp - datetime(timestamp.year, 1, 1)).total_seconds() // 3600)

def extract_soc_from_clustered_model(results_path,ref_model, n_days, method, rep_method,soc_proxy: bool = False):

    model = calliope.read_netcdf(results_path)

    if soc_proxy:
        rep_method = rep_method + '_with_soc_proxy'
    
    #process the clustering map
    cluster_map = pd.read_csv(f"SoC_proxy_TSA/cache/cluster_maps/{ref_model}_n_{n_days}_{method}_{rep_method}.csv")
    cluster_map = cluster_map.rename(columns={
    'timesteps': 'datesteps',
    'PeriodNum': 'mapped_datesteps'
    })
    cluster_map['datesteps'] = pd.to_datetime(cluster_map['datesteps'], format='%Y-%m-%d')
    cluster_map['mapped_datesteps'] = pd.to_datetime(cluster_map['mapped_datesteps'], format='%Y-%m-%d')

    #pull the intracluster soc
    df_intracluster_soc = (   
            (model.results['storage'].fillna(0))
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
        (model.results['storage_inter_cluster'].fillna(0))
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

    return df_result

#function to get ldes capacity and power capacities from calliope result file
def get_capacities(calliope_model, type: str = 'standard'):
    df_storage_caps = (   
            (calliope_model.results['storage_cap'].fillna(0))
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame('storage_cap')
            .reset_index()
        )

    df_energy_caps = (   
            (calliope_model.results['flow_cap'].fillna(0))
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame('flow_cap')
            .reset_index()
        ) 
    
    if type == 'standard':
        df_state_of_charge = (   
            (calliope_model.results['storage'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('soc')
            .reset_index()
        )
    else:
        df_state_of_charge = (   
            (calliope_model.results['storage'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('soc')
            .reset_index()
        )
        # df_state_of_charge = (   
        #     (calliope_model.results['storage_inter_cluster'].fillna(0))
        #     .to_series()
        #     # .where(lambda x: x != 0)
        #     .dropna()
        #     .to_frame('soc')
        #     .reset_index()
        # )

        #TODO: compute the SoC profile for clustered models as per inter-cluster storage maths
        # https://calliope.readthedocs.io/en/v0.7.0.dev5/math/storage_inter_cluster/?h=inter+c#storage


    #process SoC
    df_state_of_charge=df_state_of_charge[df_state_of_charge['techs'] == 'h2_salt_cavern']
    # df_state_of_charge['timesteps']=tt.convert_to_hour_of_year(df_state_of_charge['timesteps'])
    df_state_of_charge = df_state_of_charge.set_index('timesteps')
    # df_state_of_charge['soc'] = df_state_of_charge['soc']-df_state_of_charge['soc'].iloc[0] #baseline soc to 0 starting point
    # df_state_of_charge['soc'] = df_state_of_charge['soc'] /np.mean(df_state_of_charge['soc'] ) #normalise by mean

    ldes_capacity = df_storage_caps.loc[df_storage_caps['techs'] == 'h2_salt_cavern', 'storage_cap'].iloc[0] #storage cap
    capacities = np.array(df_energy_caps.loc[df_energy_caps['carriers'] == 'power', 'flow_cap']) #array of power caps

    return ldes_capacity, capacities, df_state_of_charge