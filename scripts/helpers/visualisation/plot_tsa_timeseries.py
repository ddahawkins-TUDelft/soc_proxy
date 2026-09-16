
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import calliope
from calliope import Model as CalliopeModel
from soc_proxy.calliope.io import read_clustered_netcdf

def _get_first_present_var(model: CalliopeModel, candidates: List[str]):
    for name in candidates:
        if name in model.results.data_vars:
            return name
    raise KeyError(f"None of the candidate result variables are present: {candidates}")

def _expand_clustered_intra_series(model: CalliopeModel, cluster_map_csv: Path,
                                   tech: str, candidates: List[str],
                                   carriers: Optional[List[str]] = None,
                                   value_name: str = "value") -> pd.Series:
    # Load cluster map: original day ('datesteps') -> representative day ('mapped_datesteps')
    df_map = pd.read_csv(cluster_map_csv)
    df_map = df_map.rename(columns={'timesteps':'datesteps','PeriodNum':'mapped_datesteps'})
    df_map['datesteps'] = pd.to_datetime(df_map['datesteps'])
    df_map['mapped_datesteps'] = pd.to_datetime(df_map['mapped_datesteps'])

    varname = _get_first_present_var(model, candidates)

    df_intra = (
        model.results[varname].fillna(0).to_series().dropna()
        .to_frame(value_name).reset_index()
    )

    # Filter to tech + optional carriers (e.g., power)
    df_intra = df_intra[df_intra['techs'] == tech].copy()
    if carriers is not None and 'carriers' in df_intra.columns:
        df_intra = df_intra[df_intra['carriers'].isin(carriers)].copy()

    # Map representative day to actual calendar day
    df_intra['mapped_datesteps'] = pd.to_datetime(df_intra['timesteps']).dt.normalize()
    df = df_map.merge(df_intra, on='mapped_datesteps', how='left')

    # Rebuild hourly timestamps on the true dates using the time-of-day from the rep day
    t_only = pd.to_datetime(df['timesteps']).dt.time
    ts = df['datesteps'].dt.normalize() + pd.to_timedelta(t_only.astype(str))
    s = df[value_name].rename(value_name)
    s.index = pd.DatetimeIndex(ts, name='timesteps')
    return s.sort_index()

def _simple_reference_series(model: CalliopeModel, tech: str,
                             candidates: List[str], carriers: Optional[List[str]] = None,
                             value_name: str = "value") -> pd.Series:
    varname = _get_first_present_var(model, candidates)
    df = (
        model.results[varname].fillna(0).to_series().dropna()
        .to_frame(value_name).reset_index()
    )
    df = df[df['techs'] == tech]
    if carriers is not None and 'carriers' in df.columns:
        df = df[df['carriers'].isin(carriers)]
    s = df.groupby('timesteps')[value_name].sum().sort_index()
    s.index.name = 'timesteps'
    return s

def _clustered_soc_series(model: CalliopeModel, cluster_map_csv: Path, storage_tech: str) -> pd.Series:
    # Keep your previous SoC stitch (inter + intra)
    df_map = pd.read_csv(cluster_map_csv)
    df_map = df_map.rename(columns={'timesteps':'datesteps','PeriodNum':'mapped_datesteps'})
    df_map['datesteps'] = pd.to_datetime(df_map['datesteps'])
    df_map['mapped_datesteps'] = pd.to_datetime(df_map['mapped_datesteps'])

    df_intra = (
        model.results['storage'].fillna(0).to_series().dropna()
        .to_frame('intra').reset_index()
    )
    df_intra = df_intra[df_intra['techs'] == storage_tech].copy()
    df_intra['mapped_datesteps'] = pd.to_datetime(df_intra['timesteps']).dt.normalize()

    df_inter = (
        model.results['storage_inter_cluster'].fillna(0).to_series().dropna()
        .to_frame('inter').reset_index()
    )
    df_inter = df_inter[df_inter['techs'] == storage_tech].copy()

    df = df_inter.merge(df_map, on='datesteps', how='left')
    df = df.merge(df_intra, on='mapped_datesteps', how='left')

    t_only = df['timesteps'].dt.time
    ts = df['datesteps'].dt.normalize() + pd.to_timedelta(t_only.astype(str))
    soc = (df['inter'].fillna(0) + df['intra'].fillna(0)).rename('soc')
    soc.index = pd.DatetimeIndex(ts, name='timesteps')
    return soc.sort_index()

def _reference_soc_series(model: CalliopeModel, storage_tech: str) -> pd.Series:
    df = (
        model.results['storage'].fillna(0).to_series().dropna()
        .to_frame('soc').reset_index()
    )
    df = df[df['techs'] == storage_tech]
    df = df.set_index('timesteps').sort_index()
    return df['soc']

def load_model(nc_path: Path) -> CalliopeModel:
    return read_clustered_netcdf(str(nc_path))

def get_series_for_models(
    cluster_id: str,
    reference_nc_path: Path,
    base_path: Path = Path("SoC_proxy_TSA/data"),
    storage_tech: str = "h2_salt_cavern",
    solar_tech: str = "solar"
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    clustered_nc = base_path / "calliope_models" / f"{cluster_id}.nc"
    cluster_map_csv = base_path / "cluster_maps" / f"{cluster_id}.csv"
    m_cluster = load_model(clustered_nc)
    m_ref = load_model(reference_nc_path)

    # Solar: use carrier production where available; reconstruct full year for clustered
    solar_candidates = ["carrier_prod", "flow_out", "flow"]  # fallbacks for different Calliope versions
    solar_c = _expand_clustered_intra_series(
        m_cluster, cluster_map_csv, tech=solar_tech,
        candidates=solar_candidates, carriers=["power"], value_name="solar"
    )
    solar_r = _simple_reference_series(
        m_ref, tech=solar_tech,
        candidates=solar_candidates, carriers=["power"], value_name="solar"
    )

    # SoC
    soc_c = _clustered_soc_series(m_cluster, cluster_map_csv=cluster_map_csv, storage_tech=storage_tech)
    soc_r = _reference_soc_series(m_ref, storage_tech=storage_tech)

    # Align indices
    solar_c, solar_r = solar_c.align(solar_r, join='inner')
    soc_c, soc_r = soc_c.align(soc_r, join='inner')

    return solar_c, solar_r, soc_c, soc_r

def plot_cluster_vs_reference_timeseries(
    cluster_id: str,
    reference_nc_path: Path = Path("SoC_proxy_TSA/data/calliope_models/standard_2016_2017_reference.nc"),
    base_path: Path = Path("SoC_proxy_TSA/data"),
    storage_tech: str = "h2_salt_cavern",
    solar_tech: str = "solar",
    title_suffix: Optional[str] = None
) -> plt.Figure:
    solar_c, solar_r, soc_c, soc_r = get_series_for_models(
        cluster_id=cluster_id,
        reference_nc_path=reference_nc_path,
        base_path=base_path,
        storage_tech=storage_tech,
        solar_tech=solar_tech
    )
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(solar_c.index, solar_c.values, label=f"Clustered ({cluster_id})", linestyle='-')
    ax1.plot(solar_r.index, solar_r.values, label="Reference", linestyle='--')
    ax1.set_ylabel("Solar generation (MW)")
    ax1.set_title("Solar generation over time")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.plot(soc_c.index, soc_c.values, label=f"Clustered ({cluster_id})", linestyle='-')
    ax2.plot(soc_r.index, soc_r.values, label="Reference", linestyle='--')
    ax2.set_ylabel("LDES SoC")
    ax2.set_title("LDES state of charge over time")
    ax2.grid(True)
    ax2.legend(loc="upper right")
    ax2.set_xlabel("Time")
    if title_suffix:
        fig.suptitle(title_suffix, y=1.02)
    fig.tight_layout()
    return fig
