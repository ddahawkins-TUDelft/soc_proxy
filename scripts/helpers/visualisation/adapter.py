# utility_functions/visual_adapters.py
from dataclasses import dataclass
import pandas as pd
import calliope
from typing import Optional, Dict, Any, Union
from soc_proxy.tsa_model import tsa_model
from soc_proxy.calliope.timeseries import (
    calliope_ts_to_pandas,
    extrapolate_ts_from_cluster_map,
)


@dataclass
class ModelAdapter:
    """Unified view over a model (clustered TSA, unclustered, or reference)."""

    raw: Union[tsa_model, calliope.Model]
    kind: str  # 'clustered', 'unclustered', or 'reference'
    params: Dict[str, Any]  # free-form; use to pass soc_proxy params, paths, etc.

    @property
    def calliope_model(self) -> calliope.Model:
        if isinstance(self.raw, tsa_model):
            return self.raw.calliope_model.model
        elif isinstance(self.raw, calliope.Model):
            return self.raw
        else:
            raise TypeError("Unsupported raw model type")

    def get_storage_soc_series(self, tech: str = "h2_salt_cavern") -> pd.Series:
        """Unified SoC: if clustered, combine inter+intra; else use standard result."""
        m = self.calliope_model
        if self.kind == "clustered":
            # build from cluster map (inter + intra) as you already do
            df_map = pd.read_csv(self.params["path_cluster_map"])
            df_map = df_map.rename(
                columns={"timesteps": "datesteps", "PeriodNum": "mapped_datesteps"}
            )
            df_map["datesteps"] = pd.to_datetime(df_map["datesteps"])
            df_map["mapped_datesteps"] = pd.to_datetime(df_map["mapped_datesteps"])

            df_intra = (
                m.results["storage"]
                .fillna(0)
                .to_series()
                .dropna()
                .to_frame("intra")
                .reset_index()
            )
            df_intra = df_intra[df_intra["techs"] == tech]
            df_intra["mapped_datesteps"] = pd.to_datetime(
                df_intra["timesteps"]
            ).dt.normalize()

            df_inter = (
                m.results["storage_inter_cluster"]
                .fillna(0)
                .to_series()
                .dropna()
                .to_frame("inter")
                .reset_index()
            )
            df_inter = df_inter[df_inter["techs"] == tech]

            df = df_inter.merge(df_map, on="datesteps", how="left")
            df = df.merge(df_intra, on="mapped_datesteps", how="left")
            t_only = df["timesteps"].dt.time
            ts = df["datesteps"].dt.normalize() + pd.to_timedelta(t_only.astype(str))
            soc = (df["inter"].fillna(0) + df["intra"].fillna(0)).rename("soc")
            soc.index = pd.DatetimeIndex(ts, name="timesteps")
            return soc.sort_index()

        else:
            df = (
                m.results["storage"]
                .fillna(0)
                .to_series()
                .dropna()
                .to_frame("soc")
                .reset_index()
            )
            df = df[df["techs"] == tech]
            df = df.set_index("timesteps").sort_index()
            return df["soc"]

    def get_timeseries_df(self) -> pd.DataFrame:
        """Raw exogenous time series for proxy generation."""
        if isinstance(self.raw, tsa_model):
            if self.kind == "clustered":
                df, _ = extrapolate_ts_from_cluster_map(
                    self.raw.paths["cluster_map"], self.raw.paths["timeseries"]
                )
            else:
                df = calliope_ts_to_pandas(
                    source=self.raw.paths["timeseries"],
                    date_range_lower_bound=f"{self.raw.calliope_model.params['date_range'][0]}-01-01",
                    date_range_upper_bound=f"{self.raw.calliope_model.params['date_range'][-1]}-12-31",
                )
        else:  # reference
            df = calliope_ts_to_pandas(
                source=self.params["path_timeseries"],
                date_range_lower_bound=f"{self.params['date_range'][0]}-01-01",
                date_range_upper_bound=f"{self.params['date_range'][-1]}-12-31",
            )
        df.set_index("timesteps", inplace=True)
        df.columns.name = None
        return df.sort_index()

    def get_capacities(self, power_techs, energy_techs=("h2_salt_cavern",)):
        """Return (power_cap: pd.Series by tech, energy_cap: pd.Series by tech)."""
        m = self.calliope_model
        power_cap = (
            m.results["flow_cap"]
            .fillna(0)
            .to_series()
            .dropna()
            .to_frame("cap")
            .reset_index()
        )
        power_cap = power_cap[
            (power_cap["techs"].isin(power_techs)) & (power_cap["carriers"] == "power")
        ]
        power_cap = (power_cap.groupby("techs")["cap"].sum()).sort_index()

        energy_cap = (
            m.results["storage_cap"]
            .fillna(0)
            .to_series()
            .dropna()
            .to_frame("cap")
            .reset_index()
        )
        energy_cap = energy_cap[energy_cap["techs"].isin(energy_techs)]
        energy_cap = (energy_cap.groupby("techs")["cap"].sum()).sort_index()
        return power_cap, energy_cap
