# utility_functions/class_targets.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING, List
import pandas as pd
import numpy as np

from soc_proxy.calliope.timeseries import calliope_ts_to_pandas, extrapolate_ts_from_cluster_map
from soc_proxy import generate_soc_proxy

if TYPE_CHECKING:
    from soc_proxy.tsa_model import tsa_model


@dataclass
class TargetRegistry:
    model: "tsa_model"
    cache: Dict[str, pd.Series] = field(default_factory=dict)

    # ---------------- base loaders ----------------
    def _hourly_timeseries(self) -> pd.DataFrame:
        # Test-model data (clustered models get extrapolated back to full horizon)
        if self.model.tsa.type == "cluster":
            df, _ = extrapolate_ts_from_cluster_map(
                self.model.paths["cluster_map"],
                self.model.paths["timeseries"]
            )
        else:
            dr = self.model.calliope_model.params.get("date_range", None)
            low = f"{dr[0]}-01-01" if dr else None
            high = f"{dr[-1]}-12-31" if dr else None
            df = calliope_ts_to_pandas(source=self.model.paths["timeseries"],
                                       date_range_lower_bound=low,
                                       date_range_upper_bound=high)
        df = df.set_index("timesteps").sort_index()
        df.columns.name = None
        return df

    def hourly(self, name: str) -> pd.Series:
        if name in self.cache:
            return self.cache[name]
        df = self._hourly_timeseries()
        if name == "soc_proxy_ldes":
            p = self.model.soc_proxy.params
            demand = self.model.tsa.params["name_demand"][0]
            df_proxy, _, _ = generate_soc_proxy(
                df=df,
                demand_field=demand,
                renewables_fields_and_weights=p["capacity_weights"],
                dispatchable_techs=p["dispatchable_techs"],
                storage_process_losses=p["storage_process_losses"],
                soc_decomposition=p["soc_decomposition"],
                timestamp_col=None,
            )
            s = df_proxy["soc_proxy_LDES"]
        else:
            if name not in df.columns:
                raise KeyError(f"Target '{name}' not found in test model timeseries.")
            s = df[name]
        self.cache[name] = s
        return s

    # ---------------- daily aggregations ----------------
    def daily(self, name: str, how: str = "mean") -> pd.Series:
        h = self.hourly(name)
        if how == "mean":
            return h.resample("D").mean()
        elif how == "sum":
            return h.resample("D").sum()
        else:
            raise ValueError(f"Unknown aggregation: {how}")

    def daily_stacked(self, names: List[str], how: str = "mean") -> pd.DataFrame:
        return pd.DataFrame({nm: self.daily(nm, how=how) for nm in names}).sort_index()

    # ---------------- daily profile (vector) ----------------
    def daily_profile(self, name: str, hours_per_period: int = 24) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by day with columns name_h00..h{H-1}.
        Assumes evenly spaced data and at least 'hours_per_period' rows per day.
        """
        s = self.hourly(name)
        # infer base step and expected steps per day
        diffs = s.index.to_series().diff().dropna()
        step = diffs.median()
        if step <= pd.Timedelta(0):
            step = pd.Timedelta(hours=1)
        expected = int(pd.Timedelta(days=1) / step)
        H = hours_per_period or expected

        rows, days = [], []
        for day, seg in s.groupby(pd.Grouper(freq="D")):
            seg = seg.iloc[:H]
            if len(seg) == H:
                rows.append(seg.to_numpy())
                days.append(day.normalize())
        cols = [f"{name}_h{h:02d}" for h in range(H)]
        return pd.DataFrame(rows, index=pd.DatetimeIndex(days, name="timesteps"), columns=cols)

    def daily_profile_stacked(self, names: List[str], hours_per_period: int = 24) -> pd.DataFrame:
        parts = [self.daily_profile(nm, hours_per_period=hours_per_period) for nm in names]
        return pd.concat(parts, axis=1).sort_index()


@dataclass
class ReferenceTargetRegistry(TargetRegistry):
    """Loads targets from the *reference* (original) timeseries by default."""
    ref_cfg: Dict = field(default_factory=dict)

    def _hourly_timeseries(self) -> pd.DataFrame:
        # Default to original (pre-cluster) data saved in the model
        src = (self.ref_cfg or {}).get("path_timeseries") \
              or self.model.paths.get("original_timeseries") \
              or self.model.paths["timeseries"]

        # date range: default to the model's date_range
        dr = (self.ref_cfg or {}).get("date_range", self.model.calliope_model.params.get("date_range", None))
        low = f"{dr[0]}-01-01" if dr else None
        high = f"{dr[-1]}-12-31" if dr else None

        df = calliope_ts_to_pandas(source=src,
                                   date_range_lower_bound=low,
                                   date_range_upper_bound=high)
        df = df.set_index("timesteps").sort_index()
        df.columns.name = None
        return df

    def hourly(self, name: str) -> pd.Series:
        if name in self.cache:
            return self.cache[name]
        df = self._hourly_timeseries()
        if name == "soc_proxy_ldes":
            cfg = self.ref_cfg or {}
            p = cfg.get("soc_proxy_params", self.model.soc_proxy.params)
            demand = cfg.get("demand_field", "demand_power")
            df_proxy, _, _ = generate_soc_proxy(
                df=df,
                demand_field=demand,
                renewables_fields_and_weights=p["capacity_weights"],
                dispatchable_techs=p["dispatchable_techs"],
                storage_process_losses=p["storage_process_losses"],
                soc_decomposition=p["soc_decomposition"],
                timestamp_col=None,
            )
            s = df_proxy["soc_proxy_LDES"]
        else:
            if name not in df.columns:
                raise KeyError(f"Reference target '{name}' not found in original timeseries.")
            s = df[name]
        self.cache[name] = s
        return s
