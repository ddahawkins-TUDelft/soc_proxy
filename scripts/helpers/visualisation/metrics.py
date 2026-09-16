# utility_functions/visual_metrics.py
from typing import Callable, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from scripts.helpers.visualisation.adapter import ModelAdapter
from soc_proxy import generate_soc_proxy

@dataclass
class MetricResult:
    value: Any
    extras: Dict[str, Any] = field(default_factory=dict)

def _unwrap_metric_output(obj: Any) -> Tuple[Any, Dict[str, Any]]:
    """
    Accept raw scalars/Series/DataFrames or MetricResult.
    Always return (value, extras_dict).
    """
    if isinstance(obj, MetricResult):
        return obj.value, obj.extras
    return obj, {}

@dataclass
class EvalContext:
    model: ModelAdapter
    ref: Optional[ModelAdapter]  # can be None if not needed
    cache: Dict[str, Any]
    user_params: Dict[str, Any]  # free slot for ad-hoc settings

MetricFn = Callable[[EvalContext], Any]

class MetricRegistry:
    def __init__(self):
        self._fns: Dict[str, MetricFn] = {}
        self._needs_ref: Dict[str, bool] = {}

    def register(self, name: str, fn: MetricFn, needs_reference: bool = False):
        self._fns[name] = fn
        self._needs_ref[name] = needs_reference

    def get(self, name: str) -> MetricFn:
        return self._fns[name]

    def needs_ref(self, name: str) -> bool:
        return self._needs_ref.get(name, False)

METRICS = MetricRegistry()

def _cache(ctx: EvalContext, key: str, builder: Callable[[], Any]) -> Any:
    """
    Memoize a computed value per (metric_key, model_id).
    This prevents different models of the same 'kind' from sharing the same series.
    """
    model_id = id(ctx.model.raw)
    k = (key, model_id)
    if k not in ctx.cache:
        ctx.cache[k] = builder()
    return ctx.cache[k]

# ---------- Base variables

def _soc_series(ctx: EvalContext) -> pd.Series:
    return _cache(ctx, "soc", lambda: ctx.model.get_storage_soc_series())

METRICS.register("soc", _soc_series)

def _soc_proxy_series(ctx: EvalContext) -> pd.Series:
    def _build():
        df = ctx.model.get_timeseries_df()

        # choose params from tsa_model or from reference params
        if hasattr(ctx.model.raw, "soc_proxy"):  # tsa_model
            p = ctx.model.raw.soc_proxy.params
            demand = ctx.model.raw.tsa.params['name_demand'][0]
        else:  # reference adapter
            p = ctx.model.params['soc_proxy_params']
            demand = 'demand_power'

        df_proxy, _, _ = generate_soc_proxy(
            df=df,
            demand_field=demand,
            renewables_fields_and_weights=p['capacity_weights'],
            dispatchable_techs=p['dispatchable_techs'],
            storage_process_losses=p['storage_process_losses'],
            soc_decomposition=p['soc_decomposition'],
            timestamp_col=None
        )
        s = df_proxy['soc_proxy_LDES'].rename('soc_proxy_LDES')
        return s

    return _cache(ctx, "soc_proxy", _build)

METRICS.register("soc_proxy", _soc_proxy_series)

def _time_from_y(ctx: EvalContext) -> pd.DatetimeIndex:
    # resolved later: the visual layer can use y.index as x if x == 'time'
    raise NotImplementedError("x='time' is handled by the visual layer")

METRICS.register("time", _time_from_y)

# ---------- Example scalar metric: installed power capacity total
def _total_power_cap(ctx: EvalContext) -> float:
    power_techs = (
        list(ctx.model.raw.soc_proxy.params['capacity_weights'].keys())
        if hasattr(ctx.model.raw, "soc_proxy")
        else list(ctx.model.params['soc_proxy_params']['capacity_weights'].keys())
    )
    pcap, _ = ctx.model.get_capacities(power_techs)
    return float(pcap.sum())
METRICS.register("total_power_cap", _total_power_cap)

# ---------- Error builder utilities

def _align(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    A, B = a.sort_index(), b.sort_index()
    A2, B2 = A.align(B, join="inner")
    return A2, B2

def mae(a: pd.Series, b: pd.Series) -> float:
    A, B = _align(a, b)
    return float(np.mean(np.abs(A - B)))

# ---------- Error metrics

def _error_wrapper(series_metric_name: str, reducer=mae) -> MetricFn:
    def fn(ctx: EvalContext):
        if ctx.ref is None:
            raise ValueError(f"Metric '{series_metric_name}_error' requires a reference.")
        s_model = METRICS.get(series_metric_name)(ctx)
        # Build a separate context for the reference to compute the same underlying series
        ref_ctx = EvalContext(model=ctx.ref, ref=None, cache=ctx.cache, user_params=ctx.user_params)
        s_ref = METRICS.get(series_metric_name)(ref_ctx)
        return reducer(s_model, s_ref)
    return fn

# common errors you might need
METRICS.register("soc_error_mae", _error_wrapper("soc"), needs_reference=True)
METRICS.register("socproxy_error_mae", _error_wrapper("soc_proxy"), needs_reference=True)

# ---------- MAGMe
def _magme(ctx: EvalContext) -> float:
    """
    'Mean Absolute Generation Mix Error' example:
    Compare normalized generation mix (or installed mix) to reference.
    """
    if ctx.ref is None:
        raise ValueError("MAGMe requires a reference.")
    # Example: compare installed *power* mix proportions
    power_techs = (
        list(ctx.model.raw.soc_proxy.params['capacity_weights'].keys())
        if hasattr(ctx.model.raw, "soc_proxy")
        else list(ctx.model.params['soc_proxy_params']['capacity_weights'].keys())
    )
    pcap_m, _ = ctx.model.get_capacities(power_techs)
    pcap_r, _ = ctx.ref.get_capacities(power_techs)

    # normalize to shares, an error offset of 1e-12 is added to avoid computational errrors where reference capacity installed is 0
    e_m = pcap_m / (pcap_r+1e-12) #pcap_m / pcap_m.sum() if pcap_m.sum() else pcap_m*0 #alternative looks at share mix error
    e_r = pcap_r / (pcap_r+1e-12) #pcap_r / pcap_r.sum() if pcap_r.sum() else pcap_r*0 #alternative looks at share mix error

    # align tech sets and compute MAE across techs
    A, B = e_m.align(e_r, fill_value=0.0)
    extras = {
        "shares_model": A,          # model caps per tech
        "shares_ref": B,            # reference caps per tech
        "delta_shares": (A - B),    # error % per tech
    }
    return MetricResult(value=float(np.mean(np.abs(A - B))), extras=extras)
METRICS.register("MAGMe", _magme, needs_reference=True)

# ---------- MACMe
def _macme(ctx: EvalContext) -> float:
    """
    'Mean Absolute Capacity Mix Error' example:
    Compare normalized generation and storage mix (or installed mix) to reference.
    """
    if ctx.ref is None:
        raise ValueError("MACMe requires a reference.")
    # Example: compare installed *power* mix proportions
    power_techs = (
        list(ctx.model.raw.soc_proxy.params['capacity_weights'].keys())
        if hasattr(ctx.model.raw, "soc_proxy")
        else list(ctx.model.params['soc_proxy_params']['capacity_weights'].keys())
    )
    pcap_m, ecap_m = ctx.model.get_capacities(power_techs)
    pcap_r, ecap_r = ctx.ref.get_capacities(power_techs)

    cap_m = pd.concat([pcap_m,ecap_m])
    cap_r = pd.concat([pcap_r,ecap_r])

    # normalize to shares, an error offset of 1e-12 is added to avoid computational errrors where reference capacity installed is 0
    e_m = cap_m / (cap_r+1e-12) #pcap_m / pcap_m.sum() if pcap_m.sum() else pcap_m*0 #alternative looks at share mix error
    e_r = cap_r / (cap_r+1e-12)

    # align tech sets and compute MAE across techs
    A, B = e_m.align(e_r, fill_value=0.0)
    extras = {
        "shares_model": A,          # model caps per tech
        "shares_ref": B,            # reference caps per tech
        "delta_shares": (A - B),    # error % per tech
    }
    return MetricResult(value=float(np.mean(np.abs(A - B))), extras=extras)
METRICS.register("MACMe", _macme, needs_reference=True)
