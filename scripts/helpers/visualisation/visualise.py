# utility_functions/helper_visualise.py (new visualise)
from typing import List, Dict, Any, Optional,Tuple
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import mplcursors
import math

from calliope import Model as CalliopeModel
from soc_proxy.tsa_model import tsa_model
from scripts.helpers.visualisation.adapter import ModelAdapter
from scripts.helpers.visualisation.metrics import METRICS, EvalContext, _unwrap_metric_output
matplotlib.use("TkAgg")




_FIELD_MAP = {
    "Time": "time",
    "State of Charge": "soc",
    "SoC Proxy": "soc_proxy",
    "MAGMe": "MAGMe",
    "MACMe": "MACMe"
}

DEFAULT_ERROR_CMAP = "plasma"   # options: "cividis", "plasma", "viridis"
DEFAULT_GOOD_IS_LOW = True       # lower error = better

# Auto-scale colour range based on data (0 -> ceil(max to nearest step))
AUTO_SCALE_ERROR_RANGE = True
ERROR_COLOUR_ROUND_STEP = 0.2  # round max up to nearest 20%
ERROR_COLOUR_MIN_SPAN   = 0.02  # ensure at least a 2% span for visibility

def _is_tsa_model(obj) -> bool:
    # primary: class check; fallback: duck-typed attributes
    return isinstance(obj, tsa_model) or (hasattr(obj, "calliope_model") and hasattr(obj, "tsa"))


def _is_calliope_model(obj) -> bool:
    # primary: class check; fallback: duck-typed attributes used by calliope.Model
    if isinstance(obj, CalliopeModel):
        return True
    return hasattr(obj, "results") and hasattr(obj, "inputs")


def build_adapters(list_model_dict: List[Dict[str, Any]]) -> Tuple[List[ModelAdapter], Optional[ModelAdapter]]:
    adapters: List[ModelAdapter] = []
    ref_adapter: Optional[ModelAdapter] = None

    for md in list_model_dict:
        m = md["model"]
        name = md.get("name", "model")

        if _is_tsa_model(m):
            # Prefer m.tsa.type if available, else fall back to params['type']
            kind = getattr(m.tsa, "type", None) or m.calliope_model.params.get("type", "unclustered")
            params = {"name": name}
            if kind in ['cluster','optimisation','cluster_with_optimisation']:
                # your code used 'cluster' vs 'clustered' in places — normalize to 'clustered' for plotting
                kind = "clustered"
            if kind == "clustered":
                params["path_cluster_map"] = m.paths["cluster_map"]
            adapters.append(ModelAdapter(raw=m, kind=kind, params=params))

        elif _is_calliope_model(m):
            # Treat any raw calliope.Model in the list as the reference unless specified otherwise
            params = {"name": name} | md.get("params", {})
            adapter = ModelAdapter(raw=m, kind="reference", params=params)
            adapters.append(adapter)
            # If you have multiple calliope.Models and only one is the ref, you could select by name here
            if ref_adapter is None or name.lower() == "reference":
                ref_adapter = adapter

        else:
            # Print some debugging info to help if it happens again
            raise TypeError(
                f"Unknown model type in list_model_dict: {type(m)}; "
                f"keys available: {list(md.keys())}"
            )

    return adapters, ref_adapter

def _resolve_field(name: Optional[str]) -> Optional[str]:
    return None if name is None else _FIELD_MAP.get(name, name)

def _eval_metric(name: str, ctx: EvalContext):
    fn = METRICS.get(name)
    out = fn(ctx)
    return _unwrap_metric_output(out)

def visualise(
    list_model_dict: list,
    x_field: str,
    y_field: str,
    colour_field: Optional[str] = None,
    user_params: Optional[Dict[str, Any]] = None,
    show_tsa_internal_surplus_accumulation: bool = False,
    save_fig: bool = False

):
    if save_fig:
        matplotlib.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],  # fallbacks
        "font.size": 32,          # default text size
        "axes.titlesize": 36,
        "axes.labelsize": 32,
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "legend.fontsize": 28,

        # keep text editable in vector exports
        "pdf.fonttype": 42,       # embed TrueType
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        })
    
    x_field = _resolve_field(x_field)
    y_field = _resolve_field(y_field)
    colour_field = _resolve_field(colour_field)
    user_params = user_params or {}

    adapters, ref_adapter = build_adapters(list_model_dict)

    # Decide if colour_field is scalar (e.g., MAGMe); if so we'll color lines by it
    scalar_colour = False
    if colour_field:
        # pick the first non-reference adapter if possible for probing
        probe_adapter = next((a for a in adapters if a.kind != "reference"), adapters[0])
        tmp_ctx = EvalContext(model=probe_adapter, ref=ref_adapter, cache={}, user_params={})
        test_val, _ = _eval_metric(colour_field, tmp_ctx)
        scalar_colour = not isinstance(test_val, (pd.Series, pd.DataFrame))

    # Only set a cycling palette if we are NOT error-coloring the lines
    if not scalar_colour:
        matplotlib.rcParams['axes.prop_cycle'] = cycler(
            color=plt.cm.plasma(np.linspace(0.0, 0.92, len(adapters)))
        )

    fig, ax = plt.subplots(figsize=(12, 5))
    cache = {}

    colour_values = None
    sm = None     # ScalarMappable for the colorbar

    if scalar_colour:
        # 1) compute scalar error for each model
        vals = []
        for adapter in adapters:
            ctx = EvalContext(model=adapter, ref=ref_adapter, cache=cache, user_params={})
            val, extras = _eval_metric(colour_field, ctx)
            vals.append(float(val))
        colour_values = np.array(vals)        

        # 2) normalization + colormap (auto-scaled 0 → ceil(max to nearest step))
        finite_vals = colour_values[np.isfinite(colour_values)]
        max_err = float(np.nanmax(finite_vals)) if finite_vals.size else 0.0

        if AUTO_SCALE_ERROR_RANGE:
            # round up to nearest ERROR_COLOUR_ROUND_STEP (e.g., 0.10 = 10%)
            step = ERROR_COLOUR_ROUND_STEP
            vmax = step if max_err == 0 else max(step, math.ceil(max_err / step) * step)
            vmin = 0.0

            # enforce a minimum span so colours don’t collapse when errors are tiny
            if vmax - vmin < ERROR_COLOUR_MIN_SPAN:
                vmax = vmin + ERROR_COLOUR_MIN_SPAN
        else:
            # fallback fixed range
            vmin = 0.0
            vmax = 1.0

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        cmap = plt.get_cmap(DEFAULT_ERROR_CMAP)
        if not DEFAULT_GOOD_IS_LOW:
            cmap = cmap.reversed()

        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])   # required for colorbar


    lines = []
    for i, adapter in enumerate(adapters):
        ctx = EvalContext(model=adapter, ref=ref_adapter, cache=cache, user_params=user_params)

        # y
        y_metric = METRICS.get(y_field)
        if METRICS.needs_ref(y_field) and ref_adapter is None:
            raise ValueError(f"Metric '{y_field}' needs a reference but none was provided.")
        y_val, y_extras = _eval_metric(y_field, ctx)
        if isinstance(y_val, pd.DataFrame):
            y_val = y_val.iloc[:, 0]
        y_val = y_val.sort_index()

        # x
        if x_field == "time":
            x_val = y_val.index
        else:
            x_val, x_extras = _eval_metric(x_field, ctx)
            if isinstance(x_val, pd.Series):
                x_val, y_val = x_val.align(y_val, join='inner')
            else:
                raise ValueError(f"x metric '{x_field}' must be a series or 'time'.")

        # label + color
        base_label = adapter.params.get("name", f"model_{i}")

        if adapter.kind == 'reference':
            # Always grey for the reference, regardless of colour scheme
            color = 'grey'
            label = base_label
        else:
            if scalar_colour and colour_values is not None:
                val = colour_values[i]
                color = sm.to_rgba(val)
                label = f"{base_label} | {colour_field}={val*100:.1f}%"
            else:
                color = None  # fall back to cycle
                label = base_label

        z = 100 if adapter.kind == 'reference' else 1
        line, = plt.plot(x_val, y_val, label=label, zorder=z, color=color, picker=5)

        line._hover_info = {
            "Name": base_label,
            "Type": adapter.kind,
            "Peak": f"{float(np.nanmax(y_val)):.2e}",
            "Peak Date": f"{y_val.idxmax()}",
        }
        if scalar_colour and colour_values is not None and adapter.kind != 'reference':
            line._hover_info[f"{colour_field}"] = f"{val*100:.2f}%"

        # if y_field == 'soc' and adapter.kind == 'reference':
        #     plt.axhline(y=0.9*float(np.nanmax(y_val)), color='lightgrey', linestyle='--') 


        lines.append(line)
    
    # Test code for exploring how the tsa internally considers soc proxy
    if show_tsa_internal_surplus_accumulation:
        m = list_model_dict[0]['model']
        if m.tsa.type == 'optimisation' and m.tsa.params['soc_proxy']['optimisation_proxy_mode'] == 'exogenous': 
            df_internal_surpluses = m.soc_proxy.df
            df_internal_surpluses['accum_surplus'] = df_internal_surpluses['surplus_LDES'].cumsum()
            df_internal_surpluses['accum_surplus'] += df_internal_surpluses['soc_proxy_LDES'].iloc[0]
            plt.plot(df_internal_surpluses.index, df_internal_surpluses['accum_surplus'], label='TSA Internal SoC Proxy')
        else:
            raise Exception('show_tsa_internal_surplus_accumulation set to True, but tsa method is not an exogenous optimisation')

    plt.xlabel(x_field)
    plt.ylabel(y_field)
    plt.title(f'{y_field} vs. {x_field}')
    plt.grid(True)
    # plt.legend()
    ax.legend(loc="upper right")
    plt.tight_layout()

    

    # colorbar for scalar colour_field
    if scalar_colour and (sm is not None):
        # Use the *figure* to add the colorbar, and pass ax=<current axes>
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        # ticks in %
        ticks = np.linspace(sm.norm.vmin, sm.norm.vmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t*100:.0f}%" for t in ticks])
        cbar.set_label(f"{colour_field} (lower is better)" if DEFAULT_GOOD_IS_LOW
                    else f"{colour_field} (higher is better)")

    # hover
    ax = plt.gca()
    cursor = mplcursors.cursor(ax.lines, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        info = getattr(sel.artist, "_hover_info", None)
        sel.annotation.set_text("\n".join(f"{k}: {v}" for k, v in info.items()) if info else sel.artist.get_label())
        sel.annotation.get_bbox_patch().set_alpha(0.9)

    if save_fig:
        fig.set_size_inches(56/2.54, 22/2.54)       
        fig.set_dpi(300)                 # bump DPI
        plt.savefig("poster.svg", dpi=300, bbox_inches="tight")
    else:
        plt.show()