import calliope
from pathlib import Path
from netCDF4 import Dataset
import  yaml


def read_clustered_netcdf(path):

    p = Path(path).resolve()
    if not p.is_file():
        raise Exception('File does not exist at: {path}')
    with Dataset(p, "a") as nc:  # append mode
        g = nc.groups["attrs"]
        cfg = yaml.safe_load(g.getncattr("config"))
        cfg.get("init", {}).pop("time_cluster", None)  # remove lingering clustering
        g.setncattr("config", yaml.safe_dump(cfg))

    return calliope.read_netcdf(path)

def hotfix_unify_clusters_universe(m) -> "xr.Dataset":
    """
    Unify the 'clusters' label universe so that both 'timestep_cluster' and
    'lookup_datestep_cluster' use the SAME coord labels. Expands the coord if needed
    and reindexes all cluster-indexed variables.

    Call this BEFORE model.build().

    Returns the updated xarray Dataset (inputs) that was written back.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr
    import inspect


    # --- 1) Locate the inputs dataset (robustly) ------------------------------
    ds = None
    owners = []

    # Most likely places in dev7, BEFORE build:
    try:
        ds = getattr(m.calliope_model.model, "inputs", None)
        if ds is not None:
            owners.append(("calliope_model.model", m.calliope_model.model, "inputs"))
    except Exception:
        pass

    # Some builds keep a private _model_data dataset pre-build:
    if ds is None:
        try:
            ds = getattr(m.calliope_model.model, "_model_data", None)
            if ds is not None:
                owners.append(("calliope_model.model", m.calliope_model.model, "_model_data"))
        except Exception:
            pass

    # AFTER build(), inputs also exists under backend; we try but we won't require it:
    if ds is None:
        try:
            backend = getattr(m.calliope_model.model, "backend", None)
            if backend is not None:
                ds = getattr(backend, "inputs", None)
                if ds is not None:
                    owners.append(("calliope_model.model.backend", backend, "inputs"))
        except Exception:
            pass

    if ds is None or not isinstance(ds, xr.Dataset):
        raise RuntimeError(
            "Could not locate the model inputs xarray.Dataset. "
            "Make sure you call this *before* build() and that m.calliope_model.model exists."
        )

    # --- 2) Sanity checks -----------------------------------------------------
    required = ["timestep_cluster", "lookup_datestep_cluster"]
    for r in required:
        if r not in ds:
            raise RuntimeError(f"Inputs is missing required variable '{r}'. Found vars: {list(ds.data_vars)}")
    if "clusters" not in ds.coords:
        raise RuntimeError("Inputs is missing 'clusters' coordinate.")

    tc = ds["timestep_cluster"].values
    ldc = ds["lookup_datestep_cluster"].values
    have = ds.coords["clusters"].values

    # --- 3) Build the unified label universe ---------------------------------
    # Drop NaNs in mapping arrays
    tc_clean = tc[~pd.isna(tc)]
    ldc_clean = ldc[~pd.isna(ldc)]

    # Decide on coord dtype and how to normalize labels
    coord_dtype = have.dtype

    def _to_target_dtype(arr, target_dtype):
        arr = np.asarray(arr)
        if np.issubdtype(target_dtype, np.integer):
            # If floats, cast to int (e.g., 11.0 -> 11)
            return arr.astype("float64").astype(target_dtype, copy=False)
        elif np.issubdtype(target_dtype, np.datetime64):
            return pd.to_datetime(arr).to_numpy(dtype=target_dtype)
        else:
            # strings / objects: cast to same dtype
            return arr.astype(target_dtype, copy=False)

    tc_norm  = _to_target_dtype(tc_clean, coord_dtype)
    ldc_norm = _to_target_dtype(ldc_clean, coord_dtype)
    have_norm = _to_target_dtype(have, coord_dtype)

    # Unified, sorted unique labels
    unified = pd.Index(pd.unique(pd.Series(np.concatenate([have_norm, tc_norm, ldc_norm])))).sort_values()
    unified = unified.to_numpy()

    # If nothing to change, exit early
    if len(unified) == len(have_norm) and np.array_equal(unified, have_norm):
        print("[hotfix] clusters coord already consistent. No changes made.")
        return ds

    print(f"[hotfix] Expanding 'clusters' coord: {len(have_norm)} -> {len(unified)}")

    # --- 4) Reindex ALL variables that have a 'clusters' dimension ------------
    ds2 = ds.copy()
    ds2 = ds2.assign_coords(clusters=("clusters", unified))

    # Reindex data variables carrying the 'clusters' dimension
    for name, da in ds.data_vars.items():
        if "clusters" in da.dims:
            ds2[name] = da.reindex({"clusters": unified})

    # Optionally: reindex coordinates that might be defined over clusters (rare)
    for cname, cval in ds.coords.items():
        if cname == "clusters":
            continue
        if hasattr(cval, "dims") and "clusters" in getattr(cval, "dims", ()):
            ds2 = ds2.assign_coords({cname: cval.reindex({"clusters": unified})})

    # --- 5) Quick integrity checks -------------------------------------------
    # The mapping arrays must now be int-like (or same dtype as coord) and contain only valid labels
    def _check_mapping(varname):
        arr = ds2[varname].values
        if np.issubdtype(coord_dtype, np.integer):
            if not np.issubdtype(arr.dtype, np.integer):
                # cast in-place to coord dtype (safe)
                ds2[varname] = (ds2[varname].dims, _to_target_dtype(arr, coord_dtype))
        # Missing labels report
        missing = sorted(set(np.unique(ds2[varname].values)) - set(unified.tolist()))
        if missing:
            print(f"[hotfix][WARN] Mapping '{varname}' still contains labels not in 'clusters': {missing[:10]}")

    _check_mapping("timestep_cluster")
    _check_mapping("lookup_datestep_cluster")

    # --- 6) Write back into the model (best-effort) ---------------------------
    wrote_somewhere = False
    for label, owner, attr in owners:
        try:
            setattr(owner, attr, ds2)
            print(f"[hotfix] Wrote updated inputs to {label}.{attr}")
            wrote_somewhere = True
            # Don’t break early; write to all discovered holders to keep them in sync
        except Exception as ex:
            print(f"[hotfix] Could not write to {label}.{attr}: {ex}")

    if not wrote_somewhere:
        raise RuntimeError("Failed to write updated inputs back to the model.")

    # --- 7) Report a short summary -------------------------------------------
    def _head(a, n=37):
        try:
            return a[:n].tolist()
        except Exception:
            return list(a)[:n]

    print("[hotfix] clusters coord dtype:", ds2.coords["clusters"].values.dtype,
          "size:", ds2.dims.get("clusters"))
    print("[hotfix] clusters head:", _head(ds2.coords["clusters"].values))

    return ds2

def hotfix_normalize_timestep_selectors(m):
    """
    Ensure any arrays used as timestep slices (e.g. $final_step) contain *labels*
    from the 'timesteps' coord, not positional indices or wrong dtypes.
    Applies to likely candidates found in inputs.

    Call this BEFORE model.build().
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    # --- find the inputs dataset (pre-build) ---
    ds = None
    try:
        ds = getattr(m.calliope_model.model, "inputs", None)
    except Exception:
        pass
    if ds is None:
        try:
            ds = getattr(m.calliope_model.model, "_model_data", None)
        except Exception:
            pass
    if ds is None or not isinstance(ds, xr.Dataset):
        raise RuntimeError("Could not find model inputs Dataset. Call before build().")

    if "timesteps" not in ds.coords:
        raise RuntimeError("Inputs has no 'timesteps' coordinate.")

    tcoord = ds.coords["timesteps"].values
    t_dtype = tcoord.dtype
    t_len = tcoord.size

    # Variables that often act as `$final_step` / `$previous_step` etc.
    candidates = [
        "cluster_first_timestep",
        "lookup_cluster_last_timestep",
        "lookup_datestep_last_cluster_timestep",
        "final_step",            # in case your pipeline names it directly
        "previous_step",         # ditto
    ]
    candidates = [v for v in candidates if v in ds.data_vars]

    def _to_labels_like_timesteps(vals):
        vals = np.asarray(vals)

        # Case A: already datetime-like or same dtype as timesteps
        if np.issubdtype(t_dtype, np.datetime64):
            # If vals look datetime-like (strings or datetimes), convert to the exact dtype
            try:
                out = pd.to_datetime(vals).to_numpy(dtype=t_dtype)
                return out
            except Exception:
                pass

        # Case B: integer positions -> map to labels
        if np.issubdtype(vals.dtype, np.integer):
            if ((vals >= 0) & (vals < t_len)).all():
                return tcoord[vals]
            # fall through if out of bounds

        # Case C: float positions that are actually integers (e.g., 11.0)
        if np.issubdtype(vals.dtype, np.floating):
            as_int = vals.astype("int64")
            if np.allclose(vals, as_int, equal_nan=False) and ((as_int >= 0) & (as_int < t_len)).all():
                return tcoord[as_int]

        # Case D: strings that match exactly the coord after coercion
        if vals.dtype.kind in ("U", "S", "O"):
            try:
                if np.issubdtype(t_dtype, np.datetime64):
                    out = pd.to_datetime(vals).to_numpy(dtype=t_dtype)
                    return out
                else:
                    # If timesteps are strings (rare), cast to same dtype
                    return vals.astype(t_dtype, copy=False)
            except Exception:
                pass

        # If nothing matched, return as-is and let caller error with diagnostics
        return vals

    # Apply normalization to each candidate
    for name in candidates:
        da = ds[name]
        mapped = _to_labels_like_timesteps(da.values)

        # Quick membership check
        want = np.asarray(mapped)
        have = tcoord
        missing = [x for x in np.unique(want).tolist() if x not in have.tolist()]
        if missing:
            print(f"[hotfix] WARNING: '{name}' still has labels not in 'timesteps' coord (showing up to 10): {missing[:10]}")

        # Replace if changed dtype or mapped
        try:
            ds[name] = xr.DataArray(mapped, dims=da.dims, coords=da.coords)
            # Align dtype exactly to the timesteps dtype if datetime-like
            if np.issubdtype(t_dtype, np.datetime64) and ds[name].dtype != t_dtype:
                ds[name] = ds[name].astype(t_dtype)
            print(f"[hotfix] Normalized '{name}' to timestep labels (dtype {ds[name].dtype}).")
        except Exception as ex:
            print(f"[hotfix] Could not normalize '{name}': {ex}")

    return ds

