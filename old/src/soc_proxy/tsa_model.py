import pandas as pd
import calliope
from typing import Literal, Iterable, Dict
import copy, json, hashlib, os
from .calliope.model_config import clustered_model_config
import shutil
from .calliope.timeseries import calliope_ts_to_pandas
from .soc_proxy import generate_soc_proxy
import numpy as np
from .clustering import cluster_tsa_with_extremes
import time
import re


DEBUG_MODE = True


class tsa_model:
    def __init__(
        self,
        path_timeseries: str,
        description: str = "",
        tsa_type=Literal["cluster", "optimisation", "none"],
    ):

        # establishing the type
        if tsa_type not in (
            "cluster",
            "optimisation",
            "cluster_with_optimisation",
            "none",
        ):
            raise ValueError(f"Invalid type: {type}")

        # create the id and description for this model run
        self.id = ""
        self.description = description

        self.calliope_model = calliope_model()
        self.soc_proxy = soc_proxy()
        self.tsa = tsa(type=tsa_type)

        if not os.path.exists(path_timeseries):
            raise Exception(
                f"The provided timeseries path ({path_timeseries}) does not exist."
            )

        self.paths = {
            "directory": "",
            "subdirectories": {
                "calliope_models": "calliope_models",
                "cluster_maps": "cluster_maps",
                "timeseries": "timeseries",
                "parameters": "parameters",
            },
            "calliope_model": "",
            "cluster_map": "",
            "timeseries": path_timeseries,
            "parameters": "",
            "original_timeseries": path_timeseries,
        }

    def set_directory(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.paths["directory"] = directory

    def compute_id(self, length: int = 20, include_timeseries_hash: bool = True):
        """
        Build a stable hash from:
        - canonicalized subsets of calliope/soc_proxy/tsa params
        - an optional hash of the *source* timeseries file contents
        """
        # 1) Start from deep copies so we don't mutate your originals
        calliope_p = copy.deepcopy(self.calliope_model.params)
        soc_p = copy.deepcopy(self.soc_proxy.params)
        tsa_p = copy.deepcopy(self.tsa.params)

        # 2) Drop ephemeral keys that should NOT affect identity
        _drop_keys(
            calliope_p,
            {
                "output_model_name",
                "path_netcdf",
                "path_cluster_map",
                "path_timeseries",
                "calliope_full_log",
                "horizon_start",
                "horizon_end",
            },
        )

        # 3) Canonicalize order-insensitive fields (sort lists / dicts)
        _canonicalize_params(calliope_p)
        _canonicalize_params(soc_p)
        _canonicalize_params(tsa_p)

        # 4) Optional: include a content hash of the *source* timeseries (not the copied <id>.csv)
        payload = {
            "calliope": calliope_p,
            "soc_proxy": soc_p,
            "tsa": tsa_p,
        }
        src_ts = self.paths.get("timeseries")
        if include_timeseries_hash and src_ts and os.path.exists(src_ts):
            payload["timeseries_sha"] = _hash_file(src_ts)[
                :16
            ]  # short content fingerprint

        # 5) JSON with sorted keys + plain Python scalars (no numpy)
        blob = json.dumps(_to_jsonable(payload), sort_keys=True, separators=(",", ":"))
        self.id = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]

    # Function which generates all the path names for various aspects of the model including calliope, cluster_maps, and timeseries
    def assign_paths(self, directory: str = None):
        if directory:
            self.set_directory(directory=directory)

        if not self.paths["directory"]:
            raise Exception(
                "No high-level directory has been assigned to the model. Please assign a directory using tsa_model.set_directory() or pass the parameter into this function"
            )
        if not self.id:
            raise Exception(
                "No unique id has been assigned to this model. Please assign an id using tsa_model.compute_id()"
            )

        # assign and create the relevant directories
        if not os.path.exists(
            f"{self.paths['directory']}/{self.paths['subdirectories']['calliope_models']}"
        ):
            os.makedirs(
                f"{self.paths['directory']}/{self.paths['subdirectories']['calliope_models']}",
                exist_ok=True,
            )
        self.paths["calliope_model"] = (
            f"{self.paths['directory']}/{self.paths['subdirectories']['calliope_models']}/{self.id}.nc"
        )

        if not os.path.exists(
            f"{self.paths['directory']}/{self.paths['subdirectories']['cluster_maps']}"
        ):
            os.makedirs(
                f"{self.paths['directory']}/{self.paths['subdirectories']['cluster_maps']}",
                exist_ok=True,
            )
        self.paths["cluster_map"] = (
            f"{self.paths['directory']}/{self.paths['subdirectories']['cluster_maps']}/{self.id}.csv"
        )

        if not os.path.exists(
            f"{self.paths['directory']}/{self.paths['subdirectories']['parameters']}"
        ):
            os.makedirs(
                f"{self.paths['directory']}/{self.paths['subdirectories']['parameters']}",
                exist_ok=True,
            )
        self.paths["parameters"] = (
            f"{self.paths['directory']}/{self.paths['subdirectories']['parameters']}/{self.id}.json"
        )

        # here we copy the source timeseries data into the model's directory and reassign the path id
        # ts_dir = f"{self.paths['directory']}/{self.paths['subdirectories']['timeseries']}"
        # if not os.path.exists(ts_dir):
        #     os.makedirs(ts_dir)
        # dest = f"{ts_dir}/{self.id}.csv"
        # if not os.path.exists(dest):
        #     shutil.copy2(self.paths['timeseries'], dest)
        # else:
        #     if not os.path.exists(self.paths['cluster_map']):
        #         print('[Dispatch] timeseries file was overwritten as clustermap does not exist yet.')

        # self.paths['timeseries'] = dest

    # CALLIOPE FUNCTIONS -------------------------------------------------------------------------------------------------

    def configure_calliope(self):
        # TODO: function that exports or generates the relevant parameters for calliope

        # check id exists
        if not self.id:
            raise Exception(
                "A unique ID has not been assigned, please fully configure the model with appropriate parameters and then call tsa_model.compute_id()."
            )
        print(f"[Calliope] Validating and Configuring Calliope model: {self.id}")

        # if os.path.exists(self.paths['calliope_model']):
        #     print(f'> Calliope: Results already exist for: {self.id}, loading model instead.')
        # else:

        # checks model type and checks relevant files exist
        if not os.path.exists(self.paths["timeseries"]):
            raise Exception(
                f"The timeseries file does not exist at path: {self.paths['timeseries']}"
            )
        if self.tsa.type != "none" and not os.path.exists(self.paths["cluster_map"]):
            raise Exception(
                f"The cluster map file does not exist at path: {self.paths['cluster_map']}"
            )
        if not self.paths["calliope_model"]:
            raise Exception("A valid save path has not been configured.")

        # check calliope parameters exist and validate:
        validate_params(
            self.calliope_model.params,
            allow_extra=True,
            template={
                "type": str,
                "config_yaml_name": str,
                "date_range": list,
                "calliope_full_log": list,
            },
        )

        # generate additional parameters necessary for model
        self.calliope_model.params.update(
            {
                "horizon_start": f"{self.calliope_model.params['date_range'][0]}-01-01",
                "horizon_end": f"{self.calliope_model.params['date_range'][-1]}-12-31",
                "output_model_name": self.id,
                "path_netcdf": self.paths["calliope_model"],
                "path_cluster_map": self.paths["cluster_map"],
                "path_timeseries": self.paths["timeseries"],
            }
        )

        self.calliope_model.model = clustered_model_config(self.calliope_model.params)
        self.calliope_model.status = "configured"

    def build_calliope(self):
        if not self.paths["calliope_model"]:
            raise Exception("A valid save path has not been configured.")
        if os.path.exists(self.paths["calliope_model"]):
            print(
                f"[Calliope] Results already exist for: {self.id}, loading model instead."
            )
            self.calliope_model.model = calliope.read_netcdf(
                self.paths["calliope_model"]
            )
            self.calliope_model.status = "solved"
        else:
            if self.calliope_model.status != "configured":
                raise Exception("Calliope model has not been configured.")
            print(f"[Calliope] Building Calliope model: {self.id}")
            self.calliope_model.model.build()
            self.calliope_model.model.backend.shadow_prices.activate()  # for tracking of duals
            self.calliope_model.status = "built"

    def solve_and_save_calliope(self):
        if not self.paths["calliope_model"]:
            raise Exception("A valid save path has not been configured.")
        if os.path.exists(self.paths["calliope_model"]):
            print("[Calliope] skipping...")
            self.calliope_model.model = calliope.read_netcdf(
                self.paths["calliope_model"]
            )
            self.calliope_model.status = "solved"
        else:
            if self.calliope_model.status != "built":
                raise Exception("Calliope model has not been built.")
            print(f"[Calliope] Solving Calliope model: {self.id}", flush=True)
            self.calliope_model.model.solve()
            if self.calliope_model.model.results.nbytes > 0:
                self.calliope_model.status = "solved"
                self.calliope_model.model.to_netcdf(self.paths["calliope_model"])
            else:
                self.calliope_model.status = "failed"
                raise Exception("Calliope failed due to error")

            print(
                f"[Calliope] Solution saved to: {self.paths['calliope_model']}",
                flush=True,
            )

    # SOC Proxy FUNCTIONS -------------------------------------------------------------------------------------------------

    # none necessary

    def generate_soc_proxy_expost(self):

        if self.tsa.params["soc_proxy"]["use_soc_proxy"]:
            if self.calliope_model.params["type"] == "cluster":
                from .calliope.timeseries import extrapolate_ts_from_cluster_map

                df_timeseries, _ = extrapolate_ts_from_cluster_map(
                    self.paths["cluster_map"], self.paths["timeseries"]
                )

            else:
                from .calliope.timeseries import calliope_ts_to_pandas

                df_timeseries = calliope_ts_to_pandas(
                    source=self.paths["timeseries"],
                )

            df_timeseries.set_index("timesteps", inplace=True)
            df_timeseries.columns.name = None
            original_columns = df_timeseries.columns.values.tolist()
            original_columns.extend(
                self.tsa.params["soc_proxy"]["proxy_inputs_to_consider"]
            )
            original_columns.append("soc_proxy_LDES")

            df_timeseries, _, _ = generate_soc_proxy(
                df=df_timeseries,
                demand_field=self.tsa.params["name_demand"][0],
                renewables_fields_and_weights=self.soc_proxy.params["capacity_weights"],
                dispatchable_techs=self.soc_proxy.params["dispatchable_techs"],
                storage_process_losses=self.soc_proxy.params["storage_process_losses"],
                soc_decomposition=self.soc_proxy.params["soc_decomposition"],
                timestamp_col=None,
                margin_value=self.soc_proxy.params.get("margin_value", 0.04),
                margin_mode=self.soc_proxy.params.get("margin_mode", "fixed"),
            )
            df_timeseries = df_timeseries[original_columns]

            self.soc_proxy.df = df_timeseries

        else:
            self.soc_proxy.df = None

    # TSA FUNCTIONS -------------------------------------------------------------------------------------------------

    def compute_features_dataframe(self):

        if os.path.exists(self.paths["cluster_map"]):
            print(f"[TSA] Warning, {self.paths['cluster_map']} already exists.")
        print(f"[TSA] Extracting features dataframe for {self.id}")  # TODO:

        df_timeseries, original_columns = self._build_timeseries()
        df_timeseries = self._resample_timeseries(df_timeseries, original_columns)

        # assign the output
        self.tsa.df_features = df_timeseries

    def apply_tsa_pipeline(self):

        if os.path.exists(self.paths["cluster_map"]):
            print(
                f"[TSA] Skipping TSA as cluster map already exists at {self.paths['cluster_map']}"
            )
            return

        start_time = time.time()

        self.compute_features_dataframe()

        result = None

        weightDict = self._compute_weights_dictionary()

        if self.tsa.params["soc_proxy"]["use_soc_proxy"]:
            _soc_proxy_params = self.soc_proxy.params
        else:
            _soc_proxy_params = {}

        # CLUSTERING: Runs if mode is set to cluster or cluster with optimisation
        if self.tsa.type in ["cluster", "cluster_with_optimisation"]:
            # Compute weights dictionary

            print(
                f"[TSA]: Applying {self.tsa.params['cluster_method']} clustering method, with {self.tsa.params['representation_method']} representation."
            )
            result = cluster_tsa_with_extremes(
                df_timeseries=self.tsa.df_features,
                number_typical_periods=self.tsa.params["k_periods"],
                hours_per_period=self.tsa.params["hours_per_period"],
                cluster_method=self.tsa.params["cluster_method"],
                rep_method=self.tsa.params["representation_method"],
                path_to_original_timeseries="",
                path_to_cluster_csv=self.paths["cluster_map"],
                path_to_new_timeseries=self.paths["timeseries"],
                soc_proxy_dict=self.tsa.params["soc_proxy"],
                weightDict=weightDict,
                soc_features=self.tsa.params["soc_features"],
                extremes_spec=self.tsa.params["extremes_spec"],
                soft_prune=self.tsa.params["soft_prune"],
            )

        if self.tsa.type in ["optimisation", "cluster_with_optimisation"]:
            from .optimisation.dispatch import optimisation_dispatch

            if self.tsa.type == "optimisation":
                result = optimisation_dispatch(
                    tsa_config=self.tsa,
                    path_clustermap=self.paths["cluster_map"],
                    path_timeseries=self.paths["timeseries"],
                    pre_cluster_result=None,
                    feature_weights=weightDict,
                    soc_proxy_params=_soc_proxy_params,
                )

            else:
                # When debugging, we delete this file because its annoying to manually delete this when re-running code.
                if DEBUG_MODE:
                    os.remove(self.paths["cluster_map"])

                # ensure that df_features is in a daily format where pre-clustering may have permitted hourly clustering.
                self.tsa.df_features = self._resample_timeseries(
                    self.tsa.df_features, original_columns=[], force=True
                )

                result = optimisation_dispatch(
                    tsa_config=self.tsa,
                    path_clustermap=self.paths["cluster_map"],
                    path_timeseries=self.paths["timeseries"],
                    pre_cluster_result=result,
                    feature_weights=weightDict,
                    soc_proxy_params=_soc_proxy_params,
                )

        print(f"[TSA] {self.tsa.type} completed in {time.time() - start_time:.2f}")

        return result

    def _compute_weights_dictionary(self):
        λ_dict = self.tsa.params.get("lambda_soc", None)

        λ = max(0.0, min(1.0, float(λ_dict["cluster"])))

        names_ren = list(self.tsa.params.get("names_renewables", []))
        names_dem = list(self.tsa.params.get("name_demand", []))
        n_series = len(names_ren) + len(names_dem)

        # If there are no demand/renewables:
        if n_series == 0:
            if λ == 0.0:
                # Nothing would have positive weight → fail fast with a helpful error
                raise ValueError(
                    "lambda_proxy=0 but no demand/renewable series found; all weights would be zero."
                )
            # Otherwise just put all weight on proxy and move on
            w_each = 0.0
        else:
            w_each = (1.0 - λ) / n_series

        weightDict = {
            **{r: w_each for r in names_ren},
            **{d: w_each for d in names_dem},
        }

        if self.tsa.params.get("soc_proxy", {}).get("use_soc_proxy", False):
            proxy_cols = list(
                self.tsa.params["soc_proxy"].get("proxy_inputs_to_consider", [])
            )
            if not proxy_cols:
                raise KeyError(
                    "soc_proxy.use_soc_proxy=True but 'proxy_inputs_to_consider' is empty."
                )
            for proxy_param in proxy_cols:
                weightDict[proxy_param] = λ

        return weightDict

    def _build_timeseries(self):

        # load the timeseries
        df_timeseries = calliope_ts_to_pandas(
            self.paths["timeseries"],
            date_range_lower_bound=f"{self.calliope_model.params['date_range'][0]}-01-01",
            date_range_upper_bound=f"{self.calliope_model.params['date_range'][-1]}-12-31",
        )
        df_timeseries.set_index("timesteps", inplace=True)
        df_timeseries.columns.name = None
        original_columns = df_timeseries.columns.values.tolist()

        # if soc proxy is to be used, append the requested column.
        if self.tsa.params["soc_proxy"]["use_soc_proxy"]:
            original_columns.extend(
                self.tsa.params["soc_proxy"]["proxy_inputs_to_consider"]
            )
            df_timeseries, _, _ = generate_soc_proxy(
                df=df_timeseries,
                demand_field=self.tsa.params["name_demand"][0],
                renewables_fields_and_weights=self.soc_proxy.params["capacity_weights"],
                dispatchable_techs=self.soc_proxy.params["dispatchable_techs"],
                storage_process_losses=self.soc_proxy.params["storage_process_losses"],
                soc_decomposition=self.soc_proxy.params["soc_decomposition"],
                timestamp_col=None,
                margin_value=self.soc_proxy.params.get("margin_value", 0.04),
                margin_mode=self.soc_proxy.params.get("margin_mode", "fixed"),
            )

            if (
                self.tsa.type in ["optimisation", "cluster_with_optimisation"]
                and self.tsa.params["soc_proxy"]["optimisation_proxy_mode"]
                == "endogenous"
            ):
                self.tsa._surplus_by_day = (
                    df_timeseries["surplus_LDES"].resample("D").sum()
                )
                self.tsa._surplus_by_day_index = self.tsa._surplus_by_day.index

                days, rows = [], []
                for day, g in df_timeseries.groupby(pd.Grouper(freq="D")):
                    if len(g) != 24:
                        continue  # skip incomplete days
                    rows.append(g["surplus_LDES"].to_numpy(dtype=float))
                    days.append(day.normalize())
                self.tsa._surplus_hourly_by_day = (
                    np.vstack(rows) if rows else np.empty((0, 24))
                )
                self.tsa._surplus_hourly_index = pd.DatetimeIndex(
                    days, name="timesteps"
                )

        return df_timeseries[original_columns], original_columns

    def _resample_timeseries(
        self, df_timeseries, original_columns, force: bool = False
    ):
        # if aggregating daily bool is True, then aggregate otherwise transpose the hourly data into daily profiles to reduce MILP load
        if self.tsa.type == "optimisation" or force:
            if self.tsa.params["resample_to_daily_resolution"]:
                # Select only numeric columns (e.g., drop metadata if present)
                df_timeseries = df_timeseries.select_dtypes(include=[np.number])
                # Group by day and apply aggregation
                df_timeseries = df_timeseries.resample("D").agg("mean")
            else:
                # Create containers
                daily_rows = []
                days = []

                # Group by day (normalize keeps midnight timestamps)
                for day, group in df_timeseries.groupby(pd.Grouper(freq="D")):
                    if len(group) != 24:
                        # Skip incomplete days (DST or edges)
                        continue
                    # Build a 24h vector per variable and concatenate
                    row = np.concatenate(
                        [group[var].to_numpy() for var in original_columns]
                    )
                    daily_rows.append(row)
                    days.append(day.normalize())

                # Build column names once
                col_names = [
                    f"{var}_h{h:02d}" for var in original_columns for h in range(24)
                ]

                # Build daily dataframe with a proper Date index
                df_timeseries = pd.DataFrame(
                    daily_rows,
                    columns=col_names,
                    index=pd.DatetimeIndex(days, name="timesteps"),
                )

        #  realign cached hourly-surplus templates to the resampled/transposed daily index ---
        if (
            hasattr(self.tsa, "_surplus_hourly_by_day")
            and hasattr(self.tsa, "_surplus_hourly_index")
            and len(self.tsa._surplus_hourly_by_day) > 0
        ):
            # Reindex to the current df_timeseries index (which is daily in both branches)
            S = pd.DataFrame(
                self.tsa._surplus_hourly_by_day,
                index=self.tsa._surplus_hourly_index,
                columns=[f"h{h:02d}" for h in range(24)],
            )
            S = S.reindex(df_timeseries.index)  # align to whatever resampling produced
            # Store back as a tight ndarray for fast use later
            self.tsa._surplus_hourly_by_day = S.to_numpy(dtype=float)
            self.tsa._surplus_hourly_index = S.index

        return df_timeseries

    # Save/Load FUNCTIONS -------------------------------------------------------------------------------------------------

    def save_params(self):

        params = {
            "calliope_params": self.calliope_model.params,
            "soc_proxy_params": self.soc_proxy.params,
            "tsa_params": self.tsa.params,
        }

        with open(self.paths["parameters"], "w") as f:
            json.dump(params, f, indent=4)  # indent=4 makes it readable

        print(f"[Model] Parameter json saved to {self.paths['parameters']}")

    def load_params(self, assign: bool = False):

        with open(self.paths["parameters"], "r") as f:
            params = json.load(f)

        # Unpack back into your variables
        calliope_params = params["calliope_params"]
        soc_proxy_params = params["soc_proxy_params"]
        tsa_params = params["tsa_params"]

        if assign:
            self.calliope_model.params = calliope_params
            self.soc_proxy.params = soc_proxy_params
            self.tsa.params = tsa_params
            print(
                f"[Model]  Parameter json loaded from {self.paths['parameters']} and assigned to model"
            )
        else:
            print(f"[Model]  Parameter json loaded from {self.paths['parameters']}")

        return calliope_params, soc_proxy_params, tsa_params


class calliope_model:
    def __init__(self, params: dict = None):

        self.params = params if params else {}
        self.timeseries = {
            "df": pd.DataFrame,
            "paths": [],
        }
        self.model = calliope.Model  # calliope model
        self.path = ""
        self.status = ""

    def set_params(self, params: dict):
        self.params = params

    def update_param(self, param: str, value):
        self.params[param] = value

    def set_timeseries(self, timeseries: pd.DataFrame, path):
        self.timeseries["df"] = timeseries
        if self.timeseries["paths"][-1] != path:  # if its already there, ignore
            self.timeseries["paths"].append(path)

    def get_latest_timeseries_path(self):
        return self.timeseries["paths"][-1]


class soc_proxy:
    def __init__(self, params: dict = {}):
        self.params = params
        self.df = None

    # function sets the parameters variable
    def set_params(self, params: dict):
        self.params = params


class tsa:
    def __init__(
        self,
        type: Literal["cluster", "optimisation", "cluster_with_optimisation", "none"],
    ):

        # establishing the type
        if type not in ("cluster", "optimisation", "cluster_with_optimisation", "none"):
            raise ValueError(f"Invalid type: {type}")
        self.type = type
        self.status = False

        if type != "none":
            self.params = {}
            self.cluster_map = pd.DataFrame
            self.df_features = pd.DataFrame
            self.distance_matrix = pd.DataFrame

    # function sets the parameters variable
    def set_params(self, params):
        if self.type == "none":
            raise Exception(
                "tsa is set to type none, tsa parameters cannot be assigned."
            )
        else:
            self.params = params

    # function that saves the cluster map results and save path
    def set_cluster_map(self, cluster_map: pd.DataFrame):
        if self.type == "none":
            raise Exception(
                "tsa is set to type none, tsa parameters cannot be assigned."
            )
        else:
            self.cluster_map = cluster_map


# general function for verifying an input dictionary contains the right keys and value typès given a template
def validate_params(params: dict, template: dict, allow_extra=False):
    """
    Validate a params dict against a template dict of {key: type or tuple of types}.
    Returns True if valid, raises ValueError otherwise.
    """
    # Check missing keys
    missing = [k for k in template if k not in params]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    # Check type of each key
    wrong_type = [k for k, t in template.items() if not isinstance(params[k], t)]
    if wrong_type:
        raise TypeError(
            f"Wrong types for keys: { {k: type(params[k]).__name__ for k in wrong_type} }"
        )

    # Check extra keys
    if not allow_extra:
        extras = [k for k in params if k not in template]
        if extras:
            raise ValueError(f"Unexpected extra keys: {extras}")

    return True


def _drop_keys(d: dict, keys: set[str]):
    for k in list(d.keys()):
        if k in keys:
            d.pop(k, None)


def _canonicalize_params(obj):
    """
    In-place canonicalization:
      - sort dict keys (handled later by json sort_keys, but we also normalize nested dicts)
      - sort lists for *known* order-insensitive fields
      - convert numpy scalars to Python
      - round floats (optional)
    """
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            # Known list fields that should be treated as sets
            if k in {"names_renewables", "name_demand"} and isinstance(v, list):
                obj[k] = sorted(map(str, v))
            # Known nested list to treat as sets
            if k == "soc_proxy" and isinstance(v, dict):
                if "proxy_inputs_to_consider" in v and isinstance(
                    v["proxy_inputs_to_consider"], list
                ):
                    v["proxy_inputs_to_consider"] = sorted(
                        map(str, v["proxy_inputs_to_consider"])
                    )
            # Dicts whose *item order* shouldn't matter
            if k in {"capacity_weights"} and isinstance(v, dict):
                obj[k] = dict(sorted((str(kk), float(vv)) for kk, vv in v.items()))
            # Recurse
            _canonicalize_params(v)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            _canonicalize_params(obj[i])
    elif isinstance(obj, (np.floating, np.integer)):
        # Convert numpy scalars to builtin
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


def _to_jsonable(x):
    """Convert numpy/pandas types to JSON-safe Python scalars/strings."""
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    # If you have pd.Timestamp in params:
    try:
        import pandas as pd

        if isinstance(x, pd.Timestamp):
            return x.isoformat()
    except Exception:
        pass
    return x


def _hash_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
