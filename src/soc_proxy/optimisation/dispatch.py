from ..tsa_model import tsa
import pandas as pd
import numpy as np
from . import tsa as opt
from ..clustering import ClusterResult
import multiprocessing


max_threads = multiprocessing.cpu_count()


def optimisation_dispatch(
    tsa_config: tsa,
    path_clustermap: str,
    path_timeseries: str,
    feature_weights: dict,
    soc_proxy_params: dict,
    pre_cluster_result: ClusterResult = None,
):

    result = {"cluster_map": pd.DataFrame}

    df_features = tsa_config.df_features
    features = df_features.columns.values

    # check and validate the mode
    mode = tsa_config.params["soc_proxy"]["optimisation_proxy_mode"]
    if mode not in ["endogenous", "exogenous"]:
        raise Exception(
            f"{mode} is not a valid value for tsa.params[soc_proxy][optimisation_proxy_mode]. Options are [endogenous,exogenous]"
        )

    use_soc_proxy = tsa_config.params["soc_proxy"]["use_soc_proxy"]
    proxy_inputs = tsa_config.params["soc_proxy"]["proxy_inputs_to_consider"]
    lambda_soc = float(tsa_config.params.get("lambda_soc", 0.5)["optimisation"])
    print("[TSA] Configuring MILP")

    # ======================== ENDOGENOUS MODE ========================
    if mode == "endogenous":
        print(
            f"[TSA] Dispatching MILP with endogenous variables{' including ' + ' '.join(proxy_inputs) if proxy_inputs else ''}."
        )

        # ------------------------------ INPUT CHECKS -------------------------------------

        target_columns = (
            tsa_config.params["name_demand"] + tsa_config.params["names_renewables"]
        )

        if use_soc_proxy:
            # check all non-proxy inputs are present in dataframe
            missing = []
            for feature in target_columns:
                missing.append(feature) if feature not in features else None

            if missing:
                raise Exception(
                    f"Mode is set to endogenous but the fields {missing} are missing from the dataframe of features"
                )

            # setup the removal of exogenous soc proxy features
            extra = []
            for feature in proxy_inputs:
                extra.append(feature) if feature in features else None
            if extra:
                print(
                    f"[TSA] In endogenous mode, all exogenous soc_proxy_features are removed from the optimisation as not to conflict with the endogenous generation of features. The following fields were removed from features dataframe prior to optimisation: {extra}"
                )

            df_features = df_features[target_columns]

        else:
            raise Exception(
                "Mode cannot be set to endogenous whilst use_soc_proxy is set to false"
            )

        # ------------------------------ Main -------------------------------------

        if pre_cluster_result:
            # 1) Extract indices
            C, rep_for_day, dates_index, rep_dates_kept, missing = (
                precluster_to_row_indices(
                    df_features=tsa_config.df_features,  # df features
                    representatives=pre_cluster_result.representatives,  # DatetimeIndex
                    assignment=pre_cluster_result.assignment,  # Series (optional)
                )
            )

            fix_reps_flag = len(C) == tsa_config.params["k_periods"]

            # pull surplus (N) from cache, same day order as df_features (daily)
            S_by_day = np.zeros(tsa_config._surplus_hourly_by_day.shape[0])
            if tsa_config.params["resample_to_daily_resolution"]:
                S_by_day = tsa_config._surplus_by_day.to_numpy(dtype="float64")
            else:
                S_by_day = tsa_config._surplus_hourly_by_day
            if S_by_day is None or S_by_day.shape[0] != len(tsa_config.df_features):
                raise RuntimeError(
                    "Endogenous-restricted path needs cached hourly surplus (N x 24) aligned with df_features."
                )

            eta_ch = float(
                soc_proxy_params["storage_process_losses"]["charging_efficiency"]
            )
            eta_dis = float(
                soc_proxy_params["storage_process_losses"]["discharging_efficiency"]
            )

            if tsa_config.params.get("k_periods_optimisation", ""):
                k = tsa_config.params["k_periods_optimisation"]
                print(
                    f"[TSA] Optimisation will select {k} rep days from {tsa_config.params['k_periods']} candidates identified by pre-clustering."
                )
            else:
                k = tsa_config.params["k_periods"]

            result = opt.solve_ordo_with_endogenous_soc_restricted(
                df_features=tsa_config.df_features,  # DAILY
                k=k,
                feature_weights=feature_weights,
                preferred_features=target_columns,  # same set as exogenous ORDO
                candidates=C,  # GLOBAL ids from precluster
                fix_reps=fix_reps_flag,  # True if |C|==k → pure reassignment
                warm_start_rep_for_day=rep_for_day,  # GLOBAL per-day rep, feasible start
                eta_ch=eta_ch,
                eta_dis=eta_dis,
                lambda_soc=lambda_soc,
                surplus_by_day=S_by_day,  # (N) or (N, 24) if resample_to_daily_resolution == False
                normalize="minmax_signed",
                solver="gurobi",
                MIPGap=tsa_config.params.get("mipgap", 0.01),
                threads=10,
                timelimit=tsa_config.params.get("timelimit", 1200),
                verbose=True,
                lp_method="barrier",
            )

        else:
            print("[TSA] Solving MILP with endogenous soc proxy features")

            eta_ch = float(
                soc_proxy_params["storage_process_losses"]["charging_efficiency"]
            )
            eta_dis = float(
                soc_proxy_params["storage_process_losses"]["discharging_efficiency"]
            )

            # pull the cached (N_days x 24) surplus directly ---
            # pull surplus (N) from cache, same day order as df_features (daily)
            S_by_day = np.zeros(tsa_config._surplus_hourly_by_day.shape[0])
            if tsa_config.params["resample_to_daily_resolution"]:
                S_by_day = tsa_config._surplus_by_day.to_numpy(dtype="float64")
            else:
                S_by_day = tsa_config._surplus_hourly_by_day
            if S_by_day is None or S_by_day.shape[0] != len(tsa_config.df_features):
                raise RuntimeError(
                    "Endogenous-restricted path needs cached hourly surplus (N x 24) aligned with df_features."
                )
            reference_surplus = S_by_day.reshape(-1)  # (T,)

            use_endogenous_biases = tsa_config.params.get(
                "apply_endogenous_biases", False
            )

            result = opt.solve_tsa(
                # --- Required core inputs ---
                df_features=tsa_config.df_features,  # daily rows (N), feature columns (hourly or daily)
                k=tsa_config.params["k_periods"],  # number of representatives
                feature_weights=feature_weights,
                preferred_features=None,
                normalize="minmax_signed",  # for feature blocks -> D
                # --- Candidate restriction (None => unrestricted) ---
                candidates=None,
                # --- SoC / endogenous proxy toggle & data ---
                use_soc_term=True,
                surplus_by_day=S_by_day,  # shape (N,), daily net surplus (same order as df_features)
                reference_surplus=reference_surplus,  # optional; if None, uses surplus_by_day as the reference
                eta_ch=eta_ch,  # efficiencies if you re-enable them in _build_a_and_soc_ref
                lambda_soc=lambda_soc,  # weight on SoC term when use_soc_term=True
                # --- Optional endogenous biasing ---
                use_endogenous_biases=use_endogenous_biases,
                # biases_params = None,          # params for generate_endogenous_biases(...)
                # bias_minmax = (0.0, 1.0),
                # bias_zero_threshold  = 0.25,
                # --- Optional distance normalization ---
                normalize_D_to_unit=True,  # if True, scale/clamp D to [0,1] (global)
                D_unit_method="p99_clip",
                # --- Solver knobs ---
                solver="gurobi",
                MIPGap=0.01,
                threads=max_threads - 4 if max_threads > 4 else 2,
                timelimit=tsa_config.params.get("timelimit", 1200),
                verbose=True,
                root_lp="barrier",
            )

        print("[TSA] Solution Found")

        # get index for saving
        dates_index = df_features.resample("D").agg("mean").index

        opt.save_milp_result_to_cluster_map(
            result=result, dates_index=dates_index, output_path=path_clustermap
        )

        print(f"[TSA] Saving cluster map to {path_clustermap}")

        return result

    # ======================== EXOGENOUS MODE ========================
    elif mode == "exogenous":
        # ------------------------------ INPUT CHECKS -------------------------------------

        # columns for the exogenous solver to consider, if use_soc_proxy is true, these additional columns will be captured next
        target_columns = (
            tsa_config.params["name_demand"] + tsa_config.params["names_renewables"]
        )

        if use_soc_proxy:
            missing = []
            for feature in proxy_inputs:
                missing.append(feature) if feature not in features else None

            if missing:
                raise Exception(
                    f"Mode is set to exogenous but the fields are {missing} are missing from the dataframe of features"
                )

            target_columns.extend(proxy_inputs)
        else:
            extra = []
            for feature in features:
                extra.append(feature) if feature not in target_columns else None
            if extra:
                print(
                    f"[TSA] use_soc_proxy was set to False. The following fields were removed from features dataframe prior to optimisaiton: {extra}"
                )

        # ------------------------------ Main -------------------------------------

        if pre_cluster_result:
            # 1) Extract indices
            # C, rep_for_day, dates_index, rep_dates_kept, missing = precluster_to_row_indices(
            #     df_features=tsa_config.df_features,                 # df features
            #     representatives=pre_cluster_result.representatives, # DatetimeIndex
            #     assignment=pre_cluster_result.assignment,           # Series (optional)
            # )

            # fix_reps_flag = (len(C) == tsa_config.params['k_periods'])

            # result = opt.solve_ordo_from_features_restricted(
            #     df_features=tsa_config.df_features,
            #     k=tsa_config.params['k_periods'],
            #     feature_weights=feature_weights,
            #     preferred_features=target_columns,     # as you already use for ORDO
            #     normalize="minmax_signed",
            #     candidates=C,
            #     fix_reps=fix_reps_flag,
            #     warm_start_rep_for_day=rep_for_day,    # <- feasible MIP start from clustering
            #     lp_method="barrier",
            # )

            raise NotImplementedError("Pre clustering not implemented")

        else:
            print("[TSA] Solving MILP using exogenous features only")

            result = opt.solve_tsa(
                # --- Required core inputs ---
                df_features=tsa_config.df_features,  # daily rows (N), feature columns (hourly or daily)
                k=tsa_config.params["k_periods"],  # number of representatives
                feature_weights=feature_weights,
                preferred_features=target_columns,
                normalize="minmax_signed",  # for feature blocks -> D
                # --- Candidate restriction (None => unrestricted) ---
                candidates=None,
                # --- SoC / endogenous proxy toggle & data ---
                use_soc_term=False,
                # --- Optional distance normalization ---
                normalize_D_to_unit=True,  # if True, scale/clamp D to [0,1] (global)
                D_unit_method="p99_clip",
                # --- Solver knobs ---
                solver="gurobi",
                MIPGap=0.01,
                threads=max_threads - 4 if max_threads > 4 else 2,
                timelimit=tsa_config.params.get("timelimit", 1200),
                verbose=True,
                root_lp="barrier",
            )

        print("[TSA] Solution Found")

        # get index for saving
        dates_index = df_features.resample("D").agg("mean").index

        opt.save_milp_result_to_cluster_map(
            result=result, dates_index=dates_index, output_path=path_clustermap
        )

        print(f"[TSA] Saving cluster map to {path_clustermap}")

        return result

    else:
        raise Exception("Invalid mode.")

    return result


def _normalize_dates_like(x: pd.Index | pd.Series) -> pd.DatetimeIndex:
    """
    Ensure tz-naive daily datetimes for safe equality & lookup.
    """
    dx = pd.to_datetime(x, errors="coerce")
    # strip time & tz
    if isinstance(dx, pd.DatetimeIndex):
        dx = dx.tz_localize(None) if dx.tz is not None else dx
        return dx.normalize()
    else:
        dx = dx.dt.tz_localize(None) if getattr(dx.dt, "tz", None) is not None else dx
        return dx.dt.normalize()


def precluster_to_row_indices(
    df_features: pd.DataFrame,
    representatives: pd.DatetimeIndex,
    assignment: pd.Series | None = None,
):
    """
    Map pre-cluster outputs to row indices of df_features.

    Parameters
    ----------
    df_features : DataFrame with one row per day (index should be daily dates or convertible)
    representatives : DatetimeIndex of representative days
    assignment : Series mapping original day -> representative day (optional; for hard RR export)

    Returns
    -------
    C : list[int]                 # row indices for reps (order matches 'representatives')
    rep_for_day : np.ndarray|None # N-vector of rep row index per day (for hard RR)
    dates_index : pd.DatetimeIndex# normalized daily index for df_features (useful elsewhere)
    rep_dates_kept : pd.DatetimeIndex # reps actually found in df_features (same length as C)
    missing_reps : list[pd.Timestamp]  # reps that were not found (if any)
    """
    # normalize df_features index to daily
    if not isinstance(df_features.index, pd.DatetimeIndex):
        dates_index = _normalize_dates_like(df_features.index)
        df_features = df_features.copy()
        df_features.index = dates_index
    else:
        dates_index = _normalize_dates_like(df_features.index)

    # build lookup: date -> row integer
    row_of_date = {d: i for i, d in enumerate(dates_index)}

    # normalize representatives and map to row ids
    reps_norm = _normalize_dates_like(representatives)
    C = []
    rep_dates_kept = []
    missing_reps = []
    for d in reps_norm:
        i = row_of_date.get(d, None)
        if i is None:
            missing_reps.append(d)
        else:
            C.append(i)
            rep_dates_kept.append(d)
    rep_dates_kept = pd.DatetimeIndex(rep_dates_kept)

    rep_for_day = None
    if assignment is not None:
        # normalize assignment index & values
        asg_idx = _normalize_dates_like(assignment.index)
        asg_val = _normalize_dates_like(assignment)

        # initialize with -1 to detect holes
        N = len(df_features)
        rep_for_day = np.full(N, -1, dtype=int)

        # build a fast map for rep day -> its row index
        rep_row = {d: row_of_date[d] for d in rep_dates_kept}

        # iterate through assignment; map each original day (row j) to rep row i
        for d_day, d_rep in zip(asg_idx, asg_val):
            j = row_of_date.get(d_day, None)
            i = rep_row.get(d_rep, row_of_date.get(d_rep, None))
            if j is not None and i is not None:
                rep_for_day[j] = i

        # if any remain -1 (e.g., missing in df_features), fallback: nearest calendar date or first rep
        if (rep_for_day < 0).any() and len(C) > 0:
            fallback = C[0]
            rep_for_day[rep_for_day < 0] = fallback

    return C, rep_for_day, dates_index, rep_dates_kept, missing_reps
