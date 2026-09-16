import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from soc_proxy.calliope.timeseries import calliope_ts_to_pandas


def sample_days_by_season(year, n, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Define grouped seasons
    grouped_seasons = {
        'winter': [(1, 1), (3, 20), (12, 21), (12, 31)],
        'spring': [(3, 21), (6, 20)],
        'summer': [(6, 21), (9, 22)],
        'autumn': [(9, 23), (12, 20)]
    }

    season_names = list(grouped_seasons.keys())
    samples_per_season = n // len(season_names)
    remainder = n % len(season_names)

    sampled_dates = []

    for i, (season, ranges) in enumerate(grouped_seasons.items()):
        count = samples_per_season + (1 if i < remainder else 0)
        season_days = []
        for j in range(0, len(ranges), 2):
            start = pd.Timestamp(year=year, month=ranges[j][0], day=ranges[j][1])
            end = pd.Timestamp(year=year, month=ranges[j+1][0], day=ranges[j+1][1])
            days = pd.date_range(start, end)
            season_days.extend(days)
        sampled = np.random.choice(season_days, size=count, replace=False)
        sampled_dates.extend(sampled)

    sampled_dates = sorted(sampled_dates)
    return pd.to_datetime(sampled_dates)


def preprocess_df(source, date_lower_bound, date_upper_bound):
    df = calliope_ts_to_pandas(source, date_lower_bound, date_upper_bound)
    df['hour'] = df['timesteps'].dt.hour
    df['dayofyear'] = df['timesteps'].dt.dayofyear
    return df


def build_daily_profiles(df, feature_cols):
    daily_slices = []
    for col in feature_cols:
        daily = df.pivot(index='dayofyear', columns='hour', values=col)
        daily.columns = [f'{col}_h{h:02d}' for h in daily.columns]
        daily_slices.append(daily)
    daily_profiles = pd.concat(daily_slices, axis=1)
    return daily_profiles


def monte_carlo_tsa(df_source, date_lower_bound, date_upper_bound, feature_cols, n_rep_days, seed):
    np.random.seed(seed)

    df = preprocess_df(df_source, date_lower_bound, date_upper_bound)
    daily_profiles = build_daily_profiles(df, feature_cols)

    year = df['timesteps'].dt.year.iloc[0]
    rep_dates = sample_days_by_season(year=year, n=n_rep_days, seed=seed)
    rep_dayofyear = pd.to_datetime(rep_dates).dayofyear

    rep_profiles = daily_profiles.loc[rep_dayofyear].to_numpy()

    distances = pairwise_distances(daily_profiles.to_numpy(), rep_profiles)
    closest_idx = np.argmin(distances, axis=1)

    dayofyear_to_date = pd.date_range(start=f'{year}-01-01', periods=366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365)

    assignment = pd.DataFrame({
        'dayofyear': daily_profiles.index,
        'rep_day_label': closest_idx,
        'rep_dayofyear': [rep_dayofyear[i] for i in closest_idx],
    })

    assignment['timesteps'] = assignment['dayofyear'].apply(lambda d: dayofyear_to_date[d - 1].strftime('%Y-%m-%d'))
    assignment['PeriodNum'] = assignment['rep_dayofyear'].apply(lambda d: dayofyear_to_date[d - 1].strftime('%Y-%m-%d'))

    assignment = assignment[['timesteps', 'rep_day_label', 'PeriodNum']]
    return assignment

# def multiyear_tsa(source, clustering_method, n_clusters):
