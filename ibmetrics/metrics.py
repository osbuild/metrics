"""
Functions for calculating metrics based on the build data.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas


def summarise(summary: Dict[str, Any]) -> str:
    """
    Return a human-readable text summary of the provided values.
    """
    out = ["Summary"]
    out += ["=======\n"]
    out += [f"Period: {summary['start']} - {summary['end']}\n"]

    out += [f"- Total builds: {summary['n builds']}"]
    out += [f"- Number of users: {summary['n users']}"]

    out += [f"- Builds with packages: {summary['n builds with packages']}"]

    out += [f"- Builds with filesystem customizations: {summary['n builds with fs customizations']}"]

    out += [f"- Builds with custom repos: {summary['n builds with custom repos']}"]
    return "\n".join(out)


def summarize(summary: Dict[str, Any]) -> str:
    """
    Alias for summarise().
    """
    return summarise(summary)


def make_summary(builds: pandas.DataFrame) -> Dict[str, Any]:
    """
    Return a dictionary that summarises the data in builds.
    The dictionary can be consumed by summarise() to create a human-readable text summary of the data.
    """
    summary = {
        "start": builds["created_at"].min(),
        "end": builds["created_at"].max(),
        "n builds": builds.shape[0],
        "n users": builds["org_id"].nunique(),
        "n builds with packages": builds["packages"].apply(bool).sum(),
        "n builds with fs customizations": builds["filesystem"].apply(bool).sum(),
        "n builds with custom repos": builds["payload_repositories"].apply(bool).sum(),
    }

    return summary


def monthly_value(builds: pandas.DataFrame, column: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the number of unique values for the given column that appear in the data for each calendar month within the
    date ranges found in the build data.
    The second return value is an array of the start dates of each month corresponding to the counts in the first value.
    """
    month_offset = pandas.DateOffset(months=1)

    t_start = builds["created_at"].min()
    m_start = pandas.Timestamp(year=t_start.year, month=t_start.month, day=1)  # start of month of first data point

    t_end = builds["created_at"].max()
    # start of month following last data point
    m_end = pandas.Timestamp(year=t_end.year, month=t_end.month, day=1) + pandas.DateOffset(months=1)

    month_starts = []
    n_values = []
    m_current = m_start
    while m_current < m_end:
        idxs = (builds["created_at"] >= m_current) & (builds["created_at"] < m_current+month_offset)
        n_values.append(builds[column].loc[idxs].nunique())
        month_starts.append(m_current)
        m_current += month_offset

    return np.array(n_values), np.array(month_starts)


def monthly_users(builds: pandas.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the number of unique users that appear in the data for each calendar month within the date ranges found in
    the build data.
    The second return value is an array of the start dates of each month corresponding to the counts in the first value.
    """
    return monthly_value(builds, "org_id")


def monthly_builds(builds: pandas.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the number of builds in the data for each calendar month within the date ranges found in the build data.
    The second return value is an array of the start dates of each month corresponding to the counts in the first value.
    """
    return monthly_value(builds, "job_id")


def monthly_new_users(builds: pandas.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the number of new users that appear each calendar month within the date ranges found in the build data.
    The second return value is an array of the start dates of each month corresponding to the counts in the first value.
    """
    # get the first build date for each org_id, then calculate monthly_users from that data only
    first_builds: List[Dict[str, Any]] = []
    for org_id in builds["org_id"].unique():
        org_builds = builds.loc[builds["org_id"] == org_id]
        first_date = org_builds["created_at"].min()
        first_builds.append({"org_id": org_id, "created_at": first_date})

    df = pandas.DataFrame.from_dict(first_builds)
    return monthly_users(df)


def value_sliding_window(builds: pandas.DataFrame, column: str, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the number of unique values for the given column that appear in the data within a sliding window of a given
    width (in days).
    The second return value is an array of the end dates for each window corresponding to each element in the first
    value.
    """
    window = pandas.Timedelta(days=window)
    t_start = builds["created_at"].min()
    t_end = builds["created_at"].max()
    step = pandas.Timedelta(days=1)  # slide the window 1 day each time

    end_dates = []
    n_values = []
    t_current = t_start + window  # start with a full window
    while t_current < t_end:
        idxs = (builds["created_at"] >= t_current-window) & (builds["created_at"] < t_current)
        n_values.append(builds[column].loc[idxs].nunique())
        end_dates.append(t_current)
        t_current += step

    return np.array(n_values), np.array(end_dates)


def builds_over_time(builds: pandas.DataFrame, period: timedelta) -> Tuple[np.ndarray, np.ndarray]:
    t_start = builds["created_at"].min()
    t_end = builds["created_at"].max()
    bin_starts = []
    n_builds = []
    while t_start+period < t_end:
        idxs = (builds["created_at"] >= t_start) & (builds["created_at"] < t_start+period)
        n_builds.append(sum(idxs))
        bin_starts.append(t_start)
        t_start += period

    return np.array(bin_starts), np.array(n_builds)


def repeat_orgs(builds: pandas.DataFrame, min_builds: int, period: timedelta) -> Set[str]:
    """
    Return a list of org_ids that have built at least 'min_builds' in a period of 'period'.
    """
    orgs = builds["org_id"].unique()

    rep_orgs = set()

    pd_period = pandas.Timedelta(period)  # convert for compatibility with numpy types

    for org in orgs:
        org_build_idxs = builds["org_id"] == org
        org_build_dates = builds["created_at"].loc[org_build_idxs]
        periods = np.diff(org_build_dates.sort_values())

        # if a sum of min_builds-1 periods is less than period, then the org is identified as a repeat/active org
        for p_idx, _ in enumerate(periods):
            p_sum = np.sum(periods[p_idx:p_idx+min_builds-1])

            if p_sum < pd_period:
                rep_orgs.add(org)

    return rep_orgs


def org_build_days(builds: pandas.DataFrame) -> pandas.DataFrame:
    """
    Org IDs associated with the dates where they had at least one build.
    """
    build_days: List[Dict[str, Any]] = []
    for org_id in builds["org_id"].unique():
        org_builds = builds.loc[builds["org_id"] == org_id]
        dates = np.unique(org_builds["created_at"].values.astype("datetime64[D]"))  # round to day
        build_days.append({"org_id": org_id, "build_dates": dates})

    return pandas.DataFrame.from_dict(build_days)


def active_orgs(builds: pandas.DataFrame, min_days: int, recent_limit: int) -> pandas.Series:
    """
    Returns a Series of org_ids for orgs that have builds on at least min_days separate days and the most recent one was
    after recent_limit days ago.
    """
    build_days = org_build_days(builds)
    counts = build_days["build_dates"].apply(len)
    build_days = build_days.loc[counts >= min_days]
    cutoff = datetime.now() - timedelta(days=recent_limit)
    most_recent_dates = build_days["build_dates"].apply(max)
    recent_idxs = most_recent_dates > cutoff
    recent_orgs = build_days["org_id"].loc[recent_idxs]
    return recent_orgs


def dau_over_mau(builds: pandas.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the ratio of daily active users (DAU) over monthly active users (MAU, 30-day sliding window) for each day in
    the data.
    The second return value is an array of the end dates for each window corresponding to each element in the first
    value.
    """
    dau, _ = value_sliding_window(builds, "org_id", 1)
    mau, mau_dates = value_sliding_window(builds, "org_id", 30)

    # slice off first 30 days of the DAUs since we don't have MAUs before that
    dau = dau[-len(mau):]
    return dau/mau, mau_dates


def footprints(builds: pandas.DataFrame) -> pandas.DataFrame:
    """
    Returns a new DataFrame that replaces the image type with a corresponding footprint.
    Footprints are groups of image types:
    - edge: rhel-edge-commit and rhel-edge-installer
    - private-cloud: vsphere and guest-image
    - bare-metal: image-installer
    - gcp: gcp
    - aws: aws
    - azure: azure and vhd
    """
    type_footprint = {
        "rhel-edge-commit": "edge",
        "rhel-edge-installer": "edge",
        "vsphere": "private-cloud",
        "guest-image": "private-cloud",
        "image-installer": "bare-metal",
        "gcp": "gcp",
        "aws": "aws",
        "azure": "azure",
        "vhd": "azure",
    }

    fp_df = builds.replace({"image_type": type_footprint})
    fp_df.rename(columns={"image_type": "footprint"}, inplace=True)
    return fp_df
