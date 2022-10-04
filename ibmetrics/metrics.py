"""
Functions for calculating metrics based on the build data.
"""
import pandas

import numpy as np

from typing import Any, Dict, List, Set, Tuple
from datetime import datetime, timedelta


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


def monthly_users(builds: pandas.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the number of unique users that appear in the data for each calendar month within the date ranges found in
    the build data.
    The second return value is an array of the start dates of each month corresponding to the counts in the first value.
    """

    month_offset = pandas.DateOffset(months=1)

    t_start = builds["created_at"].min()
    m_start = pandas.Timestamp(year=t_start.year, month=t_start.month, day=1)  # start of month of first data point

    t_end = builds["created_at"].max()
    # start of month following last data point
    m_end = pandas.Timestamp(year=t_end.year, month=t_end.month, day=1) + pandas.DateOffset(months=1)

    month_starts = []
    n_users = []
    m_current = m_start
    while m_current < m_end:
        idxs = (builds["created_at"] >= m_current) & (builds["created_at"] < m_current+month_offset)
        n_users.append(builds["org_id"].loc[idxs].nunique())
        month_starts.append(m_current)
        m_current += month_offset

    return np.array(n_users), np.array(month_starts)


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

    active_orgs = set()

    pd_period = pandas.Timedelta(period)  # convert for compatibility with numpy types

    for org in orgs:
        org_build_idxs = builds["org_id"] == org
        org_build_dates = builds["created_at"].loc[org_build_idxs]
        periods = np.diff(org_build_dates.sort_values())

        # if a sum of min_builds-1 periods is less than period, then the org is identified as a repeat/active org
        for p_idx, _ in enumerate(periods):
            p_sum = np.sum(periods[p_idx:p_idx+min_builds-1])

            if p_sum < pd_period:
                active_orgs.add(org)

    return active_orgs


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
