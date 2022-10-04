"""
Functions for calculating metrics based on the build data.
"""
import pandas

import numpy as np

from typing import Any, Dict, Tuple


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


def get_summary(builds: pandas.DataFrame) -> Dict[str, Any]:
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


def get_monthly_users(builds: pandas.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
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
