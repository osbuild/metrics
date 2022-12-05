"""
Functions for cleaning or transforming the build data.
"""
from datetime import datetime
from typing import List

import pandas


def filter_users(builds: pandas.DataFrame, users: pandas.DataFrame, patterns: List[str]) -> pandas.DataFrame:
    """
    Filter users with name matching provided patterns from builds and return a filtered view of the data.
    """
    if users is None or not patterns:
        # no filtering possible
        return builds

    users = users.fillna({"name": "---"})

    def get_ids(value: str) -> pandas.Series:
        matching_idxs = users["name"].str.match(value, case=False)
        return users["accountNumber"].loc[matching_idxs].astype(str)

    for pattern in patterns:
        if not pattern:
            continue

        for rm_id in get_ids(pattern):
            builds = builds.loc[builds["account_number"] != rm_id]

    return builds


def get_filter_ids(users: pandas.DataFrame, patterns: List[str]) -> List[str]:
    if users is None or not patterns:
        # no filtering possible
        return []

    users = users.fillna({"name": "---"})

    filter_ids: List[str] = []
    for pattern in patterns:
        if not pattern:
            continue

        matching_idxs = users["name"].str.match(pattern, case=False)
        matching_ids = users["org_id"].loc[matching_idxs].unique()
        filter_ids.extend(matching_ids)

    return filter_ids


def filter_orgs(data: pandas.DataFrame, filter_ids: List[str]) -> pandas.DataFrame:
    """
    Removes rows from data that have an 'org_id' that's included in the filter_ids.
    """
    for f_id in filter_ids:
        keep_idxs = data["org_id"] != f_id
        data = data.loc[keep_idxs]

    return data


def slice_time(builds: pandas.DataFrame, start: datetime, end: datetime) -> pandas.DataFrame:
    """
    Return a filtered view of the data that only includes builds made between the given start and end time.
    """
    idxs = (builds["created_at"] >= start) & (builds["created_at"] <= end)
    return builds.loc[idxs]
