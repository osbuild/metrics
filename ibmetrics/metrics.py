"""
Functions for calculating metrics based on the build data.
"""
import pandas

from typing import Any, Dict


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
