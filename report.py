import argparse
import os

from datetime import datetime, timedelta
from typing import List, Set, Tuple, Optional

import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.signal as sp


from ibmetrics import data, metrics, reader


def print_weekly_users(builds: pandas.DataFrame, users: pandas.DataFrame, start: datetime):
    end = start + timedelta(days=7)  # one week
    week_idxs = (builds["created_at"] >= start) & (builds["created_at"] < end)
    week_users = set(builds["org_id"].loc[week_idxs])

    pre_week_idxs = (builds["created_at"] < start)
    pre_users = set(builds["org_id"].loc[pre_week_idxs])  # users seen before start day

    start_str = start.strftime("%A, %d %B %Y")
    print(f"Number of unique users for week of {start_str}: {len(week_users)}")

    new_users = week_users - pre_users
    print(f"Number of new users for week of {start_str}: {len(new_users)}")


def builds_over_time(builds: pandas.DataFrame,
                     start: datetime, end: datetime, period: timedelta) -> Tuple[np.ndarray, np.ndarray]:
    t_start = start
    bin_starts = []
    n_builds = []
    while t_start+period < end:
        idxs = (builds["created_at"] >= t_start) & (builds["created_at"] < t_start+period)
        n_builds.append(sum(idxs))
        bin_starts.append(t_start)
        t_start += period

    return np.array(bin_starts), np.array(n_builds)


def users_over_time(builds: pandas.DataFrame,
                    start: datetime, end: datetime, period: timedelta) -> Tuple[np.ndarray, np.ndarray]:
    t_start = start
    bin_starts = []
    n_users = []
    while t_start+period < end:
        idxs = (builds["created_at"] >= t_start) & (builds["created_at"] < t_start+period)
        n_users.append(len(set(builds["org_id"].loc[idxs])))
        bin_starts.append(t_start)
        t_start += period

    return np.array(bin_starts), np.array(n_users)


def read_file(fname: os.PathLike) -> pandas.DataFrame:
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = os.path.join(cache_home, "osbuild-metrics")
    os.makedirs(cache_dir, exist_ok=True)
    cache_fname = os.path.join(cache_dir, os.path.basename(os.path.splitext(fname)[0]) + ".pkl")
    if os.path.exists(cache_fname):
        print(f"Using cached pickle file at {cache_fname}")
        # TODO: handle exceptions
        return pandas.read_pickle(cache_fname)

    builds = reader.read_dump(fname)
    print(f"Saving cached pickle file at {cache_fname}")
    builds.to_pickle(cache_fname)
    return builds


def trendline(values):
    values = list(values)
    n_points = len(values)
    half = n_points//2
    kernel = sp.gaussian(n_points, std=7)
    kernel /= sum(kernel)
    # pad the original values with the last value for half the kernel size
    values = values + ([values[-1]] * half)
    tline = sp.convolve(values, kernel, mode="same")
    tline = tline[:-half]
    return tline.tolist()


def moving_average(values):
    sums = np.cumsum(values)
    weights = np.arange(1, len(sums)+1, 1)
    return sums / weights


def plot_build_counts(builds: pandas.DataFrame, start: datetime, end: datetime, p_days: int):
    t_starts, build_counts = builds_over_time(builds, start=start, end=end, period=timedelta(days=p_days))
    ax = plt.axes()
    ax.plot(t_starts, build_counts, ".b", markersize=12, label="n builds")

    builds_trend = moving_average(build_counts)
    ax.plot(t_starts, builds_trend, "-b", label="builds mov. avg.")

    ax.set_xticks(t_starts)
    # rotate xtick labels 45 degrees cw for readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    ax.axis(ymin=0, xmin=start)
    ax.set_xlabel("dates")
    ax.legend(loc="best")
    ax.grid(True)


def plot_user_counts(builds: pandas.DataFrame, start: datetime, end: datetime, p_days: int):
    t_starts, user_counts = users_over_time(builds, start=start, end=end, period=timedelta(days=p_days))
    ax = plt.axes()
    ax.plot(t_starts, user_counts, ".g", markersize=12, label="n users")

    user_trend = moving_average(user_counts)
    ax.plot(t_starts, user_trend, "-g", label="users mov. avg.")

    ax.set_xticks(t_starts)
    # rotate xtick labels 45 degrees cw for readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    ax.axis(ymin=0, xmin=start)
    ax.set_xlabel(f"beginning of {p_days} day period")
    ax.set_ylabel("")
    ax.legend(loc="best")
    ax.grid(True)


def plot_image_types(builds: pandas.DataFrame):
    image_types = builds["image_type"].value_counts()
    plt.pie(image_types.values, labels=image_types.index)


def plot_weekly_users(builds: pandas.DataFrame, start: datetime, end: datetime,
                      ax: Optional[plt.Axes] = None):
    last_date = builds["created_at"].max()

    users_so_far = set()
    n_week_users = []
    n_new_users = []

    start_dates = []

    p_start = start
    while p_start < last_date:
        end = p_start + timedelta(days=7)  # one week
        week_idxs = (builds["created_at"] >= p_start) & (builds["created_at"] < end)
        week_users = set(builds["org_id"].loc[week_idxs])

        new_users = week_users - users_so_far

        n_week_users.append(len(week_users))
        n_new_users.append(len(new_users))
        start_dates.append(p_start)

        users_so_far.update(week_users)
        p_start = end

    if not ax:
        ax = plt.axes()

    bar_shift = timedelta(days=1)
    ax.bar(np.array(start_dates)+bar_shift, n_week_users, width=2, color="blue", label="n users")
    ax.bar(start_dates, n_new_users, width=2, color="red", label="n new users")
    ax.legend(loc="best")
    start_month = start.replace(day=1)
    end_month = last_date.replace(month=last_date.month+1, day=1)
    xticks = []
    tick = start_month
    while tick <= end_month:
        xticks.append(tick)
        month = tick.month
        year = tick.year
        if month + 1 > 12:
            tick = tick.replace(year=year+1, month=1)
        else:
            tick = tick.replace(month=month+1)

    ax.set_xticks(xticks)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # rotate xtick labels 45 degrees cw for readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)


def print_frequent_packages(builds: pandas.DataFrame, limit=20):
    all_packages = []
    for pkg_list in builds["packages"]:
        all_packages.extend(set(pkg_list))

    print("## Most frequently selected packages")
    pkg_counts = pandas.value_counts(all_packages)
    for idx, (name, count) in enumerate(pkg_counts.iloc[:limit].items()):
        print(f"{idx+1:3d}. {name:40s} {count:5d}")
    print("---------------------------------")


def print_image_type_counts(builds):
    print("## Image types")
    type_counts = builds["image_type"].value_counts()
    for idx, (name, count) in enumerate(type_counts.items()):
        print(f"{idx+1:3d}. {name:40s} {count:5d}")
    print("---------------------------------")


def print_frequent_orgs(builds: pandas.DataFrame, users: pandas.DataFrame, limit=20):

    print("## Biggest orgs")
    org_counts = builds["account_number"].value_counts()
    for idx, (acc_num, count) in enumerate(org_counts.iloc[:limit].items()):
        name = acc_num
        if len(users):
            user_idx = users["accountNumber"].astype(str) == acc_num
            if sum(user_idx) == 1:
                name = users["name"][user_idx].values.item()
            elif sum(user_idx) > 1:
                raise ValueError(f"Multiple ({sum(user_idx)}) entries with same "
                                 "account_number ({acc_num}) in user data")
        print(f"{idx+1:3d}. {name:40s} {count:5d}")
    print("------------")


def get_repeat_orgs(builds: pandas.DataFrame, min_builds: int, period: timedelta) -> Set[str]:
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


def get_org_build_days(builds: pandas.DataFrame) -> pandas.DataFrame:
    """
    Org IDs associated with the dates where they had at least one build.
    """
    build_days: List[Dict[str, Any]] = []
    for org_id in builds["org_id"].unique():
        org_builds = builds.loc[builds["org_id"] == org_id]
        dates = np.unique(org_builds["created_at"].values.astype("datetime64[D]"))  # round to day
        build_days.append({"org_id": org_id, "build_dates": dates})

    return pandas.DataFrame.from_dict(build_days)


def get_active_orgs(builds: pandas.DataFrame, min_days: int, recent_limit: int) -> pandas.Series:
    """
    Returns a Series of org_ids for orgs that have builds on at least min_days separate days and the most recent one was
    after recent_limit days ago.
    """
    build_days = get_org_build_days(builds)
    counts = build_days["build_dates"].apply(len)
    build_days = build_days.loc[counts >= min_days]
    cutoff = datetime.now() - timedelta(days=recent_limit)
    most_recent_dates = build_days["build_dates"].apply(max)
    recent_idxs = most_recent_dates > cutoff
    recent_orgs = build_days["org_id"].loc[recent_idxs]
    return recent_orgs


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report from Image Builder usage data")
    parser.add_argument("data", help="File containing data dump")
    parser.add_argument("--start", default=None, help="Start date to use (older data are ignored)")
    parser.add_argument("--end", default=None, help="End date to use (newer data are ignored)")
    parser.add_argument("--userinfo", default=None, help="File containing user info (json)")
    parser.add_argument("--userfilter", default=None, help="File containing user names to remove from data")

    args = parser.parse_args()
    return args


# pylint: disable=too-many-statements,too-many-locals
def main():
    args = parse_args()
    data_fname = args.data

    builds = read_file(data_fname)
    print(f"Imported {len(builds)} records")

    users = None
    if args.userinfo:
        users = pandas.read_json(args.userinfo)

    user_filter = []
    if args.userfilter:
        with open(args.userfilter, encoding="utf-8") as filterfile:
            user_filter = filterfile.read().split("\n")

    builds = data.filter_users(builds, users, user_filter)
    print(f"{len(builds)} records after user filtering")

    if args.start:
        start = datetime.fromisoformat(args.start)
    else:
        start = builds["created_at"].min()

    if args.end:
        end = datetime.fromisoformat(args.end)
    else:
        end = builds["created_at"].max()

    builds = data.slice_time(builds, start, end)

    print(metrics.summarise(metrics.get_summary(builds)))

    print_frequent_packages(builds)
    print_image_type_counts(builds)
    print_frequent_orgs(builds, users)

    # find the last Monday before the start of the data
    first_mon = start
    while first_mon.isoweekday() != 1:
        first_mon = first_mon - timedelta(days=1)

    # plot weekly counts
    p_days = 7  # 7 day period

    img_basename = os.path.splitext(os.path.basename(data_fname))[0]

    # builds counts
    plt.figure(figsize=(16, 9), dpi=100)
    plot_build_counts(builds, first_mon, end, p_days)
    imgname = img_basename + "-builds.png"
    plt.savefig(imgname)
    print(f"Saved figure {imgname}")

    # user counts
    plt.figure(figsize=(16, 9), dpi=100)
    plot_user_counts(builds, first_mon, end, p_days)
    imgname = img_basename + "-users.png"
    plt.savefig(imgname)
    print(f"Saved figure {imgname}")

    # image type breakdown
    plt.figure(figsize=(16, 9), dpi=100)
    plot_image_types(builds)
    imgname = img_basename + "-image_types.png"
    plt.savefig(imgname)
    print(f"Saved figure {imgname}")

    plt.figure(figsize=(16, 9), dpi=100)
    plot_weekly_users(builds, first_mon, end)
    imgname = img_basename + "-weekly_users.png"
    plt.savefig(imgname)
    print(f"Saved figure {imgname}")


if __name__ == "__main__":
    main()
