"""
Plotting functions
"""
from datetime import datetime, timedelta
from typing import Optional, Set

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas

from . import metrics


def build_counts(builds: pandas.DataFrame, p_days: int, ax: Optional[plt.Axes] = None):
    """
    Bar graph of the number of builds in a given period specified by p_days.
    """
    if not ax:
        ax = plt.axes()

    t_starts, counts = metrics.builds_over_time(builds, period=timedelta(days=p_days))
    counts_plot = ax.plot(t_starts, counts, ".", markersize=12, label="n builds")

    dot_color = counts_plot[0].get_color()
    builds_trend = _moving_average(counts)  # reuse dot colour for trendline
    ax.plot(t_starts, builds_trend, "-", color=dot_color, label="builds mov. avg.")

    ax.set_xticks(t_starts)
    ax.set_xlabel("dates")
    ax.legend(loc="best")


def _moving_average(values):
    """
    Calculate the moving average for a series of values.
    """
    sums = np.cumsum(values)
    weights = np.arange(1, len(sums)+1, 1)
    return sums / weights


def monthly_users(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    """
    Bar graph of the number of users that appear in each calendar month.
    """
    if not ax:
        ax = plt.axes()

    user_counts, months = metrics.monthly_users(builds)
    ax.bar(months, user_counts, width=20, zorder=2)
    for mo, nu in zip(months, user_counts):
        plt.text(mo, nu, str(nu), size=16, ha="center")

    xlabels = [f"{mo.month_name()} {mo.year}" for mo in months]
    ax.set_xticks(months, xlabels)
    ax.set_title("Monthly users")


def monthly_users_stacked(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    """
    Bar graph of the number of users that appear in each calendar month; new users stacked on old.
    """

    if not ax:
        ax = plt.axes()

    user_counts, months = metrics.monthly_users(builds)
    new_user_counts, months = metrics.monthly_new_users(builds)
    old_user_counts = user_counts - new_user_counts
    ax.bar(months, new_user_counts, width=27, bottom=old_user_counts)
    ax.bar(months, old_user_counts, width=27, label="Recurring")

    font_size = matplotlib.rcParams["font.size"]
    for mo, count in zip(months, user_counts):
        ax.text(mo, count-font_size/2, str(count), size=font_size*1.2, ha="center", va="top", color="white")

    xlabels = [f"{mo.month_name()}" for mo in months]
    ax.set_xticks(months, xlabels)
    ax.set_title("Organizations building at least one image")
    plt.figtext(0.1, -0.2, "Source: Image Builder Production Database",
                wrap=False, horizontalalignment='left', color="#9a9a9a")
    ax.legend()


def monthly_builds(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    """
    Bar graph of the number of builds in each calendar month.
    """
    if not ax:
        ax = plt.axes()

    counts, months = metrics.monthly_builds(builds)
    ax.bar(months, counts, width=20, zorder=2)
    for mo, nu in zip(months, counts):
        plt.text(mo, nu, str(nu), size=16, ha="center")

    xlabels = [f"{mo.month_name()} {mo.year}" for mo in months]
    ax.set_xticks(months, xlabels)
    ax.set_title("Monthly builds")


def monthly_new_users(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    """
    Bar graph of the number of new users that appear in each calendar month.
    """
    if not ax:
        ax = plt.axes()

    user_counts, months = metrics.monthly_new_users(builds)
    ax.bar(months, user_counts, width=20, zorder=2)
    for mo, nu in zip(months, user_counts):
        plt.text(mo, nu, str(nu), size=16, ha="center")

    xlabels = [f"{mo.month_name()} {mo.year}" for mo in months]
    ax.set_xticks(months, xlabels)
    ax.set_title("Monthly new users")


def users_sliding_window(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    if not ax:
        ax = plt.axes()

    user_counts, dates = metrics.value_sliding_window(builds, "org_id", 30)
    ax.plot(dates, user_counts, zorder=2)
    ax.set_xlabel("Window end date")
    ax.set_title("Number of users in the previous 30 days")


def imagetype_builds(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    """
    Pie chart of the distribution of image types built.
    """
    if not ax:
        ax = plt.axes()

    types = builds["image_type"].value_counts()
    labels = [f"{idx} ({val})" for idx, val in types.items()]
    ax.pie(types.values, labels=labels)


def footprint_builds(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    """
    Pie chart of the distribution of footprints. See metrics.footprints() for details.
    """
    if not ax:
        ax = plt.axes()

    builds_footprints = metrics.footprints(builds)
    feet = builds_footprints["footprint"].value_counts()
    labels = [f"{idx} ({val})" for idx, val in feet.items()]
    ax.pie(feet.values, labels=labels)


def weekly_users(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    """
    Bar graph of users per seven day period. Shows new users alongside total users per period.
    This function does not align periods to calendar weeks.
    """
    first_date = builds["created_at"].min()
    last_date = builds["created_at"].max()

    users_so_far: Set[str] = set()
    n_week_users = []
    n_new_users = []

    start_dates = []

    p_start = first_date
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

    n_ret_users = np.subtract(n_week_users, n_new_users)
    ax.bar(start_dates, n_ret_users, width=2, label="returning users")
    ax.bar(start_dates, n_new_users, bottom=n_ret_users, width=2, label="new users")
    ax.legend(loc="best")
    month_offset = pandas.DateOffset(months=1)

    start_month = first_date.replace(day=1)
    end_month = last_date.replace(month=last_date.month+1, day=1)
    end_month = last_date.replace(day=1) + month_offset
    xticks = []
    tick = start_month
    while tick <= end_month:
        xticks.append(tick)
        tick += month_offset

    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def dau_over_mau(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    if not ax:
        ax = plt.axes()

    mod, dates = metrics.dau_over_mau(builds)
    ax.plot(dates, mod)
    ax.set_xlabel("Window end date")
    ax.set_title("Daily users / Users in the previous 30 days")


def single_footprint_distribution(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    """
    Bar graph showing the number of users that only build images for a single footprint, separated by footprint.
    """
    if not ax:
        ax = plt.axes()

    sfp_users = metrics.single_footprint_users(builds)
    fp_counts = sfp_users["footprint"].value_counts()
    ax.bar(fp_counts.index, fp_counts.values)
    for idx, count in zip(fp_counts.index, fp_counts.values):
        ax.text(idx, count, str(count), size=16, ha="center")
    ax.set_xlabel("Footprints")
    ax.set_title("Single-footprint user counts")
    ax.set_ylim(ymax=max(fp_counts)+30)


def single_footprint_monthly_users(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    """
    Multi-bar graph of the number of single-footprint users that appear in each calendar month, separated by footprint.
    """

    if not ax:
        ax = plt.axes()

    # get org_ids of orgs that only build one footprint
    sfp_users = metrics.single_footprint_users(builds, split_cloud=False)

    # Add footprint column to each build
    builds_wfp = metrics.footprints(builds, split_cloud=False)

    # Filter out multi-footprint org_ids
    builds_wfp = builds_wfp.loc[builds_wfp["org_id"].isin(sfp_users["org_id"])]

    shift = 0
    for footprint in sorted(builds_wfp["footprint"].unique()):
        # filter builds for the given footprint
        fp_builds = builds_wfp.loc[builds_wfp["footprint"] == footprint]
        # plot monthly users for the filtered set
        user_counts, months = metrics.monthly_users(fp_builds)
        months += pandas.Timedelta(days=shift)
        # ax.plot(months, user_counts, linewidth=3, label=footprint)
        ax.bar(months, user_counts, width=3, zorder=2, label=footprint)
        shift += 3

        # add numbers to bars
        # for mo, nu in zip(months, user_counts):
        #     plt.text(mo, nu, str(nu), size=16, ha="center")

    xlabels = [f"{mo.month_name()} {mo.year}" for mo in months]
    ax.set_xticks(months, xlabels)
    ax.set_title("Monthly users")
    ax.legend()


def footprint_monthly_builds(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    if not ax:
        ax = plt.axes()

    # Add footprint column to each build
    builds_wfp = metrics.footprints(builds, split_cloud=False)

    shift = 0
    for footprint in sorted(builds_wfp["footprint"].unique()):
        # filter builds for the given footprint
        fp_builds = builds_wfp.loc[builds_wfp["footprint"] == footprint]
        # plot monthly builds for the filtered set
        counts, months = metrics.monthly_builds(fp_builds)
        months += pandas.Timedelta(days=shift)
        ax.bar(months, counts, width=3, zorder=2, label=footprint)
        shift += 3
        # for mo, nu in zip(months, counts):
        #     plt.text(mo, nu, str(nu), size=16, ha="center")

    xlabels = [f"{mo.month_name()} {mo.year}" for mo in months]
    ax.set_xticks(months, xlabels)
    ax.set_title("Monthly builds")
    ax.legend()


def active_time(subscriptions: pandas.DataFrame):
    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["font.size"] = 8

    now = datetime.now()
    month_start = datetime(year=now.year, month=now.month, day=1)
    # start at the month-start of the first record
    start = subscriptions["created"].values.min().astype("datetime64[M]")  # truncate to month
    end = np.datetime64(str(month_start)).astype("datetime64[M]")  # convert and truncate

    months = []
    durations = []
    for mstart in np.arange(start, end):
        mstop = mstart + 1

        # clip created to month
        created = subscriptions["created"].astype("datetime64[s]")
        created = created.clip(mstart, mstop)

        # clip lastcheckin to month
        lastcheckin = subscriptions["lastcheckin"].astype("datetime64[s]")
        lastcheckin = lastcheckin.clip(mstart, mstop)

        duration = np.sum(lastcheckin - created)
        if duration.total_seconds() == 0:
            continue
        monthname = datetime.strptime(f"{mstart}", "%Y-%m").strftime("%B")
        months.append(monthname)
        durations.append(duration.total_seconds())

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.grid(axis="y", color="#dddddd")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set(linewidth=1.1)
    ax.xaxis.set_tick_params(size=0, pad=6)
    ax.yaxis.set_tick_params(size=0)
    ax.set_axisbelow(True)
    ax.set_title("Total runtime of RHEL instances\ncreated from Image Builder, in days", loc="left", fontweight="bold")

    bar_width = 0.66
    plt.bar(months, np.array(durations)/3600/24, width=bar_width)
    ax.set_xlim(-bar_width * 2/3, len(months) - 1 + bar_width * 2/3)
    fig.text(0.12, 0,  "Source: Red Hat Subscription Manager data", fontsize="small", color="#777777")
