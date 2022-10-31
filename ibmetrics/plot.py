"""
Plotting functions
"""
from datetime import timedelta
from typing import Optional, Set

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
    ax.bar(months, old_user_counts, width=27)
    ax.bar(months, new_user_counts, width=27, bottom=old_user_counts, label="New users")

    xlabels = [f"{mo.month_name()} {mo.year}" for mo in months]
    ax.set_xticks(months, xlabels)
    ax.set_title("Monthly Unique Users")
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

    bar_shift = timedelta(days=1)
    ax.bar(np.array(start_dates)+bar_shift, n_week_users, width=2, label="n users")
    ax.bar(start_dates, n_new_users, width=2, label="n new users")
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
