"""
Plotting functions
"""
import pandas

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from datetime import timedelta
from typing import Optional, Set

from . import metrics


def build_counts(builds: pandas.DataFrame, p_days: int):
    t_starts, counts = metrics.builds_over_time(builds, period=timedelta(days=p_days))
    ax = plt.axes()
    ax.plot(t_starts, counts, ".b", markersize=12, label="n builds")

    builds_trend = moving_average(counts)
    ax.plot(t_starts, builds_trend, "-b", label="builds mov. avg.")

    ax.set_xticks(t_starts)
    # rotate xtick labels 45 degrees cw for readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    ax.set_xlabel("dates")
    ax.legend(loc="best")
    ax.grid(True)


def moving_average(values):
    sums = np.cumsum(values)
    weights = np.arange(1, len(sums)+1, 1)
    return sums / weights


def monthly_users(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    if not ax:
        ax = plt.axes()

    mau, months = metrics.monthly_users(builds)
    ax.bar(months, mau, width=20, zorder=2)
    for mo, nu in zip(months, mau):
        plt.text(mo, nu, str(nu), size=16, ha="center")

    xlabels = [f"{mo.month_name()} {mo.year}" for mo in months]
    ax.set_xticks(months, xlabels, rotation=45, ha="right")
    ax.grid(True)
    ax.set_title("Monthly users")


def image_types(builds: pandas.DataFrame):
    image_types = builds["image_type"].value_counts()
    plt.pie(image_types.values, labels=image_types.index)


def weekly_users(builds: pandas.DataFrame, ax: Optional[plt.Axes] = None):
    p_start = builds["created_at"].min()
    last_date = builds["created_at"].max()

    users_so_far: Set[str] = set()
    n_week_users = []
    n_new_users = []

    start_dates = []

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
    start_month = p_start.replace(day=1)
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
