from datetime import datetime

import pandas
import matplotlib.pyplot as plt

import ibmetrics as ib
import report


def monthly_users_stacked(builds: pandas.DataFrame):
    """
    Bar graph of the number of users that appear in each calendar month; new users stacked on old.
    """
    # import seaborn as sns

    user_counts, months = ib.metrics.monthly_users(builds)
    new_user_counts, months = ib.metrics.monthly_new_users(builds)
    old_user_counts = user_counts - new_user_counts
    plt.bar(months, old_user_counts, width=27, color="#9b0000")
    plt.bar(months, new_user_counts, width=27, color="#EE0000", bottom=old_user_counts, label="New users")

    xlabels = [f"{mo.month_name()} {mo.year}" for mo in months]
    plt.set_xticks(months, xlabels, rotation=45, ha="right")
    plt.set_title("Monthly Unique Users")
    plt.legend(frameon=False)
    for mo, new, old in zip(months, new_user_counts, old_user_counts):
        print(f"{mo.month_name()}: {new+old}")



def main():
    builds = report.read_file("./data/dump-2022-10-24.log")
    users = pandas.read_json("./data/userinfo.json", dtype=False)
    with open("./data/userfilter.txt", encoding="utf-8") as filterfile:
        user_filter = filterfile.read().split("\n")

    print(f"Read {len(builds)} records")

    # filtering: remove builds before GA
    ga_date = datetime(2022, 5, 4)
    ga_idxs = builds["created_at"] >= ga_date
    full_month_cutoff = datetime.now().replace(day=1, hour=0, minute=0, second=0)
    full_month_idxs = builds["created_at"] < full_month_cutoff
    builds = builds.loc[ga_idxs & full_month_idxs]
    print(f"Using {len(builds)} records (since GA)")

    # filtering: remove internal users
    builds = ib.data.filter_users(builds, users, user_filter)
    print(f"Using {len(builds)} records (after filtering)")

    plt.figure()
    ib.plot.monthly_users_stacked(builds)
    plt.savefig("./monthly_users.png")
    plt.savefig("./monthly_users.svg")


if __name__ == "__main__":
    main()
