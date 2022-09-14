import os
import sys

from datetime import datetime, timedelta

import pandas
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp


from ibmetrics import reader


def print_summary(builds):
    print("## Summary")
    start = builds.created_at.min()
    end = builds.created_at.max()
    print(f"_Period: {start} - {end}:_")

    print(f"Number of users: {len(set(builds.org_id))}")

    n_with_packages = sum(1 if len(pkg) else 0 for pkg in builds.packages)
    print(f"- Builds with packages: {n_with_packages}")

    avg_packages = np.mean([len(pkg) for pkg in builds.packages])
    print(f"- Average number of packages per build: {avg_packages:.2f}")
    avg_packages_nonempty = np.mean([len(pkg) for pkg in builds.packages if len(pkg)])
    print(f"- Average number of packages per build (excluding empty): {avg_packages_nonempty:.2f}")

    n_with_fs = sum(1 if len(fs) else 0 for fs in builds.filesystem)
    print(f"- Builds with filesystem customizations: {n_with_fs}")

    n_with_repos = sum(1 if len(repos) else 0 for repos in builds.payload_repositories)
    print(f"- Builds with custom repos: {n_with_repos}")


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


def slice_time(builds: pandas.DataFrame, start: datetime, end: datetime):
    idxs = (builds.created_at >= start) & (builds.created_at <= end)
    return builds.loc[idxs]


# pylint: disable=too-many-statements,too-many-locals
def main():
    cust_dtypes = {
        "org_id": str,
        "org_name": str,
        "strategic": str,
    }
    customers = pandas.read_csv("Customers.csv", delimiter=",",
                                header=0, names=["org_id", "org_name", "strategic"], dtype=cust_dtypes)

    fname = sys.argv[1]
    builds = read_file(fname)
    print(f"Imported {len(builds)} records")

    start = builds.created_at.min()
    end = builds.created_at.max()

    builds = slice_time(builds, start, end)
    print(f"{len(builds)} between {start} and {end}")

    print_summary(builds)

    all_packages = []
    for pkg_list in builds.packages:
        all_packages.extend(set(pkg_list))

    print("## Most frequently selected packages")
    pkg_counts = pandas.value_counts(all_packages)
    for idx, (name, count) in enumerate(pkg_counts.iloc[:20].items()):
        print(f"{idx+1:3d}. {name:40s} {count:5d}")
    print("---------------------------------")

    print("## Image types")
    type_counts = builds.image_type.value_counts()
    for idx, (name, count) in enumerate(type_counts.items()):
        print(f"{idx+1:3d}. {name:40s} {count:5d}")
    print("---------------------------------")

    print("## Biggest orgs")
    org_counts = builds.org_id.value_counts()
    for idx, (org_id, count) in enumerate(org_counts.iloc[:20].items()):
        name = org_id
        user_idx = customers.org_id == org_id
        if sum(user_idx) == 1:
            name = customers.org_name[user_idx].values.item()
        elif sum(user_idx) > 1:
            raise ValueError(f"Multiple ({sum(user_idx)}) entries with same org_id ({org_id}) in customer data")
        print(f"{idx+1:3d}. {name:40s} {count:5d}")
    print("------------")

    build_dates = builds.created_at.values.astype("datetime64[D]")
    all_dates = np.arange(build_dates[0], build_dates[-1], dtype="datetime64[D]")
    builds_per_day = np.array([sum(d == build_dates) for d in all_dates])

    plt.figure(figsize=(16, 9), dpi=100)

    plt.plot(all_dates, builds_per_day, ".", markersize=12, label="n builds")
    trend = trendline(builds_per_day)
    plt.plot(all_dates, trend, label="trendline")

    now = datetime.now()
    nowline = [now, now]
    plt.plot(nowline, [0, np.max(builds_per_day)], linestyle="--", color="black", label="now")

    plt.grid()
    plt.axis(ymin=0)
    plt.xlabel("dates")
    plt.ylabel("builds per day")
    plt.legend(loc="best")

    imgname = os.path.splitext(fname)[0] + ".png"
    plt.savefig(imgname)
    print(f"Saved plot as {imgname}")


if __name__ == "__main__":
    main()
