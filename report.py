import os
import sys

from datetime import datetime

import pandas
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp


from ibmetrics import reader


def print_summary(builds):
    print("## Summary")
    dates = builds["created_at"].astype("datetime64[D]")
    start = min(dates)
    end = max(dates)
    print(f"_Period: {start} - {end}:_")
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

    print_summary(builds)

    dates = np.array([d.date() for d in builds.created_at], dtype=np.datetime64)

    all_dates = np.arange(dates[0], dates[-1], dtype=np.datetime64)
    counts = np.array([sum(d == dates) for d in all_dates])

    pkgfreq = {}
    for pkgs in builds.packages:
        for pkg in set(pkgs):
            count = pkgfreq.get(pkg, 0)
            pkgfreq[pkg] = count + 1

    print("## Most frequently selected packages")
    for idx, (name, count) in enumerate(sorted(pkgfreq.items(), key=lambda item: item[1], reverse=True)):
        print(f"{idx+1:3d}. {name:40s} {count:5d}")
        if idx == 9:
            break
    print("---------------------------------")

    imgfreq = {}
    for img in builds.image_type:
        count = imgfreq.get(img, 0)
        imgfreq[img] = count + 1

    print("## Image types")
    for idx, (name, count) in enumerate(sorted(imgfreq.items(), key=lambda item: item[1], reverse=True)):
        print(f"{idx+1:3d}. {name:40s} {count:5d}")
    print("---------------------------------")

    userfreq = {}
    for user in builds.org_id:
        count = userfreq.get(user, 0)
        userfreq[user] = count + 1

    print("## Biggest orgs")
    for idx, (org_id, count) in enumerate(sorted(userfreq.items(), key=lambda item: item[1], reverse=True)):
        name = org_id
        user_idx = customers.org_id == org_id
        if sum(user_idx) == 1:
            name = customers.org_name[user_idx].values.item()
        elif sum(user_idx) > 1:
            raise ValueError(f"Multiple ({sum(user_idx)}) entries with same org_id ({org_id}) in customer data")
        print(f"{idx+1:3d}. {name:40s} {count:5d}")
        if idx == 19:
            break
    print("------------")

    plt.figure(figsize=(16, 9), dpi=100)

    plt.plot(all_dates, counts, ".", markersize=12, label="n builds")
    trend = trendline(counts)
    plt.plot(all_dates, trend, label="trendline")

    now = datetime.now()
    nowline = [now, now]
    plt.plot(nowline, [0, np.max(counts)], linestyle="--", color="black", label="now")

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
