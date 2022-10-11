import argparse
import os
import sys

from datetime import datetime

import pandas
import matplotlib.pyplot as plt
import scipy.signal as sp

import ibmetrics as ib


def read_file(fname: os.PathLike, recreate_cache=False) -> pandas.DataFrame:
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = os.path.join(cache_home, "osbuild-metrics")
    os.makedirs(cache_dir, exist_ok=True)
    cache_fname = os.path.join(cache_dir, os.path.basename(os.path.splitext(fname)[0]) + ".pkl")
    if os.path.exists(cache_fname) and not recreate_cache:
        print(f"Using cached pickle file at {cache_fname}")
        try:
            return pandas.read_pickle(cache_fname)
        # pylint: disable=broad-except
        except Exception as exc:
            print(f"Error reading cached pickle file {cache_fname}: {exc}")
            print("File may have been corrupted.")
            print("You can recreate the cache with --recreate-cache")
            sys.exit(1)

    builds = ib.reader.read_dump(fname)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report from Image Builder usage data")
    parser.add_argument("data", help="File containing data dump")
    parser.add_argument("--start", default=None, help="Start date to use (older data are ignored)")
    parser.add_argument("--end", default=None, help="End date to use (newer data are ignored)")
    parser.add_argument("--userinfo", default=None, help="File containing user info (json)")
    parser.add_argument("--userfilter", default=None, help="File containing user names to remove from data")
    parser.add_argument("--recreate-cache", action="store_true", help="Recreate cache file even if it exists")

    args = parser.parse_args()
    return args


# pylint: disable=too-many-statements,too-many-locals
def main():
    args = parse_args()
    data_fname = args.data

    builds = read_file(data_fname, args.recreate_cache)
    print(f"Imported {len(builds)} records")

    users = None
    if args.userinfo:
        users = pandas.read_json(args.userinfo)

    user_filter = []
    if args.userfilter:
        with open(args.userfilter, encoding="utf-8") as filterfile:
            user_filter = filterfile.read().split("\n")

    builds = ib.data.filter_users(builds, users, user_filter)
    print(f"{len(builds)} records after user filtering")

    if args.start:
        start = datetime.fromisoformat(args.start)
    else:
        start = builds["created_at"].min()

    if args.end:
        end = datetime.fromisoformat(args.end)
    else:
        end = builds["created_at"].max()

    builds = ib.data.slice_time(builds, start, end)

    print(ib.metrics.summarise(ib.metrics.make_summary(builds)))

    # plot weekly counts
    p_days = 7  # 7 day period

    img_basename = os.path.splitext(os.path.basename(data_fname))[0]

    # builds counts
    plt.figure(figsize=(16, 9), dpi=100)
    ib.plot.build_counts(builds, p_days)
    imgname = img_basename + "-builds.png"
    plt.savefig(imgname)
    print(f"Saved figure {imgname}")

    # user counts
    plt.figure(figsize=(16, 9), dpi=100)
    ib.plot.monthly_users(builds)
    imgname = img_basename + "-users.png"
    plt.savefig(imgname)
    print(f"Saved figure {imgname}")

    # image type breakdown
    plt.figure(figsize=(16, 9), dpi=100)
    ib.plot.imagetype_builds(builds)
    imgname = img_basename + "-image_types.png"
    plt.savefig(imgname)
    print(f"Saved figure {imgname}")

    plt.figure(figsize=(16, 9), dpi=100)
    ib.plot.weekly_users(builds)
    imgname = img_basename + "-weekly_users.png"
    plt.savefig(imgname)
    print(f"Saved figure {imgname}")


if __name__ == "__main__":
    main()
