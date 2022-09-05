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


def trendline(values):
    values = list(values)
    m = 41
    halfm = m//2
    kernel = sp.gaussian(m, std=7)
    kernel /= sum(kernel)
    # pad the original values with the last value for half the kernel size
    values = values + ([values[-1]] * halfm)
    tl = sp.convolve(values, kernel, mode="same")
    tl = tl[:-halfm]
    return tl.tolist()


cust_dtypes = {
    "org_id": str,
    "org_name": str,
    "strategic": str,
}
customers = pandas.read_csv("Customers.csv", delimiter=",",
                            header=0, names=["org_id", "org_name", "strategic"], dtype=cust_dtypes)

fname = sys.argv[1]
builds = reader.read_dump(fname)
print(f"Imported {len(builds)} records")

print_summary(builds)


dates = np.array([d.date() for d in builds.created_at], dtype=np.datetime64)

all_dates = np.arange(dates[0], dates[-1], dtype=np.datetime64)
counts = np.array([sum(d == dates) for d in all_dates])

pkgfreq = {}
for idx, pkgs in enumerate(builds.packages):
    for pkg in list(pkgs):
        n = pkgfreq.get(pkg, 0)
        pkgfreq[pkg] = n + 1


print()
print("## Most frequently selected packages")
for idx, (name, count) in enumerate(sorted(pkgfreq.items(), key=lambda item: item[1], reverse=True)):
    print(f"{idx+1:3d}. {name:40s} {count:5d}")
    if idx == 9:
        break
print("---------------------------------")

imgfreq = {}
for img in builds.image_type:
    n = imgfreq.get(img, 0)
    imgfreq[img] = n + 1

print("## Image types")
for idx, (name, count) in enumerate(sorted(imgfreq.items(), key=lambda item: item[1], reverse=True)):
    print(f"{idx+1:3d}. {name:40s} {count:5d}")
print("---------------------------------")

userfreq = {}
for user in builds.org_id:
    n = userfreq.get(user, 0)
    userfreq[user] = n + 1

org_dict = {}
for _, user in customers.iterrows():
    org_dict[user.org_id] = user.org_name

print("## Biggest orgs")
for idx, (org_id, count) in enumerate(sorted(userfreq.items(), key=lambda item: item[1], reverse=True)):
    name = org_dict.get(org_id, org_id)
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

# plt.show()
