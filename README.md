# osbuild-metrics

Scripts and utilities for exploring Image Builder usage data and creating reports.

## How to use

The `ibmetrics/reader` module can parse the output from an weekly SQL query against the image builder production database. The current weekly query can be found in the [internal app-interface repo](https://gitlab.cee.redhat.com/service/app-interface/-/tree/master/data/services/insights/image-builder/sql-queries) (note that old, inactive queries are marked with `delete: true` and generally only the most recent one is active at any given time).

You can load the data from a log into a [Pandas](https://pandas.pydata.org/) DataFrame with:
```python
import reader

builds = reader.read_dump(fname)
```

The `report.read_file()` method adds caching on top of this for speed, since the log file parsing can take a few seconds. On first read of a data dump, it saves the data as a Python pickle in `${XDG_CACHE_HOME}/osbuild-metrics/` using the basename of the input file. Future loads of the same file load the pickle (no cache freshness checks are made; it is assumed that the log file from the data dump never changes).

Running `report.py` against a log file will produce some stats and figures. The `main()` function in this file can be used as a playground to explore the data.
Alternatively, you can load the data and the `report` module in an interactive environment and explore it there:
```python
import report
import pandas

builds = report.read_file("./data/dump-2022-09-26.log")
users = pandas.read_json("./data/userinfo.json")  # maps account numbers to account names and other info

print(f"Read {len(builds)} records")

report.print_summary(builds)
...
```

See also the [explore](./explore.ipynb) notebook in the root of the repo.

## Getting the data

The data can be downloaded from the [Openshift Console output](https://console-openshift-console.apps.crcp01ue1.o9m8.p1.openshiftapps.com/k8s/ns/image-builder-prod/cronjobs) (under Workloads > Pods). Note that jobs run every Monday and the output is only available for 3 days. Ask @achilleas-k for convenient access to data dumps.

## Planned features

The plan is for the repo to contain libraries and functions for conveniently exploring the data. Currently, most useful code is in the `report.py` file, but this should be modularised. Scripts and Jupyter Notebooks will be added that produce sample stats and figures.
