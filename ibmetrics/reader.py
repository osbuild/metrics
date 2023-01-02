"""
The reader module provides functions for loading data.
"""
import glob
import json
import os
import re
import sys

from typing import Dict, List, Union

import numpy as np
import pandas


def _list_reader(col):
    return np.array([json.loads(val) if val else [] for val in col], dtype=object)


def _date_reader(col):
    return np.array(col, dtype=np.datetime64)


CONVERTERS = {
    "created_at": _date_reader,
    "packages": _list_reader,
    "filesystem": _list_reader,
    "payload_repositories": _list_reader,
}


def convert(name, data):
    if f := CONVERTERS.get(name):
        return f(data)

    return data


def _parse_dump_row(line: str) -> List[str]:
    return [s.strip() for s in line.split("|")]


def read_dump(fname: Union[os.PathLike, str]) -> pandas.DataFrame:
    """
    Parses a database dump and returns the data formatted
    """
    with open(fname, encoding="utf-8") as dumpfile:
        # first line are column names
        names = _parse_dump_row(next(dumpfile))

        # second line is a row of dashes; will ignore
        next(dumpfile)

        row_count_re = re.compile(r"\((?P<rows>[0-9]+) rows\)\n")

        data: Dict[str, List] = {n: [] for n in names}
        row_count = -1
        for line in dumpfile:
            if match := row_count_re.fullmatch(line):
                row_count = int(match.group("rows"))
                break
            for i, v in enumerate(_parse_dump_row(line)):
                data[names[i]].append(v)

    for name, column in data.items():
        data[name] = convert(name, column)

    df = pandas.DataFrame(data=data)

    if row_count == -1:
        print("WARNING: failed to parse row count", file=sys.stderr)
    elif row_count != len(df):
        print(f"WARNING: read {len(df)} records but row count in dump footer states {row_count} rows",
              file=sys.stderr)

    return df


def read_parquet(path: Union[os.PathLike, str]) -> pandas.DataFrame:
    """
    Read data from a directory of parquet files. Data in the files are concatenated into a single DataFrame.
    Only files in the directory with the .parquet extension are loaded. Other files are ignored. Subdirectories are not
    traversed.
    """
    filelist = glob.glob(os.path.join(path, "*.parquet"))
    if not filelist:
        print(f"WARNING: no .parquet files found in {path}")
        return pandas.DataFrame()

    dataframes = []
    for fname in filelist:
        dataframes.append(pandas.read_parquet(fname))

    builds = pandas.concat(dataframes, ignore_index=True)

    return builds


def read(path: Union[os.PathLike, str]) -> pandas.DataFrame:
    """
    Alias for read_parquet()
    """
    return read_parquet(path)
