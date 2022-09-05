"""
The reader module provides functions for loading data.
"""
import os
import re
import sys
import json

from typing import List

import pandas

import numpy as np


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


def _parse_dump_row(line: str) -> List[str]:
    return [s.strip() for s in line.split("|")]


def _make_data_frame(headers: List[str], data: List[List[str]]) -> pandas.DataFrame:
    data_str = np.array(data)
    data_dict = {}
    for name, column_str in zip(headers, data_str.transpose()):
        if converter := CONVERTERS.get(name):
            column = converter(column_str)
        else:
            column = column_str
        data_dict[name] = column

    return pandas.DataFrame(data=data_dict)


def read_dump(fname: os.PathLike) -> pandas.DataFrame:
    """
    Parses a database dump and returns the data formatted
    """
    with open(fname, encoding="utf-8") as dumpfile:
        lines = dumpfile.read().split("\n")

    # first line should be headers (column names)
    headers = _parse_dump_row(lines[0])

    row_count_re = re.compile(r"\((?P<rows>[0-9]+) rows\)")

    # second line is a row of dashes; will ignore
    rows = []
    row_count = -1
    for line in lines[2:]:
        if match := row_count_re.fullmatch(line):
            groups = match.groupdict()
            if row_count_str := groups.get("rows"):
                row_count = int(row_count_str)
        if "|" in line:  # columns are separated by pipes; use it to identify data rows
            rows.append(_parse_dump_row(line))

    if row_count == -1:
        print("WARNING: failed to parse row count", file=sys.stderr)
    elif row_count != len(rows):
        print(f"WARNING: read {len(rows)} records but row count in dump footer states {row_count} rows",
              file=sys.stderr)

    data_df = _make_data_frame(headers, rows)

    return data_df
