#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""utils.py

"""

from io import StringIO

from pandas import read_csv, concat


def read_cqpweb_table(path_in):

    with open(path_in, "rt") as f_in:
        table_rows = list()
        for line in f_in:
            if len(line.split("\t")) > 3:
                table_rows.append(line)

    df = read_csv(StringIO("".join(table_rows)), sep="\t", index_col=1)
    df = df[['Stat 1.']]
    df.columns = ['am']
    df.index.name = 'item'
    return df


def read_cqpweb_tables(path1, path2, magnitude_path, names=None):

    print("reading data")
    df1 = read_cqpweb_table(path1)
    df2 = read_cqpweb_table(path2)
    print("%d items in first df, %d items in second df" % (len(df1), len(df2)))
    if names is None:
        names = ["1", "2"]
    df = df1.join(df2, lsuffix="_"+names[0], rsuffix="_"+names[1], how='outer')

    return df


def read_ccc_table(path):

    df = read_csv(path, sep="\t", index_col=0, dtype=str)

    return df


def read_ccc_tables(paths, names=None):

    if names is None:
        names = [p.split("/")[-1].split(".")[0] for p in paths]

    dfs = [read_ccc_table(path) for path in paths]
    df = concat(dfs, keys=names)

    return df
