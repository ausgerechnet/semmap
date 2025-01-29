#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""utils.py

"""

import sys
from io import StringIO
from time import time

from pandas import concat, read_csv


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


class Progress:
    """Class for showing progress in for-loops
    (1) pb = Progress() before loop
    (2) pb.update()     every loop
    (3) pb.finalize()   after loop
    """

    def __init__(self, length=None, rate=1):
        """
        :param int length: length of loop (will calculate approximate ETA)
        :param int rate: refresh rate (default: every second)
        """

        self.start_glob = time()  # start time of the progress bar
        self.length = length      # total number of items
        self.rate = rate          # refresh rate

        self.c = 0                    # number of items already encountered
        self.boundle_c = self.c       # start counter of this bundle
        self.boundle_time = time()    # start time of this bundle
        self.max_message = 0

    def up(self):
        """alias for self.update()"""
        self.update()

    def fine(self):
        """alias for self.finalize()"""
        self.finalize()

    def update(self):

        self.c += 1
        when = time()

        current_time = when - self.boundle_time
        current_size = self.c - self.boundle_c

        if current_time > self.rate:

            avg_glob = (when-self.start_glob) / self.c

            if self.length is not None:
                msg = " ".join([
                    f"{int(self.c/self.length*100)}% ({self.c}/{self.length}).",
                    f"average: {int2str(avg_glob)}.",
                    f"average last {current_size} item(s): {int2str(current_time/current_size)}.",
                    f"ETA: {int2str((self.length - self.c) * avg_glob)}"
                ])

            else:
                msg = " ".join([
                    f"{self.c}.",
                    f"average: {int2str(avg_glob)}.",
                    f"average last {current_size} item(s): {int2str(current_time/current_size)}.",
                    f"total time: {int2str(when-self.start_glob)}"
                ])

            # print output
            self.print_line(msg)

            # update bundle start counter and start time
            self.boundle_c = self.c
            self.boundle_time = when

        if self.c == self.length:
            self.finalize()

    def print_line(self, msg, end="\r", file=sys.stderr):
        self.max_message = max(self.max_message, len(msg) + 1)
        trail = " ".join("" for _ in range(self.max_message-len(msg)))
        print(msg + trail, end=end, file=file)

    def finalize(self):
        total_time = time() - self.start_glob
        msg = "done. processed %d items in %s" % (self.c, int2str(total_time))
        self.print_line(msg, "\n")


def int2str(seconds):
    """ returns an appropriately formatted str of the seconds provided """

    # very small
    if seconds < 2:
        milli_seconds = 1000 * seconds
        if milli_seconds > 2:
            return f"{int(milli_seconds)} ms"

        micro_seconds = 1000 * milli_seconds
        if micro_seconds > 2:
            return f"{int(micro_seconds)} µs"
        elif micro_seconds > .1:
            return f"{round(micro_seconds, 2)} µs"
        elif micro_seconds > .01:
            return f"{round(micro_seconds, 3)} µs"
        elif micro_seconds > .001:
            return f"{round(micro_seconds, 4)} µs"
        else:
            return "<1 ns"

    # days and hours if more than a day
    nr_days = int(seconds // (60 * 60 * 24))
    nr_hours = int(seconds // (60 * 60) % 24)
    if nr_days > 0:
        return f"{nr_days} days, {nr_hours} hours"

    # hours and minutes if more than 12 hours
    nr_minutes = int(seconds // 60 % 60)
    if nr_hours > 12:
        return f"{nr_hours} hours, {nr_minutes} minutes"

    # default: hours:minutes:seconds
    nr_seconds = int(seconds % 60)
    return "{:02}:{:02}:{:02}".format(nr_hours, nr_minutes, nr_seconds)
