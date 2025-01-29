#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""database.py

"""


import logging

from annoy import AnnoyIndex
from pandas import read_csv

logger = logging.getLogger(__name__)


def create_index(path_in, path_names, path_index, dim=512):
    """
    path_in: TODO: create from pymagnitude / C-txt
    path_names: TODO: save as sqlite
    path_index: TODO: save as .annoy
    dim: TODO: parse from path_in
    """

    logger.debug("loading data")
    data = read_csv(path_in, sep="\t").drop("Unnamed: 0", axis=1)
    embeddings = data.drop("value", axis=1)
    names = data[["value"]]

    logger.debug("creating index")
    index = AnnoyIndex(dim, "angular")
    for i, embedding in embeddings.iterrows():
        index.add_item(i, list(embedding))

    logger.debug("building trees")
    index.build(100, n_jobs=-1)

    logger.debug("saving")
    names.to_csv(path_names, sep="\t", compression="gzip")
    index.save("path_index")

    return names, index


def load_index(path_names, path_index, dim=512):
    """

    path_names: TODO sqlite
    path_index: .annoy
    dim: TODO parse from names
    """

    names = read_csv(path_names, sep="\t", index_col=0)
    index = AnnoyIndex(dim, "angular")
    index.load(path_index)

    return names, index
