#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import sqlite3
# import numpy as np
from difflib import get_close_matches

from annoy import AnnoyIndex
from numpy import matmul
from pandas import DataFrame, concat
from pymagnitude import Magnitude
from sklearn.metrics.pairwise import cosine_similarity

# def find_neighbours(name, names, index, k=5):

#     try:
#         i = names.loc[names["value"] == name].index.values[0]
#     except IndexError:
#         raise NotImplementedError(f'no embedding stored for "{name}", implement dealing with OOV')
#     similarities = index.get_nns_by_item(i, k, include_distances=True)
#     neighbours = names.loc[similarities[0]]
#     neighbours["distance"] = similarities[1]

#     return neighbours


class SemanticSpace:

    def __init__(self, magnitude_path=None):
        """

        :param str magnitude_path: Path to a .pymagnitude embeddings file.

        """

        self.database = Magnitude(magnitude_path)
        self.coordinates = None

    def _embeddings(self, items):
        """
        loads a subset of all embeddings into a DataFrame.

        :param set tokens: set of tokens to get embeddings for

        :return: Dataframe containing embeddings
        :rtype: Dataframe
        """

        if len(items) != len(set(items)):
            raise ValueError('items must be unique')

        embeddings = [self.database.query(item) for item in items]
        df = DataFrame(index=items, data=embeddings)

        return df

    def generate2d(self, items, method='tsne', parameters=None, normalise=False):
        """creates 2d-coordinates for a list of tokens

        :param list tokens: list of tokens to generate coordinates for
        :param str method: umap / tsne

        :return: pandas.Dataframe with x and y coordinates
        :rtype: pandas.Dataframe

        """

        # load vectors
        embeddings = self._embeddings(items)

        # if no vectors are loaded
        if embeddings.empty:
            return DataFrame()

        # just in case
        embeddings = embeddings.dropna()

        # set up transformer
        if method == 'tsne':
            from sklearn.manifold import TSNE
            parameters_ = dict(
                n_components=2,
                perplexity=min(30.0, len(embeddings) - 1),
                early_exaggeration=12.0,
                learning_rate='auto',
                max_iter=1000,
                n_iter_without_progress=300,
                min_grad_norm=1e-07,
                metric='euclidean',
                metric_params=None,
                init='pca',
                verbose=0,
                random_state=None,
                method='barnes_hut',
                angle=0.5,
                n_jobs=4
            )
            if parameters is not None:
                parameters_.update(parameters)

            transformer = TSNE(**parameters_)

        elif method == 'umap':
            from umap import UMAP
            transformer = UMAP()

        else:
            raise NotImplementedError(f'transformation "{method}" not supported')

        # generate 2d coordinates as data frame
        coordinates = DataFrame(
            data=transformer.fit_transform(embeddings),
            index=embeddings.index,
            columns=['x', 'y']
        )
        coordinates.index.name = 'item'

        if normalise:
            coordinates.x = coordinates.x / coordinates.x.abs().max()
            coordinates.y = coordinates.y / coordinates.y.abs().max()

        # save coordinates
        self.coordinates = coordinates

        return coordinates

    def add(self, items, cutoff=0.2):
        """caclulates coordinates for new items based on their cosine
        similarity to the items spanning self.coordinates.
        # TODO deduplicate

        :param str items: items to add
        :param float cutoff: cut-off value for cosine similarity

        :return: new coordinates (columns 'x' and 'y', indexed by items)
        :rtype: DataFrame

        """

        # get embedding for item
        item_embeddings = self._embeddings(items)
        base_embeddings = self._embeddings(self.coordinates.index)

        # cosine similarity matrix (n_items times n_base)
        sim = cosine_similarity(item_embeddings, base_embeddings)

        # apply cut-off
        # sim = where(sim < cutoff, 0, sim)

        # norm rows to use as convex combination
        simsum = sim.sum(axis=1)
        sim = (sim.T/simsum).T

        # matrix multiplication takes care of linear combination
        new_coordinates = matmul(sim, self.coordinates)

        # convert to DataFrame
        new_coordinates = DataFrame(new_coordinates)
        new_coordinates.index = items

        # append
        self.coordinates = concat([self.coordinates, new_coordinates])

        return new_coordinates


class SemMap:

    def __init__(self, path_to_annoy, path_to_db):

        self.path_to_annoy = path_to_annoy
        self.path_to_db = path_to_db

        # Connect to the SQLite database
        self.conn = sqlite3.connect(self.path_to_db)
        self.cursor = self.conn.cursor()

        # Load settings from the database
        self.dim, self.random_seed = self._load_settings()

        # Initialize Annoy index
        self.index = AnnoyIndex(self.dim, 'angular')
        self.index.load(self.path_to_annoy)

        # Seed the random number generator
        random.seed(self.random_seed)

    def _load_settings(self):
        """Load dimension and random seed settings from the database."""
        self.cursor.execute("SELECT dim, random_seed FROM settings")
        dim, random_seed = self.cursor.fetchone()
        return dim, random_seed

    def _embeddings(self, items, similarity_threshold=0.8):
        """Retrieve embeddings for given items, OOV via similarity (fallback: random vector)."""

        embeddings = list()
        oovs = list()

        for item in items:

            # get index for the item if it exists
            index = self._get_item_index(item)
            if index is not None:
                embeddings = self.index.get_item_vector(index)
                oov = None

            else:

                if similarity_threshold:
                    # search for a similar item
                    similar_item = self._find_similar_item(item, similarity_threshold)
                    if similar_item:
                        # use embedding of similar item
                        similar_index = self._get_item_index(similar_item)
                        embedding = self.index.get_item_vector(similar_index)
                        oov = 'similar'
                    else:
                        # generate a random vector
                        embedding = self._generate_random_vector()
                        oov = 'random'
                else:
                    embedding = self._generate_random_vector()
                    oov = 'random'

            embeddings.append(embedding)
            oovs.append(oov)

        df = DataFrame(index=items, data=embeddings)

        return df

    def _get_item_index(self, item):
        """Retrieve the index of an item from the database."""
        self.cursor.execute("SELECT index FROM items WHERE item = ?", (item,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def _find_similar_item(self, item, threshold):
        """Find a similar item in the database based on string similarity."""
        self.cursor.execute("SELECT item FROM items")
        all_items = [row[0] for row in self.cursor.fetchall()]

        # Get the closest matches within the threshold
        matches = get_close_matches(item, all_items, n=1, cutoff=threshold)
        return matches[0] if matches else None

    def _generate_random_vector(self, item):
        """Generate a reproducible random vector for the given item using the random seed."""

        item_seed = hash(item) + self.random_seed
        random.seed(item_seed)

        # Generate and return a random vector
        return [random.uniform(-1, 1) for _ in range(self.dim)]

    def close(self):
        """Close the database connection."""
        self.conn.close()
