#!/usr/bin/python3
# -*- coding: utf-8 -*-

from numpy import matmul, where
from pandas import DataFrame, concat
from pymagnitude import Magnitude
from sklearn.metrics.pairwise import cosine_similarity


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

    def generate2d(self, items, method='tsne', parameters=None):
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
                n_iter=1000,
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
