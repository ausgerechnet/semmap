#!/usr/bin/python3
# -*- coding: utf-8 -*-

from numpy import matmul, where, errstate
from pandas import DataFrame, concat
from pymagnitude import Magnitude
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings_store import EmbeddingsStore


class SemanticSpace:

    def __init__(self, path=None, normalise=False):
        """

        :param str path: path to ".magnitude" or ".semmap" file
        :param boolean normalise: restrict coordinates to [-1,1]^2?

        """

        if path is None or path.endswith("magnitude"):
            self.database = Magnitude(path)
        elif path.endswith("semmap"):
            self.database = EmbeddingsStore(path)
        else:
            raise ValueError()

        self.normalise = normalise
        self.coordinates = None

    def _embeddings(self, items):
        """Get embeddings of provided items as a DataFrame.

        :param list items: set of tokens to get embeddings for

        :return: DataFrame containing embeddings
        :rtype: DataFrame
        """

        if len(items) != len(set(items)):
            raise ValueError('items must be unique')

        embeddings = [self.database.query(item) for item in items]
        df = DataFrame(index=items, data=embeddings)

        return df

    def most_similar(self, positive, negative=[], n=10):
        """Get similar items of the ones provided.

        :param list positive: list of positive items ("similar to")
        :param list negative: list of negative items ("not similar to")

        :return: DataFrame of items and similarities, ordered by decreasing similarity
        :rtype: DataFrame
        """

        neighbours = self.database.most_similar(positive=positive, negative=negative, topn=n)
        items = [n[0] for n in neighbours]
        similarities = [n[1] for n in neighbours]

        return DataFrame({'item': items, 'similarity': similarities}).set_index('item')

    def generate2d(self, items, method='tsne', parameters={}):
        """Create 2d-coordinates for list of items.

        :param list items: list of items to generate coordinates for
        :param str method: ["tsne"] | "umap" | "openTSNE"
        :param dict parameters: parameters to pass to dim reduction algorithm

        :return: DataFrame with x and y coordinates, indexed by items
        :rtype: DataFrame
        """

        # load vectors
        embeddings = self._embeddings(items)

        # if no vectors are loaded
        if embeddings.empty:
            return DataFrame()

        # set up transformer
        if method == 'tsne':
            from sklearn.manifold import TSNE
            parameters_ = dict(
                n_components=2,
                perplexity=min(25, len(embeddings) - 1),
                early_exaggeration=12.0,
                learning_rate='auto',
                max_iter=1000,
                n_iter_without_progress=300,
                min_grad_norm=1e-07,
                metric='cosine',
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
            data2d = transformer.fit_transform(embeddings)

        elif method == 'openTSNE':
            from openTSNE import TSNE
            transformer = TSNE()
            data2d = transformer.fit(embeddings)

        elif method == 'umap':
            from umap import UMAP
            transformer = UMAP()
            data2d = transformer.fit_transform(embeddings)

        else:
            raise NotImplementedError(f'transformation "{method}" not supported')

        # generate 2d coordinates as data frame
        coordinates = DataFrame(
            data=data2d,
            index=embeddings.index,
            columns=['x', 'y']
        )
        coordinates.index.name = 'item'

        if self.normalise:
            coordinates.x = coordinates.x / coordinates.x.abs().max()
            coordinates.y = coordinates.y / coordinates.y.abs().max()

        # save coordinates
        self.coordinates = coordinates

        return coordinates

    def add(self, items, cutoff=0):
        """caclulates coordinates for new items based on their cosine
        similarity to the items spanning self.coordinates.

        :param str items: items to add
        :param float cutoff: cut-off value for cosine similarity

        :return: new coordinates (columns 'x' and 'y', indexed by items)
        :rtype: DataFrame
        """

        if cutoff < 0 or cutoff > 1:
            raise ValueError()

        # get embeddings for items
        item_embeddings = self._embeddings(items)
        base_embeddings = self._embeddings(self.coordinates.index)

        # cosine similarity matrix (n_items times n_base)
        sim = cosine_similarity(item_embeddings, base_embeddings)

        # prune
        sim = where(sim < cutoff, 0, sim)

        # norm rows to use as convex combination
        simsum = sim.sum(axis=1)  # can be 0 -- resulting rows will be replace by 1/sim.shape[1]
        with errstate(divide='ignore', invalid='ignore'):
            sim = (sim.T / simsum).T
        sim[simsum == 0] = 1 / sim.shape[1]

        # matrix multiplication takes care of linear combination
        new_coordinates = matmul(sim, self.coordinates)

        # convert to DataFrame
        new_coordinates = DataFrame(new_coordinates)
        new_coordinates.index = items

        # append
        self.coordinates = concat([self.coordinates, new_coordinates])

        return new_coordinates
