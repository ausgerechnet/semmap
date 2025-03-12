#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random
import sqlite3
from difflib import get_close_matches
from hashlib import sha256

import numpy as np
from annoy import AnnoyIndex
from pandas import DataFrame

from .embeddings import create_embeddings
from .utils import Progress


def create_items_table(path_db, indices, items):
    """

    """
    connection = sqlite3.connect(path_db)
    cursor = connection.cursor()

    # Create items table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS items (
            index_id INTEGER PRIMARY KEY,
            item TEXT
        )
    ''')

    # Create indices on both columns
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_index_id ON items (index_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_item ON items (item)')

    # Insert data into items table
    items = [str(i) for i in items]
    cursor.executemany('INSERT INTO items (index_id, item) VALUES (?, ?)', zip(indices, items))

    # Commit and close connection
    connection.commit()
    connection.close()


def store_embeddings(items, vectors, path_settings, path_db, path_annoy, dim, n_trees, metric, random_seed, model_name):

    if len(items) != len(vectors):
        raise ValueError()

    dim = len(vectors[0]) if dim is None else dim

    print("creating annoy index")
    index = AnnoyIndex(dim, metric=metric)
    pb = Progress(len(items))
    indices = list()
    for i, vector in enumerate(vectors):
        if len(vector) != dim:
            raise ValueError()
        index.add_item(i, vector)
        indices.append(i)
        pb.up()

    print(".. building trees")
    index.build(n_trees=n_trees, n_jobs=-1)

    print(".. saving index")
    index.save(path_annoy)

    print("saving database")
    create_items_table(path_db, indices, items)

    # Write settings to settings file
    print("saving settings")
    settings = {
        "dim": dim,
        "random_seed": random_seed,
        "path_annoy": path_annoy,
        "path_db": path_db,
        "metric": metric,
        "model_name": model_name,
        "n_trees": n_trees
    }
    with open(path_settings, "wt") as f:
        json.dump(settings, f, indent=4)


def create_embeddings_store(path_items, path_settings, path_db, path_annoy, n_trees=100, metric='angular',
                            model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', random_seed=42):
    """Create and store embeddings from a file containing items.

    Parameters:
    param str path_items: Path to file containing items (one per line).
    param str path_settings: Path to settings file.
    param str path_db: Path to SQLite database.
    param str path_annoy: Path to Annoy index file.
    param int n_trees: Number of trees for Annoy index.
    param str metric: Distance metric for Annoy nearest neighbours.
    param str model_name: Name of the embedding model.
    param int random_seed: Random seed for reproducibility.

    Returns:
    EmbeddingStore: An instance of the EmbeddingStore class.
    """

    items = open(path_items, "rt").read().strip().split("\n")
    vectors = create_embeddings(items, model_name, as_is=True)
    store_embeddings(items, vectors, path_settings, path_db, path_annoy, dim=None,
                     n_trees=n_trees, metric=metric, random_seed=random_seed, model_name=model_name)

    return EmbeddingsStore(path_settings)


class EmbeddingsStore:

    def __init__(self, path_settings):

        self.path_settings = path_settings

        # load settings from file
        self.dim, self.metric, self.random_seed, self.path_annoy, self.path_db, self.model_name, self.n_trees = self._load_settings()

        # connect to SQLite database
        self.conn = sqlite3.connect(self.path_db)
        self.cursor = self.conn.cursor()

        # load Annoy index
        self.index = AnnoyIndex(self.dim, self.metric)
        self.index.load(self.path_annoy)

        # use random seed
        random.seed(self.random_seed)

    def _load_settings(self):
        """Load settings."""
        settings = json.load(open(self.path_settings, "rt"))
        return settings['dim'], settings['metric'], settings['random_seed'], settings['path_annoy'], \
            settings['path_db'], settings['model_name'], settings['n_trees']

    def _get_item_index(self, item):
        """Retrieve the index of an item from the database."""
        self.cursor.execute("SELECT index_id FROM items WHERE item = ?", (item,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def _get_items_by_index(self, indices):
        self.cursor.execute(f"SELECT index_id, item FROM items WHERE index_id IN ({','.join(['?']*len(indices))})", indices)
        result = self.cursor.fetchall()
        d = dict(result)
        return [(idx, d[idx]) for idx in indices]

    def _find_string_similarity(self, item, threshold):
        """Find a similar item in the database based on string similarity."""
        self.cursor.execute("SELECT item FROM items")
        all_items = [str(row[0]) for row in self.cursor.fetchall()]

        # Get the closest matches within the threshold
        matches = get_close_matches(item, all_items, n=1, cutoff=threshold)
        return matches[0] if matches else None

    def _generate_random_vector(self, item):
        """Generate a reproducible random vector for the given item using the random seed."""

        item_seed = int(sha256(item.encode('utf-8')).hexdigest(), 16) + self.random_seed
        random.seed(item_seed)

        # Generate and return a random vector
        return [random.uniform(-1, 1) for _ in range(self.dim)]

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def delete_data(self, force=False):
        """Deletes database, index, and settings.


        """
        if not force:
            confirm = input(f"Are you sure you want to delete {self.path_db}, {self.path_annoy}, and {self.path_settings}? (yes/no): ")
            force = confirm.lower() == "yes"
        if force:
            self.close()
            for path in [self.path_db, self.path_annoy, self.path_settings]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Deleted: {path}")
                else:
                    print(f"File not found: {path}")
        else:
            print("Deletion canceled.")

    def update(self, path_items, force=True):
        """update embedding store by adding new items; this means deleting the old annoy index

        """

        print("getting old items")
        self.cursor.execute("SELECT index_id, item FROM items")
        old_data = [(int(row[0]), str(row[1])) for row in self.cursor.fetchall()]
        old_indices = [d[0] for d in old_data]
        old_items = [str(d[1]) for d in old_data]
        print(f".. number of old items: {len(old_items)}")

        print("getting new items")
        new_items = open(path_items, "rt").read().strip().split("\n")
        print(f".. number of new items: {len(new_items)}")
        new_items = [str(n) for n in new_items if n not in old_items]
        print(f".. number of new items not already in store: {len(new_items)}")

        if len(new_items) == 0:
            print(".. nothing to do here")
            return self

        self.delete_data(force=force)

        print("getting vectors")
        old_vectors = [self.index.get_item_vector(i) for i in old_indices]
        new_vectors = create_embeddings(new_items, self.model_name, as_is=True)

        all_items = old_items + new_items
        all_vectors = np.vstack([old_vectors, new_vectors])

        store_embeddings(all_items, all_vectors, self.path_settings, self.path_db, self.path_annoy, dim=None,
                         n_trees=self.n_trees, metric=self.metric, random_seed=self.random_seed, model_name=self.model_name)

        return EmbeddingsStore(self.path_settings)

    def get_embeddings(self, items, similarity_threshold=None, oov_info=False):
        """Retrieve embeddings for given items, OOV via string similarity (fallback: random vector)."""

        vectors = list()
        oovs = list()

        for item in items:

            oov = False

            # get index for the item if it exists
            index = self._get_item_index(item)
            if index is not None:
                vector = self.index.get_item_vector(index)

            else:
                oov = 'random'

                if similarity_threshold:
                    # search for a similar item
                    similar_item = self._find_string_similarity(item, similarity_threshold)
                    if similar_item:
                        # use embedding of similar item
                        similar_index = self._get_item_index(similar_item)
                        vector = self.index.get_item_vector(similar_index)
                        oov = 'string-similarity'

                if oov == 'random':
                    # generate a random vector
                    vector = self._generate_random_vector(item)
                    oov = 'random'

            vectors.append(vector)
            oovs.append(oov)

        df = DataFrame(index=items, data=vectors)
        if oov_info:
            df['oov'] = oovs

        return df

    def find_neighbours(self, item, n=10):

        idx = self._get_item_index(item)
        if idx is None:
            raise NotImplementedError(f'no embedding stored for "{item}"')
        idx_neighbours, distances = self.index.get_nns_by_item(idx, n, include_distances=True)
        neighbours = self._get_items_by_index(idx_neighbours)

        return [(neighbour[1], distance) for (neighbour, distance) in zip(neighbours, distances)]

    def query(self, item):
        """alias for get_embeddings()"""

        return self.get_embeddings([item]).loc[item].values

    def most_similar(self, positive, negative=[], topn=10):
        """alias for find_neighbours()

        we mimic magnitude behaviour here: we search for the closest vectors to the centroid of given vectors
        """

        pos_vecs = neg_vecs = []
        if len(positive) > 0:
            pos_vecs = self.get_embeddings(positive, oov_info=True)
            pos_vecs = pos_vecs.loc[pos_vecs['oov'] != "random"].drop('oov', axis=1).values
        vecs = pos_vecs

        if len(negative) > 0:
            neg_vecs = self.get_embeddings(negative, oov_info=True)
            neg_vecs = neg_vecs.loc[neg_vecs['oov'] != "random"].drop('oov', axis=1).values
            neg_vecs = -1.0 * neg_vecs
            vecs = np.vstack([pos_vecs, neg_vecs])

        mean_vector = np.mean(vecs, axis=0)

        idx_neighbours, distances = self.index.get_nns_by_vector(mean_vector, n=topn, search_k=1000, include_distances=True)
        neighbours = self._get_items_by_index(idx_neighbours)

        return [(neighbour[1], 1 - distance ** 2 / 2) for (neighbour, distance) in zip(neighbours, distances)]
