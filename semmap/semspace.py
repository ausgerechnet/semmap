from io import StringIO

from numpy import matmul, where
from pandas import DataFrame, concat, read_csv
from pymagnitude import Magnitude
# transformers
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


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


class SemanticSpace(object):

    def __init__(self, magnitude_path):
        """

        :param str magnitude_path: Path to a .pymagnitude embeddings file.

        """

        self.database = Magnitude(magnitude_path)
        self.coordinates = None

    def embeddings(self, tokens):
        """
        loads a subset of all embeddings into a DataFrame.

        :param set tokens: set of tokens to get embeddings for

        :return: Dataframe containing embeddings
        :rtype: Dataframe
        """

        embeddings = [
            self.database.query(token) for token in tokens
        ]
        df = DataFrame(
            index=tokens,
            data=embeddings
        )

        return df

    def generate2d(self, tokens, method='tsne', parameters=None):
        """creates 2d-coordinates for a list of tokens

        :param list tokens: list of tokens to generate coordinates for
        :param str method: umap / tsne

        :return: pandas.Dataframe with x and y coordinates
        :rtype: pandas.Dataframe

        """

        # load vectors
        embeddings = self.embeddings(tokens)

        # if no vectors are loaded
        if embeddings.empty:
            return DataFrame()

        # set up transformer
        if method == 'tsne':
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
            transformer = UMAP()

        else:
            raise NotImplementedError(
                'transformation "%s" not supported' % method
            )

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

    def add(self, items, cutoff=.2):
        """caclulates coordinates for new items based on their cosine
        similarity to the items spanning self.coordinates.

        :param str items: items to add
        :param float cutoff: cut-off value for cosine similarity

        :return: new coordinates (columns 'x' and 'y', indexed by items)
        :rtype: DataFrame

        """

        # only for items that are not already in semantic space
        new_items = [i for i in items if i not in self.coordinates.index]
        if len(new_items) != len(items):
            raise ValueError()

        # get embedding for item
        item_embeddings = self.embeddings(items)
        base_embeddings = self.embeddings(self.coordinates.index)

        # cosine similaritiy matrix (n_items times n_base)
        sim = cosine_similarity(item_embeddings, base_embeddings)

        # apply cut-off
        sim = where(sim < cutoff, 0, sim)

        # norm rows to use as convex combination
        # TODO catch global 0 error
        sim = (sim.T/sim.sum(axis=1)).T

        # matrix multiplication takes care of linear combination
        new_coordinates = matmul(sim, self.coordinates)

        # convert to DataFrame
        new_coordinates = DataFrame(new_coordinates)
        new_coordinates.index = items

        # append
        self.coordinates = concat([self.coordinates, new_coordinates])

        return new_coordinates

    def visualize(self, size):
        """
        :param Series size: DataFrame containing label sizes
        """
        pass
