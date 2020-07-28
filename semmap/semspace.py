from pandas import DataFrame
from pymagnitude import Magnitude
from scipy.spatial.distance import cosine
# visualization
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, output_file, show
# transformers
from sklearn.manifold import TSNE
from umap import UMAP


class SemanticSpace(object):

    def __init__(self, magnitude_path=None):
        """
        :param str magnitude_path: Path to a .pymagnitude embeddings file.
        """

        self.database = magnitude_path
        if self.database is not None:
            self.embeddings = Magnitude(self.database)

    def _embeddings(self, tokens):
        """
        loads a subset of all embeddings into a DataFrame.

        :param set tokens: set of tokens to get embeddings for

        :return: Dataframe containing embeddings
        :rtype: Dataframe
        """

        tokens = list(set(tokens))
        vectors = [self.embeddings.query(token) for token in tokens]
        df = DataFrame(data=vectors, index=tokens)

        return df

    def generate2d(self, tokens, method='umap'):
        """
        creates 2d-coordinates for a list of tokens

        :param list tokens: list of tokens to generate coordinates for
        :param str method: umap / tsne

        :return: pandas.Dataframe with x and y coordinates
        :rtype: pandas.Dataframe
        """

        # load vectors
        embeddings = self._embeddings(tokens)

        # if no vectors are loaded
        if embeddings.empty:
            return DataFrame()

        # just in case
        embeddings = embeddings.dropna()

        # set up transformer
        if method == 'tsne':
            transformer = TSNE(n_components=2,
                               metric='euclidean',
                               perplexity=10.,
                               verbose=0)

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

    def add(self, item, cutoff=.2):
        """
        Calculate new coordinates for one embedding, based on cosine similarity.

        :param str item: token to add
        :param float cutoff: cut-off value for cosine similarity

        :return: pandas.Series with ['tsne_x', 'tsne_y', 'user_x', 'user_y'] for
        :rtype: pandas.Series
        """

        # get embedding for items
        item_embedding = self._embeddings([item])[0]

        # gather all similar embeddings
        similarities = []
        for base_embedding in self.coordinates.values:
            similarity = 1 - cosine(item_embedding, base_embedding)

            if similarity >= cutoff:
                similarities.append(similarity)
            else:
                similarities.append(0)

        global_similarity_index = sum(similarities)

        if global_similarity_index == 0:
            # put in global center
            new_coordinates = self.coordinates.sum() / len(self.coordinates)
        else:
            # weighted average
            tmp_coordinates = self.coordinates.apply(lambda x: x * similarities)
            new_coordinates = tmp_coordinates.sum() / global_similarity_index

        # append new coordinates
        self.coordinates = self.coordinates.append(
            DataFrame(
                data={
                    'x': new_coordinates['x'],
                    'y': new_coordinates['y'],
                },
                index=[item]
            )
        )

        return new_coordinates

    def visualize(self, size,
                  title='semantic map',
                  path="/tmp/vis.html"):
        """
        :param Series size: pd.Series containing label sizes
        """

        output_file(path)
        print(self.coordinates.join(size).columns)
        source = ColumnDataSource(self.coordinates.join(size))
        p = figure(title=title)
        p.scatter(x='x', y='y', size=size.name, source=source)
        p.xaxis[0].axis_label = ''
        p.yaxis[0].axis_label = ''
        # coordinate labels = items
        labels = LabelSet(x='x', y='y', text='item', level='glyph',
                          x_offset=5, # text_font_size=size.name,
                          y_offset=5, source=source,
                          render_mode='canvas')
        p.add_layout(labels)
        show(p)
        print(p)
