from openTSNE import TSNE
from pandas import read_csv
from pymagnitude import Magnitude
import numpy as np


MAGNITUDE_PATH = "tests/data/deWiki-small.magnitude"


def test_projection():

    mag = Magnitude(MAGNITUDE_PATH)

    base_items = read_csv("tests/data/base-items.tsv", sep="\t", index_col=0, keep_default_na=False)
    base_items = list(base_items.head(500).index)

    embeddings = [mag.query(item) for item in base_items]
    embeddings_array = np.vstack(embeddings)

    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )

    projection = tsne.fit(embeddings_array)
    print(projection)
