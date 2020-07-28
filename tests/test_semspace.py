from semmap.semspace import SemanticSpace
from pandas import read_csv
import pytest


def test_embeddings(brexit_corpus):
    semspace = SemanticSpace(brexit_corpus['magnitude_path'])
    embeddings = semspace._embeddings(['angela', 'merkel'])
    print(embeddings)


def test_generate2d(brexit_corpus):
    ams = read_csv("tests/data/BREXIT_merkel-ll.tsv", index_col=0, sep="\t")
    semspace = SemanticSpace(brexit_corpus['magnitude_path'])
    embeddings = semspace.generate2d(ams.index)
    df = embeddings.join(ams)
    df.to_csv("tests/data/BREXIT_merkel-ll-2d.tsv", sep="\t")


def test_generate2d_tsne(brexit_corpus):
    ams = read_csv("tests/data/BREXIT_merkel-ll.tsv", index_col=0, sep="\t")
    semspace = SemanticSpace(brexit_corpus['magnitude_path'])
    embeddings = semspace.generate2d(ams.index, method='tsne')
    df = embeddings.join(ams)
    print(df)


def test_add_item(brexit_corpus):
    semspace = SemanticSpace(brexit_corpus['magnitude_path'])
    df = read_csv("tests/data/BREXIT_merkel-ll-2d.tsv", sep="\t", index_col=0)
    semspace.coordinates = df[['x', 'y']]
    print(semspace.add('test'))
    print(semspace.coordinates)


@pytest.mark.now
def test_vis(brexit_corpus):
    df = read_csv("tests/data/BREXIT_merkel-ll-2d.tsv", sep="\t", index_col=0)
    semspace = SemanticSpace(brexit_corpus['magnitude_path'])
    semspace.coordinates = df[['x', 'y']]
    semspace.visualize(size=df['log_ratio'])
