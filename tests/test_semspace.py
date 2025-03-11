from pandas import read_csv
from semmap.semspace import SemanticSpace
# import pytest


MAGNITUDE_PATH = "tests/data/deWiki-small.magnitude"


# EMBEDDINGS
def test_embeddings():
    items = ['angela', 'merkel']
    semspace = SemanticSpace()
    embeddings = semspace._embeddings(items)
    assert all(x in embeddings.index for x in items)


# DIMENSIONALITY REDUCTION
def test_generate2d_tsne():

    semspace = SemanticSpace(MAGNITUDE_PATH)
    base_items = open("tests/data/base-items.voc", "rt").read().strip().split("\n")
    assert all(isinstance(x, str) for x in base_items)
    base_items = base_items[:100]
    coordinates = semspace.generate2d(base_items, method='tsne')
    assert all(x in coordinates.index for x in base_items)


def test_generate2d_openTSNE():

    semspace = SemanticSpace(MAGNITUDE_PATH)
    base_items = open("tests/data/base-items.voc", "rt").read().strip().split("\n")
    assert all(isinstance(x, str) for x in base_items)
    base_items = base_items[:100]
    coordinates = semspace.generate2d(base_items, method='openTSNE')
    assert all(x in coordinates.index for x in base_items)


def test_generate2d_umap():

    semspace = SemanticSpace(MAGNITUDE_PATH)
    base_items = open("tests/data/base-items.voc", "rt").read().strip().split("\n")
    assert all(isinstance(x, str) for x in base_items)
    base_items = base_items[:100]
    coordinates = semspace.generate2d(base_items, method='umap')
    assert all(x in coordinates.index for x in base_items)


# UPDATE
def test_add_item():

    semspace = SemanticSpace(MAGNITUDE_PATH)
    df = read_csv("tests/data/BREXIT_merkel-ll-2d.tsv", sep="\t", index_col=0)
    semspace.coordinates = df[['x', 'y']]
    assert 'test' not in semspace.coordinates
    coordinates = semspace.add(['test'])
    assert 'test' in coordinates.index
    assert 'test' in semspace.coordinates.index


# def test_vis(brexit_corpus):
#     df = read_csv("tests/data/BREXIT_merkel-ll-2d.tsv", sep="\t", index_col=0)
#     semspace = SemanticSpace(brexit_corpus['magnitude_path'])
#     semspace.coordinates = df[['x', 'y']]


# def test_pipeline_sz(sz_corpus):
#     ams = read_csv("tests/data/ufa-sz-atomkraft/201308.tsv", sep="\t", index_col=0)
#     semspace = SemanticSpace(sz_corpus['magnitude_path'])
#     semspace.generate2d(ams.index)
#     print(semspace.coordinates)


# NORMALISATION
def test_normalise():

    semspace = SemanticSpace(MAGNITUDE_PATH, normalise=True)
    base_items = open("tests/data/base-items.voc", "rt").read().strip().split("\n")
    new_items = open("tests/data/new-items.voc", "rt").read().strip().split("\n")

    semspace.generate2d(base_items)
    wrong = semspace.coordinates.loc[(semspace.coordinates.x.abs() > 1) | (semspace.coordinates.y.abs() > 1)]
    assert len(wrong) == 0

    new_coordinates = semspace.add(new_items)
    wrong = new_coordinates.loc[(new_coordinates.x.abs() > 1) | (new_coordinates.y.abs() > 1)]
    assert len(wrong) == 0
