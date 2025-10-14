from pandas import read_csv
from semmap.semspace import SemanticSpace
from semmap.embeddings_store import create_embeddings_store
import random
import string
import os
import pytest


MAGNITUDE_PATH = "tests/data/deWiki-small.magnitude"


# EMBEDDINGS
def test_embeddings():
    items = ['angela', 'merkel']
    semspace = SemanticSpace()
    embeddings = semspace._embeddings(items)
    assert all(x in embeddings.index for x in items)


def test_embeddings_store():
    path_items = "tests/data/germaparl.voc"
    path_settings = "tests/data/germaparl.semmap"
    path_db = "tests/data/germaparl.sqlite"
    path_annoy = "tests/data/germaparl.annoy"
    if not os.path.exists(path_settings):
        create_embeddings_store(path_items, path_settings, path_db, path_annoy)
    semspace = SemanticSpace(path_settings)
    items = ['CDU', 'CSU']
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


# PERFORMANCE
@pytest.mark.now
def test_many_new():

    path_settings = "tests/data/germaparl.semmap"
    semspace = SemanticSpace(path_settings, normalise=True)
    from time import time

    random.seed(42)
    new_items = [
        ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        for _ in range(100)
    ]
    semspace.database.model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    start = time()
    semspace._embeddings(new_items)
    end = time()
    elapsed = end - start
    print(f"elapsed time: {elapsed:.6f} seconds")

    start = time()
    semspace.generate2d(new_items)
    end = time()
    elapsed = end - start
    print(f"elapsed time: {elapsed:.6f} seconds")
