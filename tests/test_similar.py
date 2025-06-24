import pytest
from pandas import DataFrame
import os
from semmap.semspace import SemanticSpace
from semmap.embeddings_store import create_embeddings_store

MAGNITUDE_PATH = "tests/data/deWiki-small.magnitude"


def test_most_similar_magnitude():

    semspace = SemanticSpace(MAGNITUDE_PATH)
    items = ['gehen', 'laufen']
    df = semspace.most_similar(items, n=1000)
    assert isinstance(df, DataFrame)
    assert df.index[0] == 'lassen'


def test_most_similar():

    path_items = "tests/data/germaparl.voc"
    path_settings = "tests/data/germaparl.semmap"
    path_db = "tests/data/germaparl.sqlite"
    path_annoy = "tests/data/germaparl.annoy"

    if not os.path.exists(path_settings):
        create_embeddings_store(path_items, path_settings, path_db, path_annoy)

    items = ['gehen', 'laufen']
    semspace = SemanticSpace(path_settings)
    df = semspace.most_similar(items, n=1000)
    assert isinstance(df, DataFrame)
    assert df.index[0] == 'laufen'
    assert df.index[1] == 'gehen'
    assert df.index[2] == 'treten'


@pytest.mark.now
def test_most_similar_mwu():

    path_items = "tests/data/germaparl.voc"
    path_settings = "tests/data/germaparl.semmap"
    path_db = "tests/data/germaparl.sqlite"
    path_annoy = "tests/data/germaparl.annoy"

    if not os.path.exists(path_settings):
        create_embeddings_store(path_items, path_settings, path_db, path_annoy)

    items = ["nach Hause gehen"]
    semspace = SemanticSpace(path_settings)
    df = semspace.most_similar(items, n=1000)
    assert isinstance(df, DataFrame)
    print(df)
