from semmap.embeddings_store import create_embeddings_store, EmbeddingsStore
import numpy as np
import os
import pytest


def test_create_embeddings_store():

    path_items = "tests/data/germaparl.voc"
    path_settings = "tests/data/germaparl.semmap"
    path_db = "tests/data/germaparl.sqlite"
    path_annoy = "tests/data/germaparl.annoy"
    if not os.path.exists(path_settings):
        store = create_embeddings_store(path_items, path_settings, path_db, path_annoy)
    else:
        store = EmbeddingsStore(path_settings)
    print(store)


@pytest.mark.now
def test_update_embeddings_nothing_to_do():

    path_settings = "tests/data/germaparl.semmap"
    path_items = "tests/data/germaparl.voc"
    store = EmbeddingsStore(path_settings)
    store = store.update(path_items)


@pytest.mark.now
def test_update_embeddings_store():

    path_settings = "tests/data/germaparl.semmap"
    path_items = "tests/data/new-items.voc"
    store = EmbeddingsStore(path_settings)
    store = store.update(path_items)


def test_get_embeddings():
    path_settings = "tests/data/germaparl.semmap"
    store = EmbeddingsStore(path_settings)

    store.get_embeddings(["gehen"], oov_info=True)[[1, 'oov']]
    store.get_embeddings(["gehes"], similarity_threshold=.8, oov_info=True)[[1, 'oov']]
    store.get_embeddings(["gehend"], similarity_threshold=.8, oov_info=True)[[1, 'oov']]
    store.get_embeddings(["tion2sdaf"], similarity_threshold=.8, oov_info=True)[[1, 'oov']]


def test_query():
    path_settings = "tests/data/germaparl.semmap"
    store = EmbeddingsStore(path_settings)
    embeddings = store.query("CDU")
    assert isinstance(embeddings, np.ndarray)


def test_most_similar():
    path_settings = "tests/data/germaparl.semmap"
    store = EmbeddingsStore(path_settings)
    a = store.find_neighbours("gehen")
    b = store.most_similar(["gehen"])
    assert a == b

    new = store.most_similar(["gehen"], ["kommen"])
    assert new[0][0] == "gehen"
    assert new[1][0] == "verlassen"
