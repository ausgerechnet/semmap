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


def test_update_embeddings_nothing_to_do():

    path_settings = "tests/data/germaparl.semmap"
    path_items = "tests/data/germaparl.voc"
    store = EmbeddingsStore(path_settings)
    store = store.update(path_items)


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


def test_random_reproducible():
    path_settings = "tests/data/germaparl.semmap"
    store = EmbeddingsStore(path_settings)
    a = store._generate_random_vector("Covid-19")
    b = store._generate_random_vector("Covid-19")
    store = EmbeddingsStore(path_settings)
    c = store._generate_random_vector("Covid-19")
    assert a == b == c
    assert a[0] == -0.2022167449939105


def test_get_embeddings_oov():
    path_settings = "tests/data/germaparl.semmap"
    store = EmbeddingsStore(path_settings)
    positive = ["gehen", "rennen", "laufen"]
    vecs = store.get_embeddings(positive, oov_info=True)
    oovs = vecs['oov']
    assert oovs.loc['gehen'] == oovs.loc['laufen'] is False
    assert oovs.loc['rennen'] == "random"


@pytest.mark.now
def test_most_similar():
    path_settings = "tests/data/germaparl.semmap"
    store = EmbeddingsStore(path_settings)
    a = store.find_neighbours("gehen")
    b = store.most_similar(["gehen"])
    # convert to similarities
    c = [(_[0], 1 - _[1] ** 2 / 2) for _ in a]
    assert b == c

    new = store.most_similar(["gehen", "laufen"], topn=200)
    assert (len(new) == 200)
    assert new[0][0] == "laufen"
    assert new[1][0] == "gehen"
    assert new[2][0] == "treten"

    new = store.most_similar(["gehen"], ["kommen"], topn=200)
    assert (len(new) == 200)
    assert new[0][0] == "gehen"
    assert new[1][0] == "verlassen"


def test_most_similar_2():

    path_settings = "tests/data/germaparl.semmap"
    store = EmbeddingsStore(path_settings)

    new = store.most_similar(["gehen", "laufen"], ["kommen", "verlassen", "wollen"], topn=200)
    assert (len(new) == 200)
    assert new[0][0] == "Mindestlohn"
