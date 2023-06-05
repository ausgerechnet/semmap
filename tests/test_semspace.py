from pandas import read_csv

from semmap.semspace import SemanticSpace
import pytest


def test_embeddings(brexit_corpus):
    semspace = SemanticSpace(brexit_corpus['magnitude_path'])
    embeddings = semspace.embeddings(['angela', 'merkel'])
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
    print(semspace.add(['test']))
    print(semspace.coordinates)


def test_vis(brexit_corpus):
    df = read_csv("tests/data/BREXIT_merkel-ll-2d.tsv", sep="\t", index_col=0)
    semspace = SemanticSpace(brexit_corpus['magnitude_path'])
    semspace.coordinates = df[['x', 'y']]


def test_pipeline_sz(sz_corpus):
    ams = read_csv("tests/data/ufa-sz-atomkraft/201308.tsv", sep="\t", index_col=0)
    semspace = SemanticSpace(sz_corpus['magnitude_path'])
    semspace.generate2d(ams.index)
    print(semspace.coordinates)
