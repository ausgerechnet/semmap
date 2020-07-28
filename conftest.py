import pytest


@pytest.fixture(scope='function')
def brexit_corpus():
    """ settings for BREXIT_V20190522_DEDUP """

    registry_path = "/home/ausgerechnet/corpora/cwb/registry/"

    corpus_name = "BREXIT_V20190522_DEDUP"

    lib_path = (
        "/home/ausgerechnet/repositories/spheroscope/library/"
        "BREXIT_V20190522_DEDUP/"
    )
    meta_path = (
        "/home/ausgerechnet/corpora/cwb/upload/"
        "brexit/brexit-preref-rant/brexit_v20190522_dedup.tsv.gz"
    )
    magnitude_path = (
        "/home/ausgerechnet/corpora/embeddings/magnitude/"
        "enTwitterWord2Vec.magnitude"
    )

    return {
        'registry_path': registry_path,
        'corpus_name': corpus_name,
        'lib_path': lib_path,
        'meta_path': meta_path,
        'magnitude_path': magnitude_path
    }
