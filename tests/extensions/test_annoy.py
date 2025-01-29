from semmap.semspace import SemMap


def test_embeddings(brexit_corpus):
    semspace = SemMap()
    embeddings = semspace._embeddings(['angela', 'merkel'])
    print(embeddings)
