# Semantic Map

This is a Python3 module for creating and updating coordinates of semantic maps. (Semantic maps are 2-dimensional representations of 

tackles two problems:
- OOV in embeddings look-up (coordinates can be generated for all keys, including keys without embeddings)
- iterative projection onto lower-dimensional co-oordinates (keys can be added to an existing projection)

## OOV functionality
- construct character n-gram embeddings during training time (e.g. FastText)
  + yields OOV support for all words if at least one character n-gram has been observed (trivial for unigrams: alphabet)
  + but: pre-computed embeddings might come without n-gram character encodings
- pymagnitude:
  + construct character n-gram embeddings during import time
  + center and normalise randomly (but reproducible)
  + interpolate with in-vocabulary words via string similarity
  + improved string similarity: morphology-aware for English, shrinking repeated characters, ...

## Iterative projection
- random projection (but reproducible)
- convex combination of similar keys (cosine similarity of embeddings)
- iterative t-SNE
- iterative UMAP

NB: no functionality for context-aware embeddings (ELMo, BERT)

## Features

- creation of embeddings via `transformers`
  + input: .tsv
  + output: embeddings-keys.txt, embeddings-representations.tsv → embeddings.tsv
- storage of embeddings via `annoy`
  + input: embeddings.tsv
  + output: embeddings.ann
- OOV via `levenshtein`: edit distance
- OOV via `transformers` and / or `fasttext`: sub-token (character-level) embeddings
- nearest-neighbour lookup via `annoy`

- dimensionality reduction via `sklearn`
- adding points via cosine similarity
- iterative dimensionality reduction via `openTSNE` and `umap-learn`

## Roadmap

- [ ] iterative projections
- [ ] PyPI
- [x] github tests
- [ ] use openTSNE instead of sklearn -- fails when there's many items ("IndexError: Vector has wrong length (expected 300, got 1000)")
- [ ] iterative projection (add-item) with openTSNE and UMAP
- [ ] OOV
- [ ] annoy instead of pymagnitude
