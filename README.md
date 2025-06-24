# Semantic Map

This is a Python3 module for creating and updating coordinates of semantic maps. Semantic maps are 2-dimensional representations of sets of type embeddings, the types usually gained from collocation or keyword analyses.

The module tackles two problems:
- OOV in embeddings look-up (coordinates can be generated for all items, including items without embeddings)
- iterative projection onto lower-dimensional coordinates (items can be added to an existing projection)

Limitations:
- no functionality for context-aware embeddings (ELMo, BERT)
- for updating the store dynamically (e.g. for MWUs), I will have to migrate e.g. to FAISS

## Installation

```
pip install git+https://github.com/ausgerechnet/semmap.git
```

## Embeddings

### Import
- module supports creation of embeddings via SentenceTransformers 
- it can also read in pre-created embeddings from FastText ("C-text") format
  + NB: pre-computed embeddings might come without n-gram character encodings (or other subtoken representations)

### CLI for creating embeddings
- creation of embeddings via `transformers`
  + input: .tsv
  + output: embeddings-keys.txt, embeddings-representations.tsv → embeddings.tsv
- storage of embeddings via `annoy`
  + input: embeddings.tsv
  + output: embeddings.ann

### Storage
- by default, embeddings are stored in a custom storage (EmbeddingsStore) based on annoy
  + EmbeddingsStore config file ends on ".semmap", links to annoy database and type dictionary
- alternatively, embeddings can be stored in a pymagnitude database
  
### OOV functionality
- pymagnitude:
  + construct character n-gram embeddings during import time
  + center and normalise randomly (but reproducibly)
  + interpolate with in-vocabulary words via string similarity
  + improved string similarity: morphology-aware for English, shrinking repeated characters, ...

- EmbeddingsStore:
  + default: create on the fly -- only reasonable for `SentenceTransformers` (or `FastText`, but NotImplemented)
  + random (but reproducible) init
  + based on string similarity via `levenshtein` (edit distance)

- NB FastText:
  + constructs character n-gram embeddings during initial encoding (e.g. FastText)
  + yields OOV support for all words if at least one character n-gram has been observed (trivial for unigrams: alphabet)

## Nearest neighbours

- annoy functionality (fixed lookup trees)
- pymagnitude also uses annoy

## Working with semantic maps

### API
- central API offered by semmap.SemanticSpace
- init with `path` (must end on "magnitude" or "semmap")

### dimensionality reduction
- default: `sklearn.manifold.TSNE`
- `umap.UMAP`
- `openTSNE.TSNE`

### iterative projection
- default: convex combination of 2d mapping of similar types (cosine similarity of high-dimensional embeddings)
- random (but reproducible) projection
- iterative t-SNE (NotImplemented, `openTSNE`)
- iterative UMAP (NotImplemented, `umap-learn`)

## Roadmap

- [ ] PyPI
- [x] github tests
- [x] OOV
- [x] use annoy instead of pymagnitude
- [ ] use openTSNE instead of sklearn -- fails when there's many items ("IndexError: Vector has wrong length (expected 300, got 1000)")
- [ ] iterative projection (add-item) with `openTSNE` and `UMAP`
