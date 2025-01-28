"""create embeddings from LLM"""

from argparse import ArgumentParser
import gzip

from ccc import Corpus
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("cwb_id")
    parser.add_argument("--p_att", default="lemma")
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--path_out", default=None)
    args = parser.parse_args()

    cwb_id = args.cwb_id
    p_att = args.p_att
    model = args.model
    path_out = args.path_out

    if path_out is None:
        path_out = f'{cwb_id}-{p_att}.txt.gz'

    # cwb_id = "GERMAPARL-1949-2021"
    # p_att = "lemma"
    # model = "paraphrase-multilingual-MiniLM-L12-v2"
    # path_out = "test.txt"

    print(f"getting marginals of p-att '{p_att}' in corpus '{cwb_id}'")
    corpus = Corpus(cwb_id)
    items = list(corpus.marginals(p_atts=[p_att]).index)

    print(f"loading model ({model})")
    encoder = SentenceTransformer(model)

    print(f"creating embeddings for {len(items)} items")
    embeddings = encoder.encode(items)

    print(f"writing to '{path_out}'")
    with gzip.open(path_out, "wt") as f:
        f.write(f"{str(len(embeddings))} {str(len(embeddings[0]))}\n")
        for item, embedding in zip(items, embeddings):
            f.write(item + " ")
            f.write(" ".join([str(i) for i in embedding.tolist()]) + "\n")
