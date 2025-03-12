#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip

import torch
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizerFast

from .utils import Progress


def create_contextual_embeddings(tokens, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    """create embeddings from list of tokens

    """
    # Load pre-trained model and tokenizer
    print(f"loading model ({model_name})")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Tokenize input text and convert to tensor
    print(f"creating embeddings for {len(tokens)} items")
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    subtokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract last hidden state (contextual embeddings)
    subtoken_embeddings = outputs.last_hidden_state

    # Align embeddings with whole words
    token_ids = inputs.word_ids(batch_index=0)
    token_embeddings = []
    current_token_id = None
    current_token_embedding = []

    for i, token_id in enumerate(token_ids):

        if token_id is None:
            continue  # Skip special tokens like [CLS], [SEP]

        if token_id != current_token_id:
            if current_token_embedding:
                # aggregate subtoken embeddings and append mean to token embedding
                token_embeddings.append(torch.mean(torch.stack(current_token_embedding), dim=0))
            current_token_embedding = [subtoken_embeddings[0, i, :]]
            current_token_id = token_id
        else:
            current_token_embedding.append(subtoken_embeddings[0, i, :])

    # Append last token embedding
    if current_token_embedding:
        token_embeddings.append(torch.mean(torch.stack(current_token_embedding), dim=0))

    # convert to dataframes
    df_tokens = DataFrame(index=tokens, data=torch.stack(token_embeddings))
    df_tokens.index.name = 'token'
    df_subtokens = DataFrame(index=subtokens, data=subtoken_embeddings[0])
    df_subtokens['token_id'] = token_ids
    df_subtokens['token_id'] = df_subtokens['token_id'].fillna(-1).astype(int)
    df_subtokens.index.name = 'subtoken'

    return df_tokens, df_subtokens


def create_embeddings(items, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', path_out=None, mode='wt', as_is=False):
    """create context-free embeddings from list of types

    """
    print(f"loading model ({model_name})")
    encoder = SentenceTransformer(model_name)

    print(f"creating embeddings for {len(items)} items")
    embeddings = encoder.encode(items)

    if not path_out:
        if as_is:
            return embeddings
        df_embeddings = DataFrame(index=items, data=embeddings)
        df_embeddings.index.name = 'token'
        return df_embeddings

    print(f"writing to '{path_out}'")
    pb = Progress(len(items))
    with gzip.open(path_out, mode) as f:
        f.write(f"{str(len(embeddings))} {str(len(embeddings[0]))}\n")
        for item, embedding in zip(items, embeddings):
            f.write(item + " ")
            f.write(" ".join([str(i) for i in embedding.tolist()]) + "\n")
            pb.up()


def read_text_embeddings(path_in):
    """read from file where
    - first line contains number of rows and number of dimensions
    - each following line contains item and vector

    """
    with gzip.open(path_in, 'rt') as f:

        line = f.readline()
        row = line.rstrip().split(" ")
        dim_row = int(row[0])
        dim_col = int(row[1])

        items = list()
        vectors = list()
        pb = Progress(dim_row)
        for line in f:
            # skip empty lines
            if line.strip() == "":
                continue
            row = line.split(" ")
            dims = len(row)
            if not dims >= dim_col + 1:
                raise ValueError(f'line does not have enough dimensions:\n{row}')
            dim_token = dims - dim_col
            item = " ".join(row[:dim_token])
            items.append(item)
            vectors.append(row[dim_token:])
            pb.up()

    return items, vectors
