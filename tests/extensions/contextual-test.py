from sentence_transformers import SentenceTransformer
from ccc import Corpus
from sklearn.manifold import TSNE
from pandas import DataFrame
from pandas import concat
from semmap.embeddings import create_contextual_embeddings, create_embeddings


def save_experiment(name_sentence, tokens):

    df_cf = create_embeddings(tokens)
    df_cf['method'] = 'contextfree'
    df_cf['token_id'] = list(range(len(tokens)))

    df_tokens, df_subtokens = create_contextual_embeddings(tokens)
    df_tokens['method'] = 'contextual'
    df_tokens['token_id'] = list(range(len(tokens)))
    df_subtokens['method'] = 'contextual_subtokens'

    dfs_cf_tokens = list()
    dfs_cf_subtokens = list()
    for i, token in enumerate(tokens):
        df_cf_tokens, df_cf_subtokens = create_contextual_embeddings([tokens[i]])
        df_cf_tokens['method'] = 'contextual_contextfree'
        df_cf_tokens['token_id'] = i
        df_cf_subtokens['method'] = 'contextual_contextfree_subtokens'
        df_cf_subtokens['token_id'] = i
        dfs_cf_tokens.append(df_cf_tokens)
        dfs_cf_subtokens.append(df_cf_subtokens)

    df = concat([df_cf, df_tokens, df_subtokens] + dfs_cf_tokens + dfs_cf_subtokens)
    df['name_sentence'] = name_sentence
    df.index.name = 'token'
    df.to_csv(name_sentence + '.tsv.gz', sep="\t", compression="gzip")


# Embeddings of "Korpus" / "Corpus"
save_experiment("Korpus-Text", "Das Korpus besteht aus vielen Texten .".split(" "))
save_experiment("Korpus-Text-sentence", ["Das Korpus besteht aus vielen Texten ."])
save_experiment("Korpus-Schrauben", "Der Korpus muss mit Schrauben befestigt werden .".split(" "))
save_experiment("Korpus-Schrauben-sentence", ["Das Korpus besteht aus vielen Texten ."])
save_experiment("Corpus-Text", "The corpus consists of several texts .".split(" "))
save_experiment("corpus-Text-sentence", ["The corpus consists of several texts ."])
save_experiment("Corpus-Screws", "The carcass must be fastened with screws .".split(" "))
save_experiment("Corpus-Screws-sentence", ["The carcass must be fastened with screws ."])


# we search for economy / Wirtschaft in English tweets and GermaParl
germaparl = Corpus("GERMAPARL-1949-2021")
wirtschaft = germaparl.query('[lemma="Wirtschaft"]', context_break='s')
wirtschaft_profile = wirtschaft.collocates(cut_off=500, order='conservative_log_ratio')

# encoding
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
wirtschaft_embeddings = model.encode(list(wirtschaft_profile.index))

# dataframe conversion
df = DataFrame(wirtschaft_embeddings)
df.index = wirtschaft_profile.index

# TSNE
parameters_ = dict(
    n_components=2,
    perplexity=min(30.0, len(df) - 1),
    early_exaggeration=12.0,
    learning_rate='auto',
    n_iter=1000,
    n_iter_without_progress=300,
    min_grad_norm=1e-07,
    metric='cosine',
    metric_params=None,
    init='pca',
    verbose=0,
    random_state=None,
    method='barnes_hut',
    angle=0.5,
    n_jobs=4
)
transformer = TSNE(**parameters_)
data2d = transformer.fit_transform(df)

# save as coordinates
coordinates = DataFrame(data=data2d, index=df.index, columns=['x', 'y'])
coordinates.to_csv("multilingual-test.tsv", sep="\t")
