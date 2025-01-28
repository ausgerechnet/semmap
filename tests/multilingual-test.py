from sentence_transformers import SentenceTransformer
from ccc import Corpus
from sklearn.manifold import TSNE
from pandas import DataFrame, concat

# we search for economy / Wirtschaft in English tweets and GermaParl
germaparl = Corpus("GERMAPARL-1949-2021")
wirtschaft = germaparl.query('[lemma="Wirtschaft"]', context_break='s')
wirtschaft_profile = wirtschaft.collocates(cut_off=500, order='conservative_log_ratio')

austerity = Corpus("AUSTERITY_0925_FINAL")
economy = austerity.query('[tt_lemma="economy"]', context_break='s')
economy_profile = economy.collocates(p_query=['tt_lemma'], cut_off=500, order='conservative_log_ratio')

# model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
wirtschaft_embeddings = model.encode(list(wirtschaft_profile.index))
economy_embeddings = model.encode(list(economy_profile.index))

wirtschaft_df = DataFrame(wirtschaft_embeddings)
wirtschaft_df.index = wirtschaft_profile.index
economy_df = DataFrame(economy_embeddings)
economy_df.index = economy_profile.index

df = concat([economy_df.head(50), wirtschaft_df.head(50)])

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
coordinates = DataFrame(data=data2d, index=df.index, columns=['x', 'y'])
coordinates.to_csv("multilingual-test.tsv", sep="\t")
