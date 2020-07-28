from ccc import Corpus


corpus = Corpus("BREXIT_V20190522_DEDUP")
matches = corpus.query('[lemma="Merkel"%cd]', context=20, s_context='tweet')
collocates = corpus.collocates(matches, p_query="lemma")
df = collocates.show(order='log_likelihood')
df.to_csv("BREXIT_merkel-ll.tsv", sep="\t")
