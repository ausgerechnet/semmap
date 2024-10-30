from docarray import BaseDoc
from docarray.typing import NdArray
from docarray import DocList
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB
import numpy as np


class ToyDoc(BaseDoc):
    text: str = ''
    embedding: NdArray[128]


# Specify your workspace path
db = InMemoryExactNNVectorDB[ToyDoc](workspace='./workspace_path')

# Index a list of documents with random embeddings
doc_list = [ToyDoc(text=f'toy doc {i}', embedding=np.random.rand(128)) for i in range(1000)]
db.index(inputs=DocList[ToyDoc](doc_list))

# Perform a search query
query = ToyDoc(text='query', embedding=np.random.rand(128))
results = db.search(inputs=DocList[ToyDoc]([query]), limit=10)

# Print out the matches
for m in results[0].matches:
    print(m)
