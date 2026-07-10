This is a testable retrieval-augmented generation system over
local PDF sources: it loads and splits documents, embeds chunks, stores vectors, retrieves relevant evidence,
and produces an answer that either cites the available information or explicitly refuses to answer when
the data is insufficient. The embeding model is the light sentence-transformer
model sentence-transformers/all-MiniLM-L6-v2. First, a cheap retrieval model scores a large number of
chunks and keeps only a small set of promising candidates, for example the best 10 chunks. Then a more
expensive second-stage model is applied only to those candidates to find the most valuable ones. In real
RAG systems this second stage can be implemented either with an LLM judge or with a cross-encoder
reranker. In this project the reranking stage uses a cross-encoder model.
