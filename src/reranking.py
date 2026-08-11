# Our reranking model is a cross-encoder that takes a query and a candidate chunk and outputs a score.
#  Indicates how relevant the chunk is to the query. 
#  We use this to rerank the top-k candidates returned by the bi-encoder retrieval model.

from FlagEmbedding import FlagReranker
from src.retrieval import RetrievedChunk

_reranker: FlagReranker | None = None


#We load the reranker model globally to avoid reloading it every time
def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
    return _reranker


# We rerank the top-k candidates returned by the bi-encoder retrieval model using the cross-encoder reranker.   
def rerank(query: str, chunks: list[RetrievedChunk], top_n: int = 3) -> list[RetrievedChunk]:
    reranker = get_reranker()
    pairs = [[query, c.text] for c in chunks]
    scores = reranker.compute_score(pairs, normalize=True)  # [0,1] higher = more relevant
    scored = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in scored[:top_n]]