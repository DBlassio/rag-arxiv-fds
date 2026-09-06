"""Reranking module: cross-encoder rescoring of bi-encoder candidates."""

from FlagEmbedding import FlagReranker
from src.retrieval import RetrievedChunk

_reranker: FlagReranker | None = None


def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
    return _reranker


def rerank(query: str, chunks: list[RetrievedChunk], top_n: int = 3) -> list[RetrievedChunk]:
    reranker = get_reranker()
    pairs = [[query, c.text] for c in chunks]
    scores = reranker.compute_score(pairs, normalize=True)
    scored = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in scored[:top_n]]