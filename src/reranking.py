# Our reranking model is a cross-encoder that takes
# a query and a candidate chunk and outputs a score.

from FlagEmbedding import FlagReranker
from src.retrieval import RetrievedChunk


# Local model baked into the Docker image
MODEL_PATH = "/opt/models/bge-reranker-v2-m3"

_reranker: FlagReranker | None = None


def get_reranker() -> FlagReranker:
    global _reranker

    if _reranker is None:
        _reranker = FlagReranker(
            MODEL_PATH,
            use_fp16=False,
        )

    return _reranker


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int = 3,
) -> list[RetrievedChunk]:

    reranker = get_reranker()

    pairs = [[query, c.text] for c in chunks]

    scores = reranker.compute_score(
        pairs,
        normalize=True,
    )

    scored = sorted(
        zip(chunks, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )

    return [
        chunk
        for chunk, _ in scored[:top_n]
    ]