#This is our retrieval module.
#It takes a question, embeds it, and searches the vector store for the most relevant chunks of text.

from dataclasses import dataclass
from src.embeddings import embed_texts
from src.vectorstore import get_client, get_or_create_collection, query as vs_query


#We first create a dataclass to hold the retrieved chunks of text, along with their metadata and distance from the query vector.
@dataclass
class RetrievedChunk:
    text: str
    title: str
    arxiv_id: str
    distance: float

#Our retrieve function but now with Reranking
#We add a second layer of retrieval that combines the bi-encoder retrieval with the cross-encoder reranking.
def retrieve(question: str,n_results: int = 3,use_reranking: bool = False,n_candidates: int = 20) -> list[RetrievedChunk]:
    query_vec = embed_texts([question])[0]
    client = get_client()
    collection = get_or_create_collection(client)

    #If we rerank, we fetch more candidates from the vector store to allow the reranker to select the best ones.
    fetch_n = n_candidates if use_reranking else n_results
    results = vs_query(collection, query_vec, n_results=fetch_n)

    chunks = [
        RetrievedChunk(text=doc, title=meta.get("title", "unknown"),arxiv_id=meta.get("arxiv_id", ""), distance=dist)
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])]

    if use_reranking:
        from src.reranking import rerank
        chunks = rerank(question, chunks, top_n=n_results)

    return chunks

