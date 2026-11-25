#This is our retrieval module.
#It takes a question, embeds it, and searches the vector store for the most relevant chunks of text.

from dataclasses import dataclass
from src.embeddings import embed_texts
from src.vectorstore import get_client, get_or_create_collection, query as vs_query

#We first create a dataclass to hold the retrieved chunks of text, along with their metadata and distance from the query vector.
@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    title: str
    arxiv_id: str
    distance: float
    rerank_score: float | None = None

@dataclass
class RetrievalTrace:
    candidates: list[RetrievedChunk]
    final_chunks: list[RetrievedChunk]


#Our retrive funcion. Retrieves candidates (n_candidates) and preserve the final selection (n_results).
def retrieve_with_trace(question: str, n_results:int=3, use_reranking: bool = False, n_candidates: int = 20) -> RetrievalTrace:

    #We convert the question into an embedding
    query_vec = embed_texts([question])[0]

    #Connect to ChromaDB
    client = get_client()
    collection = get_or_create_collection(client)

    #Retrive the candidates
    results = vs_query(collection,query_vec,n_results=n_candidates)

    #Convert ChormaDB results into RetrivedChunk objects
    candidates = [RetrievedChunk(chunk_id=chunk_id,text=doc,title=meta.get("title", "unknown"),arxiv_id=meta.get("arxiv_id", ""),distance=dist)
                  for chunk_id, doc, meta, dist in zip(results["ids"][0],results["documents"][0],results["metadatas"][0],results["distances"][0])]

    #Select the final
    if use_reranking:
        from src.reranking import rerank
        final_chunks = rerank(question, candidates,top_n=n_results)
    else:
        final_chunks = candidates[:n_results]

    return RetrievalTrace(candidates=candidates, final_chunks=final_chunks)



def retrieve(question: str,n_results: int = 3,use_reranking: bool = False,n_candidates: int = 20) -> list[RetrievedChunk]:
    candidate_count = n_candidates if use_reranking else n_results

    trace = retrieve_with_trace(question=question,n_results=n_results,use_reranking=use_reranking,n_candidates=candidate_count)

    return trace.final_chunks
