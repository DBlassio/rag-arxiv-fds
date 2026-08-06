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


#Our retrieve function takes a question and an optional number of results to return

def retrieve(question: str, n_results: int = 3) -> list[RetrievedChunk]:

    query_vec = embed_texts([question])[0]
    client = get_client()
    collection = get_or_create_collection(client)

    #We retrieve the most relevant chunks of text from the vector store using the query vector
    results = vs_query(collection, query_vec, n_results=n_results)

    chunks = []

    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        chunks.append(RetrievedChunk(
            text=doc,
            title=meta.get("title", "unknown"),
            arxiv_id=meta.get("arxiv_id", ""),
            distance=dist))

    return chunks 