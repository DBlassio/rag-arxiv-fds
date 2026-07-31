#ChromaDB vector store module.

import chromadb
from langchain_core.documents import Document

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "arxiv_abstracts"


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_PATH)

# We create a collection with HNSW index and cosine similarity.

def get_or_create_collection(client: chromadb.ClientAPI):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}}, 
    )


def add_chunks(collection, chunks: list[Document], embeddings) -> None:
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))], # IDs for each chunk
        documents=[c.page_content for c in chunks], # Store the text content of each chunk
        embeddings=embeddings.tolist(), # Store the embeddings as a list of lists
        metadatas=[c.metadata for c in chunks]) # Store the metadata for each chunk

#This is our query function, retrieves the top n_results most similar documents to the query_embedding.
def query(collection, query_embedding, n_results: int = 3):
    return collection.query(query_embeddings=[query_embedding.tolist()],
                            n_results=n_results)