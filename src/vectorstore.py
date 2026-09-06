"""Vectorstore module: ChromaDB persistent client wrapper."""

import chromadb
from langchain_core.documents import Document

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "arxiv_abstracts"


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection(client: chromadb.ClientAPI):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )


def add_chunks(collection, chunks: list[Document], embeddings) -> None:
    collection.add(
        ids=[make_chunk_id(c) for c in chunks],
        documents=[c.page_content for c in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[c.metadata for c in chunks],
    )


def query(collection, query_embedding, n_results: int = 3):
    return collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)


def make_chunk_id(chunk) -> str:
    safe_id = chunk.metadata["arxiv_id"].split("/")[-1]
    return f"{safe_id}_{chunk.metadata['chunk_index']}"


def reset_collection(client) -> None:
    """Borra la colección si existe, para reconstruir limpio."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass