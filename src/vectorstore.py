# ChromaDB vector store module.

from pathlib import Path
import shutil

import chromadb
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Original database bundled inside the Docker image.
# Lambda's /var/task filesystem is read-only at runtime.
SOURCE_CHROMA_PATH = Path("/var/task/chroma_db")

# Writable directory available during Lambda execution.
RUNTIME_CHROMA_PATH = Path("/tmp/chroma_db")

COLLECTION_NAME = "arxiv_abstracts"


# ---------------------------------------------------------------------------
# Chroma initialization
# ---------------------------------------------------------------------------

def prepare_chroma() -> None:
    """
    Copy the read-only Chroma database from the Docker image to /tmp.

    The copy is performed only once per Lambda execution environment.
    """

    if RUNTIME_CHROMA_PATH.exists():
        return

    if not SOURCE_CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"Chroma database not found at {SOURCE_CHROMA_PATH}"
        )

    temp_path = Path("/tmp/chroma_db_init")

    if temp_path.exists():
        shutil.rmtree(temp_path)

    shutil.copytree(
        SOURCE_CHROMA_PATH,
        temp_path,
    )

    temp_path.rename(RUNTIME_CHROMA_PATH)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_client = None


def get_client() -> chromadb.ClientAPI:
    """
    Return a reusable Chroma client for this Lambda execution environment.
    """

    global _client

    if _client is None:
        prepare_chroma()

        _client = chromadb.PersistentClient(
            path=str(RUNTIME_CHROMA_PATH)
        )

    return _client


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def get_or_create_collection(client: chromadb.ClientAPI):
    """
    Create or retrieve the arXiv collection using cosine similarity.
    """

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

def add_chunks(
    collection,
    chunks: list[Document],
    embeddings,
) -> None:

    collection.add(
        ids=[make_chunk_id(c) for c in chunks],
        documents=[c.page_content for c in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[c.metadata for c in chunks],
    )


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query(
    collection,
    query_embedding,
    n_results: int = 3,
):
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
    )


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------

def make_chunk_id(chunk) -> str:
    safe_id = chunk.metadata["arxiv_id"].split("/")[-1]
    return f"{safe_id}_{chunk.metadata['chunk_index']}"


# ---------------------------------------------------------------------------
# Development utility
# ---------------------------------------------------------------------------

def reset_collection(client) -> None:
    """
    Useful only for local development/re-indexing.
    """

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass