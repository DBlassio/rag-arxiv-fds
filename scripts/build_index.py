# Our Orchestration to run the entire pipeline: fetch -> chunk -> embed -> index into ChromaDB.

from src.ingestion import fetch_arxiv_papers
from src.chunking import chunk_documents
from src.embeddings import embed_texts
from src.vectorstore import get_client, get_or_create_collection, add_chunks, reset_collection


#Since ChromaDB only accepts scalar metadata, we need to serialize any list metadata into strings.
def _serialize_metadata(chunks):
    """Chroma solo acepta metadata escalar — listas se convierten a string."""
    for chunk in chunks:
        for key, value in chunk.metadata.items():
            if isinstance(value, list):
                chunk.metadata[key] = ", ".join(value)
    return chunks


def main(max_results: int = 500, reset: bool = True):
    client = get_client()

    if reset:
        print("0. Resetting collection...")
        reset_collection(client)

    print(f"1. Fetching {max_results} papers...")
    docs = fetch_arxiv_papers(max_results=max_results, batch_size=100)

    print("2. Chunking...")
    chunks = chunk_documents(docs)
    chunks = _serialize_metadata(chunks)

    print("3. Embedding...")
    texts = [c.page_content for c in chunks]
    embeddings = embed_texts(texts)

    print("4. Indexing into ChromaDB...")
    collection = get_or_create_collection(client)
    add_chunks(collection, chunks, embeddings)

    print(f"Done. Collection has {collection.count()} chunks.")


if __name__ == "__main__":
    main()