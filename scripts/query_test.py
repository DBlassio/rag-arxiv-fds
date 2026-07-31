## This script is intended for manual testing of the query functionality.

from src.embeddings import embed_texts
from src.vectorstore import get_client, get_or_create_collection, query


def main():
    client = get_client()
    collection = get_or_create_collection(client)

    #Example query: "How do transformers handle long-range dependencies?"
    question = "How do transformers handle long-range dependencies?"
    print("Query:", question)
    #Then we embed the query using the same embedding model used for indexing.
    query_vec = embed_texts([question])[0]
    #We retrieve the top 3 most similar documents from the collection.
    results = query(collection, query_vec, n_results=3)


    #We extract the documents, metadata, and distances (cosine) from the results and print them.
    print("Retrieving top 3 results... \n")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"[distance={dist:.4f}] {meta.get('title', '?')}")
        print(doc[:150])
        print("---")

if __name__ == "__main__":
    main()