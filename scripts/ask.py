#End to end RAG query script:
# Takes a question, retrieves relevant chunks of text from the vector store, and generates an answer using the Claude model.

from src.retrieval import retrieve
from src.generation import generate_answer


def main():

    # Query
    question = input("Question: ").strip()
    if not question:
        question = "How do transformers handle long-range dependencies?" #Our fallback question 

    #Retrieval 
    chunks = retrieve(question, n_results=3)

    print("\n--- Retrieved Chunks ---")
    for c in chunks:
        print(f"[distance={c.distance:.4f}] {c.title}")

    #Generation
    answer = generate_answer(question, chunks)

    print("\n--- Response ---")
    print(answer)


if __name__ == "__main__":
    main()