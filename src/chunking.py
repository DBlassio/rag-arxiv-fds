# Chunking module: recursive splitting aware of the tokenizer used for embeddings
# This is important because the embedding model's tokenizer may split text differently 
# than a simple character count, which can affect the quality of embeddings and retrieval.

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer
from src.ingestion import fetch_arxiv_papers


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

#Our splitter function to create a RecursiveCharacterTextSplitter based on the tokenizer of the embedding model.
def get_splitter(chunk_size: int = 400, chunk_overlap: int = 0) -> RecursiveCharacterTextSplitter:

    """
    We set overlap = 0 by default: late evidence does not show consistent benefit.
    We test it empirically with RAGAS, instead of assuming the 10-20% "by hand".
    """
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Our actual chunking function that takes a list of LangChain Documents and splits them into smaller chunks 
def chunk_documents(documents: list[Document], chunk_size: int = 400, chunk_overlap: int = 0) -> list[Document]:
    splitter = get_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i  # trazabilidad: qué posición ocupa dentro de su doc

    return chunks


if __name__ == "__main__":

    docs = fetch_arxiv_papers(max_results=10)
    chunks = chunk_documents(docs)

    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    for c in chunks[:3]:
        print("---")
        print(c.page_content[:150])
        print(c.metadata)