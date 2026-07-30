#Ingestion Module: Fetches papers from arXiv as LangChain Documents.
import time
import requests
import xml.etree.ElementTree as ET
from langchain_core.documents import Document

ARXIV_API_URL = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_arxiv_papers(query: str = "cat:cs.CL",max_results: int = 50,batch_size: int = 20,) -> list[Document]:
    """
    Function to retrieve papers from arXiv based on a search query 
    and return them as a list of LangChain Documents.
    """

    documents = []
    start = 0

    while start < max_results:
        batch = min(batch_size, max_results - start)
        params = {"search_query": query, "start": start, "max_results": batch}
        response = requests.get(ARXIV_API_URL, params=params, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        entries = root.findall("atom:entry", NS)
        if not entries:
            break  # se acabaron los resultados disponibles

        documents.extend(_parse_entry(e) for e in entries)
        start += batch
        if start < max_results:
            time.sleep(3)

    return documents

#Auxiliar function to extract relevant information from an arXiv entry and convert it into a LangChain Document.
def _parse_entry(entry: ET.Element) -> Document:
    title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
    abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
    arxiv_id = entry.find("atom:id", NS).text.strip()
    published = entry.find("atom:published", NS).text.strip()
    authors = [a.find("atom:name", NS).text for a in entry.findall("atom:author", NS)]
    categories = [c.attrib["term"] for c in entry.findall("atom:category", NS)]

    return Document(
        page_content=abstract,
        metadata={
            "title": title,
            "authors": authors,
            "arxiv_id": arxiv_id,
            "published": published,
            "categories": categories,
            "source": "arxiv",
        },
    )


if __name__ == "__main__":
    docs = fetch_arxiv_papers(max_results=10)
    print(f"Fetched {len(docs)} documents")
    print(docs[0].page_content[:200])
    print(docs[0].metadata)