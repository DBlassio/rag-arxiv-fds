from fastapi import FastAPI
from pydantic import BaseModel
from mangum import Mangum

from src.retrieval import retrieve
from src.generation import generate_answer

app = FastAPI(title="arXiv Research Assistant")


class AskRequest(BaseModel):
    question: str
    use_reranking: bool = True


class Source(BaseModel):
    title: str
    arxiv_id: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    chunks = retrieve(request.question, use_reranking=request.use_reranking)
    answer = generate_answer(request.question, chunks)
    return AskResponse(
        answer=answer,
        sources=[Source(title=c.title, arxiv_id=c.arxiv_id, distance=c.distance) for c in chunks])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


handler = Mangum(app)