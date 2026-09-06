"""Generation module: builds a grounded prompt and calls Claude."""
#Our Generation module
#It takes a question and the retrieved chunks of text, builds a prompt, and calls the Claude model to generate an answer based on the context.

import os
import boto3
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from src.retrieval import RetrievedChunk

#We load our env credentials
load_dotenv()  

#Our LLM model 
MODEL_NAME = "claude-haiku-4-5-20251001"

#Our prompt for the LLM model, instructing it to answer questions based only on the provided context from arXiv abstracts.
SYSTEM_PROMPT = """You are a research assistant answering questions using ONLY the provided context from arXiv abstracts.

Rules:
- Base your answer strictly on the provided context.
- If the context does not contain enough information to answer, say so explicitly.
- Cite which paper (by title) supports each claim."""

# We cache the API key to avoid repeated retrievals from environment variables or AWS Secrets Manager.
_api_key_cache: str | None = None

def _get_api_key() -> str:
    global _api_key_cache
    if _api_key_cache is not None:
        return _api_key_cache
    env_key = os.getenv("ANTHROPIC_API_KEY")
    if env_key:
        _api_key_cache = env_key
    else:
        secret_name = os.getenv("ANTHROPIC_API_KEY_SECRET_NAME", "rag-arxiv/anthropic-api-key")
        _api_key_cache = boto3.client("secretsmanager").get_secret_value(SecretId=secret_name)["SecretString"]
    return _api_key_cache



#We build our prompt by combining the retrieved chunks of text into a single context string, and appending the user's question. The prompt is then sent to the Claude model for generating an answer.
#First the most relevant
def build_context(chunks: list[RetrievedChunk]) -> str:
    parts = [f"[Source {i+1}: {c.title}]\n{c.text}" for i, c in enumerate(chunks)]
    return "\n\n".join(parts)

def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context(chunks)
    llm = ChatAnthropic(model=MODEL_NAME, max_tokens=500, temperature=0, api_key=_get_api_key())

    prompt = f"""Context: {context}   
    
    Question: {question}
    
    Answer based only on the context above:"""

    response = llm.invoke([("system", SYSTEM_PROMPT),("human", prompt)])

    return response.content