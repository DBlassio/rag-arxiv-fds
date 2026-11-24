# In this script, we sample a number of indexed chunks from the vector store, and for each chunk, 
# we generate a question-answer pair using a more capable LLM (Claude Sonnet 5).
# The generated pairs are saved to a JSON file for evaluation purposes.

import json
import random
from langchain_anthropic import ChatAnthropic
from src.vectorstore import get_client, get_or_create_collection
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "claude-sonnet-5"  # more capable than our embedding model, so we can generate better questions and answers
N_SAMPLES = 20

GEN_PROMPT = """Given this abstract, write ONE specific factual question that can be
answered using ONLY this abstract, and provide the exact answer based on the abstract.

Abstract:
{text}

Respond in this exact format:
QUESTION: <question>
ANSWER: <answer>"""


#We get our sample of chunks from the vector store.
def sample_chunks(n: int, seed: int=42) -> list[dict]:
    client = get_client()
    collection = get_or_create_collection(client)
    all_ids = collection.get()["ids"]

    if not all_ids:
        return []
    
    rng = random.Random(seed) #For further reproducibility, we can set a seed for the random sampling.

    sampled_ids = rng.sample(all_ids, min(n, len(all_ids)))
    result = collection.get(ids=sampled_ids)

    sampled_chunks = []

    for chunk_id, text, metadata in zip(result["ids"], result["documents"], result["metadatas"]):
        arxiv_id = metadata.get("arxiv_id", "")
        if not arxiv_id:
            raise ValueError(f"Chunk {chunk_id} is missing arxiv_id in metadata. Skipping.")
        sampled_chunks.append({"chunk_id": chunk_id, 
                               "arxiv_id": arxiv_id,
                               "title": metadata.get("title", ""),
                               "text": text})
    return sampled_chunks

#We generate the evaluation set by invoking the LLM on each sampled chunk.
def generate_eval_set(n: int = N_SAMPLES, seed: int = 42) -> list[dict]:

    llm = ChatAnthropic(model=MODEL_NAME, max_tokens=300)
    eval_set = []

    for chunk in sample_chunks(n, seed=seed):
        response = llm.invoke(GEN_PROMPT.format(text=chunk["text"])).content
        question = answer = None
        for line in response.split("\n"):
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
        if question and answer:
            eval_set.append({
                #Original evaluation fields
                "question": question,
                "ground_truth": answer,
                "source_title": chunk["title"],

                #Retrieval ground truth
                "reference_doc_ids": [chunk["arxiv_id"]],
                "reference_chunk_ids": [chunk["chunk_id"]],

                #Original evidence
                "reference_context": chunk["text"],

                #Evaluation metadata
                "is_answerable": True,
                "question_type": "single_document",
                "review_status": "pending"
            })
            
    return eval_set


if __name__ == "__main__":
    eval_set = generate_eval_set(seed=42)
    output_path = "eval/eval_set_v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(eval_set)} evaluation pairs "f"-> {output_path}")