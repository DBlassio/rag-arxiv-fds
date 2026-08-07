#This is our code to run the RAG pipeline over the eval set and score it with RAGAS.

import asyncio
import csv
import json
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecisionWithReference, ContextRecall
from src.retrieval import retrieve
from src.generation import generate_answer

load_dotenv()
JUDGE_MODEL = "claude-sonnet-4-6"


async def score_example(item: dict, metrics: dict) -> dict:
    chunks = retrieve(item["question"], n_results=3)
    answer = generate_answer(item["question"], chunks)
    contexts = [c.text for c in chunks]

    faithfulness = await metrics["faithfulness"].ascore(user_input=item["question"], response=answer, retrieved_contexts=contexts)
    relevancy = await metrics["answer_relevancy"].ascore(user_input=item["question"], response=answer)
    precision = await metrics["context_precision"].ascore(user_input=item["question"], reference=item["ground_truth"], retrieved_contexts=contexts)
    recall = await metrics["context_recall"].ascore(user_input=item["question"], retrieved_contexts=contexts, reference=item["ground_truth"])

    return {
        "question": item["question"],
        "answer": answer,
        "contexts": " | ".join(contexts),
        "faithfulness": faithfulness.value,
        "answer_relevancy": relevancy.value,
        "context_precision": precision.value,
        "context_recall": recall.value,
    }


async def main():
    with open("eval/eval_set.json") as f:
        eval_set = json.load(f)

    llm = llm_factory(JUDGE_MODEL, provider="anthropic", client=AsyncAnthropic())
    llm.model_args.pop("top_p", None)
    embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")

    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    print(f"Scoring {len(eval_set)} eval questions...")
    results = []
    for i, item in enumerate(eval_set):
        print(f"  [{i+1}/{len(eval_set)}] {item['question'][:60]}...")
        results.append(await score_example(item, metrics))

    keys = ["question", "answer", "contexts", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    with open("eval/ragas_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print("\n--- Baseline (promedio del eval set) ---")
    for metric in keys[3:]: 
        avg = sum(r[metric] for r in results) / len(results)
        print(f"{metric}: {avg:.3f}")


if __name__ == "__main__":
    asyncio.run(main())