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


async def score_example(item: dict, metrics: dict, use_reranking: bool) -> dict:

    chunks = retrieve(item["question"], use_reranking=use_reranking)
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


async def run_eval(eval_set: list[dict], metrics: dict, use_reranking: bool, label: str) -> list[dict]:
    results = []
    for i, item in enumerate(eval_set):
        print(f"  [{label}] [{i+1}/{len(eval_set)}] {item['question'][:50]}...")
        results.append(await score_example(item, metrics, use_reranking))
    return results


def save_csv(results: list[dict], path: str) -> None:
    keys = ["question", "answer", "contexts", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


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

    print(f"Scoring {len(eval_set)} eval questions for 2 configurations...")
    baseline = await run_eval(eval_set, metrics, use_reranking=False, label="baseline")
    reranked = await run_eval(eval_set, metrics, use_reranking=True, label="reranked")

    save_csv(baseline, "eval/ragas_results_baseline.csv")
    save_csv(reranked, "eval/ragas_results_reranked.csv")

    print("\n--- Comparison between baseline and reranked ---")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        b = sum(r[metric] for r in baseline) / len(baseline)
        r = sum(r[metric] for r in reranked) / len(reranked)
        delta = r - b
        print(f"{metric}: {b:.3f} -> {r:.3f}  ({'+' if delta >= 0 else ''}{delta:.3f})")


if __name__ == "__main__":
    asyncio.run(main())