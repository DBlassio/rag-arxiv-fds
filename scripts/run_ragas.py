#This is our code to run the RAG pipeline over the eval set and score it with RAGAS.
import os
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


# Some of our metrics require us to evaluate whether relevant documents appear in the top-k results.
def calculate_retrieval_metrics(retrieved_chunks, reference_doc_ids, k=3):

    # Documents retrieved by our RAG system
    retrieved_doc_ids = [chunk.arxiv_id for chunk in retrieved_chunks[:k]]

    # Relevant documents from our evaluation dataset
    relevant_doc_ids = set(reference_doc_ids)

    # Documents that are both retrieved and relevant
    retrieved_relevant = [doc_id for doc_id in retrieved_doc_ids if doc_id in relevant_doc_ids]

    # Recall@K, its our proportion of relevant documents retrieved
    recall_at_k = (
        len(set(retrieved_relevant)) / len(relevant_doc_ids)
        if relevant_doc_ids
        else 0.0)

    # Hit Rate@K, did we retrieve at least one relevant document?
    hit_rate_at_k = 1.0 if retrieved_relevant else 0.0

    # MRR: reciprocal rank of the first relevant document
    reciprocal_rank = 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            reciprocal_rank = 1.0 / rank
            break

    return {
        f"recall_at_{k}": recall_at_k,
        f"hit_rate_at_{k}": hit_rate_at_k,
        "mrr": reciprocal_rank}


# This function scores a single example from the evaluation set. It retrieves relevant chunks, generates an answer, and computes various metrics.
async def score_example(item: dict, metrics: dict, use_reranking: bool) -> dict:

    chunks = retrieve(item["question"], use_reranking=use_reranking)

    retrieval_metrics = calculate_retrieval_metrics(retrieved_chunks=chunks,
                                                    reference_doc_ids=item["reference_doc_ids"],
                                                    k=3)

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
        **retrieval_metrics, # our retrieval metrics (recall@k, hit rate@k, MRR)
    }


async def run_eval(eval_set: list[dict],metrics: dict,use_reranking: bool,label: str,output_path: str,) -> list[dict]:

    # We check if there are previously saved results
    existing_results = load_existing_results(output_path)
    results_by_question = {result["question"]: result for result in existing_results}
    print(f"[{label}] Previously completed: {len(results_by_question)}")


    for i, item in enumerate(eval_set):
        question = item["question"]

        # Skip questions that have already been evaluated
        if question in results_by_question:
            print(f"  [{label}] [{i+1}/{len(eval_set)}] Already saved. Skipping.")
            continue

        print(f"  [{label}] [{i+1}/{len(eval_set)}] {question[:50]}...")

        result = await score_example(item,metrics,use_reranking)
        results_by_question[question] = result

        # Preserve the original evaluation order
        ordered_results = [results_by_question[item["question"]] for item in eval_set if item["question"] in results_by_question]

        # Save after every completed question
        save_csv(ordered_results, output_path)
        print(f"  Checkpoint saved: {len(ordered_results)}/{len(eval_set)}")

    return [results_by_question[item["question"]] for item in eval_set]


CSV_FIELDS = [
    "question", "answer", "contexts",
    "faithfulness", "answer_relevancy","context_precision", "context_recall",
    "recall_at_3", "hit_rate_at_3", "mrr"]

NUMERIC_FIELDS = [
    "faithfulness", "answer_relevancy","context_precision", "context_recall",
    "recall_at_3", "hit_rate_at_3", "mrr"]


#Function to load previously saved evaluation results, if they exist. 
def load_existing_results(path: str) -> list[dict]:

    if not os.path.exists(path):
        return []

    with open(path, "rb") as f:
        has_utf8_bom = f.read(3) == b"\xef\xbb\xbf"

    encoding = "utf-8-sig" if has_utf8_bom else "cp1252"

    with open(path, "r", encoding=encoding, newline="") as f:
        results = list(csv.DictReader(f))

    for result in results:
        for field in NUMERIC_FIELDS:
            result[field] = float(result[field])

    return results

#Funtion to save evaluation results to a CSV file.
def save_csv(results: list[dict], path: str) -> None:
    
    temp_path = f"{path}.tmp"

    with open(temp_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    # Replace the previous checkpoint only after the new file is complete.
    os.replace(temp_path, path)



# Main function to run the evaluation pipeline.
async def main():
    with open("eval/eval_set_v2.json", "r", encoding="utf-8") as f:
        all_examples = json.load(f)

    eval_set = [item for item in all_examples if item.get("review_status") == "approved"] #We only want to evaluate examples that have been approved manually (human approval).

    print(f"Total examples: {len(all_examples)}")
    print(f"Approved examples: {len(eval_set)}")
    print(f"Skipped examples: {len(all_examples) - len(eval_set)}")

    if not eval_set:
        raise ValueError("No approved examples found in eval_set_v2.json")


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

    #Running baseline 
    baseline = await run_eval(eval_set,metrics,use_reranking=False,label="baseline",output_path="eval/ragas_results_v2_baseline.csv")

    #Running reranked
    reranked = await run_eval(eval_set,metrics,use_reranking=True,label="reranked",output_path="eval/ragas_results_v2_reranked.csv")

    #Saving results to CSV
    #save_csv(baseline, "eval/ragas_results_v2_baseline.csv")
    #save_csv(reranked, "eval/ragas_results_v2_reranked.csv")

    print("\n--- Comparison between baseline and reranked ---")
    for metric in ["faithfulness","answer_relevancy","context_precision","context_recall",
        "recall_at_3","hit_rate_at_3","mrr",]:

        b = sum(r[metric] for r in baseline) / len(baseline)
        r = sum(r[metric] for r in reranked) / len(reranked)

        #Variation between rerranked and baseline
        delta = r - b

        print(
            f"{metric}: {b:.3f} -> {r:.3f} "
            f"({'+' if delta >= 0 else ''}{delta:.3f})"
        )


if __name__ == "__main__":
    asyncio.run(main())