# Local Retrieval Evaluation

"""
Compares:
1. Bi-encoder top 20: initial candidate retrieval.
2. Bi-encoder top 3: original ranking without reranking.
3. Cross-encoder top 3: final ranking after reranking.
"""


import csv
import json
from pathlib import Path
from src.retrieval import retrieve_with_trace
from src.retrieval_metrics import calculate_ranking_metrics

#Paths
EVAL_PATH = Path("eval/eval_set_v2.json")
OUTPUT_PATH = Path("eval/retrieval_results_v2.csv")

#Config
N_RESULTS = 3
N_CANDIDATES = 20

METRIC_NAMES = ["recall_at_k","hit_rate_at_k","mrr_at_k",]


# Helper Functions

# Extract chunk_ids
def get_chunk_ids(chunks) -> list[str]:
    return [chunk.chunk_id for chunk in chunks]

# Extract document ids
def get_document_ids(chunks) -> list[str]:
    document_ids = []
    seen = set()

    for chunk in chunks:
        doc_id = chunk.arxiv_id

        if doc_id and doc_id not in seen:
            document_ids.append(doc_id)
            seen.add(doc_id)

    return document_ids

#Convert to JSON for CSV Storage
def json_list(values: list) -> str:
    return json.dumps(values, ensure_ascii=False)

# Function to add ranking metrics to an evaluation row
def add_ranking_metrics(row: dict,prefix: str,retrieved_ids: list[str],reference_ids: list[str],k: int,) -> None:
    
    metrics = calculate_ranking_metrics( retrieved_ids=retrieved_ids,reference_ids=reference_ids,k=k,)

    # For different k we input the value
    for metric_name, value in metrics.items():
        column_name = f"{prefix}_{metric_name.replace('_at_k', f'_at_{k}')}"
        row[column_name] = value



def main():

    # We load approved evaluation examples
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        all_examples = json.load(f)

    eval_set = [item for item in all_examples if item.get("review_status") == "approved"]

    if not eval_set:
        raise ValueError("No approved evaluation examples found.")

    print(f"Total examples: {len(all_examples)}")
    print(f"Approved examples: {len(eval_set)}")
    print(f"Candidate pool: {N_CANDIDATES}")
    print(f"Final context size: {N_RESULTS}")

    results = []

    # We start to evaluate every question
    for i, item in enumerate(eval_set, start=1):

        question = item["question"]

        reference_doc_ids = item.get("reference_doc_ids", [])
        reference_chunk_ids = item.get("reference_chunk_ids", [])

        if not reference_doc_ids or not reference_chunk_ids:
            raise ValueError(f"Missing reference IDs for question: {question}")

        print(f"\n[{i}/{len(eval_set)}] {question[:65]}...")

        # We retrieve 20 candidates and rerank them
        # So this produce both the initial ranking and final top 3
        trace = retrieve_with_trace(question=question, n_results=N_RESULTS, use_reranking=True, n_candidates=N_CANDIDATES)

        candidates = trace.candidates

        # Original bi-encoder rankings.
        bi_encoder_top20 = candidates[:N_CANDIDATES]
        bi_encoder_top3 = candidates[:N_RESULTS]

        # Final cross-encoder ranking.
        reranked_top3 = trace.final_chunks

        if not candidates:
            raise RuntimeError(f"No retrieval results for question: {question}")

        # We prpeare ID lists:
        rankings = {
            "bi_encoder_top20": bi_encoder_top20,
            "bi_encoder_top3": bi_encoder_top3,
            "reranked_top3": reranked_top3}

        row = {
            "question": question,
            "source_title": item.get("source_title", ""),
            "reference_doc_ids": json_list(reference_doc_ids),
            "reference_chunk_ids": json_list(reference_chunk_ids),

            # Initial ranking.
            "bi_encoder_top20_chunk_ids": json_list(get_chunk_ids(bi_encoder_top20)),
            "bi_encoder_top20_doc_ids": json_list(get_document_ids(bi_encoder_top20)),

            # Original top 3.
            "bi_encoder_top3_chunk_ids": json_list(get_chunk_ids(bi_encoder_top3)),
            "bi_encoder_top3_doc_ids": json_list(get_document_ids(bi_encoder_top3)),

            # Final top 3.
            "reranked_top3_chunk_ids": json_list(get_chunk_ids(reranked_top3)),
            "reranked_top3_doc_ids": json_list(get_document_ids(reranked_top3)),

            # Cross-encoder scores, in final ranking order.
            "reranked_top3_scores": json_list([chunk.rerank_score for chunk in reranked_top3])}

        # We calculate the document and chunk metrics
        for ranking_name, chunks in rankings.items():

            if ranking_name == "bi_encoder_top20":
                k = N_CANDIDATES
            else:
                k = N_RESULTS

            # Document evaluation.
            add_ranking_metrics(row=row,
                                prefix=f"{ranking_name}_doc",
                                retrieved_ids=get_document_ids(chunks),
                                reference_ids=reference_doc_ids,
                                k=k)

            # Chunk evaluation.
            add_ranking_metrics(row=row,
                                prefix=f"{ranking_name}_chunk",
                                retrieved_ids=get_chunk_ids(chunks),
                                reference_ids=reference_chunk_ids,
                                k=k)

        results.append(row)

        print(f"  Candidates: {len(candidates)} | "
              f"Bi-encoder top 3: {len(bi_encoder_top3)} | "
              f"Reranked top 3: {len(reranked_top3)}")

    # We save our results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    metric_columns = []

    for ranking_name, k in [
        ("bi_encoder_top20_doc", N_CANDIDATES),
        ("bi_encoder_top20_chunk", N_CANDIDATES),
        ("bi_encoder_top3_doc", N_RESULTS),
        ("bi_encoder_top3_chunk", N_RESULTS),
        ("reranked_top3_doc", N_RESULTS),
        ("reranked_top3_chunk", N_RESULTS)]:

        for metric_name in METRIC_NAMES:
            metric_columns.append(f"{ranking_name}_{metric_name.replace('_at_k', f'_at_{k}')}")

    base_columns = ["question","source_title","reference_doc_ids","reference_chunk_ids",
                    "bi_encoder_top20_chunk_ids","bi_encoder_top20_doc_ids","bi_encoder_top3_chunk_ids","bi_encoder_top3_doc_ids",
                    "reranked_top3_chunk_ids","reranked_top3_doc_ids","reranked_top3_scores"]

    fieldnames = base_columns + metric_columns

    with open(OUTPUT_PATH,"w",newline="",encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_PATH}")

    # We print our metrics

    print("\n========== RETRIEVAL SUMMARY ==========")

    summary_groups = [
        ("Bi-encoder top 20 / Document", "bi_encoder_top20_doc", 20),
        ("Bi-encoder top 20 / Chunk", "bi_encoder_top20_chunk", 20),
        ("Bi-encoder top 3 / Document", "bi_encoder_top3_doc", 3),
        ("Bi-encoder top 3 / Chunk", "bi_encoder_top3_chunk", 3),
        ("Reranked top 3 / Document", "reranked_top3_doc", 3),
        ("Reranked top 3 / Chunk", "reranked_top3_chunk", 3)]

    for label, prefix, k in summary_groups:
        
        print(f"\n{label}")
        
        for metric_name in METRIC_NAMES:
            column = (f"{prefix}_" f"{metric_name.replace('_at_k', f'_at_{k}')}")
            average = sum(row[column] for row in results) / len(results)
            print(f"  {metric_name.replace('_at_k', f'@{k}')}: {average:.3f}")

if __name__ == "__main__":
    main()
