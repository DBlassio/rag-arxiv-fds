#This module contains functions to calculate local retrieval metrics. (Do not require an LLM or an external API - Costs)


#Function to calcuate Recall@k, Hit Rate@k and MRR@k
def calculate_ranking_metrics(retrieved_ids: list[str], reference_ids: list[str], k: int = 3) -> dict[str, float]:
    # We first validate the input and our relevant_ids
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant_ids = set(reference_ids)

    if not relevant_ids:
        raise ValueError("reference_ids cannot be empty.")

    if any(not chunk_id for chunk_id in retrieved_ids):
        raise ValueError("retrieved_ids cannot contain empty IDs.")

    # We keep only the first K retrieved results.
    top_k_ids = retrieved_ids[:k]

    # Then we identify the relevant IDs found in the top K results.
    retrieved_relevant = set(top_k_ids) & relevant_ids

    # Recall@K: proportion of relevant IDs that were retrieved.
    recall_at_k = len(retrieved_relevant) / len(relevant_ids)

    # Hit Rate@K: did we retrieve at least one relevant ID?
    hit_rate_at_k = 1.0 if retrieved_relevant else 0.0

    # MRR@K: reciprocal rank of the first relevant result.
    reciprocal_rank = 0.0

    for rank, retrieved_id in enumerate(top_k_ids, start=1):
        if retrieved_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    return {"recall_at_k": recall_at_k,
            "hit_rate_at_k": hit_rate_at_k,
            "mrr_at_k": reciprocal_rank}
