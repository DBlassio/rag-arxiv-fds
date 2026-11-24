from scripts.generate_eval_set import sample_chunks

chunks = sample_chunks(3, seed=42)

for chunk in chunks:
    print(
        chunk["chunk_id"],
        chunk["arxiv_id"],
        chunk["title"]
    )

first_run = [
    c["chunk_id"]
    for c in sample_chunks(3, seed=42)
]

second_run = [
    c["chunk_id"]
    for c in sample_chunks(3, seed=42)
]

assert first_run == second_run

print("Sampling is reproducible!")